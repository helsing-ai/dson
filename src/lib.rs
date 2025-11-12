// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! # DSON: A Delta-State CRDT for JSON-like Data Structures
//!
//! This crate provides a Rust implementation of **DSON**, a space-efficient,
//! delta-state Conflict-Free Replicated Datatype (CRDT) for JSON-like data structures.
//! It is based on the research paper ["DSON: JSON CRDT Using Delta-Mutations For Document Stores"][paper]
//! and inspired by the original author's [JavaScript implementation][js-impl].
//!
//! The primary goal of this library is to enable robust, and efficient
//! multi-writer collaboration in extremely constrained environments (high
//! latency and low bandwidth; opportunistic networking).
//!
//! Unlike other CRDT libraries that expose a single "Document" type, DSON provides a set of
//! composable primitives. This allows you to build the exact data structure you need. The most
//! common top-level structure is an [`OrMap`], which can contain other CRDTs, enabling nested,
//! JSON-like objects. The entire state is typically wrapped in a [`CausalDotStore`], which
//! tracks the causal history.
//!
//! [paper]: https://dl.acm.org/doi/10.14778/3510397.3510403
//! [oppnet]: https://hal.science/hal-03405138/document "Frédéric Guidec, Yves Mahéo, Camille Noûs. Delta-State-Based Synchronization of CRDTs in Opportunistic Networks. In 2021 IEEE 46th Conference on Local Computer Networks (LCN). doi:10.1109/LCN52139.2021.9524978"
//! [js-impl]: https://github.com/crdt-ibm-research/json-delta-crdt
//!
//! ## Core Concepts
//!
//! DSON provides three fundamental, composable CRDTs:
//!
//! - [`OrMap`]: An **Observed-Remove Map**, mapping arbitrary keys to other CRDT values.
//! - [`OrArray`]: An **Observed-Remove Array**, providing a list-like structure.
//! - [`MvReg`]: A **Multi-Value Register**, for storing primitive values. When
//!   concurrent writes occur, the register holds all conflicting values. This is
//!   the only CRDT in this library that can represent value conflicts.
//!
//! These primitives can be nested to create arbitrarily complex data structures, such as a map
//! containing an array of other maps.
//!
//! All modifications produce a **delta**. Instead of sending the entire state after each
//! change, only this small delta needs to be transmitted to other replicas.
//!
//! ## Observed-Remove Semantics
//!
//! DSON uses **Observed-Remove (OR)** semantics for its collections. This means
//! an element can only be removed if its addition has been observed. If an
//! element is updated concurrently with its removal, the update "wins," and the
//! element is preserved. OR-semantics are intuitive, and this is often the
//! desired behavior.
//! Consider a collaborative shopping list:
//!
//! 1. **Initial State**: Both Alice and Bob see `["apples", "bananas"]`.
//! 2. **Alice's Action**: Alice updates "bananas" to "blueberries".
//! 3. **Bob's Action**: Concurrently, Bob removes "bananas".
//!
//! With OR-semantics, the final list will be `["apples", "blueberries"]`. Bob's removal
//! is overridden by Alice's concurrent update because the update implies the continued
//! existence of the item.
//!
//! DSON can be extended with special CRDTs providing different semantics
//! for specific use cases though.
//!
//! ## Causal CRDTs and Tombstone-Free Removals
//!
//! DSON is a **causal** CRDT, meaning it uses causal history to resolve conflicts.
//! This history is tracked in a [`CausalContext`], which contains a set of "dots"—unique
//! identifiers for every operation.
//!
//! ### Dots
//!
//! A **dot** is a globally unique identifier for an operation (for example, adding or updating a
//! value). It is the fundamental unit for tracking causality.
//!
//! A [`Dot`] is a tuple `(Identifier, Sequence)`:
//!
//! - **[`Identifier`]**: A unique ID for the actor (a specific application instance on a
//!   specific node) that performed the operation. It is composed of a `NodeId` and an
//!   `ApplicationId`. This structure allows multiple applications on the same machine to
//!   collaborate without their histories conflicting.
//! - **`Sequence`**: A monotonically increasing number (effectively a Lamport timestamp)
//!   that is unique to that actor.
//!
//! When a replica makes a change, it generates a new dot. This dot is broadcast to other
//! replicas along with the **delta** describing the change.
//!
//! The collection of all dots a replica has observed forms its [`CausalContext`]. This
//! context represents the replica's knowledge of the document's history. By comparing its
//! local `CausalContext` with the context from a received delta, a replica can determine
//! which operations are new, which are concurrent, and which have already been seen. This
//! allows DSON to merge changes correctly and guarantee convergence.
//!
//! A key advantage of this model is the elimination of **tombstones**. In many other
//! CRDTs, when an item is deleted, a "tombstone" marker is left behind to signify
//! its removal. These tombstones are never garbage-collected and can cause unbounded
//! metadata growth in long-lived documents.
//!
//! DSON avoids this growth by tracking which operations are "live". A removal is simply the
//! absence of an operation's dot from the causal context. When replicas sync, they
//! can determine which items have been deleted by comparing their causal contexts,
//! without needing explicit tombstone markers. This ensures that the metadata size
//! remains proportional to the size of the live data, not the entire history of
//! operations.
//!
//! ## Scope of this Crate
//!
//! This crate provides the core data structures and algorithms for DSON. It is
//! responsible for generating deltas from mutations and merging them to ensure
//! eventual consistency. It is up to you to build your document structure by
//! composing the provided CRDT primitives, most commonly by wrapping an [`OrMap`]
//! in a [`CausalDotStore`].
//!
//! Note that this is a low-level library. You will likely want to build a
//! typed abstraction layer on top of `dson` rather than use it directly in your
//! application code.
//!
//! **It does not include any networking protocols.**
//!
//! You are responsible for implementing the transport layer to broadcast deltas
//! to other replicas. The correctness of this library, particularly its
//! **causal consistency** guarantees, relies on the transport layer delivering
//! deltas in an order that respects the causal history of events. This is typically
//! achieved with an anti-entropy algorithm that exchanges deltas and their
//! causal metadata ([`CausalContext`]).
//!
//! ## Getting Started: A Simple Conflict
//!
//! This example demonstrates how two users (Alice and Bob) concurrently edit the same
//! data, creating a conflict that DSON resolves gracefully using the transaction API.
//!
//! ```rust
//! use dson::{
//!     crdts::{mvreg::MvRegValue, snapshot::ToValue},
//!     CausalDotStore, Identifier, OrMap,
//! };
//!
//! // 1. SETUP: TWO REPLICAS
//! // Create two replicas, Alice and Bob, each with a unique ID.
//! let alice_id = Identifier::new(0, 0);
//! let mut alice_store = CausalDotStore::<OrMap<String>>::default();
//!
//! let bob_id = Identifier::new(1, 0);
//! let mut bob_store = CausalDotStore::<OrMap<String>>::default();
//!
//! // 2. INITIAL STATE
//! // Alice creates an initial value using the transaction API.
//! let key = "document".to_string();
//! let delta_from_alice = {
//!     let mut tx = alice_store.transact(alice_id);
//!     tx.write_register(&key, MvRegValue::String("initial value".to_string()));
//!     tx.commit()
//! };
//!
//! // 3. SYNC
//! // Bob receives Alice's initial change.
//! bob_store.join_or_replace_with(delta_from_alice.0.store, &delta_from_alice.0.context);
//! assert_eq!(alice_store, bob_store);
//!
//! // 4. CONCURRENT EDITS
//! // Now Alice and Bob make changes without syncing.
//!
//! // Alice updates the value to "from Alice".
//! let delta_alice_edit = {
//!     let mut tx = alice_store.transact(alice_id);
//!     tx.write_register(&key, MvRegValue::String("from Alice".to_string()));
//!     tx.commit()
//! };
//!
//! // Concurrently, Bob updates the value to "from Bob".
//! let delta_bob_edit = {
//!     let mut tx = bob_store.transact(bob_id);
//!     tx.write_register(&key, MvRegValue::String("from Bob".to_string()));
//!     tx.commit()
//! };
//!
//! // 5. MERGE
//! // The replicas exchange their changes.
//! alice_store.join_or_replace_with(delta_bob_edit.0.store, &delta_bob_edit.0.context);
//! bob_store.join_or_replace_with(delta_alice_edit.0.store, &delta_alice_edit.0.context);
//!
//! // After merging, both stores are identical.
//! assert_eq!(alice_store, bob_store);
//!
//! // 6. VERIFY CONFLICT
//! // The concurrent writes are preserved as a conflict in the register.
//! // The transaction API exposes this through the CrdtValue enum.
//! use dson::transaction::CrdtValue;
//!
//! let tx = alice_store.transact(alice_id);
//! match tx.get(&key) {
//!     Some(CrdtValue::Register(reg)) => {
//!         // Read all concurrent values
//!         let values: Vec<_> = reg.values().into_iter().collect();
//!         assert_eq!(values.len(), 2);
//!         assert!(values.contains(&&MvRegValue::String("from Alice".to_string())));
//!         assert!(values.contains(&&MvRegValue::String("from Bob".to_string())));
//!     }
//!     _ => panic!("Expected register with conflict"),
//! }
//! ```
//!
//! For more examples of the transaction API, including nested structures and performance
//! considerations, see the [`transaction`] module documentation.
//!
//! ## Advanced Topics
//!
//! ### The Extension System
//!
//! DSON includes an extension system that allows developers to define custom CRDTs by
//! implementing the [`ExtensionType`] trait. This is for building domain-specific data
//! structures that go beyond the standard JSON-like primitives.
//!
//! By implementing the [`ExtensionType`] trait, you define how your custom type should be
//! serialized, deserialized, and merged. The system handles conflict resolution based on
//! the rules you define.
//!
//! This can be used to implement custom data structures like counters, text objects, or
//! more efficient state representation.
//!
//! ### Validation and Observation
//!
//! DSON provides a [`Sentinel`](crate::sentinel::Sentinel) trait that allows you to observe or
//! validate changes as they are applied during a merge. This can be used for implementing
//! authorization, logging, or triggering side effects.
//!
//! ## Network and Consistency
//!
//! DSON's delta-based approach minimizes the amount of data that needs to be transmitted
//! between replicas, making it efficient for low-bandwidth or high-latency networks.
//!
//! However, much of the complexity of using DSON in practice lies in the correct design and
//! implementation of the gossip protocol used to exchange deltas between replicas. An
//! efficient gossip protocol is not trivial to implement. For guidance, refer to the
//! research on [opportunistic networking (oppnet)][oppnet].
//!
//! It is also important to understand that DSON's causal consistency guarantees are provided on
//! a per-register basis. This means that while individual values are guaranteed to be causally
//! consistent, the relationships between different values are not. This can lead to very
//! unintuitive behavior.
//! For example, if you have two registers, `x` and `y`, you write to `x` and then to `y`,
//! another replica might see the write to `y` before the write to `x`.
//!
//! ## License
//!
//! This project is licensed under either of
//!
//! - Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
//! - MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
//!
//! at your option.
//!
//! ## Features
//!
//! - `json`: Enables serialization and deserialization of DSON documents to and from
//!   `serde_json::Value`. This feature is enabled by default.
//! - `serde`: Provides `serde` support for all CRDT types.
//! - `arbitrary`: Implements `quickcheck::Arbitrary` for CRDT types, useful for property-based testing.
//! - `chrono`: Enables `chrono` support for `Timestamp`. This feature is enabled by default.
//! - `ulid`: Enables registers to hold ulids. This feature is enabled by default.
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

use ahash::RandomState;
use std::{
    fmt,
    hash::BuildHasher,
    ops::BitAnd,
    sync::atomic::{AtomicBool, Ordering},
};

// Use a constant seed for hashing to make performance benchmarks have less variance.
pub(crate) const DETERMINISTIC_HASHER: RandomState = RandomState::with_seeds(48, 1516, 23, 42);

pub mod causal_context;
pub use causal_context::{
    CausalContext, Dot, Identifier, MAX_APPLICATION_ID, NodeId, Priority, ROOT_APP_ID,
};
mod dotstores;
pub use dotstores::{
    CausalDotStore, DotChange, DotFun, DotFunMap, DotFunValueIter, DotMap, DotStore, DotStoreJoin,
    DryJoinOutput,
};
pub mod crdts;
pub use crdts::{mvreg::MvReg, orarray::OrArray, ormap::OrMap};
pub mod api;
/// Transaction-based API for ergonomic CRDT mutations.
///
/// See [`transaction`] module documentation for details and examples.
pub mod transaction;
pub use transaction::Delta;
#[cfg(feature = "chrono")]
pub mod datetime_literal;
pub mod either;
#[cfg(feature = "json")]
mod json;
/// Macros usable for tests and initialization
pub mod macros;
pub mod sentinel;

// re-export for the datetime-literal macro
#[cfg(feature = "chrono")]
pub use chrono;

// for [``] auto-linking
#[cfg(doc)]
use crdts::TypeVariantValue;

static ENABLE_DETERMINISM: AtomicBool = AtomicBool::new(false);

/// Makes all data structures behave deterministically.
///
/// This should only be enabled for testing, as it increases the odds of DoS
/// scenarios.
#[doc(hidden)]
pub fn enable_determinism() {
    ENABLE_DETERMINISM.store(true, Ordering::Release);
}

/// Checks if determinism is enabled.
///
/// Should be used internally and for testing.
#[doc(hidden)]
pub fn determinism_enabled() -> bool {
    ENABLE_DETERMINISM.load(Ordering::Acquire)
}

/// Create a random state for a hashmap.
/// If `enable_determinism` has been used, this will return a deterministic
/// decidedly non-random RandomState, useful in tests.
#[inline]
fn make_random_state() -> RandomState {
    if determinism_enabled() {
        DETERMINISTIC_HASHER
    } else {
        // Create an instance of the standard ahash random state.
        // This will be random, and will not be the same for any two runs.
        RandomState::new()
    }
}

fn create_map<K, V>() -> std::collections::HashMap<K, V, DsonRandomState> {
    std::collections::HashMap::with_hasher(DsonRandomState::default())
}

fn create_map_with_capacity<K, V>(
    capacity: usize,
) -> std::collections::HashMap<K, V, DsonRandomState> {
    std::collections::HashMap::with_capacity_and_hasher(capacity, DsonRandomState::default())
}

/// This is a small wrapper around the standard RandomState.
/// This allows us to easily switch to a non-random RandomState for use in tests.
#[derive(Clone)]
pub struct DsonRandomState {
    inner: RandomState,
}

// Implement default, falling back on regular ahash::RandomState except
// when 'enable_determinism' has been called, in which case a static
// only-for-test RandomState is used.
impl Default for DsonRandomState {
    #[inline]
    fn default() -> Self {
        Self {
            inner: make_random_state(),
        }
    }
}

// We implement BuildHasher for DsonRandomState, but all we do is delegate to
// the wrapped 'inner' RandomState.
//
// This construct allows us to easily use a deterministic RandomState (i.e, not random :-) ),
// for tests.
//
// Since DsonRandomState implements default, the user doesn't have to do anything more than
// specialize their hashmap using DsonRandomState instead of RandomState.
impl BuildHasher for DsonRandomState {
    type Hasher = <RandomState as BuildHasher>::Hasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        self.inner.build_hasher()
    }
}

/// A type that extends [`TypeVariantValue`] and friends with additional value types.
///
/// If you are looking for an implementor of this trait to stick with the standard DSON/JSON types,
/// use [`crdts::NoExtensionTypes`].
///
/// The compiler should guide you towards all the various other traits and types you need in order
/// to satisfy this trait once you add an impl of it.
///
/// In terms of mental model, think of the type that directly implements this trait as a direct
/// analogue of [`TypeVariantValue`]. That is, it should generally be a struct with one `Option`
/// field for each possible kind of custom value type. It needs to be a struct, not an enum, so
/// that it can represent conflicts in type changes (for example, one writer sets a value to custom kind A
/// and another sets it to custom kind B concurrently). [`ExtensionType::Value`] is used in
/// situations where it is known that only a single kind is held.
/// [`ExtensionType::coerce_to_value_ref`] is the main way in which such type conflicts are
/// resolved.
///
/// The sub-types ("kinds") of a custom extension type must all be CRDTs, which in turn makes the
/// implementing type also a CRDT assuming it follows the directions above. This is represented by
/// the requirement that both `Self` and `ExtensionType::Value` implement [`DotStore`].
///
/// Implementors of this trait are generally used wherever `<Custom>` or `<C>` appears.
pub trait ExtensionType: DotStore + Default {
    /// Represents the kind of the underlying type without holding any data.
    ///
    /// This is the extension equivalent of [`crdts::ValueType`], and will likely be a simple
    /// data-less enum.
    type ValueKind: Copy + fmt::Debug;

    /// Type that holds a known, single kind of this type.
    ///
    /// This is the extension equivalent of [`crdts::Value`], and will likely be an enum where each
    /// variant holds one of the field types of `Self`.
    ///
    /// Since each sub-type should be a CRDT, this type should trivially implement [`DotStore`] by
    /// forwarding to the [`DotStore`] implementation of the contained sub-type.
    ///
    /// Since `Self` is expected to be able to hold all sub-types (potentially more than one at a
    /// time), this type should be trivial to turn into `Self`.
    type Value: fmt::Debug + Clone + PartialEq + DotStore + Into<Self>;

    /// Type that holds a reference to a known, single kind of this type.
    ///
    /// This is the extension equivalent of [`crdts::ValueRef`], and will likely be an enum where
    /// each variant holds a `&` to one of the field types of `Self` (as indicated by the
    /// `From<&Self::Value>` requirement).
    ///
    /// This type is generally used to represent a view into sub-tree of a DSON document. That
    /// sub-tree is then read using [`crdts::snapshot::ToValue`].
    ///
    /// Since this type is required to implement `Copy` (it is supposed to just be a reference
    /// type), it is expected to directly implement [`Into`] for [`ExtensionType::ValueKind`] as
    /// opposed to going via a `&self` method.
    ///
    /// The requirement of `Into<Self::Value>` may seem odd, but serves as a replacement for
    /// [`Clone`]. We can't use `Clone` since `Clone` is "special" when it comes to `&` -- the
    /// compiler knows that when you call `Clone` on a `&T`, you want a `T` back, but it wouldn't
    /// be as smart for `ValueRef`.
    type ValueRef<'doc>: Copy
        + fmt::Debug
        + From<&'doc Self::Value>
        + Into<Self::Value>
        + crdts::snapshot::ToValue
        + Into<Self::ValueKind>
    where
        Self: 'doc;

    /// Coerces the potentially type-conflicted value in `self` into a single-typed
    /// [`Self::ValueRef`].
    ///
    /// This is an inherently lossy operation -- if a type conflict exists in `self`, this has to
    /// pick which type should be exposed when the document is read. This is required since the
    /// types in [`crdts::snapshot`] cannot represent type conflicts, only value conflicts.
    ///
    /// This is the extension equivalent of [`TypeVariantValue::coerce_to_value_ref`], and will
    /// generally be an `if-let` chain that returns a [`Self::ValueRef`] for the "first" sub-type
    /// of `self` that is set. The ordering of the fields checked in the chain dictates the
    /// inference-precedence for coercion in type conflicts.
    fn coerce_to_value_ref(&self) -> Self::ValueRef<'_>;

    /// Gives a short name to describe a given custom value type.
    ///
    /// Called by [`crdts::Value::type_name`] and [`crdts::ValueRef::type_name`].
    fn type_name(value: &Self::ValueRef<'_>) -> &'static str;

    /// Get the bottom value of this type
    fn bottom() -> Self;
}

// NOTE: three arguments all of the same type -- big nope to have them be regular fn args.
pub struct ComputeDeletionsArg<'a> {
    /// Should be the causal context (ie, `.context`) of the more up to date `CausalDotStore`.
    pub known_dots: &'a CausalContext,

    /// Should be `store.dots()` of the more up to date `CausalDotStore`.
    pub live_dots: &'a CausalContext,

    /// Should be `store.dots()` of the `CausalDotStore` that may be missing deletes.
    pub ignorant: &'a CausalContext,
}

/// Returns dots that `known_dots` has deleted (by virtue of not being in `live_dots`) that
/// are still present in `ignorant`.
///
/// Conceptually computes `(known_dots - live_dots) & ignorant`.
pub fn compute_deletions_unknown_to(
    ComputeDeletionsArg {
        known_dots,
        live_dots,
        ignorant,
    }: ComputeDeletionsArg,
) -> CausalContext {
    // conceptually, this is:
    //
    //     let deletes_ever = known_dots - live_dots;
    //     let relevant_deletes = deletes_ever & ignorant;
    //
    // however, deletes_ever ends up quite large, as it holds all deletes ever, which is
    // wasteful since most of those dots then go away in the following set-intersection.
    // we can use set theory to our advantage here[1], which states that (with \ denoting
    // set subtraction):
    //
    //     (L \ M) ∩ R = (L ∩ R) \ (M ∩ R)
    //                 = (L ∩ R) \ M
    //                 = L ∩ (R \ M)
    //
    //     with
    //
    //     L = known_dots
    //     M = live_dots
    //     R = ignorant
    //
    // [1]: https://en.wikipedia.org/wiki/List_of_set_identities_and_relations#(L\M)_%E2%81%8E_R
    //
    // many of these are significantly cheaper to compute than the original (both in memory
    // and compute), especially when we take into account that intersection and subtraction
    // are both O(left operand size). in particular, since ∩ is commutative, we can compute:
    let only_in_ignorant = ignorant - live_dots;
    only_in_ignorant.bitand(known_dots)
    // the first part will be O(.store.dots()), and should result in a very small set. the
    // second part iterates only over that small set, which should be cheap. at no point do
    // we materialize a big set. its worth noting that all the sets involved here _should_
    // already be fully compacted, but if that weren't the case we'd want compacted sets to
    // be on the left-hand side.
}
