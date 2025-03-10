// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! # Dot Stores
//!
//! This module defines the core data structures, known as "dot stores", that underpin the DSON
//! (JSON CRDT Using Delta-Mutations For Document Stores) implementation. The concepts and data
//! structures defined here are based on the research paper "[DSON: JSON CRDT Using
//! Delta-Mutations For Document Stores](dson_paper.txt)".
//!
//! ## Overview
//!
//! At the heart of DSON is the idea of a **dot store**, a container for data-type-specific
//! information that stores the state of a delta-based CRDT. Each dot store is paired with a
//! [`CausalContext`], which tracks the set of observed events (dots) across replicas. This
//! combination, encapsulated in the [`CausalDotStore`] struct, forms the basis for building
//! CRDTs.
//!
//! The primary dot stores defined in this module are:
//!
//! - [`DotFun`]: A map from [`Dot`]s to values, where the set of dots is its keyset. This is used
//!   to implement simple CRDTs like [`MvReg`](crate::crdts::mvreg::MvReg) (multi-value registers).
//! - [`DotMap`]: A map from an arbitrary key type to a `DotStore`, where the computed dots are the
//!   union of the dots of its values. This is used to implement OR-Maps (Observed-Remove Maps).
//! - [`DotFunMap`]: A map from [`Dot`]s to `DotStore`s, combining the properties of `DotFun` and
//!   `DotMap`. This is used to implement OR-Arrays (Observed-Remove Arrays).
//!
//! These dot stores are designed to be composable, allowing for the construction of arbitrarily
//! nested JSON-like structures.
//!
//! ## Join Operations
//!
//! The core of the CRDT logic is the `join` operation, defined in the [`DotStoreJoin`] trait. The
//! `join` operation merges the state of two `CausalDotStore`s, resolving conflicts in a
//! deterministic way. The exact semantics of the join operation vary depending on the concrete
//! dot store type, but the general principle is to keep the most up-to-date values and discard
//! those that have been causally overwritten.
//!
//! ## References
//!
//! The theoretical foundations for the dot stores and their join operations are detailed in
//! the DSON paper. In particular, see the following sections:
//!
//! - **Section 3.3**: Introduces the concept of dot stores and defines `DotFun` and `DotMap`.
//! - **Section 4**: Describes the observed-remove semantics used in DSON.
//! - **Section 5**: Introduces the `CompDotFun` (here named `DotFunMap`) and the OR-Array
//!   algorithm.
//!
//! The original work on delta-based CRDTs can be found in the 2018 paper _Delta state replicated
//! data types_ by Paulo Sérgio Almeida, Ali Shoker, and Carlos Baquero.

use crate::{
    CausalContext, Dot, DsonRandomState, create_map, create_map_with_capacity,
    sentinel::{DummySentinel, KeySentinel, Sentinel, ValueSentinel, Visit},
};
use smallvec::SmallVec;
use std::{borrow::Borrow, collections::HashMap, fmt, hash::Hash, ops::Index};

/// A [`DotStore`] paired with a [`CausalContext`].
///
/// This is the fundamental building block of the DSON CRDT. It combines a `DotStore`, which holds
/// the state of a specific data type, with a `CausalContext`, which tracks the set of observed
/// events (dots) across replicas. This pairing allows for the implementation of delta-based
// CRDTs, where changes can be calculated and transmitted as deltas rather than entire states.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct CausalDotStore<DS> {
    /// The data-type-specific information.
    pub store: DS,
    /// The causal context, tracking observed events.
    pub context: CausalContext,
}

impl<'cs, DS> From<&'cs CausalDotStore<DS>> for (&'cs DS, &'cs CausalContext) {
    fn from(cds: &'cs CausalDotStore<DS>) -> Self {
        (&cds.store, &cds.context)
    }
}

impl<'cs, DS> From<&'cs mut CausalDotStore<DS>> for (&'cs DS, &'cs mut CausalContext) {
    fn from(cds: &'cs mut CausalDotStore<DS>) -> Self {
        (&cds.store, &mut cds.context)
    }
}

impl<DS> CausalDotStore<DS>
where
    DS: DotStore,
{
    /// Returns true if this is ⊥ (that is, empty).
    ///
    /// NOTE: the DSON paper does not explicitly define what a bottom is for a Causal⟨DotStore⟩, but
    /// it does provide that "For any 𝑋 ∈ Causal⟨DotStore⟩, 𝑋 ⊔ ⊥ = 𝑋", which constrains it to ⊥ =
    /// ({}, {}), since that is the only value that satisfies that equation and Equation 4 for any
    /// arbitrary 𝑋.
    pub fn is_bottom(&self) -> bool {
        self.store.is_bottom() && self.context.is_empty()
    }

    /// Returns a subset-CRDT derived from `self` that allows inflating state at the vector time
    /// `frontier` to what is in `self`.
    ///
    /// Does not include deletions not represented by `self.context - frontier` (that is, deletions of
    /// already-known store entries); that is left as an exercise for the caller.
    pub fn subset_for_inflation_from(&self, frontier: &CausalContext) -> CausalDotStore<DS>
    where
        DS: Clone + Default,
    {
        // our goal here is to produce a CRDT that contains aspects of `self.store` that are _not_
        // known to `frontier`. this could be additions (the easy case), but could also be deletes.
        // since deletes are handled via the _absence_ of state plus the _presence_ of a dot, we
        // need to carefully construct both the (delta) store and (delta) context here.
        //
        // the state isn't too bad: we call subset_for_inflation_from recursively all the way
        // "down" the CRDT, keeping only entries whose dot is not in frontier, or that are on the
        // path _to_ such a dot.
        //
        // the context is trickier: we can't send all of self.context since we're excluding values
        // that aren't in context. more concretely, consider what happens if (0, 1) => 'a', and the
        // dot (0, 1) is in both self.context and frontier. we will _not_ include it in the delta
        // of store (since the node represented by frontier already knows it). if we nevertheless
        // include (0, 1) in the delta context, the semantics of that is a _deletion_ of (0, 1)
        // [see DotFun::join], which obviously isn't correct. instead, we produce the delta context
        //
        //     self.context - frontier + delta_store.dots()
        //
        // this indicates that we a) have included anything we know that frontier does not, and b)
        // acknowledges that some dots are included simply by virtue of being on the path to
        // new/updated values.
        let delta_store = self.store.subset_for_inflation_from(frontier);
        let mut delta_context = &self.context - frontier;

        // NOTE: it _could_ be that nothing bad happens if we don't add in delta_store.dots(),
        // but that relies on the join at the other end not getting confused by the presence of a
        // dot in store but not in context. Feels safer to just join them here anyway.
        delta_store.add_dots_to(&mut delta_context);

        // unfortunately, this doesn't capture deletions. remember from above, the way a delete is
        // represented is the _presence_ of the dot that inserted a value `context`, coupled with
        // the _absence_ of its value in `store`.
        //
        // consider what happens if A and B are fully synchronized, and both hold, say, just an
        // MVReg with (1, 1) => 'x' as well as the causal context {(1,1)}. now, A deletes 'x'. this
        // does not generate a dot, so A's causal context is the same. when A and B next
        // synchronize, A does not know solely from B's causal context ({(1,1)}) that it is missing
        // the deletion of (1, 1) => 'x'. the store won't include (1,1) [and shouldn't],
        // self.context - frontier is empty, and so is delta_store.dots() [since delta_store is
        // bottom].
        //
        // even if we _do_ associate a dot with a deletion (e.g., writing the ALIVE field like we
        // do for maps and arrays), it doesn't solve this problem. A would then generate (1,2) for
        // the deletion, and would realize B doesn't have (1,2), *but* it won't know that that
        // implies that the causal context of the delta it sends to be should therefore include
        // (1,1). it doesn't know the relationship between (1,2) and (1,1).
        //
        // we could keep a "graveyard" that holds
        //
        //   Dot(A, B) ---(deleted)----> Dot(X, Y)
        //
        // and then here _also_ add in Dot(X, Y) for any Dot(A, B) not in frontier, but that raises
        // the question of how to garbage-collect said graveyard. it's also not clear what happens
        // for types where a deletion actually implies _multiple_ removed dots.
        //
        // for now, we leave making the context reflect deletions as an exercise for the caller.

        CausalDotStore {
            store: delta_store,
            context: delta_context,
        }
    }
}

impl<DS> Default for CausalDotStore<DS>
where
    DS: Default,
{
    fn default() -> Self {
        Self {
            store: Default::default(),
            context: Default::default(),
        }
    }
}

impl<DS> CausalDotStore<DS> {
    /// Constructs a new empty [`CausalDotStore`].
    pub fn new() -> Self
    where
        DS: Default,
    {
        Self::default()
    }
}

#[cfg(any(test, feature = "arbitrary"))]
use crate::dotstores::recording_sentinel::RecordingSentinel;

impl<DS> CausalDotStore<DS> {
    /// Joins the given [`CausalDotStore`] with this one, and returns the join.
    ///
    /// This is a convenience function around [`CausalDotStore::join_with`].
    pub fn join<S>(mut self, other: Self, sentinel: &mut S) -> Result<CausalDotStore<DS>, S::Error>
    where
        S: Sentinel,
        DS: DotStoreJoin<S> + Default,
    {
        self.consume(other, sentinel)?;
        Ok(CausalDotStore {
            store: self.store,
            context: self.context,
        })
    }

    // variant of join intended for tests, so it is not built for performance (for example, it clones eagerly
    // internally to make the interface more convenient) and it exposes internal bits (like
    // `on_dot_change`)
    #[cfg(any(test, feature = "arbitrary"))]
    pub fn test_join<S>(
        &self,
        other: &Self,
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<CausalDotStore<DS>, S::Error>
    where
        S: Sentinel,
        DS: DotStoreJoin<S> + DotStoreJoin<RecordingSentinel> + Default + Clone,
    {
        let mut this = self.clone();
        this.test_join_with_and_track(
            other.store.clone(),
            &other.context,
            on_dot_change,
            sentinel,
        )?;

        Ok(CausalDotStore {
            store: this.store,
            context: this.context,
        })
    }

    #[cfg(any(test, feature = "arbitrary"))]
    pub fn test_join_with_and_track<S>(
        &mut self,
        store: DS,
        context: &CausalContext,
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<(), S::Error>
    where
        DS: DotStoreJoin<S> + DotStoreJoin<RecordingSentinel> + Clone + Default,
        S: Sentinel,
    {
        #[cfg(debug_assertions)]
        {
            // We do a dry_join here first, to ensure that dry-join
            // and join always result in the same set of calls being
            // made to Sentinel. This is an invariant that we want to always
            // hold, so we check it in debug builds for all test cases using this function.

            let mut dry_join_sentinel = RecordingSentinel::new();
            let dry_result = <DS as DotStoreJoin<RecordingSentinel>>::dry_join(
                (&self.store, &self.context),
                (&store, context),
                &mut dry_join_sentinel,
            )
            .expect("RecordingSentinel is infallible");

            let mut full_run_sentinel = RecordingSentinel::new();
            let full_result = DS::join(
                (self.store.clone(), &self.context),
                (store.clone(), context),
                &mut |_| {},
                &mut full_run_sentinel,
            )
            .expect("RecordingSentinel is infallible");

            assert_eq!(
                dry_join_sentinel.changes_seen,
                full_run_sentinel.changes_seen
            );
            assert_eq!(dry_result.is_bottom(), full_result.is_bottom());
        }

        self.join_with_and_track(store, context, on_dot_change, sentinel)?;
        Ok(())
    }

    /// Joins the given [`CausalDotStore`] into this one.
    ///
    /// This is a convenience function around [`CausalDotStore::join_with`].
    pub fn consume<S>(
        &mut self,
        other: CausalDotStore<DS>,
        sentinel: &mut S,
    ) -> Result<(), S::Error>
    where
        DS: DotStoreJoin<S> + Default,
        S: Sentinel,
    {
        self.join_with(other.store, &other.context, sentinel)
    }

    /// Joins the given [`CausalDotStore`] into this one.
    ///
    /// This is a convenience function around [`CausalDotStore::join_with`].
    #[cfg(any(test, feature = "arbitrary"))]
    pub fn test_consume(&mut self, other: CausalDotStore<DS>)
    where
        DS: DotStoreJoin<RecordingSentinel> + Clone + Default,
    {
        self.test_join_with(other.store, &other.context)
    }

    /// Joins or replaces the current [`CausalDotStore`] with the provided one.
    ///
    /// If the current value is bottom, it is replaced wholesale, bypassing the
    /// join. This method does not accept a sentinel as changes cannot always
    /// be tracked.
    pub fn join_or_replace_with(&mut self, store: DS, context: &CausalContext)
    where
        DS: DotStoreJoin<DummySentinel> + Default,
    {
        if self.is_bottom() {
            *self = CausalDotStore {
                store,
                context: context.clone(),
            };
        } else {
            self.join_with(store, context, &mut DummySentinel)
                .expect("DummySentinel is Infallible");
        }
    }

    /// Joins the given [`DotStore`]-[`CausalContext`] pair into those in `self`.
    ///
    /// Prefer this method when you need to avoid cloning the [`CausalContext`].
    pub fn join_with<S>(
        &mut self,
        store: DS,
        context: &CausalContext,
        sentinel: &mut S,
    ) -> Result<(), S::Error>
    where
        DS: DotStoreJoin<S> + Default,
        S: Sentinel,
    {
        self.join_with_and_track(store, context, &mut |_| (), sentinel)
    }

    /// Joins the given [`DotStore`]-[`CausalContext`] pair into those in `self`.
    ///
    /// Prefer this method when you need to avoid cloning the [`CausalContext`].
    #[cfg(any(test, feature = "arbitrary"))]
    pub fn test_join_with(&mut self, store: DS, context: &CausalContext)
    where
        DS: DotStoreJoin<RecordingSentinel> + Clone + Default,
    {
        self.test_join_with_and_track(store, context, &mut |_| (), &mut RecordingSentinel::new())
            .expect("RecordingSentinel is infallible");
    }

    fn join_with_and_track<S>(
        &mut self,
        store: DS,
        context: &CausalContext,
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<(), S::Error>
    where
        DS: DotStoreJoin<S> + Default,
        S: Sentinel,
    {
        let old_store = std::mem::take(&mut self.store);
        self.store = DS::join(
            (old_store, &self.context),
            (store, context),
            on_dot_change,
            sentinel,
        )?;
        self.context.union(context);
        Ok(())
    }
}

impl<DS> CausalDotStore<DS> {
    /// Constructs a new [`CausalDotStore`] by applying the given function to the current store.
    ///
    /// This method keeps the causal context as-is.
    pub fn map_store<DS2>(self, m: impl FnOnce(DS) -> DS2) -> CausalDotStore<DS2> {
        CausalDotStore {
            store: (m)(self.store),
            context: self.context,
        }
    }

    /// Constructs a new [`CausalDotStore`] by applying the given function to the current context.
    ///
    /// This method keeps the store as-is.
    pub fn map_context(self, m: impl FnOnce(CausalContext) -> CausalContext) -> CausalDotStore<DS> {
        CausalDotStore {
            store: self.store,
            context: (m)(self.context),
        }
    }

    /// Calls a function with a reference to the contained store.
    ///
    /// Returns the original [`CausalDotStore`] unchanged.
    pub fn inspect(self, f: impl FnOnce(&DS)) -> CausalDotStore<DS> {
        f(&self.store);
        self
    }
}

/// A container for data-type specific information that stores the state of a 𝛿-based CRDT.
///
/// This trait defines the common interface for all dot stores. It provides methods for querying
/// the dots contained within the store, checking if the store is empty (i.e., ⊥), and creating a
/// subset of the store for inflation.
pub trait DotStore {
    /// Queries the set of event identifiers (ie, dots) currently stored in the dot store.
    ///
    /// Has a default implementation that creates an empty [`CausalContext`] and invokes
    /// `add_dots_to`.
    fn dots(&self) -> CausalContext {
        let mut cc = CausalContext::default();
        self.add_dots_to(&mut cc);
        cc
    }

    /// Add the set of event identifiers (ie, dots) currently stored in the dot store to `other`.
    ///
    /// Should not compact the resulting `CausalContext`.
    fn add_dots_to(&self, other: &mut CausalContext);

    /// Returns true if this dot store is ⊥ (ie, empty).
    fn is_bottom(&self) -> bool;

    /// Returns a subset-CRDT derived from `self` that allows inflating state at the vector time
    /// `frontier` to what's in `self`.
    fn subset_for_inflation_from(&self, frontier: &CausalContext) -> Self;
}

/// An observed change to a dot store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DotChange {
    /// The given dot was added to the store.
    Add(Dot),
    /// The given dot was removed from the store.
    Remove(Dot),
}

/// The outcome of performing a dry-join.
///
/// When doing a dry-join, we don't perform the join completely.
/// However, we often need to know whether the join would have
/// been bottom or not, had it been carried out. This single
/// bit of information is significantly cheaper to calculate than
/// a full join.
#[derive(Debug)]
pub struct DryJoinOutput {
    /// True if the output of the dry-join was ⊥ (bottom).
    is_bottom: bool,
}

impl DryJoinOutput {
    /// Create a join-result representing ⊥
    pub fn bottom() -> Self {
        Self { is_bottom: true }
    }
    /// Update self, setting it to 'not bottom'
    pub fn set_is_not_bottom(&mut self) {
        self.is_bottom = false;
    }
    pub fn new(is_bottom: bool) -> Self {
        Self { is_bottom }
    }
    /// If parameter is false, set self.is_bottom to false.
    /// Intuitively this is a join of two dry-join results.
    pub fn union_with(&mut self, other: DryJoinOutput) {
        if !other.is_bottom {
            self.is_bottom = false;
        }
    }
    /// Like `join_with`, but returns the result instead.
    pub fn union(&self, other: Self) -> Self {
        Self {
            is_bottom: self.is_bottom && other.is_bottom,
        }
    }
    /// Returns true if this instance represents ⊥ (bottom)
    pub fn is_bottom(&self) -> bool {
        self.is_bottom
    }
}

/// A trait for dot stores that can be joined.
///
/// This trait defines the `join` and `dry_join` operations, which are the core of the CRDT
/// logic. The `join` operation merges the state of two dot stores, while `dry_join` simulates a
/// join without actually modifying the state, which is useful for validation.
pub trait DotStoreJoin<S>: DotStore {
    /// Computes the join (⊔) between two CausalDotStores.
    ///
    /// Note that for efficiency this does not take a [`CausalDotStore`] directly, but instead
    /// takes owned [`DotStore`]s and a shared reference to the [`CausalContext`] to avoid
    /// excessive cloning.
    ///
    /// Quoth the DSON paper:
    ///
    /// > For any 𝑋 ∈ Causal⟨DotStore⟩, 𝑋 ⊔ ⊥ = 𝑋.
    /// >
    /// > For two elements 𝑋1, 𝑋2 ∈ Causal⟨DotStore⟩
    /// > we say that 𝑋1 < 𝑋2 iff ∃𝑋 ≠ ⊥ ∈ Causal⟨DotStore⟩ such that 𝑋1 ⊔ 𝑋 = 𝑋2.
    ///
    /// > An example of a ⊥ value is a merge between two elements of the Causal⟨DotFun⟩
    /// > semilattice, where the domains are disjoint but all mappings are in the others causal
    /// > history. Consider for example a write 𝑤1 that precedes a write 𝑤2, i.e., 𝑤1 ≺𝜎 𝑤2, then
    /// > the dot generated by 𝑤1 is in the causal context of the delta generated by 𝑤2. By the
    /// > definition of join, the mapping doesn’t “survive” the join, and therefore the old value
    /// > (written by 𝑤1) is overwritten – it isn’t present in the range of the map after 𝑤2.
    ///
    /// The exact semantics of a DotStore's join varies depending on the concrete type used.
    ///
    /// # Observing changes
    /// Join (⊔) operations are commutative, i.e. 𝑋1 ⊔ 𝑋2 = 𝑋2 ⊔ 𝑋1, so the order of arguments
    /// ds1 and ds2 doesn't matter w.r.t. the final result. However, conventionally we interpret
    /// ds1 as the current state and ds2 as an incoming delta, so from the perspective of the
    /// sentinel, changes are applied from ds2 into ds1. The same applies to `on_dot_change`.
    fn join(
        ds1: (Self, &CausalContext),
        ds2: (Self, &CausalContext),
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<Self, S::Error>
    where
        Self: Sized,
        S: Sentinel;

    // YADR: 2024-10-10 Implementation of DryJoin
    //
    // In the context of ensuring the schema-conformance of deltas, we faced the challenge of
    // how to efficiently validate such deltas prior to merging them into the canonical root
    // document.
    //
    // We decided for adding a "dry" join that duplicates the join logic but doesn't change any
    // document state as it runs, and neglected alternatives that would require cloning the root
    // document, keeping an undo log, or allow a sentinel to exit after applying a subset of the
    // changes (further details at the end of the YADR).
    //
    // We did this to achieve minimal performance overhead (cloning), complexity (undo-log), and
    // surprise factor (partial application) of performing delta validations, accepting the
    // duplication of code between join and dry-join as well as the maintenance burden this
    // brings when adding or updating CRDTs.
    //
    // We think this is the right trade-off because adding new CRDTs and changing old ones is
    // uncommon, performance of validation is paramount given its frequency, and debugging
    // two nearly-identical implementations is likely to be easier than debugging a
    // join + undo-log combination.
    // Alternatives that were considered and rejected:
    //
    // * Do the regular join, but keep an undo-log. If a validation error is detected,
    //   the delta is undone. This was deemed relatively hard to implement.
    // * Clone the entire document before the join, then do a regular join. This
    //   does not perform well. Experiments show a cost of ~50ms on fast machines, for reasonably
    //   sized documents (in the 10s of megabytes).
    // * Do a regular join, but allow the sentinel to veto changes. This was deemed difficult
    //   to implement, and to define the semantics of. It would also result in partial updated,
    //   which is generally undesirable and hard to reason about.
    // * Implement DryJoin and regular join using the same code. Create abstractions and
    //   use generics as needed to achieve this. A simplified, but functional, prototype using this
    //   approach was written. The complexity was deemed undesirable.

    /// Simulates a [`DotStoreJoin::join`] without constructing the output of the join.
    ///
    /// This simulation allows a sentinel to observe a join without committing its result,
    /// such as to validate a delta prior to joining it.
    ///
    /// Since this method does not have to construct the join output, it does not need to take
    /// ownership of its parameters (ie, it can be run on shared references to the dot stores).
    ///
    /// This method returns an indicator determining if the result of the real join would have
    /// been the bottom type.
    fn dry_join(
        ds1: (&Self, &CausalContext),
        ds2: (&Self, &CausalContext),
        sentinel: &mut S,
    ) -> Result<DryJoinOutput, S::Error>
    where
        Self: Sized,
        S: Sentinel;
}

/// A map from [`Dot`] to `V` whose computed dots is its keyset.
///
/// Quoth the DSON paper:
///
/// > A join of Causal⟨DotFun⟩ keeps values that exist in both of the mappings and merges their
/// > respective values, or that exist in either one of the mappings and are “new” to the other in
/// > the sense that they are not in its causal history.
///
/// In practice, this means that a join of two [`DotFun`] will keep only up-to-date elements. In
/// particular, if instance X1 has observed some [`Dot`] that exists in X2, but that [`Dot`] is not
/// present in X1, then that [`Dot`] is _not_ preserved (as it has presumably been removed).
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct DotFun<V> {
    // NOTE: the store is explicitly ordered by dot so that self-healing conflicts arising due
    // to out-of-order delivery of messages can be dealt with by final consumers by just taking
    // the last value among the conflicts, thus avoiding the need to access the dots directly. This
    // implicit resolution strategy works as long as the entry is only ever mutated by a single
    // `Identifier`, as in that case it is guaranteed that later/higher dots will override their
    // predecessors once all dots have eventually been observed.
    state: SmallVec<[(Dot, V); 1]>,
}

impl<V: fmt::Debug> fmt::Debug for DotFun<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

// manual impl because auto-derive'd `Clone` requires `V: Clone`.
impl<V> Default for DotFun<V> {
    fn default() -> Self {
        Self {
            state: Default::default(),
        }
    }
}

/// An iterator over the values of a [`DotFun`].
pub struct DotFunValueIter<'df, V> {
    it: std::slice::Iter<'df, (Dot, V)>,
}

impl<'df, V> Iterator for DotFunValueIter<'df, V> {
    type Item = &'df V;

    fn next(&mut self) -> Option<Self::Item> {
        self.it.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.it.count()
    }

    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.it.last().map(|(_, v)| v)
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.it.nth(n).map(|(_, v)| v)
    }
}
impl<V> ExactSizeIterator for DotFunValueIter<'_, V> {}
impl<V> Clone for DotFunValueIter<'_, V> {
    fn clone(&self) -> Self {
        Self {
            it: self.it.clone(),
        }
    }
}
impl<V> fmt::Debug for DotFunValueIter<'_, V>
where
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.it.fmt(f)
    }
}

impl<V> DotFun<V> {
    #[doc(hidden)]
    pub fn push(&mut self, dot: Dot, v: V) {
        self.state.push((dot, v));
    }

    /// Constructs a [`DotFun`] with the given initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            state: SmallVec::with_capacity(capacity),
        }
    }

    /// Produces an iterator over the map's keys and values.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Dot, &V)> {
        self.state.iter().map(|(k, v)| (*k, v))
    }

    /// Produces an iterator over the map's keys.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = Dot> + '_ {
        self.iter().map(|(k, _)| k)
    }

    /// Produces an iterator over the map's values.
    pub fn values(&self) -> DotFunValueIter<'_, V> {
        DotFunValueIter {
            it: self.state.iter(),
        }
    }

    /// Returns the number of keys in the map.
    pub fn len(&self) -> usize {
        self.state.len()
    }

    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    fn get_index(&self, dot: &Dot) -> Option<usize> {
        self.state
            .as_slice()
            .binary_search_by_key(dot, |(k, _)| *k)
            .ok()
    }

    /// Retrieves the associated value, if any, for the given [`Dot`].
    pub fn get(&self, dot: &Dot) -> Option<&V> {
        self.get_index(dot).map(|idx| &self.state[idx].1)
    }

    /// Retrieves a mutable reference to the associated value, if any, for the given [`Dot`].
    pub fn get_mut(&mut self, dot: &Dot) -> Option<&mut V> {
        self.get_index(dot).map(|idx| &mut self.state[idx].1)
    }

    /// Returns `true` if the given [`Dot`] has a value in this map.
    pub fn has(&self, dot: &Dot) -> bool {
        self.get_index(dot).is_some()
    }

    /// Associates the value with the given [`Dot`].
    ///
    /// Returns the previous value if any.
    pub fn set(&mut self, dot: Dot, value: V) -> Option<V> {
        if let Some(v) = self.get_mut(&dot) {
            Some(std::mem::replace(v, value))
        } else {
            let idx = self.state.partition_point(|(d, _)| *d < dot);
            self.state.insert(idx, (dot, value));
            None
        }
    }

    /// Removes and returns the value associated with a [`Dot`], if the dot exists.
    pub fn remove(&mut self, dot: &Dot) -> Option<V> {
        if let Some(idx) = self.get_index(dot) {
            // as tempting as it may be, we shouldn't use swap_remove here as we
            // want to keep the list sorted
            Some(self.state.remove(idx).1)
        } else {
            None
        }
    }

    /// Retains only the values for which a predicate is true.
    pub fn retain(&mut self, mut f: impl FnMut(&Dot, &mut V) -> bool) {
        self.state.retain(|(k, v)| f(k, v))
    }
}

impl<V> DotStore for DotFun<V>
where
    V: PartialEq + fmt::Debug + Clone,
{
    fn add_dots_to(&self, other: &mut CausalContext) {
        other.insert_dots(self.keys());
    }

    fn is_bottom(&self) -> bool {
        self.is_empty()
    }

    fn subset_for_inflation_from(&self, frontier: &CausalContext) -> Self {
        Self {
            state: self
                .state
                .iter()
                .filter(|(dot, _)| !frontier.dot_in(*dot))
                .map(|(dot, v)| (*dot, v.clone()))
                .collect(),
        }
    }
}

impl<V, S> DotStoreJoin<S> for DotFun<V>
where
    S: ValueSentinel<V>,
    V: PartialEq + fmt::Debug + Clone,
{
    /// Formally (Equation 4):
    /// ```text
    /// > (𝑚, 𝑐) ⊔ (𝑚′, 𝑐′) =
    /// >   (
    /// >       {𝑑 ↦ 𝑚[𝑑] ⊔ 𝑚′ [𝑑] | 𝑑 ∈ dom 𝑚 ∩ dom 𝑚′}
    /// >     ∪ {(𝑑, 𝑣) ∈ 𝑚  | 𝑑 ∉ 𝑐′}
    /// >     ∪ {(𝑑, 𝑣) ∈ 𝑚′ | 𝑑 ∉ 𝑐}
    /// >     , 𝑐 ∪ 𝑐′
    /// >   )
    /// ```
    ///
    /// Informally:
    ///  - for dots in both stores, join the values
    ///  - for dots in store 1 that haven't been observed by store 2, keep the value
    ///  - for dots in store 2 that haven't been observed by store 1, keep the value
    ///  - don't keep other dots
    ///  - the resulting causal context is the union of the provided causal contexts
    fn join(
        (m1, cc1): (Self, &CausalContext),
        (mut m2, cc2): (Self, &CausalContext),
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<Self, S::Error>
    where
        S: Sentinel,
    {
        // NOTE! When making changes to this method, consider if corresponding
        // changes need to be done to ::dry_join as well!

        let mut res_m = Self::with_capacity(m1.len().max(m2.len()));
        for (dot, v1) in m1.state {
            if let Some(v2) = m2.remove(&dot) {
                // dots are assumed to be unique, so there's no need to join these as they must by
                // implication be identical.
                if v1 != v2 {
                    // this should be unreachable, since validation should have caught this
                    unreachable!("duplicate node id detected");
                }

                res_m.set(dot, v1);
            } else if !cc2.dot_in(dot) {
                // m1 has v, m2 does not, but m2 hasn't observed the dot, so we keep v as this can't
                // be a removal.
                // TODO(#2): is ds1's author authorized to write this value?
                res_m.set(dot, v1);
            } else {
                // m1 map has a value that m2 does not, _but_ the map that does not have the
                // value (m2) has already observed the dot in its causal context. So, it must have
                // intentionally chosen to remove the value, and thus we should not preserve it.
                // TODO(#2): is ds2's author authorized to clear this value?
                sentinel.unset(v1)?;
                on_dot_change(DotChange::Remove(dot));
            }
        }

        // m2 has v2, m1 does not, and m1 hasn't observed the dot
        // meaning v2 should be preserved (it wasn't deleted by m1)
        // TODO(#2): is ds2's author authorized to write this value?
        for (dot, v2) in m2.state.into_iter().filter(|(dot, _)| !cc1.dot_in(*dot)) {
            sentinel.set(&v2)?;
            res_m.set(dot, v2);
            on_dot_change(DotChange::Add(dot));
        }

        Ok(res_m)
    }

    fn dry_join(
        (m1, cc1): (&Self, &CausalContext),
        (m2, cc2): (&Self, &CausalContext),
        sentinel: &mut S,
    ) -> Result<DryJoinOutput, S::Error>
    where
        Self: Sized,
        S: Sentinel,
    {
        // For explanation of this method, see comments in ::join(..).

        let mut res_m = DryJoinOutput::bottom();

        for (dot, v1) in &m1.state {
            if let Some(v2) = m2.get(dot) {
                res_m.set_is_not_bottom();
                if v1 != v2 {
                    panic!("duplicate node id detected (in crdt join)")
                }
            } else if !cc2.dot_in(*dot) {
                res_m.set_is_not_bottom();
            } else {
                sentinel.unset(v1.clone())?;
            }
        }

        for (_dot, v2) in m2
            .state
            .iter()
            .filter(|(dot, _)| !m1.has(dot) && !cc1.dot_in(*dot))
        {
            res_m.set_is_not_bottom();
            sentinel.set(v2)?;
        }

        Ok(res_m)
    }
}

/// A map from [`Dot`] to `V: DotStore`, whose computed dots is the union of the dots of its
/// values.
///
/// Quoth the DSON paper:
///
/// > We combine the [`DotMap`] and [`DotFun`] to get a dot store which maps dots to dot stores.
/// > The join operation keeps keys that have not been deleted (as in the [`DotFun`]), or the
/// > values themselves haven’t been deleted (as in the [`DotMap`]).
///
/// Note that this is called `CompDotFun` in the DSON paper (section 5.1), but `DotFunMap` in their
/// prototype implementation.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[doc(alias = "CompDotFun")]
pub struct DotFunMap<V> {
    state: HashMap<Dot, V, DsonRandomState>,
}

impl<V: fmt::Debug> fmt::Debug for DotFunMap<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.state.iter()).finish()
    }
}

impl<V> Default for DotFunMap<V> {
    fn default() -> Self {
        Self {
            state: create_map(),
        }
    }
}

impl<V> DotFunMap<V> {
    /// Constructs a [`DotFunMap`] with the given initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            state: create_map_with_capacity(capacity),
        }
    }

    /// Produces an iterator over the map's keys and values.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (Dot, &V)> {
        self.state.iter().map(|(&k, v)| (k, v))
    }

    /// Produces an iterator over the map's keys.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = Dot> + '_ {
        self.state.keys().copied()
    }

    /// Produces an iterator over the map's values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> {
        self.state.values()
    }

    /// Returns the number of keys in the map.
    pub fn len(&self) -> usize {
        self.state.len()
    }

    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }
}

impl<V> DotFunMap<V> {
    /// Retrieves the associated value, if any, for the given [`Dot`].
    pub fn get(&self, dot: &Dot) -> Option<&V> {
        self.state.get(dot)
    }

    /// Retrieves a mutable reference to the associated value, if any, for the given [`Dot`].
    pub fn get_mut(&mut self, dot: &Dot) -> Option<&mut V> {
        self.state.get_mut(dot)
    }

    /// Returns `true` if the given [`Dot`] has a value in this map.
    pub fn has(&self, dot: &Dot) -> bool {
        self.state.contains_key(dot)
    }

    /// Associates the value with the given [`Dot`].
    ///
    /// Returns the previous value if any.
    pub fn set(&mut self, dot: Dot, value: V) -> Option<V> {
        self.state.insert(dot, value)
    }
}

impl<V> FromIterator<(Dot, V)> for DotFunMap<V> {
    fn from_iter<T: IntoIterator<Item = (Dot, V)>>(iter: T) -> Self {
        Self {
            state: HashMap::from_iter(iter),
        }
    }
}

impl<V> DotStore for DotFunMap<V>
where
    V: DotStore + fmt::Debug,
{
    fn add_dots_to(&self, other: &mut CausalContext) {
        // NOTE: Equation 6 in the paper suggests this should also include self.keys(),
        //            but the original implementation does not. the text just before eq6 also says
        //            "Note that the dots method returns the dots in the domain, as well as a union
        //            on recursive calls of dots on all dot stores in the range", which again,
        //            doesn't seem true in the implementation. so we do what we think is right.
        //            This was confirmed by one of the DSON paper authors (Arik Rinberg) by email
        //            on 2023-08-25.
        other.insert_dots(self.keys());
        for v in self.values() {
            v.add_dots_to(other);
        }
    }

    fn is_bottom(&self) -> bool {
        self.state.is_empty()
    }

    fn subset_for_inflation_from(&self, frontier: &CausalContext) -> Self {
        let mut delta = Self {
            state: create_map_with_capacity(self.state.len()),
        };

        for (&dot, v) in &self.state {
            let delta_v = v.subset_for_inflation_from(frontier);
            if !delta_v.is_bottom() {
                // NOTE: we do not consider whether frontier.dot_in(dot), since updates can
                // happen _under_ old dots.
                delta.state.insert(dot, delta_v);
            }
        }

        delta
    }
}

impl<V, S> DotStoreJoin<S> for DotFunMap<V>
where
    V: DotStoreJoin<S> + fmt::Debug + Default,
    S: Visit<Dot> + KeySentinel,
{
    /// Formally (Equation 7):
    ///
    /// > (𝑚, 𝑐) ⊔ (𝑚′, 𝑐′) =
    /// >   (
    /// >       {𝑑 ↦ 𝑣(𝑑) | 𝑑 ∈ dom 𝑚 ∩ dom 𝑚′ ∧ 𝑣(𝑑) ≠ ⊥}
    /// >     ∪ {(𝑑, 𝑣) ∈ 𝑚 | 𝑑 ∉ 𝑐′}
    /// >     ∪ {(𝑑, 𝑣) ∈ 𝑚′ | 𝑑 ∉ 𝑐}
    /// >     , 𝑐 ∪ 𝑐′
    /// >   )
    /// >   where 𝑣(𝑑) = fst((𝑚(𝑑), 𝑐) ⊔ (𝑚′(𝑑), 𝑐′))
    ///
    /// Informally:
    ///  - for dots in both stores, join the values and keep non-bottoms
    ///  - for dots in store 1 that haven't beeen observed by store 2, keep the value
    ///  - for dots in store 2 that haven't beeen observed by store 1, keep the value
    ///  - don't keep other dots
    ///  - the resulting causal context is the union of the provided causal contexts
    fn join(
        (m1, cc1): (Self, &CausalContext),
        (mut m2, cc2): (Self, &CausalContext),
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<Self, S::Error>
    where
        S: Sentinel,
    {
        // NOTE! When making changes to this method, consider if corresponding
        // changes need to be done to ::dry_join as well!

        let mut res_m = Self::with_capacity(m1.len().max(m2.len()));
        for (dot, v1) in m1.state {
            sentinel.enter(&dot)?;
            if let Some(v2) = m2.state.remove(&dot) {
                let new_v = V::join((v1, cc1), (v2, cc2), on_dot_change, sentinel)?;
                if !new_v.is_bottom() {
                    res_m.set(dot, new_v);
                } else {
                    on_dot_change(DotChange::Remove(dot));
                }
            } else if !cc2.dot_in(dot) {
                // m1 has v, m2 does not, but m2 hasn't observed the dot, so we keep v as this can't
                // be a removal.
                // TODO(#2): is ds1's author authorized to write this value?
                // NOTE: the original implementation does not join the value here, and neither
                // do we. this may seem odd since presumably cc2 may not have seen the root dot but
                // _may_ have seen updates inside of that dot that we need to take into account.
                // however, that is actually impossible due to how dots work -- since the dot is
                // the key, and cc2 hasn't seen the key dot, it _cannot_ have seen anything inside
                // that dot's value either, as that would mean having seen the key, which is the
                // dot. it would be _safe_ to do a join here, we just happen to know it _must_ end
                // up returning v1 as-is.
                res_m.set(dot, v1);
            } else {
                // m1 map has a value that m2 does not, _but_ the map that does not have the
                // value (m2) has already observed the dot in its causal context. So, it must have
                // intentionally chosen to remove the value, and thus we should not preserve it.
                // TODO(#2): is ds2's author authorized to clear this value?
                // NOTE: there's an important subtlety here compared to DotMap -- if the key
                // dot is removed, all nested values are also removed _by implication_. that is, we
                // do not join v1 with Default and look for bottom (like we do in DotMap for this
                // case), but rather take a removal of a root dot as a subtree removal. this is
                // reinforced by the DSON paper:
                //
                // > As the root is stored in a CompDotFun, once a root is removed it never
                // > reappears, contrary to keys in a DotMap which may remain undeleted if there is
                // > a concurrent update.
                //
                // this is awkward for sentinels, because it means they are not notified of
                // anything that happens inside of such a nested removed-by-implication value (and
                // ditto for `on_dot_change`), so we specifically choose to still join here to
                // provide that signal. we just join with Default and cc1's own `CausalContext` so
                // that we are sure that everything will be treated as a deletion (which would not
                // necessarily be the case if we used cc2's `CausalContext`).
                on_dot_change(DotChange::Remove(dot));
                let new_v = V::join((v1, cc1), (V::default(), cc1), on_dot_change, sentinel)?;
                assert!(new_v.is_bottom());
                sentinel.delete_key()?;
            }
            sentinel.exit()?;
        }

        // m2 has v2, m1 does not, and m1 hasn't observed the dot
        // meaning v2 should be preserved (it wasn't deleted by m1)
        // TODO(#2): is ds2's author authorized to write this value?
        for (dot, v2) in m2.state {
            sentinel.enter(&dot)?;
            if !cc1.dot_in(dot) {
                on_dot_change(DotChange::Add(dot));
                // NOTE: as mentioned in the earlier for loop, the DSON paper does not
                // indicate a join should be performed here. however, unlike the inverse case
                // (`!cc2.dot_in(dot)`) we do the join anyway here so that `sentinel` and
                // `on_dot_change` get information about new stuff from v2. this is only important
                // because sentinel and on_dot_change aren't symmetrical with respect to v1 and v2
                // as per the "Observing changes" section of the `join` docs.
                let new_v = V::join((V::default(), cc1), (v2, cc2), on_dot_change, sentinel)?;
                sentinel.create_key()?;
                res_m.state.insert(dot, new_v);
            }
            sentinel.exit()?;
        }

        Ok(res_m)
    }
    fn dry_join(
        (m1, cc1): (&Self, &CausalContext),
        (m2, cc2): (&Self, &CausalContext),
        sentinel: &mut S,
    ) -> Result<DryJoinOutput, S::Error>
    where
        S: Sentinel,
    {
        // For explanation of this method, see comments in ::join(..).

        let mut res_m = DryJoinOutput::bottom();
        for (dot, v1) in m1.state.iter() {
            sentinel.enter(dot)?;
            if let Some(v2) = m2.state.get(dot) {
                let new_v = V::dry_join((v1, cc1), (v2, cc2), sentinel)?;
                if !new_v.is_bottom() {
                    res_m.set_is_not_bottom();
                }
            } else if !cc2.dot_in(*dot) {
                res_m.set_is_not_bottom();
            } else {
                let default_v = V::default();
                let new_v = V::dry_join((v1, cc1), (&default_v, cc1), sentinel)?;
                assert!(new_v.is_bottom());
                sentinel.delete_key()?;
            }
            sentinel.exit()?;
        }

        for (dot, v2) in m2.state.iter().filter(|(dot, _v2)| !m1.has(dot)) {
            sentinel.enter(dot)?;
            if !cc1.dot_in(*dot) {
                let _new_v = V::dry_join((&V::default(), cc1), (v2, cc2), sentinel)?;
                sentinel.create_key()?;
                // Just inserting a value to a map makes that map be not-bottom.
                res_m.set_is_not_bottom();
            }
            sentinel.exit()?;
        }

        Ok(res_m)
    }
}

/// A map from an arbitrary key type to a `V: DotStore`, whose computed dots is the union of the
/// dots of its values.
///
/// This is used to implement OR-Maps (Observed-Remove Maps).
///
/// Quoth the DSON paper:
///
/// > The merge in the Causal⟨DotMap⟩ applies the merge recursively on each of the keys in either
/// > domains, and keeps all non-⊥ values.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct DotMap<K, V> {
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            serialize = "K: Hash + Eq + serde::Serialize, V: serde::Serialize",
            deserialize = "K: Hash + Eq + serde::Deserialize<'de>, V: serde::Deserialize<'de>"
        ))
    )]
    state: HashMap<K, DotMapValue<V>, DsonRandomState>,
}

impl<K, V> FromIterator<(K, V)> for DotMap<K, V>
where
    K: Eq + Hash,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        Self {
            state: HashMap::from_iter(iter.into_iter().map(|(key, value)| {
                (
                    key,
                    DotMapValue {
                        value,
                        dots: Default::default(),
                    },
                )
            })),
        }
    }
}

impl<K, Q, V> Index<&Q> for DotMap<K, V>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash + ?Sized,
{
    type Output = V;

    fn index(&self, index: &Q) -> &Self::Output {
        &self.state.index(index).value
    }
}

/// A value in a [`DotMap`], which includes the value itself and a cached set of dots.
#[derive(Clone, Default)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct DotMapValue<V> {
    /// The value stored in the map.
    pub(super) value: V,

    /// This field, if set, holds a cached version of `value.dots()`.
    ///
    /// Its purpose is to ensure that calls to `self.dots()` (or analogously `self.add_dots_to()`)
    /// do not need to recurse into `value.dots()`. This is hugely important for performance, as it
    /// ensures that calls to `.dots()` on map entries are (generally) quite cheap, rather than
    /// having to walk the entire subtree of objects. This, in turn, allows `.dots()` to be
    /// used to cheaply determine whether a subtree needs to be entered to discover changes based
    /// on some other `CausalContext`.
    ///
    /// The field is read primarily in `impl DotStore`, and updated in `impl DotStoreJoin` using
    /// the `on_dot_change` machinery.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(super) dots: Option<CausalContext>,
}

impl<V: fmt::Debug> fmt::Debug for DotMapValue<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(if self.dots.is_some() { "V" } else { "v" })
            .field(&self.value)
            .finish()
    }
}

impl<V> PartialEq for DotMapValue<V>
where
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl<V> Eq for DotMapValue<V> where V: Eq {}

impl<V> DotStore for DotMapValue<V>
where
    V: DotStore,
{
    fn dots(&self) -> CausalContext {
        if let Some(dots) = &self.dots {
            debug_assert_eq!(dots, &self.value.dots());
            dots.clone()
        } else {
            self.value.dots()
        }
    }

    fn add_dots_to(&self, other: &mut CausalContext) {
        if let Some(dots) = &self.dots {
            debug_assert_eq!(dots, &self.value.dots());
            other.union(dots);
        } else {
            self.value.add_dots_to(other);
        }
    }

    fn is_bottom(&self) -> bool {
        self.value.is_bottom()
    }

    fn subset_for_inflation_from(&self, frontier: &CausalContext) -> Self {
        Self {
            value: self.value.subset_for_inflation_from(frontier),
            dots: None,
        }
    }
}

impl<S, V> DotStoreJoin<S> for DotMapValue<V>
where
    V: DotStoreJoin<S> + Default + fmt::Debug,
{
    fn join(
        (s1, ctx1): (Self, &CausalContext),
        (s2, ctx2): (Self, &CausalContext),
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<Self, S::Error>
    where
        Self: Sized,
        S: crate::sentinel::Sentinel,
    {
        // NOTE! When making changes to this method, consider if corresponding
        // changes need to be done to ::dry_join as well!

        let Self {
            value: s1,
            mut dots,
        } = s1;
        let Self { value: s2, dots: _ } = s2;
        let value = V::join(
            (s1, ctx1),
            (s2, ctx2),
            &mut |change| {
                if let Some(dots) = &mut dots {
                    match change {
                        DotChange::Add(dot) => {
                            dots.insert_dot(dot);
                        }
                        DotChange::Remove(dot) => {
                            dots.remove_dot(dot);
                        }
                    }
                }
                on_dot_change(change);
            },
            sentinel,
        )?;
        dots = Some(
            dots.map(|dots| {
                debug_assert_eq!(dots, value.dots(), "{value:?}");
                dots
            })
            .unwrap_or_else(|| value.dots()),
        );
        Ok(Self { value, dots })
    }
    fn dry_join(
        (s1, ctx1): (&Self, &CausalContext),
        (s2, ctx2): (&Self, &CausalContext),
        sentinel: &mut S,
    ) -> Result<DryJoinOutput, S::Error>
    where
        Self: Sized,
        S: Sentinel,
    {
        let Self { value: s1, dots: _ } = s1;
        let Self { value: s2, dots: _ } = s2;
        let value = V::dry_join((s1, ctx1), (s2, ctx2), sentinel)?;
        Ok(value)
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for DotMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.state.iter()).finish()
    }
}

impl<K, V> Default for DotMap<K, V> {
    fn default() -> Self {
        Self {
            state: create_map(),
        }
    }
}

impl<K, V> PartialEq for DotMap<K, V>
where
    K: Eq + Hash,
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.state.eq(&other.state)
    }
}

impl<K, V> Eq for DotMap<K, V>
where
    K: Eq + Hash,
    V: Eq,
{
}

impl<K, V> DotMap<K, V> {
    /// Constructs a [`DotMap`] with the given initial capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            state: create_map_with_capacity(capacity),
        }
    }

    /// Produces an iterator over the map's keys and values.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&K, &V)> {
        self.state.iter().map(|(k, v)| (k, &v.value))
    }

    // Insert the given key and value into the map.
    //
    // If the key is already present, its value is overwritten.
    //
    // Note, this is a low level operation. CRDT types should generally
    // not be manipulated directly by user code.
    #[doc(hidden)]
    pub fn insert(&mut self, key: K, value: V)
    where
        K: Eq + Hash,
    {
        self.state.insert(key, DotMapValue { value, dots: None });
    }

    /// Produces a mutable iterator over the map's keys and values.
    ///
    /// Invalidates the dots cache for all the map's entries, so calling `.dots()` on this
    /// collection after invoking this method may be quite slow (it has to call `.dots()` on all
    /// the entries).
    pub fn iter_mut_and_invalidate(&mut self) -> impl ExactSizeIterator<Item = (&K, &mut V)> {
        self.state.iter_mut().map(|(k, v)| {
            // see `get_mut` for why we need this
            v.dots = None;
            (k, &mut v.value)
        })
    }

    /// Produces an iterator over the map's keys.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> + '_ {
        self.state.keys()
    }

    /// Produces an iterator over the map's values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> {
        self.state.values().map(|v| &v.value)
    }

    /// Returns the number of keys in the map.
    pub fn len(&self) -> usize {
        self.state.len()
    }

    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    #[cfg(any(test, feature = "arbitrary"))]
    pub(crate) fn shrink(&self) -> Box<dyn Iterator<Item = Self>>
    where
        K: Clone + quickcheck::Arbitrary + Hash + Eq,
        V: Clone + quickcheck::Arbitrary,
    {
        Box::new(quickcheck::Arbitrary::shrink(&self.state).map(|state| Self { state }))
    }
}

impl<K, V> DotMap<K, V>
where
    K: Hash + Eq,
{
    /// Retrieves the associated value, if any, for the given key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.state.get(key).map(|v| &v.value)
    }

    /// Retrieves a mutable reference to the associated value, if any, for the given key.
    ///
    /// Invalidates the dots cache for the given map entry, so calling `.dots()` on this collection
    /// after invoking this method may be slower as it has to call `.dots()` on this entry to
    /// re-compute.
    pub fn get_mut_and_invalidate<Q>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.state.get_mut(key).map(|v| {
            // giving out `&mut v.value` ultimately permits changing `.value`, which in turn might
            // change its dots. if `v.value`'s dots change, then `v.dots` also needs to change to
            // match, but since we don't here know what that change (if any) might be, all we can
            // do is invalidate our cache so that it'll be re-computed at the next read.
            //
            // this problem (and solution) applies inductively: changing `v.value` changes
            // `v.dots`, which also changes `self.dots()`, which may in turn be cached somewhere
            // further up. however, since such a cache must _also_ have gone through `get_mut` (or
            // `iter_mut`), that cache must already have been invalidated for us to get here in the
            // first place.
            //
            // another option would be to somehow have a handle to those parents here (or
            // specifically, `&mut DotMapValue.dots`), but threading that through would be a
            // lifetime nightmare.
            v.dots = None;
            &mut v.value
        })
    }

    /// Returns `true` if the given key has a value in this map.
    pub fn has<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.state.contains_key(key)
    }

    /// Associates the value with the given key.
    ///
    /// Returns the previous value if any.
    pub fn set(&mut self, key: K, value: V) -> Option<V>
    where
        V: DotStore,
    {
        let dots = Some(value.dots());
        self.state
            .insert(key, DotMapValue { value, dots })
            .map(|v| v.value)
    }

    /// Removes the value with the given key.
    ///
    /// Returns the previous value if any.
    ///
    /// Be mindful that removing a key from a `DotMap` also changes its set of dots, but does _not_
    /// change the [`CausalContext`] in which the `DotMap` resides. As a result, removing a key in
    /// this way will make the CRDT that this `DotMap` represents imply the deletion of, not just
    /// the absence of, `key`.
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.state.remove(key).map(|v| v.value)
    }

    /// Retains only key-value pairs for which `f` returns `true`.
    ///
    /// See [`DotMap::remove`] for the CRDT implications of removing keys in this way.
    ///
    /// Note that this does not invalidate the cache, even though it may become inaccurate. For an
    /// invalidating alternative, use [`Self::retain_and_invalidate`].
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.state.retain(|k, v| f(k, &mut v.value))
    }

    /// Retains only key-value pairs for which `f` returns `true`.
    ///
    /// See [`DotMap::remove`] for the CRDT implications of removing keys in this way.
    ///
    /// Invalidates the dots cache for all entries, so calling `.dots()` on this collection
    /// after invoking this method may be slower as it has to call `.dots()` on each entry to
    /// re-compute.
    pub fn retain_and_invalidate<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut V) -> bool,
    {
        self.state.retain(|k, v| {
            // see comment in [`Self::get_mut_and_invalidate`]
            v.dots = None;
            f(k, &mut v.value)
        })
    }
}

impl<K, V> DotStore for DotMap<K, V>
where
    K: Hash + Eq + fmt::Debug + Clone,
    V: DotStore,
{
    fn add_dots_to(&self, other: &mut CausalContext) {
        for v in self.values() {
            v.add_dots_to(other);
        }
    }

    fn is_bottom(&self) -> bool {
        self.state.values().all(DotStore::is_bottom)
    }

    fn subset_for_inflation_from(&self, frontier: &CausalContext) -> Self {
        let mut delta = Self {
            state: create_map_with_capacity(self.state.len()),
        };

        for (k, v) in &self.state {
            let delta_v = v.subset_for_inflation_from(frontier);
            if !delta_v.is_bottom() {
                delta.state.insert(k.clone(), delta_v);
            }
        }

        delta
    }
}

impl<K, V, S> DotStoreJoin<S> for DotMap<K, V>
where
    K: Hash + Eq + fmt::Debug + Clone,
    V: DotStoreJoin<S> + Default + fmt::Debug,
    S: Visit<K> + KeySentinel,
    // needed for debug assertions of relevant_deletes optimization
    V: Clone + PartialEq,
{
    /// Formally (Equation 5):
    ///
    /// > (𝑚, 𝑐) ⊔ (𝑚′, 𝑐′) =
    /// >   (
    /// >       {𝑘 ↦ 𝑣(𝑘) | 𝑘 ∈ dom 𝑚 ∪ dom 𝑚′ ∧ 𝑣(𝑘) ≠ ⊥}
    /// >     , 𝑐 ∪ 𝑐′
    /// >   )
    /// >   where 𝑣(𝑘) = fst((𝑚(𝑘),𝑐) ⊔ (𝑚′(𝑘), 𝑐′))
    ///
    /// Informally:
    ///  - take the union of keys across the two stores:
    ///    - compute v as the join of the keys' values in the two maps (one may be ⊥)
    ///    - if v.store is ⊥, skip
    ///    - otherwise, include the k -> v.store mapping
    ///  - the resulting causal context is the union of the provided causal contexts
    fn join(
        (m1, cc1): (Self, &CausalContext),
        (mut m2, cc2): (Self, &CausalContext),
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<Self, S::Error> {
        // NOTE! When making changes to this method, consider if corresponding
        // changes need to be done to ::dry_join as well!

        // NOTE: the original code collects all the keys first, but that would require V: Clone
        // (and is unnecessarily inefficient), so we take a different approach that _should_ have
        // the exact same semantics: we iterate over m1 first, removing anything that's also in m2,
        // and then we iterate over what's left in m2.
        // TODO: we really shouldn't allocate a new Self here and then do a tonne of inserts,
        // since self can be quite large.
        let mut res_m = Self::with_capacity(m1.len().max(m2.len()));
        for (k, v1) in m1.state {
            sentinel.enter(&k)?;
            let v2 = m2.state.remove(&k).unwrap_or_default();
            if v2.is_bottom() {
                // TODO: We could maybe use something like a bloom filter here
                // rather than have to fully compare the dots. I think this would mean keeping a
                // bloom filter inside of `DotMapValue` that we also keep up to date, and then
                // taking the intersection of the bloom filters here, which should hopefully be
                // cheaper than what we currently do. An idea worth exploring at least!
                // NOTE: if v1 is also bottom (which _can_ happen), there is nothing to
                // delete, so there are definitely no relevant deletes.
                let relevant_deletes =
                    !v1.is_bottom() && v1.dots.as_ref().is_none_or(|dots| dots.any_dot_in(cc2));
                if !relevant_deletes {
                    // since v2 is bottom, if it contains any of the dots in v1, then that means it
                    // deletes _something_ under k, and so we need to recurse into k. otherwise, we
                    // can avoid recursing into the CRDT subtree under k entirely!
                    //
                    // in debug mode, validate that v1 indeed does not change as a result of
                    // joining with v2:
                    if cfg!(debug_assertions) {
                        let new_v = DotMapValue::<V>::join(
                            (v1.clone(), cc1),
                            (v2, cc2),
                            &mut |_| {},
                            // TODO: this should arguably be DummySentinel so that tests don't
                            // rely on the sentinel getting to see these operations just because
                            // the code is compiled in debug mode. however, doing so would require
                            // that we add `V: DotStoreJoin<DummySentinel>` as well, which would
                            // bubble up in a bunch of places.
                            sentinel,
                        )?;

                        if v1.is_bottom() {
                            // there are multiple ways to be bottom, so don't check strict equality
                            assert!(new_v.is_bottom(), "{v1:?}\nis bottom, but not\n{new_v:?}");
                        } else {
                            assert_eq!(v1, new_v);
                        }
                    }
                    // NOTE: this also preserves v1.dots, ensuring that we never have to walk
                    // all of v1 as part of the join (which would be bad as v1 could be the entire
                    // CRDT state not just delta size)!
                    res_m.state.insert(k, v1);
                    sentinel.exit()?;
                    continue;
                }
            }
            let new_v = DotMapValue::<V>::join((v1, cc1), (v2, cc2), on_dot_change, sentinel)?;
            if !new_v.is_bottom() {
                // Value was in m1, still is alive, so this is an update (but possibly v2 == v1)
                // TODO(#2): is ds1's author authorized to write this value?
                // TODO(#2): if v2 is Some, is ds2's author authorized to write this value?
                res_m.state.insert(k, new_v);
            } else {
                // If a value was previously bottom, it wouldn't have been in m1.state, as we don't
                // store bottom values. This means the value is being removed by ds2.
                // TODO(#2): is ds2's author authorized to clear this value?
                sentinel.delete_key()?;
            }
            sentinel.exit()?;
        }
        // NOTE: this now only contains keys that weren't in m1
        for (k, v2) in m2.state {
            sentinel.enter(&k)?;
            // TODO: implement relevant_deletes optimization here too (v1 and v2 are just
            // swapped). didn't do that initially since m1 tends to be the "main document" and m2
            // the "delta", so it's more important we avoid exhaustively walking m1 than m2.
            //
            // NOTE: However, consider the need for sentinels to observe all inserts. If we do
            // the optimization described above, a sentinel will no longer observe the contents
            // of new trees that are added to the document. We could add an associated const
            // to the Sentinel trait, that describes if the optimization above is allowed or not.
            let v1 = DotMapValue::<V>::default();
            let new_v = DotMapValue::<V>::join((v1, cc1), (v2, cc2), on_dot_change, sentinel)?;
            if !new_v.is_bottom() {
                // TODO(#2): is ds2's author authorized to write this value?
                sentinel.create_key()?;
                res_m.state.insert(k, new_v);
            } else {
                // TODO(#2): is ds2's author authorized to clear this value?
                // NOTE: do we even care that ds2 is trying to remove a key that doesn't exist?
            }
            sentinel.exit()?;
        }

        Ok(res_m)
    }

    fn dry_join(
        (m1, cc1): (&Self, &CausalContext),
        (m2, cc2): (&Self, &CausalContext),
        sentinel: &mut S,
    ) -> Result<DryJoinOutput, S::Error> {
        // For explanation of this method, see comments in ::join(..).

        let mut result = DryJoinOutput::bottom();
        for (k, v1) in m1.state.iter() {
            sentinel.enter(k)?;
            let default_v = Default::default();
            let v2: &DotMapValue<V> = m2.state.get(k).unwrap_or(&default_v);
            if v2.is_bottom() {
                let relevant_deletes =
                    !v1.is_bottom() && v1.dots.as_ref().is_none_or(|dots| dots.any_dot_in(cc2));
                if !relevant_deletes {
                    if cfg!(debug_assertions) {
                        let _new_v = DotMapValue::<V>::dry_join((v1, cc1), (v2, cc2), sentinel)?;
                    }
                    if !v1.is_bottom() {
                        result.set_is_not_bottom();
                    }

                    sentinel.exit()?;
                    continue;
                }
            }
            let new_v = DotMapValue::<V>::dry_join((v1, cc1), (v2, cc2), sentinel)?;
            if !new_v.is_bottom() {
                result.set_is_not_bottom();
            } else {
                sentinel.delete_key()?;
            }
            sentinel.exit()?;
        }

        for (k, v2) in m2.state.iter().filter(|(k, _v2)| !m1.has(k)) {
            sentinel.enter(k)?;
            let v1 = DotMapValue::<V>::default();
            let new_v = DotMapValue::<V>::dry_join((&v1, cc1), (v2, cc2), sentinel)?;
            if !new_v.is_bottom() {
                sentinel.create_key()?;
                result.set_is_not_bottom();
            }
            sentinel.exit()?;
        }

        Ok(result)
    }
}

#[cfg(any(test, feature = "arbitrary"))]
pub mod recording_sentinel;

#[expect(clippy::wildcard_enum_match_arm)]
#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, HashSet},
        num::NonZeroU64,
    };

    const SEQ_1: NonZeroU64 = NonZeroU64::MIN;
    const SEQ_2: NonZeroU64 = NonZeroU64::MIN.saturating_add(1);

    mod dotfun {
        use super::{super::*, *};
        use crate::sentinel::test::{NoChangeValidator, ValueCountingValidator};

        #[test]
        fn basic() {
            let mut map = DotFun::default();
            assert!(map.is_empty());
            assert!(map.is_bottom());
            assert_eq!(map.len(), 0);

            let dot = Dot::from(((0, 0), SEQ_1));
            assert!(!map.has(&dot));
            assert!(!map.dots().dot_in(dot));
            assert_eq!(map.get(&dot), None);
            assert_eq!(map.get_mut(&dot), None);

            assert_eq!(map.set(dot, "foo"), None);
            assert!(map.has(&dot));
            assert!(map.dots().dot_in(dot));
            assert_eq!(map.get(&dot).copied(), Some("foo"));
            assert_eq!(map.get_mut(&dot).copied(), Some("foo"));
            assert!(!map.is_empty());
            assert!(!map.is_bottom());
            assert_eq!(map.len(), 1);
        }

        #[test]
        fn join_bottoms() {
            let bottom = CausalDotStore::<DotFun<()>>::default();

            // We don't expect to see anything created or deleted
            // joining bottom with bottom with no causal context
            // should produce bottom and an empty causal context
            let join = bottom
                .test_join(
                    &bottom,
                    &mut |_| unreachable!("no dots added or removed"),
                    &mut NoChangeValidator,
                )
                .unwrap();
            assert_eq!(join.context, Default::default());
            assert!(join.store.is_bottom());
        }

        #[test]
        fn join_with_bottom() {
            let mut validator = ValueCountingValidator::default();

            let mut ds = CausalDotStore::<DotFun<_>>::default();
            let bottom = CausalDotStore::<DotFun<_>>::default();

            // joining non-bottom x with bottom should produce x (no changes are observed)
            let dot = Dot::from(((0, 0), SEQ_1));
            ds.store.set(dot, ());
            ds.context.insert_next_dot(dot);
            let join = ds
                .test_join(
                    &bottom,
                    &mut |_| unreachable!("no dots added or removed"),
                    &mut validator,
                )
                .unwrap();
            assert_eq!(join.context, ds.context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&dot).copied(), Some(()));
            assert!(validator.added.is_empty()); // we started with something and nothing was added
            assert!(validator.removed.is_empty());

            // joining bottom with non-bottom x should also produce x (but a change is observed)
            let join = bottom
                .test_join(
                    &ds,
                    &mut |change| assert_eq!(change, DotChange::Add(dot)),
                    &mut validator,
                )
                .unwrap();
            assert_eq!(join.context, ds.context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&dot).copied(), Some(()));
            assert_eq!(validator.added.len(), 1); // we started with nothing and something was added
            assert!(validator.removed.is_empty());
        }

        #[test]
        fn join_idempotecy() {
            let mut ds = CausalDotStore::<DotFun<_>>::default();
            let dot = Dot::from(((0, 0), SEQ_1));
            ds.store.set(dot, ());
            ds.context.insert_next_dot(dot);
            let join = ds
                .test_join(
                    &ds,
                    &mut |_| unreachable!("self-join means no dot changes"),
                    &mut NoChangeValidator,
                )
                .unwrap();
            assert_eq!(join.context, ds.context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&dot).copied(), Some(()));
        }

        #[test]
        fn join_keeps_independent() {
            let mut validator = ValueCountingValidator::default();

            let mut ds1 = CausalDotStore::<DotFun<_>>::default();
            let mut ds2 = CausalDotStore::<DotFun<_>>::default();

            let dot1 = Dot::from(((0, 0), SEQ_1));
            ds1.store.set(dot1, "dot1");
            ds1.context.insert_next_dot(dot1);
            let dot2 = Dot::from(((1, 0), SEQ_1));
            ds2.store.set(dot2, "dot2");
            ds2.context.insert_next_dot(dot2);

            let mut expected_context = ds1.context.clone();
            expected_context.union(&ds2.context);

            let join = ds1
                .test_join(
                    &ds2,
                    &mut |change| assert_eq!(change, DotChange::Add(dot2)),
                    &mut validator,
                )
                .unwrap();
            assert_eq!(validator.added, BTreeMap::from([("dot2", 1)]));
            assert!(validator.removed.is_empty());

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&dot1).copied(), Some("dot1"));
            assert_eq!(join.store.get(&dot2).copied(), Some("dot2"));
        }

        #[test]
        fn join_drops_removed() {
            let mut validator = ValueCountingValidator::default();

            let mut ds1 = CausalDotStore::<DotFun<_>>::default();
            let mut ds2 = CausalDotStore::<DotFun<_>>::default();

            let dot1 = Dot::from(((0, 0), SEQ_1));
            ds1.store.set(dot1, "dot1");
            ds1.context.insert_next_dot(dot1);
            let dot2 = Dot::from(((1, 0), SEQ_1));
            ds2.store.set(dot2, "dot2");
            ds2.context.insert_next_dot(dot2);

            // make it so that ds1 has seen dot2, but does not have a value for it,
            // which implies that ds1 explicitly removed dot2.
            ds1.context.insert_next_dot(dot2);

            let mut expected_context = ds1.context.clone();
            expected_context.union(&ds2.context);

            // also check that the join is symmetrical
            // (modulo the semantics of on_dot_change and sentinels)
            let join = ds1
                .test_join(
                    &ds2,
                    &mut |_| unreachable!("ds1 has seen all of ds2, so no dot changes"),
                    &mut validator,
                )
                .unwrap();

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&dot1).copied(), Some("dot1"));
            assert_eq!(join.store.get(&dot2), None);

            let mut added = HashSet::new();
            let mut removed = HashSet::new();
            let join = ds2
                .test_join(
                    &ds1,
                    &mut |change| match change {
                        DotChange::Add(d) if d == dot1 => {
                            assert!(added.insert(d), "on_dot_change added {d:?} twice");
                        }
                        DotChange::Remove(d) if d == dot2 => {
                            assert!(removed.insert(d), "on_dot_change removed {d:?} twice");
                        }
                        diff => unreachable!("{diff:?}"),
                    },
                    &mut validator,
                )
                .unwrap();

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&dot1).copied(), Some("dot1"));
            assert_eq!(join.store.get(&dot2), None);

            assert_eq!(added, HashSet::from_iter([dot1]));
            assert_eq!(removed, HashSet::from_iter([dot2]));

            assert_eq!(validator.added, BTreeMap::from([("dot1", 1)]));
            assert_eq!(validator.removed, BTreeMap::from([("dot2", 1)]));
        }
    }

    mod dotfunmap {
        use super::{super::*, *};
        use crate::sentinel::{DummySentinel, test::NoChangeValidator};

        #[test]
        fn basic() {
            let mut map = DotFunMap::default();
            assert!(map.is_empty());
            assert!(map.is_bottom());
            assert_eq!(map.len(), 0);

            let key = Dot::from(((9, 0), SEQ_1));
            let dot = Dot::from(((0, 0), SEQ_1));
            assert!(!map.has(&key));
            assert!(!map.dots().dot_in(dot));
            assert_eq!(map.get(&key), None);
            assert_eq!(map.get_mut(&key), None);

            assert_eq!(map.set(key, DotFun::default()), None);
            assert!(map.has(&key));
            assert_ne!(map.get(&key), None);
            assert_ne!(map.get_mut(&key), None);
            assert!(!map.is_empty());
            assert!(!map.is_bottom());
            assert_eq!(map.len(), 1);

            // since we inserted an empty DotStore, there are still no dots:
            assert!(!map.dots().dot_in(dot));
            // until we insert one:
            map.get_mut(&key).unwrap().set(dot, "bar");
            assert!(map.dots().dot_in(dot));
        }

        #[test]
        fn join_bottoms() {
            let bottom = CausalDotStore::<DotFunMap<DotFun<()>>>::default();

            // joining bottom with bottom with no causal context
            // should produce bottom and an empty causal context
            let join = bottom
                .test_join(
                    &bottom,
                    &mut |_| unreachable!("no dots added or removed"),
                    &mut NoChangeValidator,
                )
                .unwrap();
            assert_eq!(join.context, Default::default());
            assert!(join.store.is_bottom());
        }

        #[test]
        fn join_with_bottom() {
            let mut ds = CausalDotStore::<DotFunMap<DotFun<_>>>::default();
            let bottom = CausalDotStore::<DotFunMap<DotFun<_>>>::default();

            // joining non-bottom x with bottom should produce x
            let key = Dot::from(((9, 0), SEQ_1));
            let dot = Dot::from(((0, 0), SEQ_1));
            let mut v = DotFun::default();
            v.set(dot, ());
            ds.store.set(key, v.clone());
            ds.context.insert_next_dot(dot);
            let join = ds
                .test_join(
                    &bottom,
                    &mut |_| unreachable!("no dots added or removed"),
                    &mut DummySentinel,
                )
                .unwrap();
            assert_eq!(join.context, ds.context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&key), Some(&v));

            // and that should be symmetric
            // (modulo the semantics of on_dot_change and sentinels)
            let mut added = HashSet::new();
            let join = bottom
                .test_join(
                    &ds,
                    &mut |change| match change {
                        DotChange::Add(d) if [key, dot].contains(&d) => {
                            assert!(added.insert(d), "on_dot_change added {d:?} twice");
                        }
                        diff => unreachable!("{diff:?}"),
                    },
                    &mut DummySentinel,
                )
                .unwrap();
            assert_eq!(join.context, ds.context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&key), Some(&v));
            assert_eq!(added, HashSet::from_iter([key, dot]));
        }

        #[test]
        fn join_idempotecy() {
            let mut ds = CausalDotStore::<DotFunMap<DotFun<_>>>::default();

            let key = Dot::from(((9, 0), SEQ_1));
            let dot = Dot::from(((0, 0), SEQ_1));
            let mut v = DotFun::default();
            v.set(dot, ());
            ds.store.set(key, v.clone());
            ds.context.insert_next_dot(dot);

            let join = ds
                .test_join(
                    &ds,
                    &mut |_| unreachable!("self-join means no dot changes"),
                    &mut DummySentinel,
                )
                .unwrap();
            assert_eq!(join.context, ds.context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&key), Some(&v));
        }

        #[test]
        fn join_keeps_independent() {
            let mut ds1 = CausalDotStore::<DotFunMap<DotFun<_>>>::default();
            let mut ds2 = CausalDotStore::<DotFunMap<DotFun<_>>>::default();

            let key1 = Dot::from(((9, 0), SEQ_1));
            let dot1 = Dot::from(((0, 0), SEQ_1));
            let mut v1 = DotFun::default();
            v1.set(dot1, "dot1");
            ds1.store.set(key1, v1.clone());
            ds1.context.insert_next_dot(dot1);
            let key2 = Dot::from(((8, 0), SEQ_1));
            let dot2 = Dot::from(((1, 0), SEQ_1));
            let mut v2 = DotFun::default();
            v2.set(dot2, "dot2");
            ds2.store.set(key2, v2.clone());
            ds2.context.insert_next_dot(dot2);

            let mut expected_context = ds1.context.clone();
            expected_context.union(&ds2.context);

            let mut changes = HashMap::new();
            let join = ds1
                .test_join(
                    &ds2,
                    &mut |change| {
                        *changes.entry(change).or_default() += 1;
                    },
                    &mut DummySentinel,
                )
                .unwrap();

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&key1), Some(&v1));
            assert_eq!(join.store.get(&key2), Some(&v2));
            assert_eq!(
                changes,
                HashMap::from_iter([(DotChange::Add(dot2), 1), (DotChange::Add(key2), 1)])
            );
        }

        #[test]
        fn join_drops_removed() {
            let mut ds1 = CausalDotStore::<DotFunMap<DotFun<_>>>::default();
            let mut ds2 = CausalDotStore::<DotFunMap<DotFun<_>>>::default();

            let key1 = Dot::from(((9, 0), SEQ_1));
            let dot1 = Dot::from(((0, 0), SEQ_1));
            let mut v1 = DotFun::default();
            v1.set(dot1, "dot1");
            ds1.store.set(key1, v1.clone());
            ds1.context.insert_next_dot(dot1);
            let key2 = Dot::from(((8, 0), SEQ_1));
            let dot2 = Dot::from(((1, 0), SEQ_1));
            let mut v2 = DotFun::default();
            v2.set(dot2, "dot2");
            ds2.store.set(key2, v2.clone());
            ds2.context.insert_next_dot(dot2);

            // make it so that ds1 has seen dot2, but does not have a value for key1,
            // which implies that ds1 explicitly removed key1.
            ds1.context.insert_next_dot(key2);
            ds1.context.insert_next_dot(dot2);

            let mut expected_context = ds1.context.clone();
            expected_context.union(&ds2.context);

            // also check that the join is symmetrical
            // (modulo the semantics of on_dot_change and sentinels)
            let join = ds1
                .test_join(
                    &ds2,
                    &mut |_| unreachable!("ds1 has seen all of ds2, so no dot changes"),
                    &mut DummySentinel,
                )
                .unwrap();

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&key1), Some(&v1));
            assert_eq!(join.store.get(&key2), None);

            let mut added = HashSet::new();
            let mut removed = HashSet::new();
            let join = ds2
                .test_join(
                    &ds1,
                    &mut |change| match change {
                        DotChange::Add(d) if [dot1, key1].contains(&d) => {
                            assert!(added.insert(d), "on_dot_change added {d:?} twice");
                        }
                        DotChange::Remove(d) if [dot2, key2].contains(&d) => {
                            assert!(removed.insert(d), "on_dot_change removed {d:?} twice");
                        }
                        diff => unreachable!("{diff:?}"),
                    },
                    &mut DummySentinel,
                )
                .unwrap();

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&key1), Some(&v1));
            assert_eq!(join.store.get(&key2), None);
            assert_eq!(added, HashSet::from_iter([dot1, key1]));
            assert_eq!(removed, HashSet::from_iter([dot2, key2]));
        }

        #[test]
        fn nested_join() {
            let mut ds1 = CausalDotStore::<DotFunMap<DotFun<_>>>::default();
            let mut ds2 = CausalDotStore::<DotFunMap<DotFun<_>>>::default();

            // a single shared key this time so the join will need to join the DotFuns
            let key = Dot::from(((9, 0), SEQ_1));

            let dot1 = Dot::from(((0, 0), SEQ_1));
            let mut v1 = DotFun::default();
            v1.set(dot1, "dot1");
            ds1.store.set(key, v1.clone());
            ds1.context.insert_next_dot(dot1);

            // since we're testing nested join here too, make sure that join is _entirely_ obvious
            // by make it so that ds1 has seen dot2 but not repeated it, which implies that dot2 is
            // removed. we also add an extra value in ds2 for extra spice.
            let dot2 = Dot::from(((1, 0), SEQ_1));
            let dot3 = Dot::from(((1, 0), SEQ_2));
            let mut v2 = DotFun::default();
            v2.set(dot2, "dot2");
            v2.set(dot3, "dot3");
            ds2.store.set(key, v2.clone());
            ds2.context.extend([dot2, dot3]);
            ds1.context.insert_next_dot(dot2);

            let mut expected_context = ds1.context.clone();
            expected_context.union(&ds2.context);

            let expected_v = DotFun::join(
                (v1, &ds1.context),
                (v2, &ds2.context),
                &mut |change| assert_eq!(change, DotChange::Add(dot3)),
                &mut DummySentinel,
            )
            .unwrap();

            // also check that the join is symmetrical
            // (modulo the semantics of on_dot_change and sentinels)
            let mut added = HashSet::new();
            let mut removed = HashSet::new();
            let join = ds1
                .test_join(
                    &ds2,
                    &mut |change| match change {
                        DotChange::Add(d) if d == dot3 => {
                            assert!(added.insert(d), "on_dot_change added {d:?} twice");
                        }
                        diff => unreachable!("{diff:?}"),
                    },
                    &mut DummySentinel,
                )
                .unwrap();

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&key), Some(&expected_v));
            assert_eq!(added, HashSet::from_iter([dot3]));
            assert_eq!(removed, HashSet::from_iter([]));

            added.clear();
            removed.clear();
            let join = ds2
                .test_join(
                    &ds1,
                    &mut |change| match change {
                        DotChange::Add(d) if d == dot1 => {
                            assert!(added.insert(d), "on_dot_change added {d:?} twice");
                        }
                        DotChange::Remove(d) if d == dot2 => {
                            assert!(removed.insert(d), "on_dot_change removed {d:?} twice");
                        }
                        diff => unreachable!("{diff:?}"),
                    },
                    &mut DummySentinel,
                )
                .unwrap();

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(&key), Some(&expected_v));
            assert_eq!(added, HashSet::from_iter([dot1]));
            assert_eq!(removed, HashSet::from_iter([dot2]));
        }
    }

    mod dotmap {
        use super::{super::*, *};
        use crate::sentinel::test::{
            KeyCountingValidator, NoChangeValidator, ValueCountingValidator,
        };

        #[test]
        fn basic() {
            let mut map = DotMap::default();
            assert!(map.is_empty());
            assert!(map.is_bottom());
            assert_eq!(map.len(), 0);

            let key = "foo";
            let dot = Dot::from(((0, 0), SEQ_1));
            assert!(!map.has(key));
            assert!(!map.dots().dot_in(dot));
            assert_eq!(map.get(key), None);
            assert_eq!(map.get_mut_and_invalidate(key), None);

            assert_eq!(map.set(key, DotFun::default()), None);
            assert!(map.has(key));
            assert_ne!(map.get(key), None);
            assert_ne!(map.get_mut_and_invalidate(key), None);
            assert!(!map.is_empty());
            assert!(map.is_bottom(), "map is bottom since all values are bottom");
            assert_eq!(map.len(), 1);

            // since we inserted an empty DotStore, there are still no dots:
            assert!(!map.dots().dot_in(dot));
            // until we insert one:
            map.get_mut_and_invalidate(key).unwrap().set(dot, "bar");
            assert!(map.dots().dot_in(dot));
            assert!(!map.is_bottom());
        }

        #[test]
        fn join_bottoms() {
            let bottom = CausalDotStore::<DotMap<(), DotFun<()>>>::default();

            // joining bottom with bottom with no causal context
            // should produce bottom and an empty causal context
            let join = bottom
                .test_join(
                    &bottom,
                    &mut |_| unreachable!("no dots added or removed"),
                    &mut NoChangeValidator,
                )
                .unwrap();
            assert_eq!(join.context, Default::default());
            assert!(join.store.is_bottom());
        }

        #[test]
        fn join_with_bottom() {
            let mut validator = KeyCountingValidator::default();

            let mut ds = CausalDotStore::<DotMap<_, DotFun<_>>>::default();
            let bottom = CausalDotStore::<DotMap<_, DotFun<_>>>::default();

            // joining non-bottom x with bottom should produce x (and no changes are observed)
            let key = "foo";
            let dot = Dot::from(((0, 0), SEQ_1));
            let mut v = DotFun::default();
            v.set(dot, ());
            ds.store.set(key, v.clone());
            ds.context.insert_next_dot(dot);
            let join = ds
                .test_join(
                    &bottom,
                    &mut |_| unreachable!("no dots added or removed"),
                    &mut validator,
                )
                .unwrap();
            assert_eq!(join.context, ds.context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(key), Some(&v));
            assert_eq!(validator.added, 0);
            assert_eq!(validator.removed, 0);

            // joining bottom with non-bottom x should produce x (and a change is observed)
            let join = bottom
                .test_join(
                    &ds,
                    &mut |change| assert_eq!(change, DotChange::Add(dot)),
                    &mut validator,
                )
                .unwrap();
            assert_eq!(join.context, ds.context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(key), Some(&v));
            assert_eq!(validator.added, 1);
            assert_eq!(validator.removed, 0);
        }

        #[test]
        fn join_idempotecy() {
            let mut ds = CausalDotStore::<DotMap<_, DotFun<_>>>::default();

            let key = "foo";
            let dot = Dot::from(((0, 0), SEQ_1));
            let mut v = DotFun::default();
            v.set(dot, ());
            ds.store.set(key, v.clone());
            ds.context.insert_next_dot(dot);

            let join = ds
                .test_join(
                    &ds,
                    &mut |_| unreachable!("self-join means no dot changes"),
                    &mut NoChangeValidator,
                )
                .unwrap();
            assert_eq!(join.context, ds.context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(key), Some(&v));
        }

        #[test]
        fn join_keeps_independent() {
            let mut validator = KeyCountingValidator::default();

            let mut ds1 = CausalDotStore::<DotMap<_, DotFun<_>>>::default();
            let mut ds2 = CausalDotStore::<DotMap<_, DotFun<_>>>::default();

            let key1 = "foo";
            let dot1 = Dot::from(((0, 0), SEQ_1));
            let mut v1 = DotFun::default();
            v1.set(dot1, "dot1");
            ds1.store.set(key1, v1.clone());
            ds1.context.insert_next_dot(dot1);
            let key2 = "bar";
            let dot2 = Dot::from(((1, 0), SEQ_1));
            let mut v2 = DotFun::default();
            v2.set(dot2, "dot2");
            ds2.store.set(key2, v2.clone());
            ds2.context.insert_next_dot(dot2);

            let mut expected_context = ds1.context.clone();
            expected_context.union(&ds2.context);

            let join = ds1
                .test_join(
                    &ds2,
                    &mut |change| assert_eq!(change, DotChange::Add(dot2)),
                    &mut validator,
                )
                .unwrap();
            assert_eq!(validator.added, 1);
            assert_eq!(validator.removed, 0);

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(key1), Some(&v1));
            assert_eq!(join.store.get(key2), Some(&v2));
        }

        #[test]
        fn join_drops_removed() {
            let mut validator = KeyCountingValidator::default();

            let mut ds1 = CausalDotStore::<DotMap<_, DotFun<_>>>::default();
            let mut ds2 = CausalDotStore::<DotMap<_, DotFun<_>>>::default();

            let key1 = "foo";
            let dot1 = Dot::from(((0, 0), SEQ_1));
            let mut v1 = DotFun::default();
            v1.set(dot1, "dot1");
            ds1.store.set(key1, v1.clone());
            ds1.context.insert_next_dot(dot1);
            let key2 = "bar";
            let dot2 = Dot::from(((1, 0), SEQ_1));
            let mut v2 = DotFun::default();
            v2.set(dot2, "dot2");
            ds2.store.set(key2, v2.clone());
            ds2.context.insert_next_dot(dot2);

            // make it so that ds1 has seen dot2, but does not have a value for key1,
            // which implies that ds1 explicitly removed key1.
            ds1.context.insert_next_dot(dot2);

            let mut expected_context = ds1.context.clone();
            expected_context.union(&ds2.context);

            // also check that the join is symmetrical
            let join = ds1
                .test_join(
                    &ds2,
                    &mut |_| unreachable!("ds1 has seen all of ds2, so no dot changes"),
                    &mut validator,
                )
                .unwrap();

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(key1), Some(&v1));
            assert_eq!(join.store.get(key2), None);

            let mut added = HashSet::new();
            let mut removed = HashSet::new();
            let join = ds2
                .test_join(
                    &ds1,
                    &mut |change| match change {
                        DotChange::Add(d) if d == dot1 => {
                            assert!(added.insert(d), "on_dot_change added {d:?} twice");
                        }
                        DotChange::Remove(d) if d == dot2 => {
                            assert!(removed.insert(d), "on_dot_change removed {d:?} twice");
                        }
                        diff => unreachable!("{diff:?}"),
                    },
                    &mut validator,
                )
                .unwrap();

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(key1), Some(&v1));
            assert_eq!(join.store.get(key2), None);

            assert_eq!(added, HashSet::from_iter([dot1]));
            assert_eq!(removed, HashSet::from_iter([dot2]));

            assert_eq!(validator.added, 1);
            assert_eq!(validator.removed, 1);
        }

        #[test]
        fn nested_join() {
            let mut validator = ValueCountingValidator::default();

            let mut ds1 = CausalDotStore::<DotMap<_, DotFun<_>>>::default();
            let mut ds2 = CausalDotStore::<DotMap<_, DotFun<_>>>::default();

            // a single shared key this time so the join will need to join the DotFuns
            let key = "foo";

            let dot1 = Dot::from(((0, 0), SEQ_1));
            let mut v1 = DotFun::default();
            v1.set(dot1, "dot1");
            ds1.store.set(key, v1.clone());
            ds1.context.insert_next_dot(dot1);

            // since we're testing nested join here too, make sure that join is _entirely_ obvious
            // by make it so that ds1 has seen dot2 but not repeated it, which implies that dot2 is
            // removed. we also add an extra value in ds2 for extra spice.
            let dot2 = Dot::from(((1, 0), SEQ_1));
            let dot3 = Dot::from(((1, 0), SEQ_2));
            let mut v2 = DotFun::default();
            v2.set(dot2, "dot2");
            v2.set(dot3, "dot3");
            ds2.store.set(key, v2.clone());
            ds2.context.extend([dot2, dot3]);
            ds1.context.insert_next_dot(dot2);

            let mut expected_context = ds1.context.clone();
            expected_context.union(&ds2.context);

            let expected_v = DotFun::join(
                (v1, &ds1.context),
                (v2, &ds2.context),
                &mut |_| {},
                &mut validator,
            )
            .unwrap();
            assert_eq!(validator.added, BTreeMap::from([("dot3", 1)]));
            assert!(validator.removed.is_empty());

            // also check that the join is symmetrical
            // (modulo the semantics of on_dot_change and sentinels)
            let mut added = HashSet::new();
            let mut removed = HashSet::new();
            let join = ds1
                .test_join(
                    &ds2,
                    &mut |change| match change {
                        DotChange::Add(d) if d == dot3 => {
                            assert!(added.insert(d), "on_dot_change added {d:?} twice");
                        }
                        diff => unreachable!("{diff:?}"),
                    },
                    &mut validator,
                )
                .unwrap();

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(key), Some(&expected_v));
            assert_eq!(added, HashSet::from_iter([dot3]));
            assert_eq!(removed, HashSet::from_iter([]));

            added.clear();
            removed.clear();
            let join = ds2
                .test_join(
                    &ds1,
                    &mut |change| match change {
                        DotChange::Add(d) if d == dot1 => {
                            assert!(added.insert(d), "on_dot_change added {d:?} twice");
                        }
                        DotChange::Remove(d) if d == dot2 => {
                            assert!(removed.insert(d), "on_dot_change removed {d:?} twice");
                        }
                        diff => unreachable!("{diff:?}"),
                    },
                    &mut validator,
                )
                .unwrap();

            assert_eq!(join.context, expected_context);
            assert!(!join.store.is_bottom());
            assert_eq!(join.store.get(key), Some(&expected_v));

            assert_eq!(added, HashSet::from_iter([dot1]));
            assert_eq!(removed, HashSet::from_iter([dot2]));

            assert_eq!(validator.added, BTreeMap::from([("dot1", 1), ("dot3", 2)]));
            assert_eq!(validator.removed, BTreeMap::from([("dot2", 1)]));
        }
    }
}
