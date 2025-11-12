//! Transaction-based API for ergonomic CRDT mutations.
//!
//! This module provides a transaction-based API for making changes to DSON stores.
//! Unlike the callback-based `api` module, transactions provide:
//!
//! - **Method chaining** - No nested callbacks
//! - **Explicit conflict handling** - Enums force handling of type conflicts
//! - **Automatic rollback** - Changes drop unless you call `commit()`
//! - **Automatic delta management** - Deltas accumulate and return on commit
//!
//! # Example
//!
//! ```
//! use dson::{CausalDotStore, Identifier, OrMap, crdts::mvreg::MvRegValue, transaction::CrdtValue};
//! use dson::crdts::snapshot::ToValue;
//!
//! let mut store = CausalDotStore::<OrMap<String>>::default();
//! let id = Identifier::new(0, 0);
//!
//! // Create a transaction
//! let mut tx = store.transact(id);
//!
//! // Write values
//! tx.write_register("name", MvRegValue::String("Alice".to_string()));
//! tx.write_register("age", MvRegValue::U64(30));
//!
//! // IMPORTANT: You must call commit() or changes are lost
//! let delta = tx.commit();
//!
//! // Read with explicit type handling
//! let tx = store.transact(id);
//! match tx.get(&"name".to_string()) {
//!     Some(CrdtValue::Register(reg)) => {
//!         println!("Name: {:?}", reg.value().unwrap());
//!     }
//!     Some(CrdtValue::Conflicted(conflicts)) => {
//!         println!("Type conflict!");
//!     }
//!     None => {
//!         println!("Key not found");
//!     }
//!     _ => {}
//! }
//! ```
//!
//! # Transaction Semantics
//!
//! Both [`MapTransaction`] and [`ArrayTransaction`] clone the store and work on the copy.
//! Changes apply immediately to the clone, enabling reads within the transaction to see
//! uncommitted changes. Call `commit()` to apply changes permanently. Drop the transaction
//! without committing to discard all changes (automatic rollback).
//!
//! ## How Transactions Work
//!
//! - **On creation**: The store is cloned
//! - **During operations**: Changes apply to the cloned store
//! - **On commit**: The clone swaps back into the original store
//! - **On drop**: Changes discard automatically if not committed
//!
//! ## Why This Design
//!
//! This provides:
//! - **Automatic rollback**: Drop the transaction to undo changes
//! - **Isolation**: Reads see uncommitted changes within the same transaction
//! - **Simplicity**: What you write is what you read
//!
//! ## Performance Tradeoff
//!
//! The transaction API trades performance for ergonomics. Top-level transactions clone the store
//! on creation and apply each operation eagerly to the clone. This enables rollback support
//! and ensures reads within the transaction see uncommitted changes.
//!
//! Benchmarks on an empty map show **2-2.5x overhead** compared to the raw API:
//!
//! | Operation | Raw API | Transaction | Overhead |
//! |-----------|---------|-------------|----------|
//! | Insert    | 156 ns  | 347 ns      | 2.2x     |
//! | Update    | 159 ns  | 344 ns      | 2.2x     |
//! | Remove    | 50 ns   | 69 ns       | 1.4x     |
//!
//! The overhead stems from the clone-and-swap implementation. Top-level transactions clone the
//! store on creation and apply each operation eagerly to the clone. This ensures reads within
//! the transaction see uncommitted changes and enables automatic rollback on drop.
//!
//! ### Nested Transaction Optimization
//!
//! Nested transactions (`in_map`, `in_array`, `insert_map`, `insert_array`) use `mem::take`
//! instead of cloning the parent store. This moves nested structures without copying:
//!
//! - **Shallow nesting (1-2 levels)**: Minimal impact
//! - **Deep nesting (3+ levels)**: Savings from avoided parent store clones
//! - **Large nested collections**: Saves proportional to parent store size
//!
//! The ~200-300ns overhead per operation is acceptable for most applications. For
//! latency-critical single-field updates, use [`api`](crate::api). For complex mutations
//! where clarity and safety outweigh microseconds, use transactions
//!
//! # Type Conflict Handling
//!
//! DSON's unique feature is preserving type conflicts. When different replicas
//! concurrently write different types to the same key, DSON preserves both.
//! The transaction API exposes this through the [`CrdtValue`] enum:
//!
//! ```no_run
//! # use dson::transaction::{MapTransaction, CrdtValue};
//! # let tx: MapTransaction<String> = todo!();
//! match tx.get(&"field".to_string()) {
//!     Some(CrdtValue::Map(map)) => { /* single type: map */ }
//!     Some(CrdtValue::Array(array)) => { /* single type: array */ }
//!     Some(CrdtValue::Register(reg)) => { /* single type: register */ }
//!     Some(CrdtValue::Conflicted(c)) => {
//!         // Type conflict!
//!         if c.has_map() && c.has_array() {
//!             // Application must resolve
//!         }
//!     }
//!     None => { /* key doesn't exist */ }
//!     Some(CrdtValue::Empty) => { /* key exists but is empty */ }
//! }
//! ```
//!
//! # Nested Operations
//!
//! The transaction API provides uniform ergonomics at all nesting levels:
//!
//! ```
//! # use dson::{CausalDotStore, Identifier, OrMap};
//! # use dson::crdts::mvreg::MvRegValue;
//! # let mut store = CausalDotStore::<OrMap<String>>::default();
//! # let id = Identifier::new(0, 0);
//! let mut tx = store.transact(id);
//!
//! tx.in_map("user", |user_tx| {
//!     user_tx.write_register("email", MvRegValue::String("alice@example.com".to_string()));
//!     user_tx.write_register("age", MvRegValue::U64(30));
//!
//!     user_tx.in_array("tags", |tags_tx| {
//!         tags_tx.insert_register(0, MvRegValue::String("admin".to_string()));
//!         // Nested transaction commits automatically when closure returns
//!     });
//!     // Nested transaction commits automatically when closure returns
//! });
//!
//! // Top-level transaction requires explicit commit
//! let delta = tx.commit();
//! ```
//!
//! **Important**: Nested transactions (`in_map`, `in_array`, `insert_map`, `insert_array`)
//! commit automatically when their closure returns. Only the top-level transaction requires
//! an explicit `commit()` call.
//!
//! Use [`MapTransaction::in_map`] and [`MapTransaction::in_array`] for nesting.
//! Use [`ArrayTransaction::insert_map`] and [`ArrayTransaction::insert_array`]
//! for arrays containing collections.

mod array_transaction;
mod conflicted;
mod crdt_value;
mod delta;
mod map_transaction;

pub use array_transaction::ArrayTransaction;
pub use conflicted::ConflictedValue;
pub use crdt_value::CrdtValue;
pub use delta::Delta;
pub use map_transaction::MapTransaction;
