use crate::crdts::Value;
use crate::crdts::mvreg::MvRegValue;
use crate::dotstores::DotStoreJoin;
use crate::sentinel::DummySentinel;
use crate::transaction::{CrdtValue, Delta, MapTransaction};
use crate::{CausalDotStore, ExtensionType, Identifier, OrArray, OrMap};
use std::fmt;

/// Transaction for mutating an [`OrArray`].
///
/// Similar to [`MapTransaction`](crate::transaction::MapTransaction) but for arrays.
/// Operations apply eagerly to a cloned store, and `commit()` swaps the modified clone
/// back to the original store.
///
/// See the [module documentation](crate::transaction) for details on eager application
/// semantics with automatic rollback support.
///
/// # Example
///
/// ```rust
/// use dson::{CausalDotStore, Identifier, OrArray, transaction::ArrayTransaction};
///
/// let mut store = CausalDotStore::<OrArray>::default();
/// let id = Identifier::new(0, 0);
///
/// let mut tx = ArrayTransaction::new(&mut store, id);
/// // Array operations...
/// let delta = tx.commit();
/// ```
pub struct ArrayTransaction<'a, C = crate::crdts::NoExtensionTypes>
where
    C: ExtensionType,
{
    original_store: &'a mut CausalDotStore<OrArray<C>>,
    working_store: CausalDotStore<OrArray<C>>,
    id: Identifier,
    changes: Vec<CausalDotStore<OrArray<C>>>,
}

impl<'a, C> ArrayTransaction<'a, C>
where
    C: ExtensionType,
{
    /// Creates a new transaction for the given array store.
    ///
    /// The transaction clones the store and exclusively borrows the original until committed.
    /// Changes apply to the clone, enabling automatic rollback on drop.
    pub fn new(store: &'a mut CausalDotStore<OrArray<C>>, id: Identifier) -> Self
    where
        C: Clone,
    {
        let working_store = store.clone();
        Self {
            working_store,
            original_store: store,
            id,
            changes: Vec::new(),
        }
    }

    /// Creates a nested transaction without cloning.
    ///
    /// Used internally for nested transactions (`insert_map`, `insert_array`).
    /// Nested transactions don't need rollback support since they commit
    /// automatically when the closure returns. The store is swapped back on commit.
    pub(crate) fn new_nested(store: &'a mut CausalDotStore<OrArray<C>>, id: Identifier) -> Self {
        // Take ownership of store contents via mem::take, leaving default in its place
        let working_store = std::mem::take(store);
        Self {
            working_store,
            original_store: store,
            id,
            changes: Vec::new(),
        }
    }

    /// Combines accumulated deltas, swaps the working store to the original, and returns the delta.
    ///
    /// All changes have been applied to the working store clone. This method swaps the
    /// working store back to the original store reference, making the changes permanent.
    /// The combined delta is returned for network transmission.
    pub fn commit(mut self) -> Delta<CausalDotStore<OrArray<C>>>
    where
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        // Swap the working store back to the original to make changes permanent
        *self.original_store = self.working_store;

        if self.changes.is_empty() {
            return Delta::new(CausalDotStore::default());
        }

        let mut combined = self.changes.remove(0);
        for delta in self.changes.drain(..) {
            combined = combined
                .join(delta, &mut DummySentinel)
                .expect("DummySentinel is infallible");
        }

        Delta::new(combined)
    }

    fn record_change(&mut self, delta: CausalDotStore<OrArray<C>>) {
        self.changes.push(delta);
    }

    /// Inserts a register value at the given index.
    ///
    /// If `idx` is greater than the array length, the value is appended.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use dson::{CausalDotStore, Identifier, OrArray, transaction::ArrayTransaction};
    /// # use dson::crdts::mvreg::MvRegValue;
    /// # let mut store = CausalDotStore::<OrArray>::default();
    /// # let id = Identifier::new(0, 0);
    /// let mut tx = ArrayTransaction::new(&mut store, id);
    /// tx.insert_register(0, MvRegValue::String("first".to_string()));
    /// tx.insert_register(1, MvRegValue::U64(42));
    /// let delta = tx.commit();
    /// ```
    pub fn insert_register(&mut self, idx: usize, value: impl Into<MvRegValue>)
    where
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        let delta = self.working_store.store.insert_idx_register(
            idx,
            value.into(),
            &self.working_store.context,
            self.id,
        );

        // Apply delta to working store immediately so subsequent operations see updated state
        self.working_store
            .join_or_replace_with(delta.store.clone(), &delta.context);

        self.record_change(delta);
    }

    /// Removes the element at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= len()`.
    pub fn remove(&mut self, idx: usize)
    where
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        let delta = self
            .working_store
            .store
            .remove(idx, &self.working_store.context, self.id);

        // Apply delta to working store immediately so subsequent operations see updated state
        self.working_store
            .join_or_replace_with(delta.store.clone(), &delta.context);

        self.record_change(delta);
    }

    /// Inserts a map at the given index.
    ///
    /// The closure receives a mutable reference to a `MapTransaction` for
    /// configuring the nested map.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use dson::{CausalDotStore, Identifier, OrArray, transaction::ArrayTransaction};
    /// # use dson::crdts::mvreg::MvRegValue;
    /// # let mut store = CausalDotStore::<OrArray>::default();
    /// # let id = Identifier::new(0, 0);
    /// let mut tx = ArrayTransaction::new(&mut store, id);
    /// tx.insert_map(0, |task_tx| {
    ///     task_tx.write_register("title", MvRegValue::String("Write docs".to_string()));
    ///     task_tx.write_register("done", MvRegValue::Bool(false));
    /// });
    /// let delta = tx.commit();
    /// ```
    pub fn insert_map<F>(&mut self, idx: usize, f: F)
    where
        F: FnOnce(&mut MapTransaction<'_, String, C>),
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        // Create empty map for child
        let mut child_store = CausalDotStore {
            store: OrMap::<String, C>::default(),
            context: self.working_store.context.clone(),
        };

        let mut child_tx = MapTransaction::new_nested(&mut child_store, self.id);
        f(&mut child_tx);
        let child_delta = child_tx.commit();

        // Insert the map into the array
        let delta = self.working_store.store.insert_idx_with(
            idx,
            |_ctx, _id| child_delta.0.map_store(Value::Map),
            &self.working_store.context,
            self.id,
        );

        // Apply delta to working store immediately so subsequent operations see updated state
        self.working_store
            .join_or_replace_with(delta.store.clone(), &delta.context);

        self.record_change(delta);
    }

    /// Inserts an array at the given index.
    ///
    /// The closure receives a mutable reference to an `ArrayTransaction` for
    /// configuring the nested array.
    pub fn insert_array<F>(&mut self, idx: usize, f: F)
    where
        F: FnOnce(&mut ArrayTransaction<'_, C>),
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        // Create empty array for child
        let mut child_store = CausalDotStore {
            store: OrArray::<C>::default(),
            context: self.working_store.context.clone(),
        };

        let mut child_tx = ArrayTransaction::new_nested(&mut child_store, self.id);
        f(&mut child_tx);
        let child_delta = child_tx.commit();

        // Insert the array into the parent array
        let delta = self.working_store.store.insert_idx_with(
            idx,
            |_ctx, _id| child_delta.0.map_store(Value::Array),
            &self.working_store.context,
            self.id,
        );

        // Apply delta to working store immediately so subsequent operations see updated state
        self.working_store
            .join_or_replace_with(delta.store.clone(), &delta.context);

        self.record_change(delta);
    }

    /// Gets the element at the given index.
    ///
    /// Returns `None` if the index is out of bounds.
    ///
    /// # Isolation Semantics
    ///
    /// This reads the current state of the array, which includes all changes
    /// made during this transaction. Array operations apply immediately to the store,
    /// so `get` sees uncommitted changes (eager/read-uncommitted semantics).
    ///
    /// This matches the behavior of [`MapTransaction::get`](crate::transaction::MapTransaction::get)
    /// and enables consistent behavior in nested transactions.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use dson::{CausalDotStore, Identifier, OrArray, transaction::{ArrayTransaction, CrdtValue}};
    /// # use dson::crdts::mvreg::MvRegValue;
    /// # let mut store = CausalDotStore::<OrArray>::default();
    /// # let id = Identifier::new(0, 0);
    /// let mut tx = ArrayTransaction::new(&mut store, id);
    /// tx.insert_register(0, MvRegValue::String("first".to_string()));
    ///
    /// // Array tx sees uncommitted changes
    /// match tx.get(0) {
    ///     Some(CrdtValue::Register(reg)) => {
    ///         // Can read the value we just inserted
    ///     }
    ///     _ => {}
    /// }
    /// # let _ = tx.commit();
    /// ```
    pub fn get(&self, idx: usize) -> Option<CrdtValue<'_, usize, C>> {
        let value = self.working_store.store.get(idx)?;
        Some(CrdtValue::from_type_variant(value))
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.working_store.store.len()
    }

    /// Returns `true` if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.working_store.store.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crdts::NoExtensionTypes;
    use crate::{CausalDotStore, Identifier, OrArray};

    #[test]
    fn array_transaction_new() {
        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);
        let _tx = ArrayTransaction::new(&mut store, id);
    }

    #[test]
    fn array_transaction_commit_empty() {
        use crate::crdts::NoExtensionTypes;

        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);

        let tx = ArrayTransaction::new(&mut store, id);
        let delta = tx.commit();

        // Empty transaction should produce empty delta
        assert!(delta.0.store.is_empty());
    }

    #[test]
    fn array_transaction_insert_register() {
        use crate::crdts::{NoExtensionTypes, mvreg::MvRegValue, snapshot::ToValue};

        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = ArrayTransaction::new(&mut store, id);
            tx.insert_register(0, MvRegValue::String("hello".to_string()));
            let _delta = tx.commit();
        }

        // Verify insertion
        let val = store.store.get(0).expect("item should exist");
        assert_eq!(
            val.reg.value().unwrap(),
            &MvRegValue::String("hello".to_string())
        );
    }

    #[test]
    fn array_transaction_remove() {
        use crate::crdts::{NoExtensionTypes, mvreg::MvRegValue};

        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);

        // Insert two items
        {
            let mut tx = ArrayTransaction::new(&mut store, id);
            tx.insert_register(0, MvRegValue::U64(1));
            tx.insert_register(1, MvRegValue::U64(2));
            let _delta = tx.commit();
        }

        assert_eq!(store.store.len(), 2);

        // Remove first item
        {
            let mut tx = ArrayTransaction::new(&mut store, id);
            tx.remove(0);
            let _delta = tx.commit();
        }

        assert_eq!(store.store.len(), 1);
    }

    #[test]
    fn array_transaction_sequential_inserts_preserve_order() {
        // Regression test: ensure sequential inserts work correctly
        // Bug was that index clamping used stale array length (always 0)
        // causing all inserts to go to position 0
        use crate::crdts::{NoExtensionTypes, mvreg::MvRegValue, snapshot::ToValue};

        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = ArrayTransaction::new(&mut store, id);
            tx.insert_register(0, MvRegValue::String("first".to_string()));
            tx.insert_register(1, MvRegValue::String("second".to_string()));
            tx.insert_register(2, MvRegValue::String("third".to_string()));
            let _delta = tx.commit();
        }

        assert_eq!(store.store.len(), 3);

        // Verify order is deterministic and correct
        let item0 = store.store.get(0).expect("item 0 should exist");
        assert_eq!(
            item0.reg.value().unwrap(),
            &MvRegValue::String("first".to_string())
        );

        let item1 = store.store.get(1).expect("item 1 should exist");
        assert_eq!(
            item1.reg.value().unwrap(),
            &MvRegValue::String("second".to_string())
        );

        let item2 = store.store.get(2).expect("item 2 should exist");
        assert_eq!(
            item2.reg.value().unwrap(),
            &MvRegValue::String("third".to_string())
        );
    }

    #[test]
    fn array_transaction_insert_map() {
        use crate::crdts::{NoExtensionTypes, mvreg::MvRegValue, snapshot::ToValue};

        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = ArrayTransaction::new(&mut store, id);
            tx.insert_map(0, |map_tx| {
                map_tx.write_register("name", MvRegValue::String("Task 1".to_string()));
            });
            let _delta = tx.commit();
        }

        // Verify map was inserted
        let item = store.store.get(0).unwrap();
        let name = item.map.get(&"name".to_string()).unwrap();
        assert_eq!(
            name.reg.value().unwrap(),
            &MvRegValue::String("Task 1".to_string())
        );
    }

    #[test]
    fn array_transaction_insert_array() {
        use crate::crdts::{NoExtensionTypes, mvreg::MvRegValue};

        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = ArrayTransaction::new(&mut store, id);
            tx.insert_array(0, |nested_tx| {
                nested_tx.insert_register(0, MvRegValue::U64(1));
                nested_tx.insert_register(1, MvRegValue::U64(2));
            });
            let _delta = tx.commit();
        }

        // Verify nested array was created
        let item = store.store.get(0).unwrap();
        assert_eq!(item.array.len(), 2);
    }

    #[test]
    fn array_transaction_get() {
        use crate::crdts::{NoExtensionTypes, mvreg::MvRegValue, snapshot::ToValue};
        use crate::transaction::CrdtValue;

        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = ArrayTransaction::new(&mut store, id);
            tx.insert_register(0, MvRegValue::String("hello".to_string()));
            let _delta = tx.commit();
        }

        {
            let tx = ArrayTransaction::new(&mut store, id);
            let value = tx.get(0).expect("should have item at 0");

            match value {
                CrdtValue::Register(reg) => {
                    assert_eq!(
                        reg.value().unwrap(),
                        &MvRegValue::String("hello".to_string())
                    );
                }
                _ => panic!("Expected Register variant"),
            }
        }
    }

    #[test]
    fn array_transaction_get_returns_crdt_value() {
        use crate::crdts::{NoExtensionTypes, mvreg::MvRegValue, snapshot::ToValue};
        use crate::transaction::CrdtValue;

        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = ArrayTransaction::new(&mut store, id);
            tx.insert_register(0, MvRegValue::String("hello".to_string()));
            let _delta = tx.commit();
        }

        {
            let tx = ArrayTransaction::new(&mut store, id);
            let value = tx.get(0).expect("should have item at 0");

            match value {
                CrdtValue::Register(reg) => {
                    assert_eq!(
                        reg.value().unwrap(),
                        &MvRegValue::String("hello".to_string())
                    );
                }
                _ => panic!("Expected Register variant, got {value:?}"),
            }
        }
    }

    #[test]
    fn array_transaction_get_returns_map() {
        use crate::crdts::{NoExtensionTypes, mvreg::MvRegValue, snapshot::ToValue};
        use crate::transaction::CrdtValue;

        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = ArrayTransaction::new(&mut store, id);
            tx.insert_map(0, |map_tx| {
                map_tx.write_register("field", MvRegValue::U64(42));
            });
            let _delta = tx.commit();
        }

        {
            let tx = ArrayTransaction::new(&mut store, id);
            let value = tx.get(0).expect("should have item at 0");

            match value {
                CrdtValue::Map(map) => {
                    let field = map.get(&"field".to_string()).unwrap();
                    assert_eq!(field.reg.value().unwrap(), &MvRegValue::U64(42));
                }
                _ => panic!("Expected Map variant, got {value:?}"),
            }
        }
    }

    #[test]
    fn array_transaction_get_returns_array() {
        use crate::crdts::{NoExtensionTypes, mvreg::MvRegValue};
        use crate::transaction::CrdtValue;

        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = ArrayTransaction::new(&mut store, id);
            tx.insert_array(0, |nested_tx| {
                nested_tx.insert_register(0, MvRegValue::U64(1));
                nested_tx.insert_register(1, MvRegValue::U64(2));
            });
            let _delta = tx.commit();
        }

        {
            let tx = ArrayTransaction::new(&mut store, id);
            let value = tx.get(0).expect("should have item at 0");

            match value {
                CrdtValue::Array(array) => {
                    assert_eq!(array.len(), 2);
                }
                _ => panic!("Expected Array variant, got {value:?}"),
            }
        }
    }

    #[test]
    fn array_transaction_get_out_of_bounds() {
        use crate::crdts::NoExtensionTypes;

        let mut store = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
        let id = Identifier::new(0, 0);

        let tx = ArrayTransaction::new(&mut store, id);
        assert!(tx.get(0).is_none());
        assert!(tx.get(100).is_none());
    }
}
