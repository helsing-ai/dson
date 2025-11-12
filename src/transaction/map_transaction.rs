use super::{ArrayTransaction, CrdtValue, Delta};
use crate::crdts::mvreg::MvRegValue;
use crate::dotstores::DotStoreJoin;
use crate::sentinel::DummySentinel;
use crate::{CausalDotStore, ExtensionType, Identifier, OrMap};
use std::{fmt, hash::Hash};

/// A transaction for making multiple mutations to a DSON store.
///
/// Transactions provide an ergonomic API for mutations and automatically
/// manage delta generation. Operations apply eagerly to a cloned store,
/// and commit swaps the modified clone back to the original store.
///
/// # Eager Application with Rollback
///
/// Each operation (write, remove, clear) applies immediately to a cloned copy of the store.
/// This means `get()` sees uncommitted changes from the current transaction.
/// If the transaction is dropped without calling `commit()`, all changes are automatically
/// rolled back by discarding the clone.
///
/// # Borrowing
///
/// A transaction exclusively borrows the underlying store, preventing other
/// access until the transaction is committed or dropped. This follows Rust's
/// standard borrowing rules and has zero runtime overhead.
///
/// # Example
///
/// ```
/// use dson::{CausalDotStore, Identifier, OrMap, transaction::MapTransaction};
///
/// let mut store = CausalDotStore::<OrMap<String>>::default();
/// let id = Identifier::new(0, 0);
///
/// let mut tx = MapTransaction::new(&mut store, id);
/// // Make mutations...
/// let delta = tx.commit();
/// ```
pub struct MapTransaction<'a, K, C = crate::crdts::NoExtensionTypes>
where
    K: Hash + Eq + fmt::Debug + Clone,
    C: ExtensionType,
{
    original_store: &'a mut CausalDotStore<OrMap<K, C>>,
    working_store: CausalDotStore<OrMap<K, C>>,
    id: Identifier,
    // Accumulated deltas from mutations (will be joined on commit)
    changes: Vec<CausalDotStore<OrMap<K, C>>>,
}

impl<'a, K, C> MapTransaction<'a, K, C>
where
    K: Hash + Eq + fmt::Debug + Clone,
    C: ExtensionType,
{
    /// Creates a new transaction for the given store and replica identifier.
    ///
    /// The transaction clones the store and exclusively borrows the original until committed.
    /// Changes apply to the clone, enabling automatic rollback on drop.
    ///
    /// # Example
    ///
    /// ```
    /// # use dson::{CausalDotStore, Identifier, OrMap, transaction::MapTransaction};
    /// let mut store = CausalDotStore::<OrMap<String>>::default();
    /// let id = Identifier::new(0, 0);
    /// let tx = MapTransaction::new(&mut store, id);
    /// ```
    pub fn new(store: &'a mut CausalDotStore<OrMap<K, C>>, id: Identifier) -> Self
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
    /// Used internally for nested transactions (`in_map`, `in_array`).
    /// Nested transactions don't need rollback support since they commit
    /// automatically when the closure returns. The store is swapped back on commit.
    pub(crate) fn new_nested(store: &'a mut CausalDotStore<OrMap<K, C>>, id: Identifier) -> Self {
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
    ///
    /// # Example
    ///
    /// ```
    /// # use dson::{CausalDotStore, Identifier, OrMap, transaction::MapTransaction};
    /// # let mut store = CausalDotStore::<OrMap<String>>::default();
    /// # let id = Identifier::new(0, 0);
    /// let tx = MapTransaction::new(&mut store, id);
    /// // Make changes...
    /// let delta = tx.commit();
    /// // Send delta over network...
    /// ```
    pub fn commit(mut self) -> Delta<CausalDotStore<OrMap<K, C>>>
    where
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        // Swap the working store back to the original to make changes permanent
        *self.original_store = self.working_store;

        if self.changes.is_empty() {
            // No changes, return empty delta
            return Delta::new(CausalDotStore::default());
        }

        // Join all accumulated deltas into a single delta
        let mut combined = self.changes.remove(0);
        for delta in self.changes.drain(..) {
            combined = combined
                .join(delta, &mut DummySentinel)
                .expect("DummySentinel is infallible");
        }

        Delta::new(combined)
    }

    /// Records a change and applies it immediately to the working store.
    ///
    /// This enables eager (read-uncommitted) semantics, allowing subsequent
    /// operations within the transaction to see uncommitted changes.
    fn record_change(&mut self, delta: CausalDotStore<OrMap<K, C>>)
    where
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        // Apply delta to working store immediately so subsequent operations see updated state
        self.working_store
            .join_or_replace_with(delta.store.clone(), &delta.context);

        self.changes.push(delta);
    }

    /// Reads a value from the map, requiring explicit type handling.
    ///
    /// This method returns a [`CrdtValue`] enum that forces the caller to
    /// handle type conflicts and different types explicitly.
    ///
    /// Returns `None` if the key doesn't exist in the map.
    ///
    /// # Isolation Semantics
    ///
    /// This reads the current state of the working store, which includes all changes
    /// made during this transaction. Map operations apply immediately to the working store,
    /// so `get` sees uncommitted changes (eager/read-uncommitted semantics).
    ///
    /// This matches the behavior of [`ArrayTransaction::get`](crate::transaction::ArrayTransaction::get)
    /// and enables consistent behavior in nested transactions.
    ///
    /// # Example
    ///
    /// ```
    /// # use dson::{CausalDotStore, Identifier, OrMap, transaction::{MapTransaction, CrdtValue}};
    /// # use dson::crdts::snapshot::ToValue;
    /// # let mut store = CausalDotStore::<OrMap<String>>::default();
    /// # let id = Identifier::new(0, 0);
    /// let tx = store.transact(id);
    /// match tx.get(&"key".to_string()) {
    ///     Some(CrdtValue::Register(reg)) => {
    ///         // Read register value
    ///         if let Ok(value) = reg.value() {
    ///             println!("Value: {:?}", value);
    ///         }
    ///     }
    ///     Some(CrdtValue::Conflicted(conflicts)) => {
    ///         // Handle type conflict
    ///         println!("Conflict count: {}", conflicts.conflict_count());
    ///     }
    ///     None => {
    ///         println!("Key doesn't exist");
    ///     }
    ///     _ => {
    ///         // Other types (map, array, empty)
    ///     }
    /// }
    /// ```
    pub fn get(&self, key: &K) -> Option<CrdtValue<'_, K, C>> {
        let value = self.working_store.store.get(key)?;
        Some(CrdtValue::from_type_variant(value))
    }

    /// Writes a value to a register at the given key, overwriting any
    /// existing map, array, or conflicted value.
    ///
    /// Accumulates the delta for commit.
    ///
    /// # Example
    ///
    /// ```
    /// # use dson::{CausalDotStore, Identifier, OrMap, transaction::MapTransaction};
    /// # let mut store = CausalDotStore::<OrMap<String>>::default();
    /// # let id = Identifier::new(0, 0);
    /// let mut tx = MapTransaction::new(&mut store, id);
    /// tx.write_register("count", 42u64.into());
    /// tx.write_register("name", "Alice".to_string().into());
    /// let delta = tx.commit();
    /// ```
    pub fn write_register(&mut self, key: impl Into<K>, value: MvRegValue)
    where
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        let key = key.into();

        // Call OrMap::apply_to_register directly, passing a closure that calls MvReg::write
        let delta = self.working_store.store.apply_to_register(
            |reg, ctx, id| reg.write(value.clone(), ctx, id),
            key,
            &self.working_store.context,
            self.id,
        );

        // Record the delta (which also updates working_store.context)
        self.record_change(delta);
    }

    /// Removes a key from the map.
    ///
    /// This creates a CRDT tombstone that marks the key as removed. The removal
    /// will be propagated to other replicas via the delta.
    ///
    /// # Example
    ///
    /// ```
    /// # use dson::{CausalDotStore, Identifier, OrMap, transaction::MapTransaction};
    /// # use dson::crdts::mvreg::MvRegValue;
    /// # let mut store = CausalDotStore::<OrMap<String>>::default();
    /// # let id = Identifier::new(0, 0);
    /// # {
    /// #   let mut tx = MapTransaction::new(&mut store, id);
    /// #   tx.write_register("key", MvRegValue::String("value".to_string()));
    /// #   tx.commit();
    /// # }
    /// let mut tx = MapTransaction::new(&mut store, id);
    /// tx.remove("key");
    /// let delta = tx.commit();
    /// ```
    pub fn remove(&mut self, key: impl Into<K>)
    where
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        let key = key.into();

        // Call OrMap::remove directly
        let delta = self
            .working_store
            .store
            .remove(&key, &self.working_store.context, self.id);

        // Record the delta (which also updates working_store.context)
        self.record_change(delta);
    }

    /// Clears all keys from the map.
    ///
    /// This creates CRDT tombstones for all existing keys. The clear operation
    /// will be propagated to other replicas via the delta.
    ///
    /// # Example
    ///
    /// ```
    /// # use dson::{CausalDotStore, Identifier, OrMap, transaction::MapTransaction};
    /// # use dson::crdts::mvreg::MvRegValue;
    /// # let mut store = CausalDotStore::<OrMap<String>>::default();
    /// # let id = Identifier::new(0, 0);
    /// # {
    /// #   let mut tx = MapTransaction::new(&mut store, id);
    /// #   tx.write_register("a", MvRegValue::U64(1));
    /// #   tx.write_register("b", MvRegValue::U64(2));
    /// #   tx.commit();
    /// # }
    /// let mut tx = MapTransaction::new(&mut store, id);
    /// tx.clear();
    /// let delta = tx.commit();
    /// ```
    pub fn clear(&mut self)
    where
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        // Call OrMap::clear directly
        let delta = self
            .working_store
            .store
            .clear(&self.working_store.context, self.id);

        // Record the delta (which also updates working_store.context)
        self.record_change(delta);
    }

    /// Creates a nested transaction for a map at the given key.
    ///
    /// The closure receives a mutable reference to a child `MapTransaction`.
    /// All operations on the child are accumulated and applied to the parent
    /// when the closure returns.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use dson::{CausalDotStore, Identifier, OrMap, transaction::MapTransaction};
    /// # use dson::crdts::mvreg::MvRegValue;
    /// # let mut store = CausalDotStore::<OrMap<String>>::default();
    /// # let id = Identifier::new(0, 0);
    /// let mut tx = MapTransaction::new(&mut store, id);
    /// tx.in_map("user", |user_tx| {
    ///     user_tx.write_register("email", MvRegValue::String("alice@example.com".to_string()));
    ///     user_tx.write_register("age", MvRegValue::U64(30));
    /// });
    /// let delta = tx.commit();
    /// ```
    pub fn in_map<F>(&mut self, key: impl Into<K>, f: F)
    where
        F: FnOnce(&mut MapTransaction<'_, String, C>),
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        let key = key.into();

        // Get current nested map (or create empty)
        let nested_map = self
            .working_store
            .store
            .get(&key)
            .map(|v| v.map.clone())
            .unwrap_or_default();

        // Create temporary store for child transaction
        let mut child_store = CausalDotStore {
            store: nested_map,
            context: self.working_store.context.clone(),
        };

        // Create child transaction (no clone needed for nested transactions)
        let mut child_tx = MapTransaction::new_nested(&mut child_store, self.id);

        // User performs operations on child
        // If f panics, child_tx drops without commit, child_store retains Default value
        f(&mut child_tx);

        // Get delta from child
        let child_delta = child_tx.commit();

        // If child made changes, wrap in parent's map operation
        if !child_delta.0.is_bottom() {
            let delta = self.working_store.store.apply_to_map(
                |_old_map, _ctx, _id| child_delta.0.clone(),
                key,
                &self.working_store.context,
                self.id,
            );

            self.record_change(delta);
        }
    }

    /// Creates a nested transaction for an array at the given key.
    ///
    /// The closure receives a mutable reference to a child `ArrayTransaction`.
    /// All operations on the child are accumulated and applied to the parent
    /// when the closure returns.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use dson::{CausalDotStore, Identifier, OrMap, transaction::MapTransaction};
    /// # use dson::crdts::mvreg::MvRegValue;
    /// # let mut store = CausalDotStore::<OrMap<String>>::default();
    /// # let id = Identifier::new(0, 0);
    /// let mut tx = MapTransaction::new(&mut store, id);
    /// tx.in_array("tags", |tags_tx| {
    ///     tags_tx.insert_register(0, MvRegValue::String("rust".to_string()));
    ///     tags_tx.insert_register(1, MvRegValue::String("crdt".to_string()));
    /// });
    /// let delta = tx.commit();
    /// ```
    pub fn in_array<F>(&mut self, key: impl Into<K>, f: F)
    where
        F: FnOnce(&mut ArrayTransaction<'_, C>),
        C: DotStoreJoin<DummySentinel> + fmt::Debug + Clone + PartialEq,
    {
        let key = key.into();

        // Get current nested array (or create empty)
        let nested_array = self
            .working_store
            .store
            .get(&key)
            .map(|v| v.array.clone())
            .unwrap_or_default();

        // Create temporary store for child transaction
        let mut child_store = CausalDotStore {
            store: nested_array,
            context: self.working_store.context.clone(),
        };

        // Create child transaction (no clone needed for nested transactions)
        let mut child_tx = ArrayTransaction::new_nested(&mut child_store, self.id);

        // User performs operations on child
        f(&mut child_tx);

        // Get delta from child
        let child_delta = child_tx.commit();

        // If child made changes, wrap in parent's array operation
        if !child_delta.0.is_bottom() {
            let delta = self.working_store.store.apply_to_array(
                |_old_array, _ctx, _id| child_delta.0.clone(),
                key.clone(),
                &self.working_store.context,
                self.id,
            );

            self.record_change(delta);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DotStore, crdts::NoExtensionTypes};

    #[test]
    fn transaction_new() {
        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);
        let _tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
    }

    #[test]
    fn transaction_borrows_exclusively() {
        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);
        let tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);

        // This should not compile (uncomment to verify):
        // let _ = &store; // Error: cannot borrow `store` as immutable

        drop(tx);
        // After dropping tx, we can borrow again
        let _ = &store;
    }

    #[test]
    fn transaction_get_nonexistent() {
        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);
        let tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);

        assert!(tx.get(&"nonexistent".to_string()).is_none());
    }

    #[test]
    fn transaction_get_returns_correct_type() {
        use crate::crdts::mvreg::MvRegValue;
        use crate::sentinel::DummySentinel;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        // Create a register at "key"
        let delta1 = store.store.apply_to_register(
            |reg, ctx, id| reg.write(MvRegValue::String("test".to_string()), ctx, id),
            "key".to_string(),
            &store.context,
            id,
        );
        store = store.join(delta1, &mut DummySentinel).unwrap();

        let tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);

        match tx.get(&"key".to_string()) {
            Some(CrdtValue::Register(_)) => {
                // Expected: we created a register
            }
            _ => panic!("Expected Register variant"),
        }
    }

    #[test]
    fn transaction_get_register() {
        use crate::crdts::mvreg::MvRegValue;
        use crate::crdts::snapshot::ToValue;
        use crate::sentinel::DummySentinel;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        // Add a register value using proper CRDT operations
        let delta = store.store.apply_to_register(
            |reg, ctx, id| reg.write(MvRegValue::String("test".to_string()), ctx, id),
            "key".to_string(),
            &store.context,
            id,
        );
        store = store.join(delta, &mut DummySentinel).unwrap();

        let tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);

        match tx.get(&"key".to_string()) {
            Some(CrdtValue::Register(reg)) => {
                assert_eq!(
                    reg.value().unwrap(),
                    &MvRegValue::String("test".to_string())
                );
            }
            _ => panic!("Expected Register variant"),
        }
    }

    #[test]
    fn transaction_write_register() {
        use crate::crdts::mvreg::MvRegValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
        tx.write_register("count", MvRegValue::U64(42));

        assert_eq!(tx.changes.len(), 1);
    }

    #[test]
    fn transaction_write_multiple_registers() {
        use crate::crdts::mvreg::MvRegValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
        tx.write_register("count", MvRegValue::U64(42));
        tx.write_register("name", MvRegValue::String("Alice".to_string()));
        tx.write_register("active", MvRegValue::Bool(true));

        assert_eq!(tx.changes.len(), 3);
    }

    #[test]
    fn transaction_commit_applies_changes() {
        use crate::crdts::mvreg::MvRegValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.write_register("count", MvRegValue::U64(42));
            let _delta = tx.commit();
        }

        // Verify the change was applied to the store
        let value = store.store.get(&"count".to_string()).unwrap();
        assert!(!value.reg.is_bottom());
        use crate::crdts::snapshot::ToValue;
        assert_eq!(value.reg.value().unwrap(), &MvRegValue::U64(42));
    }

    #[test]
    fn transaction_commit_returns_delta() {
        use crate::crdts::mvreg::MvRegValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        let delta = {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.write_register("count", MvRegValue::U64(42));
            tx.commit()
        };

        // Delta should contain the change
        let value = delta.0.store.get(&"count".to_string()).unwrap();
        assert!(!value.reg.is_bottom());
    }

    #[test]
    fn transaction_commit_multiple_changes() {
        use crate::crdts::mvreg::MvRegValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.write_register("a", MvRegValue::U64(1));
            tx.write_register("b", MvRegValue::U64(2));
            tx.write_register("c", MvRegValue::U64(3));
            let _delta = tx.commit();
        }

        // All changes should be applied
        assert!(store.store.get(&"a".to_string()).is_some());
        assert!(store.store.get(&"b".to_string()).is_some());
        assert!(store.store.get(&"c".to_string()).is_some());
    }

    #[test]
    fn transaction_nested_map() {
        use crate::crdts::mvreg::MvRegValue;
        use crate::crdts::snapshot::ToValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.in_map("user", |user_tx| {
                user_tx
                    .write_register("email", MvRegValue::String("alice@example.com".to_string()));
            });
            let _delta = tx.commit();
        }

        // Verify nested structure
        let user_value = store.store.get(&"user".to_string()).unwrap();
        assert!(!user_value.map.is_bottom());

        let email_value = user_value.map.get(&"email".to_string()).unwrap();
        assert!(!email_value.reg.is_bottom());
        assert_eq!(
            email_value.reg.value().unwrap(),
            &MvRegValue::String("alice@example.com".to_string())
        );
    }

    #[test]
    fn transaction_nested_map_multiple_fields() {
        use crate::crdts::mvreg::MvRegValue;
        use crate::crdts::snapshot::ToValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.in_map("user", |user_tx| {
                user_tx
                    .write_register("email", MvRegValue::String("alice@example.com".to_string()));
                user_tx.write_register("age", MvRegValue::U64(30));
            });
            let _delta = tx.commit();
        }

        // Verify both fields exist
        let user_value = store.store.get(&"user".to_string()).unwrap();
        assert!(!user_value.map.is_bottom());

        let email_value = user_value.map.get(&"email".to_string()).unwrap();
        assert_eq!(
            email_value.reg.value().unwrap(),
            &MvRegValue::String("alice@example.com".to_string())
        );

        let age_value = user_value.map.get(&"age".to_string()).unwrap();
        assert_eq!(age_value.reg.value().unwrap(), &MvRegValue::U64(30));
    }

    #[test]
    fn transaction_nested_array() {
        use crate::crdts::mvreg::MvRegValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.in_array("items", |items_tx| {
                items_tx.insert_register(0, MvRegValue::String("item1".to_string()));
            });
            let _delta = tx.commit();
        }

        // Verify array was created
        let items_value = store.store.get(&"items".to_string()).unwrap();
        assert!(!items_value.array.is_bottom());
    }

    #[test]
    fn transaction_remove() {
        use crate::crdts::mvreg::MvRegValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        // First, create a key
        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.write_register("key", MvRegValue::String("value".to_string()));
            let _delta = tx.commit();
        }

        // Verify it exists
        assert!(store.store.get(&"key".to_string()).is_some());

        // Now remove it
        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.remove("key");
            let _delta = tx.commit();
        }

        // Verify it's gone
        assert!(store.store.get(&"key".to_string()).is_none());
    }

    #[test]
    fn transaction_clear() {
        use crate::crdts::mvreg::MvRegValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        // Create multiple keys
        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.write_register("a", MvRegValue::U64(1));
            tx.write_register("b", MvRegValue::U64(2));
            tx.write_register("c", MvRegValue::U64(3));
            let _delta = tx.commit();
        }

        // Verify they exist
        assert!(store.store.get(&"a".to_string()).is_some());
        assert!(store.store.get(&"b".to_string()).is_some());
        assert!(store.store.get(&"c".to_string()).is_some());

        // Clear the map
        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.clear();
            let _delta = tx.commit();
        }

        // Verify they're all gone
        assert!(store.store.get(&"a".to_string()).is_none());
        assert!(store.store.get(&"b".to_string()).is_none());
        assert!(store.store.get(&"c".to_string()).is_none());
    }

    // Property-based test: transaction API should produce same results as direct CRDT calls
    #[test]
    fn property_transaction_equals_direct_crdt() {
        use crate::crdts::mvreg::MvRegValue;

        // Test that write_register via transaction produces same result as direct call
        let mut store_tx = CausalDotStore::<OrMap<String>>::default();
        let mut store_direct = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        // Via transaction
        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store_tx, id);
            tx.write_register("key", MvRegValue::String("value".to_string()));
            let _delta = tx.commit();
        }

        // Direct CRDT call
        {
            let delta = store_direct.store.apply_to_register(
                |reg, ctx, id| reg.write(MvRegValue::String("value".to_string()), ctx, id),
                "key".to_string(),
                &store_direct.context,
                id,
            );
            store_direct.join_or_replace_with(delta.store, &delta.context);
        }

        // Should be equivalent
        assert_eq!(store_tx, store_direct);
    }

    #[test]
    fn property_concurrent_writes_converge() {
        use crate::crdts::mvreg::MvRegValue;
        use crate::crdts::snapshot::ToValue;

        // Two replicas make concurrent writes to different keys
        let id1 = Identifier::new(0, 0);
        let id2 = Identifier::new(1, 0);

        // Replica 1 writes "a"
        let mut store1 = CausalDotStore::<OrMap<String>>::default();
        let delta1 = {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store1, id1);
            tx.write_register("a", MvRegValue::U64(1));
            tx.commit()
        };

        // Replica 2 writes "b"
        let mut store2 = CausalDotStore::<OrMap<String>>::default();
        let delta2 = {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store2, id2);
            tx.write_register("b", MvRegValue::U64(2));
            tx.commit()
        };

        // Exchange deltas
        store1.join_or_replace_with(delta2.0.store.clone(), &delta2.0.context);
        store2.join_or_replace_with(delta1.0.store, &delta1.0.context);

        // Should converge to same state (CRDT property)
        assert_eq!(store1, store2);

        // Both should have both keys
        assert_eq!(
            store1
                .store
                .get(&"a".to_string())
                .unwrap()
                .reg
                .value()
                .unwrap(),
            &MvRegValue::U64(1)
        );
        assert_eq!(
            store1
                .store
                .get(&"b".to_string())
                .unwrap()
                .reg
                .value()
                .unwrap(),
            &MvRegValue::U64(2)
        );
    }

    #[test]
    fn property_remove_then_add_idempotent() {
        use crate::crdts::mvreg::MvRegValue;

        let id1 = Identifier::new(0, 0);
        let _id2 = Identifier::new(1, 0);

        // Replica 1: Add then remove
        let mut store1 = CausalDotStore::<OrMap<String>>::default();
        let delta1 = {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store1, id1);
            tx.write_register("key", MvRegValue::String("value".to_string()));
            tx.commit()
        };

        let delta2 = {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store1, id1);
            tx.remove("key");
            tx.commit()
        };

        // Replica 2: receives deltas in reverse order (remove, then add)
        let mut store2 = CausalDotStore::<OrMap<String>>::default();
        store2.join_or_replace_with(delta2.0.store, &delta2.0.context);
        store2.join_or_replace_with(delta1.0.store, &delta1.0.context);

        // Should converge - remove should win due to causal context
        assert_eq!(store1, store2);
        assert!(store1.store.get(&"key".to_string()).is_none());
    }

    #[test]
    fn test_concurrent_type_conflicts_detected() {
        use crate::crdts::mvreg::MvRegValue;

        // Two replicas concurrently write different types to the same key
        let id1 = Identifier::new(0, 0);
        let id2 = Identifier::new(1, 0);

        // Replica 1: writes a register at key "data"
        let mut store1 = CausalDotStore::<OrMap<String>>::default();
        let delta1 = {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store1, id1);
            tx.write_register("data", MvRegValue::String("text".to_string()));
            tx.commit()
        };

        // Replica 2: writes an array at key "data"
        let mut store2 = CausalDotStore::<OrMap<String>>::default();
        let delta2 = {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store2, id2);
            tx.in_array("data", |data_tx| {
                data_tx.insert_register(0, MvRegValue::U64(42));
            });
            tx.commit()
        };

        // Both replicas exchange deltas - this creates a type conflict
        store1.join_or_replace_with(delta2.0.store, &delta2.0.context);
        store2.join_or_replace_with(delta1.0.store, &delta1.0.context);

        // Both should converge to same state
        assert_eq!(store1, store2);

        // Reading should detect the conflict via CrdtValue::Conflicted
        let tx1 = MapTransaction::<String, NoExtensionTypes>::new(&mut store1, id1);
        match tx1.get(&"data".to_string()) {
            Some(CrdtValue::Conflicted(conflicts)) => {
                // Verify both types are present
                assert!(
                    conflicts.has_register(),
                    "Should have register from replica 1"
                );
                assert!(conflicts.has_array(), "Should have array from replica 2");
                assert_eq!(
                    conflicts.conflict_count(),
                    2,
                    "Should have exactly 2 conflicting types"
                );
            }
            other => panic!("Expected CrdtValue::Conflicted, got {other:?}"),
        }
    }

    #[test]
    fn map_transaction_in_map() {
        use crate::crdts::{NoExtensionTypes, mvreg::MvRegValue, snapshot::ToValue};

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.in_map("user", |user_tx| {
                user_tx.write_register("name", MvRegValue::String("Alice".to_string()));
                user_tx.write_register("age", MvRegValue::U64(30));
            });
            let _delta = tx.commit();
        }

        // Verify nested structure
        let user = store.store.get(&"user".to_string()).unwrap();
        let name = user.map.get(&"name".to_string()).unwrap();
        assert_eq!(
            name.reg.value().unwrap(),
            &MvRegValue::String("Alice".to_string())
        );

        let age = user.map.get(&"age".to_string()).unwrap();
        assert_eq!(age.reg.value().unwrap(), &MvRegValue::U64(30));
    }

    #[test]
    fn test_sequential_operations_get_unique_dots() {
        use crate::crdts::mvreg::MvRegValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.write_register("a", MvRegValue::U64(1));
            tx.write_register("b", MvRegValue::U64(2));
            let _delta = tx.commit();
        }

        // Both keys should exist
        assert!(store.store.get(&"a".to_string()).is_some());
        assert!(store.store.get(&"b".to_string()).is_some());

        // Verify we got unique dots
        let a_value = store.store.get(&"a".to_string()).unwrap();
        let b_value = store.store.get(&"b".to_string()).unwrap();

        // Get the CausalContext from each register's dots
        let a_context = a_value.reg.dots();
        let b_context = b_value.reg.dots();

        // Extract dots from contexts
        let a_dots: Vec<_> = a_context.dots().collect();
        let b_dots: Vec<_> = b_context.dots().collect();

        // Both should have exactly one dot
        assert_eq!(a_dots.len(), 1);
        assert_eq!(b_dots.len(), 1);

        // The dots should be different (unique)
        assert_ne!(a_dots[0], b_dots[0]);
    }

    #[test]
    fn map_transaction_in_array() {
        use crate::crdts::{NoExtensionTypes, mvreg::MvRegValue, snapshot::ToValue};

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.in_array("items", |items_tx| {
                items_tx.insert_register(0, MvRegValue::String("first".to_string()));
                items_tx.insert_register(1, MvRegValue::String("second".to_string()));
            });
            let _delta = tx.commit();
        }

        // Verify array was created with items
        let items_val = store.store.get(&"items".to_string()).unwrap();
        assert_eq!(items_val.array.len(), 2);

        let first = items_val.array.get(0).unwrap();
        assert_eq!(
            first.reg.value().unwrap(),
            &MvRegValue::String("first".to_string())
        );
    }

    #[test]
    fn test_in_map_empty_closure_should_not_create() {
        // Test: calling in_map with empty closure should NOT create a map
        // because the delta is completely bottom (no store, no context changes)
        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.in_map("empty", |_map_tx| {
                // Do nothing - no operations
            });
            let _delta = tx.commit();
        }

        // Should be None because child_delta.is_bottom() is true
        let result = store.store.get(&"empty".to_string());
        assert!(
            result.is_none(),
            "Empty map should NOT be created when no operations performed"
        );
    }

    #[test]
    fn test_in_array_empty_closure_should_not_create() {
        // Test: calling in_array with empty closure should NOT create an array
        // because the delta is completely bottom (no store, no context changes)
        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        {
            let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
            tx.in_array("empty", |_array_tx| {
                // Do nothing - no operations
            });
            let _delta = tx.commit();
        }

        // Should be None because child_delta.is_bottom() is true
        let result = store.store.get(&"empty".to_string());
        assert!(
            result.is_none(),
            "Empty array should NOT be created when no operations performed"
        );
    }

    // Tests for eager map semantics

    #[test]
    fn test_map_get_sees_uncommitted_writes() {
        // Test that get() sees uncommitted writes in the same transaction
        use crate::crdts::mvreg::MvRegValue;
        use crate::crdts::snapshot::ToValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);
        tx.write_register("x", MvRegValue::U64(42));

        // Should see uncommitted write (eager semantics)
        match tx.get(&"x".to_string()) {
            Some(CrdtValue::Register(reg)) => {
                assert_eq!(reg.value().unwrap(), &MvRegValue::U64(42));
            }
            other => panic!("Expected to see uncommitted write, got {other:?}"),
        }
    }

    #[test]
    fn test_nested_map_operations_see_uncommitted_state() {
        // Test that nested map operations work correctly with uncommitted parent state
        // Scenario: Write to top-level map, then try to read it in nested operation
        use crate::crdts::mvreg::MvRegValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);

        // Write a field at top level
        tx.write_register("app_name", MvRegValue::String("MyApp".to_string()));

        // Now try to read it in a nested map operation
        // With eager semantics, the child should see the uncommitted parent state
        tx.in_map("config", |config_tx| {
            // This should be able to see the parent store's uncommitted "app_name"
            // Currently this fails because in_map reads from the original store (line 305-310)
            // not the updated store with uncommitted changes
            config_tx.write_register("version", MvRegValue::U64(1));
        });

        let _delta = tx.commit();

        // Verify both fields exist
        assert!(
            store.store.get(&"app_name".to_string()).is_some(),
            "Top-level field should exist"
        );
        assert!(
            store.store.get(&"config".to_string()).is_some(),
            "Nested map should exist"
        );
    }

    #[test]
    fn test_sequential_writes_to_same_key_visible() {
        // Test that sequential writes to the same key are visible
        use crate::crdts::mvreg::MvRegValue;
        use crate::crdts::snapshot::ToValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);

        tx.write_register("counter", MvRegValue::U64(1));

        // In eager mode, this should see the previous write
        // (though in CRDT we can't read-modify-write reliably,
        // this tests visibility)
        match tx.get(&"counter".to_string()) {
            Some(CrdtValue::Register(reg)) => {
                assert_eq!(reg.value().unwrap(), &MvRegValue::U64(1));
            }
            other => panic!("Expected to see previous write, got {other:?}"),
        }

        tx.write_register("counter", MvRegValue::U64(2));

        let _delta = tx.commit();

        // Final value should be 2
        let val = store.store.get(&"counter".to_string()).unwrap();
        assert_eq!(val.reg.value().unwrap(), &MvRegValue::U64(2));
    }

    #[test]
    fn test_map_array_consistency() {
        // Test that maps and arrays have consistent eager semantics
        use crate::crdts::mvreg::MvRegValue;

        let mut store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        let mut tx = MapTransaction::<String, NoExtensionTypes>::new(&mut store, id);

        // Array operations see uncommitted state
        tx.in_array("items", |arr_tx| {
            arr_tx.insert_register(0, MvRegValue::U64(1));
            // This works because arrays are eager
            assert_eq!(arr_tx.len(), 1);
        });

        // Maps should also see uncommitted state for consistency
        tx.in_map("config", |map_tx| {
            map_tx.write_register("version", MvRegValue::U64(1));
            // This should work too (currently fails)
            match map_tx.get(&"version".to_string()) {
                Some(CrdtValue::Register(_)) => {
                    // Good - saw uncommitted write
                }
                other => panic!("Maps should see uncommitted writes like arrays do, got {other:?}"),
            }
        });

        let _delta = tx.commit();
    }
}
