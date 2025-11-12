//! Tests for transaction rollback behavior.
//!
//! When a transaction is dropped without calling commit(), all changes
//! should be rolled back, leaving the original store unchanged.

use dson::{
    CausalDotStore, Identifier, OrArray, OrMap,
    crdts::mvreg::MvRegValue,
    transaction::{ArrayTransaction, CrdtValue, MapTransaction},
};

#[test]
fn map_transaction_rollback_register() {
    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    // Create initial state
    {
        let mut tx = MapTransaction::new(&mut store, id);
        tx.write_register("name", MvRegValue::String("Alice".to_string()));
        let _delta = tx.commit();
    }

    // Clone store to compare later
    let original_store = store.clone();

    // Start transaction and make changes but DON'T commit
    {
        let mut tx = MapTransaction::new(&mut store, id);
        tx.write_register("name", MvRegValue::String("Bob".to_string()));
        tx.write_register("age", MvRegValue::U64(30));
        // Drop tx without calling commit() - should rollback
    }

    // Store should be unchanged
    assert_eq!(store, original_store);

    // Verify original value still present
    let tx = MapTransaction::new(&mut store, id);
    match tx.get(&"name".to_string()) {
        Some(CrdtValue::Register(reg)) => {
            use dson::crdts::snapshot::ToValue;
            assert_eq!(
                reg.value().unwrap(),
                &MvRegValue::String("Alice".to_string())
            );
        }
        _ => panic!("Expected register with original value"),
    }

    // Verify new key was NOT added
    assert!(tx.get(&"age".to_string()).is_none());
}

#[test]
fn map_transaction_rollback_nested_map() {
    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    // Create initial state with nested map
    {
        let mut tx = MapTransaction::new(&mut store, id);
        tx.in_map("config", |cfg_tx| {
            cfg_tx.write_register("version", MvRegValue::U64(1));
        });
        let _delta = tx.commit();
    }

    let original_store = store.clone();

    // Modify nested map but don't commit
    {
        let mut tx = MapTransaction::new(&mut store, id);
        tx.in_map("config", |cfg_tx| {
            cfg_tx.write_register("version", MvRegValue::U64(2));
            cfg_tx.write_register("debug", MvRegValue::Bool(true));
        });
        // Drop without commit
    }

    // Store should be unchanged
    assert_eq!(store, original_store);

    // Verify original nested value
    use dson::crdts::snapshot::ToValue;
    let config = store.store.get(&"config".to_string()).unwrap();
    let version = config.map.get(&"version".to_string()).unwrap();
    assert_eq!(version.reg.value().unwrap(), &MvRegValue::U64(1));
    assert!(config.map.get(&"debug".to_string()).is_none());
}

#[test]
fn map_transaction_rollback_array() {
    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    // Create initial state with array
    {
        let mut tx = MapTransaction::new(&mut store, id);
        tx.in_array("items", |arr_tx| {
            arr_tx.insert_register(0, MvRegValue::String("first".to_string()));
        });
        let _delta = tx.commit();
    }

    let original_store = store.clone();

    // Modify array but don't commit
    {
        let mut tx = MapTransaction::new(&mut store, id);
        tx.in_array("items", |arr_tx| {
            arr_tx.insert_register(1, MvRegValue::String("second".to_string()));
            arr_tx.insert_register(2, MvRegValue::String("third".to_string()));
        });
        // Drop without commit
    }

    // Store should be unchanged
    assert_eq!(store, original_store);

    // Verify array still has only one element
    use dson::crdts::snapshot::ToValue;
    let items = store.store.get(&"items".to_string()).unwrap();
    assert_eq!(items.array.len(), 1);
    assert_eq!(
        items.array.get(0).unwrap().reg.value().unwrap(),
        &MvRegValue::String("first".to_string())
    );
}

#[test]
fn array_transaction_rollback_register() {
    let mut store = CausalDotStore::<OrArray>::default();
    let id = Identifier::new(0, 0);

    // Create initial state
    {
        let mut tx = ArrayTransaction::new(&mut store, id);
        tx.insert_register(0, MvRegValue::U64(1));
        tx.insert_register(1, MvRegValue::U64(2));
        let _delta = tx.commit();
    }

    let original_store = store.clone();

    // Modify array but don't commit
    {
        let mut tx = ArrayTransaction::new(&mut store, id);
        tx.insert_register(2, MvRegValue::U64(3));
        tx.insert_register(3, MvRegValue::U64(4));
        // Drop without commit
    }

    // Store should be unchanged
    assert_eq!(store, original_store);

    // Verify array still has only 2 elements
    use dson::crdts::snapshot::ToValue;
    assert_eq!(store.store.len(), 2);
    assert_eq!(
        store.store.get(0).unwrap().reg.value().unwrap(),
        &MvRegValue::U64(1)
    );
    assert_eq!(
        store.store.get(1).unwrap().reg.value().unwrap(),
        &MvRegValue::U64(2)
    );
}

#[test]
fn array_transaction_rollback_nested_array() {
    let mut store = CausalDotStore::<OrArray>::default();
    let id = Identifier::new(0, 0);

    // Create initial state with nested array
    {
        let mut tx = ArrayTransaction::new(&mut store, id);
        tx.insert_array(0, |inner_tx| {
            inner_tx.insert_register(0, MvRegValue::U64(1));
        });
        let _delta = tx.commit();
    }

    let original_store = store.clone();

    // Modify nested array but don't commit
    {
        let mut tx = ArrayTransaction::new(&mut store, id);
        tx.insert_array(1, |inner_tx| {
            inner_tx.insert_register(0, MvRegValue::U64(2));
        });
        // Drop without commit
    }

    // Store should be unchanged
    assert_eq!(store, original_store);

    // Verify only original nested array exists
    use dson::crdts::snapshot::ToValue;
    assert_eq!(store.store.len(), 1);
    let nested = &store.store.get(0).unwrap().array;
    assert_eq!(nested.len(), 1);
    assert_eq!(
        nested.get(0).unwrap().reg.value().unwrap(),
        &MvRegValue::U64(1)
    );
}

#[test]
fn array_transaction_rollback_map() {
    let mut store = CausalDotStore::<OrArray>::default();
    let id = Identifier::new(0, 0);

    // Create initial state with map
    {
        let mut tx = ArrayTransaction::new(&mut store, id);
        tx.insert_map(0, |map_tx| {
            map_tx.write_register("id", MvRegValue::U64(1));
        });
        let _delta = tx.commit();
    }

    let original_store = store.clone();

    // Modify map but don't commit
    {
        let mut tx = ArrayTransaction::new(&mut store, id);
        tx.insert_map(1, |map_tx| {
            map_tx.write_register("id", MvRegValue::U64(2));
        });
        // Drop without commit
    }

    // Store should be unchanged
    assert_eq!(store, original_store);

    // Verify only original map exists
    use dson::crdts::snapshot::ToValue;
    assert_eq!(store.store.len(), 1);
    let map = &store.store.get(0).unwrap().map;
    let id_val = map.get(&"id".to_string()).unwrap();
    assert_eq!(id_val.reg.value().unwrap(), &MvRegValue::U64(1));
}

#[test]
fn map_transaction_commit_after_rollback() {
    // Ensure that after a rollback, a new transaction can still commit successfully
    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    // First transaction: commit
    {
        let mut tx = MapTransaction::new(&mut store, id);
        tx.write_register("count", MvRegValue::U64(1));
        let _delta = tx.commit();
    }

    // Second transaction: rollback
    {
        let mut tx = MapTransaction::new(&mut store, id);
        tx.write_register("count", MvRegValue::U64(999));
        // Drop without commit
    }

    // Third transaction: commit
    {
        let mut tx = MapTransaction::new(&mut store, id);
        tx.write_register("count", MvRegValue::U64(2));
        let _delta = tx.commit();
    }

    // Verify final value is from third transaction
    let tx = MapTransaction::new(&mut store, id);
    match tx.get(&"count".to_string()) {
        Some(CrdtValue::Register(reg)) => {
            use dson::crdts::snapshot::ToValue;
            assert_eq!(reg.value().unwrap(), &MvRegValue::U64(2));
        }
        _ => panic!("Expected register"),
    }
}

#[test]
fn array_transaction_commit_after_rollback() {
    // Ensure that after a rollback, a new transaction can still commit successfully
    let mut store = CausalDotStore::<OrArray>::default();
    let id = Identifier::new(0, 0);

    // First transaction: commit
    {
        let mut tx = ArrayTransaction::new(&mut store, id);
        tx.insert_register(0, MvRegValue::U64(1));
        let _delta = tx.commit();
    }

    // Second transaction: rollback
    {
        let mut tx = ArrayTransaction::new(&mut store, id);
        tx.insert_register(1, MvRegValue::U64(999));
        // Drop without commit
    }

    // Third transaction: commit
    {
        let mut tx = ArrayTransaction::new(&mut store, id);
        tx.insert_register(1, MvRegValue::U64(2));
        let _delta = tx.commit();
    }

    // Verify array has both committed values
    use dson::crdts::snapshot::ToValue;
    assert_eq!(store.store.len(), 2);
    assert_eq!(
        store.store.get(0).unwrap().reg.value().unwrap(),
        &MvRegValue::U64(1)
    );
    assert_eq!(
        store.store.get(1).unwrap().reg.value().unwrap(),
        &MvRegValue::U64(2)
    );
}

#[test]
fn nested_transaction_panic_safety() {
    use dson::{
        CausalDotStore, Identifier, OrMap, crdts::mvreg::MvRegValue, transaction::MapTransaction,
    };

    // Verify that if a nested transaction panics, the parent transaction
    // is not corrupted and can still be rolled back cleanly
    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    // Create initial state
    {
        let mut tx = MapTransaction::new(&mut store, id);
        tx.write_register("root", MvRegValue::U64(1));
        let _delta = tx.commit();
    }

    let original_store = store.clone();

    // Transaction with nested panic
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut tx = MapTransaction::new(&mut store, id);
        tx.write_register("root", MvRegValue::U64(2));

        tx.in_map("nested", |nested_tx| {
            nested_tx.write_register("field", MvRegValue::String("test".to_string()));
            panic!("Simulated panic in nested transaction");
        });

        #[allow(unreachable_code)]
        tx.commit()
    }));

    // Verify panic occurred
    assert!(result.is_err());

    // Store should be unchanged - automatic rollback
    assert_eq!(store, original_store);

    // Verify original value still present
    use dson::crdts::snapshot::ToValue;
    let val = store.store.get(&"root".to_string()).unwrap();
    assert_eq!(val.reg.value().unwrap(), &MvRegValue::U64(1));

    // Verify nested map was not created
    assert!(store.store.get(&"nested".to_string()).is_none());
}
