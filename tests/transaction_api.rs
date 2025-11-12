use dson::{CausalDotStore, Identifier, OrMap, crdts::mvreg::MvRegValue, transaction::CrdtValue};

#[test]
fn simple_register_write_and_read() {
    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    // Write using transaction
    {
        let mut tx = store.transact(id);
        tx.write_register("email", MvRegValue::String("alice@example.com".to_string()));
        let _delta = tx.commit();
    }

    // Read using transaction
    {
        let tx = store.transact(id);
        match tx.get(&"email".to_string()) {
            Some(CrdtValue::Register(reg)) => {
                use dson::crdts::snapshot::ToValue;
                assert_eq!(
                    reg.value().unwrap(),
                    &MvRegValue::String("alice@example.com".to_string())
                );
            }
            _ => panic!("Expected register"),
        }
    }
}

#[test]
fn two_replica_sync_with_transactions() {
    // Replica A
    let mut replica_a = CausalDotStore::<OrMap<String>>::default();
    let id_a = Identifier::new(0, 0);

    // Replica B
    let mut replica_b = CausalDotStore::<OrMap<String>>::default();
    let id_b = Identifier::new(1, 0);

    // A writes initial value
    let delta_a1 = {
        let mut tx = replica_a.transact(id_a);
        tx.write_register("count", MvRegValue::U64(0));
        tx.commit()
    };

    // B receives delta from A
    replica_b.join_or_replace_with(delta_a1.0.store, &delta_a1.0.context);

    // Both replicas should be in sync
    assert_eq!(replica_a, replica_b);

    // A and B concurrently increment
    let delta_a2 = {
        let mut tx = replica_a.transact(id_a);
        tx.write_register("count", MvRegValue::U64(1));
        tx.commit()
    };

    let delta_b1 = {
        let mut tx = replica_b.transact(id_b);
        tx.write_register("count", MvRegValue::U64(1));
        tx.commit()
    };

    // Exchange deltas
    replica_a.join_or_replace_with(delta_b1.0.store, &delta_b1.0.context);
    replica_b.join_or_replace_with(delta_a2.0.store, &delta_a2.0.context);

    // Both should converge
    assert_eq!(replica_a, replica_b);

    // Should have register with concurrent values
    let tx = replica_a.transact(id_a);
    match tx.get(&"count".to_string()) {
        Some(CrdtValue::Register(reg)) => {
            use dson::crdts::snapshot::ToValue;
            let values: Vec<_> = reg.values().into_iter().collect();
            // Both concurrent writes are preserved
            assert_eq!(values.len(), 2);
            // Both values are U64(1), but from different replicas
            assert!(values.iter().all(|v| **v == MvRegValue::U64(1)));
        }
        _ => panic!("Expected register"),
    }
}
