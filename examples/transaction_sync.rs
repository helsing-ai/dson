use dson::{
    CausalDotStore, Identifier, OrMap,
    crdts::{mvreg::MvRegValue, snapshot::ToValue},
    transaction::CrdtValue,
};

fn main() {
    // Simulate a distributed system with 3 replicas
    let mut replica_a = CausalDotStore::<OrMap<String>>::default();
    let mut replica_b = CausalDotStore::<OrMap<String>>::default();
    let mut replica_c = CausalDotStore::<OrMap<String>>::default();

    let id_a = Identifier::new(0, 0);
    let id_b = Identifier::new(1, 0);
    let id_c = Identifier::new(2, 0);

    println!("Three replicas start with empty state\n");

    // Replica A initializes a counter
    let delta_a1 = {
        let mut tx = replica_a.transact(id_a);
        tx.write_register("counter", MvRegValue::U64(0));
        tx.commit()
    };
    println!("Replica A: initialized counter to 0");

    // Broadcast delta_a1 to all replicas
    replica_b.join_or_replace_with(delta_a1.0.store.clone(), &delta_a1.0.context);
    replica_c.join_or_replace_with(delta_a1.0.store, &delta_a1.0.context);
    println!("Replicas B and C: received initialization\n");

    // All three replicas concurrently increment
    let delta_a2 = {
        let mut tx = replica_a.transact(id_a);
        tx.write_register("counter", MvRegValue::U64(1));
        tx.commit()
    };
    println!("Replica A: incremented to 1");

    let delta_b1 = {
        let mut tx = replica_b.transact(id_b);
        tx.write_register("counter", MvRegValue::U64(1));
        tx.commit()
    };
    println!("Replica B: incremented to 1");

    let delta_c1 = {
        let mut tx = replica_c.transact(id_c);
        tx.write_register("counter", MvRegValue::U64(1));
        tx.commit()
    };
    println!("Replica C: incremented to 1\n");

    // Exchange deltas (full mesh)
    println!("Synchronizing replicas...");
    replica_a.join_or_replace_with(delta_b1.0.store.clone(), &delta_b1.0.context);
    replica_a.join_or_replace_with(delta_c1.0.store.clone(), &delta_c1.0.context);

    replica_b.join_or_replace_with(delta_a2.0.store.clone(), &delta_a2.0.context);
    replica_b.join_or_replace_with(delta_c1.0.store.clone(), &delta_c1.0.context);

    replica_c.join_or_replace_with(delta_a2.0.store, &delta_a2.0.context);
    replica_c.join_or_replace_with(delta_b1.0.store, &delta_b1.0.context);

    // Verify convergence
    assert_eq!(replica_a, replica_b);
    assert_eq!(replica_b, replica_c);
    println!("All replicas converged to the same state!\n");

    // Read final value
    {
        let tx = replica_a.transact(id_a);
        if let Some(CrdtValue::Register(reg)) = tx.get(&"counter".to_string()) {
            if let Ok(MvRegValue::U64(value)) = reg.value() {
                println!("Final counter value: {value}");
            }
        }
    }

    println!("\nThe transaction API makes distributed systems easy!");
    println!("Deltas are small, composable, and automatically managed.");
}
