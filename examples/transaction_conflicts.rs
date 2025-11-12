use dson::{
    CausalDotStore, Identifier, OrMap,
    crdts::{mvreg::MvRegValue, snapshot::ToValue},
    transaction::CrdtValue,
};

fn main() {
    // Create two replicas
    let mut replica_a = CausalDotStore::<OrMap<String>>::default();
    let mut replica_b = CausalDotStore::<OrMap<String>>::default();

    let id_a = Identifier::new(0, 0);
    let id_b = Identifier::new(1, 0);

    // Replica A writes a string value
    let delta_a = {
        let mut tx = replica_a.transact(id_a);
        tx.write_register("data", MvRegValue::String("text value".to_string()));
        tx.commit()
    };

    // Replica B concurrently writes a map at the same key
    let delta_b = {
        let mut tx = replica_b.transact(id_b);
        tx.in_map("data", |data_tx| {
            data_tx.write_register("count", MvRegValue::U64(42));
        });
        tx.commit()
    };

    // Both replicas receive each other's deltas
    replica_a.join_or_replace_with(delta_b.0.store, &delta_b.0.context);
    replica_b.join_or_replace_with(delta_a.0.store, &delta_a.0.context);

    // Both replicas should converge to the same state
    assert_eq!(replica_a, replica_b);

    // Inspect the type conflict on replica A
    {
        let tx = replica_a.transact(id_a);

        match tx.get(&"data".to_string()) {
            Some(CrdtValue::Conflicted(conflicts)) => {
                let has_register = conflicts.has_register();
                let has_map = conflicts.has_map();
                let conflict_count = conflicts.conflict_count();

                println!("Type conflict detected!");
                println!("  Has register: {has_register}");
                println!("  Has map: {has_map}");
                println!("  Total conflicts: {conflict_count}");

                // Application can access both values
                if let Some(reg) = conflicts.register() {
                    if let Ok(MvRegValue::String(s)) = reg.value() {
                        println!("  Register value: {s}");
                    }
                }

                if let Some(_map) = conflicts.map() {
                    println!("  Map value is present");
                }

                println!("\nThe transaction API makes conflicts explicit!");
                println!("Your application can decide how to resolve them.");
            }
            _ => println!("Expected a type conflict"),
        }
    }
}
