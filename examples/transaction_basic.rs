use dson::{
    CausalDotStore, Identifier, OrMap,
    crdts::{mvreg::MvRegValue, snapshot::ToValue},
    transaction::CrdtValue,
};

fn main() {
    // Create a DSON store
    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    // Write some data using the transaction API
    {
        let mut tx = store.transact(id);
        tx.write_register("name", MvRegValue::String("Alice".to_string()));
        tx.write_register("age", MvRegValue::U64(30));
        tx.write_register("active", MvRegValue::Bool(true));

        let delta = tx.commit();
        println!(
            "Created delta with {} bytes",
            serde_json::to_string(&delta.0).unwrap().len()
        );
    }

    // Read the data back
    {
        let tx = store.transact(id);

        match tx.get(&"name".to_string()) {
            Some(CrdtValue::Register(reg)) => {
                if let Ok(MvRegValue::String(name)) = reg.value() {
                    println!("Name: {name}");
                }
            }
            _ => println!("Name not found or wrong type"),
        }

        match tx.get(&"age".to_string()) {
            Some(CrdtValue::Register(reg)) => {
                if let Ok(MvRegValue::U64(age)) = reg.value() {
                    println!("Age: {age}");
                }
            }
            _ => println!("Age not found or wrong type"),
        }
    }

    println!("\nTransaction API makes DSON easy to use!");
}
