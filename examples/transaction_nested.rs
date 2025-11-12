use dson::{
    CausalDotStore, Identifier, OrMap,
    crdts::{mvreg::MvRegValue, snapshot::ToValue},
    transaction::CrdtValue,
};

fn main() {
    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    // Create nested data structure
    {
        let mut tx = store.transact(id);

        // Write to nested map for user "alice"
        tx.in_map("alice", |alice_tx| {
            alice_tx.write_register("email", MvRegValue::String("alice@example.com".to_string()));
            alice_tx.write_register("age", MvRegValue::U64(30));
        });

        // Write to nested map for user "bob"
        tx.in_map("bob", |bob_tx| {
            bob_tx.write_register("email", MvRegValue::String("bob@example.com".to_string()));
            bob_tx.write_register("active", MvRegValue::Bool(true));
        });

        let _delta = tx.commit();
    }

    // Read nested data
    {
        let tx = store.transact(id);

        // Read Alice's data
        let alice_map = tx.get(&"alice".to_string()).expect("Alice should exist");
        let CrdtValue::Map(alice_map) = alice_map else {
            panic!("Alice value should be a map");
        };
        println!("Alice's data:");

        let alice_email = alice_map
            .get(&"email".to_string())
            .expect("Alice email should exist");
        let email = alice_email
            .reg
            .value()
            .expect("Alice email should have a value");
        println!("  Email: {email:?}");
        assert_eq!(email, &MvRegValue::String("alice@example.com".to_string()));

        let alice_age = alice_map
            .get(&"age".to_string())
            .expect("Alice age should exist");
        let age = alice_age
            .reg
            .value()
            .expect("Alice age should have a value");
        println!("  Age: {age:?}");
        assert_eq!(age, &MvRegValue::U64(30));

        // Read Bob's data
        let bob_map = tx.get(&"bob".to_string()).expect("Bob should exist");
        let CrdtValue::Map(bob_map) = bob_map else {
            panic!("Bob value should be a map");
        };
        println!("\nBob's data:");

        let bob_email = bob_map
            .get(&"email".to_string())
            .expect("Bob email should exist");
        let email = bob_email
            .reg
            .value()
            .expect("Bob email should have a value");
        println!("  Email: {email:?}");
        assert_eq!(email, &MvRegValue::String("bob@example.com".to_string()));

        let bob_active = bob_map
            .get(&"active".to_string())
            .expect("Bob active should exist");
        let active = bob_active
            .reg
            .value()
            .expect("Bob active should have a value");
        println!("  Active: {active:?}");
        assert_eq!(active, &MvRegValue::Bool(true));
    }

    // Deeply nested structure: map -> array -> map -> array
    // Structure: projects -> items -> properties
    {
        let mut tx = store.transact(id);

        tx.in_map("project", |project_tx| {
            project_tx.write_register("name", MvRegValue::String("Website Redesign".to_string()));

            // Array of task maps
            project_tx.in_array("tasks", |tasks_tx| {
                tasks_tx.insert_map(0, |task_tx| {
                    task_tx
                        .write_register("title", MvRegValue::String("Design mockups".to_string()));
                    task_tx.write_register("done", MvRegValue::Bool(true));
                });

                tasks_tx.insert_map(1, |task_tx| {
                    task_tx.write_register(
                        "title",
                        MvRegValue::String("Implement frontend".to_string()),
                    );
                    task_tx.write_register("done", MvRegValue::Bool(false));
                });
            });
        });

        let _delta = tx.commit();
    }

    // Read deeply nested data
    {
        let tx = store.transact(id);

        println!("\nDeeply nested structure:");

        let project = tx
            .get(&"project".to_string())
            .expect("project should exist");
        let CrdtValue::Map(project_map) = project else {
            panic!("project should be a map");
        };

        let name = project_map
            .get(&"name".to_string())
            .expect("name should exist");
        println!("  Project: {:?}", name.reg.value().unwrap());

        let tasks = project_map
            .get(&"tasks".to_string())
            .expect("tasks should exist");
        let tasks_len = tasks.array.len();
        println!("  {tasks_len} tasks");

        let task0 = tasks.array.get(0).expect("task 0 should exist");
        let title0 = task0
            .map
            .get(&"title".to_string())
            .expect("title should exist");
        println!("    Task 0: {:?}", title0.reg.value().unwrap());

        let task1 = tasks.array.get(1).expect("task 1 should exist");
        let title1 = task1
            .map
            .get(&"title".to_string())
            .expect("title should exist");
        println!("    Task 1: {:?}", title1.reg.value().unwrap());
    }

    println!("\nNested transactions make hierarchical data simple!");
}
