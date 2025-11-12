use dson::{
    CausalDotStore, Identifier, OrMap, crdts::mvreg::MvRegValue, transaction::MapTransaction,
};

fn main() {
    println!("Nested Transaction API Demo\n");

    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    // Create deeply nested structure
    {
        let mut tx = MapTransaction::new(&mut store, id);

        // Simple register
        tx.write_register("app_name", MvRegValue::String("TaskManager".to_string()));

        // Nested map
        tx.in_map("settings", |settings_tx| {
            settings_tx.write_register("theme", MvRegValue::String("dark".to_string()));
            settings_tx.write_register("notifications", MvRegValue::Bool(true));
        });

        // Array of maps
        tx.in_array("users", |users_tx| {
            users_tx.insert_map(0, |user_tx| {
                user_tx.write_register("name", MvRegValue::String("Alice".to_string()));
                user_tx.write_register("role", MvRegValue::String("admin".to_string()));
            });

            users_tx.insert_map(1, |user_tx| {
                user_tx.write_register("name", MvRegValue::String("Bob".to_string()));
                user_tx.write_register("role", MvRegValue::String("user".to_string()));
            });
        });

        // Deeply nested: map -> array -> map -> array
        tx.in_map("projects", |projects_tx| {
            projects_tx.in_array("active", |active_tx| {
                active_tx.insert_map(0, |project_tx| {
                    project_tx
                        .write_register("name", MvRegValue::String("Website Redesign".to_string()));

                    project_tx.in_array("tasks", |tasks_tx| {
                        tasks_tx
                            .insert_register(0, MvRegValue::String("Design mockups".to_string()));
                        tasks_tx.insert_register(
                            1,
                            MvRegValue::String("Implement frontend".to_string()),
                        );
                        tasks_tx.insert_register(2, MvRegValue::String("Deploy".to_string()));
                    });
                });
            });
        });

        let _delta = tx.commit();
    }

    println!("Created nested structure!");
    println!("  - Simple register: app_name");
    println!("  - Nested map: settings.theme, settings.notifications");
    println!("  - Array of maps: users[0..1]");
    println!("  - 4-level nesting: projects.active[0].tasks[0..2]");

    // No callbacks, no manual context management.
    // The same simple API at every level.

    println!("\n✓ Nested transactions eliminate callback hell!");
}
