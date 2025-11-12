//! Integration tests for nested transaction API.

use dson::{
    CausalDotStore, Identifier, OrMap, crdts::mvreg::MvRegValue, transaction::MapTransaction,
};

#[test]
fn deeply_nested_map_array_map() {
    // Structure: map -> array -> map
    // Like: { "projects": [{ "name": "DSON", "tasks": [...] }] }

    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    {
        let mut tx = MapTransaction::new(&mut store, id);

        tx.in_array("projects", |projects_tx| {
            projects_tx.insert_map(0, |project_tx| {
                project_tx.write_register("name", MvRegValue::String("DSON".to_string()));
                project_tx.write_register("priority", MvRegValue::U64(1));

                project_tx.in_array("tasks", |tasks_tx| {
                    tasks_tx
                        .insert_register(0, MvRegValue::String("Implement nested TX".to_string()));
                    tasks_tx.insert_register(1, MvRegValue::String("Write tests".to_string()));
                });
            });
        });

        let _delta = tx.commit();
    }

    // Verify structure was created
    use dson::crdts::snapshot::ToValue;

    let projects_val = store.store.get(&"projects".to_string()).unwrap();
    assert_eq!(projects_val.array.len(), 1);

    let project = projects_val.array.get(0).unwrap();
    let name = project.map.get(&"name".to_string()).unwrap();
    assert_eq!(
        name.reg.value().unwrap(),
        &MvRegValue::String("DSON".to_string())
    );

    let tasks_val = project.map.get(&"tasks".to_string()).unwrap();
    assert_eq!(tasks_val.array.len(), 2);
}

#[test]
fn array_of_arrays() {
    // Test [[1, 2], [3, 4]]

    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    {
        let mut tx = MapTransaction::new(&mut store, id);

        tx.in_array("matrix", |matrix_tx| {
            matrix_tx.insert_array(0, |row_tx| {
                row_tx.insert_register(0, MvRegValue::U64(1));
                row_tx.insert_register(1, MvRegValue::U64(2));
            });

            matrix_tx.insert_array(1, |row_tx| {
                row_tx.insert_register(0, MvRegValue::U64(3));
                row_tx.insert_register(1, MvRegValue::U64(4));
            });
        });

        let _delta = tx.commit();
    }

    // Verify 2x2 matrix
    use dson::crdts::snapshot::ToValue;

    let matrix = store.store.get(&"matrix".to_string()).unwrap();
    assert_eq!(matrix.array.len(), 2);

    let row0 = matrix.array.get(0).unwrap();
    assert_eq!(row0.array.len(), 2);
    assert_eq!(
        row0.array.get(0).unwrap().reg.value().unwrap(),
        &MvRegValue::U64(1)
    );
    assert_eq!(
        row0.array.get(1).unwrap().reg.value().unwrap(),
        &MvRegValue::U64(2)
    );

    let row1 = matrix.array.get(1).unwrap();
    assert_eq!(row1.array.len(), 2);
    assert_eq!(
        row1.array.get(0).unwrap().reg.value().unwrap(),
        &MvRegValue::U64(3)
    );
    assert_eq!(
        row1.array.get(1).unwrap().reg.value().unwrap(),
        &MvRegValue::U64(4)
    );
}

#[test]
fn concurrent_nested_modifications() {
    // Two replicas modify nested structures concurrently

    let id1 = Identifier::new(0, 0);
    let id2 = Identifier::new(1, 0);

    let mut replica1 = CausalDotStore::<OrMap<String>>::default();
    let mut replica2 = CausalDotStore::<OrMap<String>>::default();

    // Both create initial structure
    let init_delta = {
        let mut tx = MapTransaction::new(&mut replica1, id1);
        tx.in_map("config", |cfg_tx| {
            cfg_tx.write_register("version", MvRegValue::U64(1));
        });
        tx.commit()
    };

    replica1.join_or_replace_with(init_delta.0.store.clone(), &init_delta.0.context);
    replica2.join_or_replace_with(init_delta.0.store, &init_delta.0.context);

    // Replica 1: adds array to config
    let delta1 = {
        let mut tx = MapTransaction::new(&mut replica1, id1);
        tx.in_map("config", |cfg_tx| {
            cfg_tx.in_array("features", |features_tx| {
                features_tx.insert_register(0, MvRegValue::String("fast".to_string()));
            });
        });
        tx.commit()
    };

    // Replica 2: updates version concurrently
    let delta2 = {
        let mut tx = MapTransaction::new(&mut replica2, id2);
        tx.in_map("config", |cfg_tx| {
            cfg_tx.write_register("version", MvRegValue::U64(2));
        });
        tx.commit()
    };

    // Exchange deltas
    replica1.join_or_replace_with(delta2.0.store, &delta2.0.context);
    replica2.join_or_replace_with(delta1.0.store, &delta1.0.context);

    // Both should converge
    assert_eq!(replica1, replica2);

    // Verify both changes present
    use dson::crdts::snapshot::ToValue;

    let config = replica1.store.get(&"config".to_string()).unwrap();
    let version = config.map.get(&"version".to_string()).unwrap();
    assert!(version.reg.value().unwrap() == &MvRegValue::U64(2));

    let features = config.map.get(&"features".to_string()).unwrap();
    assert_eq!(features.array.len(), 1);
}
