// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! This example demonstrates how dson handles concurrent edits and resolves conflicts.
//! We simulate two replicas of a user profile, make conflicting changes to the same fields,
//! and then merge them to observe the final, converged state.
use dson::{
    CausalDotStore, Identifier, MvReg, OrMap,
    crdts::{
        NoExtensionTypes, Value,
        mvreg::MvRegValue,
        snapshot::{AllValues, ToValue},
    },
    sentinel::DummySentinel,
};
use std::error::Error;

// The data model for our user profile is a map with string keys.
// - "name": A Multi-Value Register (MvReg) for the user's name. Concurrent writes will be preserved as conflicts.
// - "tags": An Observed-Remove Array (OrArray) of MvReg<String> for tags.
// - "settings": A nested Observed-Remove Map (OrMap) for user settings.

fn main() -> Result<(), Box<dyn Error>> {
    // SETUP: TWO REPLICAS
    // We create two replicas, A and B, each with a unique identifier.
    // Both start with an empty CausalDotStore, which will hold our OrMap-based user profile.
    let replica_a_id = Identifier::new(0, 0);
    let mut replica_a_state = CausalDotStore::<OrMap<String>>::default();

    let replica_b_id = Identifier::new(1, 0);
    let mut replica_b_state = CausalDotStore::<OrMap<String>>::default();

    // INITIAL STATE on Replica A
    println!("1. Replica A creates an initial user profile.");
    // We create a "user" map and set the "name" field to "Alice".
    // This operation generates a delta (`delta_a1`) representing the change.
    let delta_a1 = dson::api::map::apply_to_map::<_, NoExtensionTypes, _>(
        |map, ctx, id| {
            // Set name in the user map
            dson::api::map::apply_to_register(
                |reg, ctx, id| reg.write("Alice".to_string().into(), ctx, id),
                "name".to_string(),
            )(map, ctx, id)
        },
        "user".to_string(),
    )(
        &replica_a_state.store,
        &replica_a_state.context,
        replica_a_id,
    );

    // Apply the delta to Replica A's state.
    replica_a_state.join_or_replace_with(delta_a1.store.clone(), &delta_a1.context);

    // SYNC: REPLICA B GETS INITIAL STATE
    println!("2. Replica B syncs with Replica A.");
    // Replica B applies the delta from Replica A to get the initial state.
    // After this, both replicas are in sync.
    replica_b_state.join_or_replace_with(delta_a1.store, &delta_a1.context);
    assert_eq!(replica_a_state, replica_b_state);
    println!("   Initial state synced: {replica_a_state:?}");

    // CONCURRENT EDITS
    println!("\n3. Replicas A and B make concurrent edits without syncing.");

    // On Replica A: Change the name to "Alice B." and add a "rust" tag.
    // These changes are based on the initial state.
    let delta_a2 = dson::api::map::apply_to_map::<_, NoExtensionTypes, _>(
        |map, ctx, id| {
            // 1. Change the name
            let map_after_name_change = dson::api::map::apply_to_register(
                |reg, ctx, id| reg.write("Alice B.".to_string().into(), ctx, id),
                "name".to_string(),
            )(map, ctx, id);

            // 2. Add a tag to the 'tags' array
            let map_after_tag_add = dson::api::map::apply_to_array(
                |array, ctx, id| {
                    dson::api::array::insert(
                        // Each element in the array is a register for the tag string
                        |ctx, _id| {
                            MvReg::default()
                                .write("rust".to_string().into(), ctx, id)
                                .map_store(Value::Register)
                        },
                        array.len(), // Insert at the end
                    )(array, ctx, id)
                },
                "tags".to_string(),
            )(
                &map_after_name_change.store,
                &map_after_name_change.context,
                id,
            );

            // Join the two operations into a single delta
            map_after_name_change
                .join(map_after_tag_add, &mut DummySentinel)
                .expect("DummySentinel is infallible")
        },
        "user".to_string(),
    )(
        &replica_a_state.store,
        &replica_a_state.context,
        replica_a_id,
    );
    // Apply the changes locally to Replica A.
    replica_a_state.join_or_replace_with(delta_a2.store.clone(), &delta_a2.context);
    println!("   Replica A: Changed name to 'Alice B.', added 'rust' tag.");

    // On Replica B: Change name to "Alice C." (a direct conflict with Replica A's change),
    // add a "crdt" tag, and add a new "dark_mode" setting.
    let delta_b1 = dson::api::map::apply_to_map::<_, NoExtensionTypes, _>(
        |map, ctx, id| {
            // 1. Change the name, creating a conflict with Replica A's edit.
            let map_after_name_change = dson::api::map::apply_to_register(
                |reg, ctx, id| reg.write("Alice C.".to_string().into(), ctx, id),
                "name".to_string(),
            )(map, ctx, id);

            // 2. Add a "crdt" tag.
            let map_after_tag_add = dson::api::map::apply_to_array(
                |array, ctx, id| {
                    dson::api::array::insert(
                        |ctx, id| {
                            MvReg::default()
                                .write("crdt".to_string().into(), ctx, id)
                                .map_store(Value::Register)
                        },
                        array.len(),
                    )(array, ctx, id)
                },
                "tags".to_string(),
            )(
                &map_after_name_change.store,
                &map_after_name_change.context,
                id,
            );

            // Join the name and tag changes
            let delta_with_name_and_tag = map_after_name_change
                .join(map_after_tag_add, &mut DummySentinel)
                .expect("DummySentinel is infallible");

            // 3. Add a "dark_mode" setting in a nested map.
            let delta_with_settings = dson::api::map::apply_to_map(
                |settings_map, ctx, id| {
                    dson::api::map::apply_to_register(
                        |reg, ctx, id| reg.write(true.into(), ctx, id),
                        "dark_mode".to_string(),
                    )(settings_map, ctx, id)
                },
                "settings".to_string(),
            )(
                &delta_with_name_and_tag.store,
                &delta_with_name_and_tag.context,
                id,
            );

            // Join all changes for Replica B into a final delta.
            delta_with_name_and_tag
                .join(delta_with_settings, &mut DummySentinel)
                .expect("DummySentinel is infallible")
        },
        "user".to_string(),
    )(
        &replica_b_state.store,
        &replica_b_state.context,
        replica_b_id,
    );
    // Apply the changes locally to Replica B.
    replica_b_state.join_or_replace_with(delta_b1.store.clone(), &delta_b1.context);
    println!("   Replica B: Changed name to 'Alice C.', added 'crdt' tag, enabled dark_mode.");

    // MERGE
    println!("\n4. Merging the concurrent changes.");
    // Replica A merges the delta from Replica B.
    replica_a_state.join_or_replace_with(delta_b1.store, &delta_b1.context);
    // Replica B merges the delta from Replica A.
    replica_b_state.join_or_replace_with(delta_a2.store, &delta_a2.context);
    // After merging, both replicas should have an identical state, demonstrating convergence.

    // VERIFICATION
    println!("\n5. Verifying the converged state.");
    assert_eq!(replica_a_state, replica_b_state);
    println!("   Replicas have converged to the same state.");
    println!("   Final state: {replica_a_state:?}");

    // Now, let's inspect the converged data structure to see how conflicts were handled.
    let user_profile = replica_a_state
        .store
        .get("user")
        .expect("key 'user' should be present");

    // --- Verify Name Conflict ---
    // The concurrent writes to the "name" field result in a conflict.
    // The MvReg preserves both values. The application can then decide how to resolve this.
    let name_values = user_profile
        .map
        .get("name")
        .unwrap()
        .reg
        .values()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();

    assert_eq!(
        name_values.len(),
        2,
        "Name should have two conflicting values"
    );
    assert!(name_values.contains(&MvRegValue::String("Alice B.".to_string())));
    assert!(name_values.contains(&MvRegValue::String("Alice C.".to_string())));
    println!("   SUCCESS: Name field correctly shows conflicting values: {name_values:?}");

    // --- Verify Tags Array ---
    // The 'tags' array should contain both "rust" and "crdt", as they were added concurrently.
    let tags = user_profile
        .map
        .get("tags")
        .expect("key 'tags' should be present");
    let tag_values = tags
        .array
        .values()
        .iter()
        .map(|v| {
            let AllValues::Register(r) = v else {
                unreachable!()
            };
            // No conflicts are expected within the tags themselves.
            assert_eq!(r.len(), 1);
            let MvRegValue::String(s) = r.get(0).unwrap() else {
                unreachable!()
            };
            s.to_owned()
        })
        .collect::<Vec<_>>();

    assert_eq!(tag_values.len(), 2, "Tags array should have two elements");
    assert!(tag_values.contains(&"rust".to_string()));
    assert!(tag_values.contains(&"crdt".to_string()));
    println!("   SUCCESS: Tags array correctly contains: {tag_values:?}");

    // --- Verify Settings Map ---
    // The 'settings' map was only modified by Replica B, so it should exist with the 'dark_mode' key.
    let settings = user_profile
        .map
        .get("settings")
        .expect("key 'settings' should be present");
    let dark_mode = settings
        .map
        .get("dark_mode")
        .expect("key 'dark_mode' should be present")
        .reg
        .value() // We expect a single value since there were no concurrent edits.
        .expect("should be no conflict in dark_mode setting");

    assert_eq!(*dark_mode, MvRegValue::Bool(true));
    println!("   SUCCESS: Settings map correctly contains: dark_mode -> {dark_mode:?}");

    Ok(())
}
