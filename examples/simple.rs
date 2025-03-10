// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! The example simulates a scenario where two replicas modify the same data and
//! then synchronize their states, arriving at a consistent final result.

use dson::{CausalDotStore, Identifier, OrMap, crdts::NoExtensionTypes};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Create a unique identifier for replica A.
    let replica_a_id = Identifier::new(0, 0);

    // Initialize the state for replica A. The `CausalDotStore` holds the CRDT data
    // and its associated causal context. We use an `OrMap` (Observed-Remove Map)
    // with String values as our top-level CRDT.
    let mut replica_a_state = CausalDotStore::<OrMap<String>>::default();

    // --- Replica A: Set email for "alice" ---
    // The following operation creates a delta that represents the change of setting
    // the email for the key "alice". This delta only contains the change set, not
    // the full state.
    let delta_from_a = dson::api::map::apply_to_map::<_, NoExtensionTypes, _>(
        |inner_map, ctx, id| {
            // Within the "alice" map, we apply a change to the "email" register.
            dson::api::map::apply_to_register(
                // The new value for the register.
                |reg, ctx, id| reg.write("alice@example.com".to_string().into(), ctx, id),
                "email".to_string(),
            )(inner_map, ctx, id)
        },
        "alice".to_string(),
    )(
        // The operation is based on the current state of replica A.
        &replica_a_state.store,
        &replica_a_state.context,
        replica_a_id,
    );

    // Apply the generated delta to replica A's own state.
    replica_a_state.join_or_replace_with(delta_from_a.store.clone(), &delta_from_a.context);

    // --- Synchronization: A -> B ---
    // In a real-world scenario, the `delta_from_a` would be sent over a network
    // to other replicas. Here, we simulate this by creating a second replica and
    // applying the delta to it.

    // Create a unique identifier for replica B.
    let replica_b_id = Identifier::new(1, 0);

    // Initialize the state for replica B.
    let mut replica_b_state = CausalDotStore::<OrMap<String>>::default();

    // Apply the delta from replica A to replica B's state.
    replica_b_state.join_or_replace_with(delta_from_a.store.clone(), &delta_from_a.context);

    // After synchronization, the states of both replicas should be identical.
    assert_eq!(replica_a_state, replica_b_state);

    // --- Replica B: Update email for "alice" ---
    // Now, replica B makes a change to the same data. This will create a new
    // delta based on replica B's current state.
    let delta_from_b = dson::api::map::apply_to_map::<_, NoExtensionTypes, _>(
        |inner_map, ctx, id| {
            dson::api::map::apply_to_register(
                |reg, ctx, id| reg.write("bob@example.com".to_string().into(), ctx, id),
                "email".to_string(),
            )(inner_map, ctx, id)
        },
        "alice".to_string(),
    )(
        &replica_b_state.store,
        &replica_b_state.context,
        replica_b_id,
    );

    // Apply the new delta to replica B's own state.
    replica_b_state.join_or_replace_with(delta_from_b.store.clone(), &delta_from_b.context);

    // --- Synchronization: B -> A ---
    // Propagate the delta from replica B back to replica A.
    replica_a_state.join_or_replace_with(delta_from_b.store.clone(), &delta_from_b.context);

    // After this final synchronization, both replicas should once again have
    // identical states, reflecting the latest change made by replica B.
    assert_eq!(replica_a_state, replica_b_state);

    Ok(())
}
