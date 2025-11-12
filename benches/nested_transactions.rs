use dson::{
    CausalDotStore, Identifier, OrMap, crdts::mvreg::MvRegValue, transaction::MapTransaction,
};
use iai_callgrind::{library_benchmark, library_benchmark_group, main};

#[library_benchmark]
fn nested_transaction_3_levels() {
    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    let mut tx = MapTransaction::new(&mut store, id);
    tx.in_map("level1", |l1_tx| {
        l1_tx.in_map("level2", |l2_tx| {
            l2_tx.write_register("value", MvRegValue::U64(42));
        });
    });
    let _delta = tx.commit();
}

#[library_benchmark]
fn direct_crdt_api_3_levels() {
    let mut store = CausalDotStore::<OrMap<String>>::default();
    let id = Identifier::new(0, 0);

    // Direct CRDT operations without transaction API
    let delta = store.store.apply_to_map(
        |l1, ctx1, id1| {
            l1.apply_to_map(
                |l2, ctx2, id2| {
                    l2.apply_to_register(
                        |reg, ctx3, id3| reg.write(MvRegValue::U64(42), ctx3, id3),
                        "value".to_string(),
                        ctx2,
                        id2,
                    )
                },
                "level2".to_string(),
                ctx1,
                id1,
            )
        },
        "level1".to_string(),
        &store.context,
        id,
    );
    store.join_or_replace_with(delta.store, &delta.context);
}

library_benchmark_group!(
    name = nested_transaction_benches;
    benchmarks = nested_transaction_3_levels, direct_crdt_api_3_levels
);

#[cfg(target_os = "linux")]
main!(library_benchmark_groups = nested_transaction_benches);

#[cfg(not(target_os = "linux"))]
fn main() {}
