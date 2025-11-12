// (c) Copyright 2025 Helsing GmbH. All rights reserved.
#![cfg_attr(not(target_os = "linux"), allow(dead_code, unused_imports))]

use dson::{
    CausalContext, CausalDotStore, Dot, Identifier, MvReg, OrArray, OrMap, api,
    crdts::{NoExtensionTypes, mvreg::MvRegValue},
    sentinel::DummySentinel,
};
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

include!(concat!(env!("OUT_DIR"), "/random_dots.rs"));

fn setup_array(n: usize) -> (Identifier, CausalDotStore<OrArray<NoExtensionTypes>>) {
    dson::enable_determinism();

    let omni_id = Identifier::new(1, 0);
    let mut omni = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
    for i in 0..n {
        let add = api::array::insert_register(
            |cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
            i,
        )(&omni.store, &omni.context, omni_id);
        omni.consume(add, &mut DummySentinel).unwrap();
    }
    (omni_id, omni)
}

#[library_benchmark]
#[bench::medium(setup_array(255))]
fn array_unshift((id, omni): (Identifier, CausalDotStore<OrArray<NoExtensionTypes>>)) {
    let omni = black_box(omni);
    let insert = api::array::insert_register(
        |cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
        0,
    )(&omni.store, &omni.context, id);
    black_box(insert);
}

#[library_benchmark]
#[bench::medium(setup_array(255))]
fn array_delete((id, omni): (Identifier, CausalDotStore<OrArray<NoExtensionTypes>>)) {
    let omni = black_box(omni);
    let delete = api::array::delete(128)(&omni.store, &omni.context, id);
    black_box(delete);
}

#[library_benchmark]
#[bench::medium(setup_array(255))]
fn array_update((id, omni): (Identifier, CausalDotStore<OrArray<NoExtensionTypes>>)) {
    let omni = black_box(omni);
    let update = api::array::apply_to_register(
        |old, cc, id| old.write(MvRegValue::Bool(false), cc, id),
        128,
    )(&omni.store, &omni.context, id);
    black_box(update);
}

#[library_benchmark]
#[bench::medium(setup_array(255))]
fn array_insert((id, omni): (Identifier, CausalDotStore<OrArray<NoExtensionTypes>>)) {
    let omni = black_box(omni);
    let insert = api::array::insert_register(
        |cc, id| MvReg::default().write(MvRegValue::Bool(false), cc, id),
        128,
    )(&omni.store, &omni.context, id);
    black_box(insert);
}

#[library_benchmark]
#[bench::medium(setup_array(255))]
fn array_push((id, omni): (Identifier, CausalDotStore<OrArray<NoExtensionTypes>>)) {
    let omni = black_box(omni);
    let insert = api::array::insert_register(
        |cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
        omni.store.len(),
    )(&omni.store, &omni.context, id);
    black_box(insert);
}

fn setup_map(n: usize) -> (Identifier, CausalDotStore<OrMap<String, NoExtensionTypes>>) {
    dson::enable_determinism();

    let omni_id = Identifier::new(1, 0);
    let mut omni = CausalDotStore::<OrMap<String, NoExtensionTypes>>::default();
    for i in 0..n {
        let add = api::map::apply_to_register(
            |_, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
            i.to_string(),
        )(&omni.store, &omni.context, omni_id);
        omni.consume(add, &mut DummySentinel).unwrap();
    }
    (omni_id, omni)
}

fn setup_direct_crdt_map(
    n: usize,
) -> (Identifier, CausalDotStore<OrMap<String, NoExtensionTypes>>) {
    dson::enable_determinism();

    let omni_id = Identifier::new(1, 0);
    let mut omni = CausalDotStore::<OrMap<String, NoExtensionTypes>>::default();
    for i in 0..n {
        let delta = omni.store.apply_to_register(
            |reg, ctx, id| reg.write(MvRegValue::Bool(true), ctx, id),
            i.to_string(),
            &omni.context,
            omni_id,
        );
        omni.consume(delta, &mut DummySentinel).unwrap();
    }
    (omni_id, omni)
}

#[library_benchmark]
#[bench::medium(setup_map(255))]
fn map_insert((id, omni): (Identifier, CausalDotStore<OrMap<String, NoExtensionTypes>>)) {
    let omni = black_box(omni);
    let insert = api::map::apply_to_register(
        |_, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
        "duck".into(),
    )(&omni.store, &omni.context, id);
    black_box(insert);
}

#[library_benchmark]
#[bench::medium(setup_map(255))]
fn map_remove((id, omni): (Identifier, CausalDotStore<OrMap<String, NoExtensionTypes>>)) {
    let omni = black_box(omni);
    let remove = api::map::remove("128")(&omni.store, &omni.context, id);
    black_box(remove);
}

#[library_benchmark]
#[bench::medium(setup_map(255))]
fn map_update((id, omni): (Identifier, CausalDotStore<OrMap<String, NoExtensionTypes>>)) {
    let omni = black_box(omni);
    let update = api::map::apply_to_register(
        |old, cc, id| old.write(MvRegValue::Bool(true), cc, id),
        "128".into(),
    )(&omni.store, &omni.context, id);
    black_box(update);
}

#[library_benchmark]
#[bench::medium(setup_direct_crdt_map(255))]
fn direct_crdt_map_insert(
    (id, omni): (Identifier, CausalDotStore<OrMap<String, NoExtensionTypes>>),
) {
    let omni = black_box(omni);
    let insert = omni.store.apply_to_register(
        |reg, ctx, id| reg.write(MvRegValue::Bool(true), ctx, id),
        "duck".to_string(),
        &omni.context,
        id,
    );
    black_box(insert);
}

#[library_benchmark]
#[bench::medium(setup_direct_crdt_map(255))]
fn direct_crdt_map_remove(
    (id, omni): (Identifier, CausalDotStore<OrMap<String, NoExtensionTypes>>),
) {
    let omni = black_box(omni);
    let remove = omni.store.remove(&"128".to_string(), &omni.context, id);
    black_box(remove);
}

#[library_benchmark]
#[bench::medium(setup_direct_crdt_map(255))]
fn direct_crdt_map_update(
    (id, omni): (Identifier, CausalDotStore<OrMap<String, NoExtensionTypes>>),
) {
    let omni = black_box(omni);
    let update = omni.store.apply_to_register(
        |reg, ctx, id| reg.write(MvRegValue::Bool(true), ctx, id),
        "128".to_string(),
        &omni.context,
        id,
    );
    black_box(update);
}

fn setup_register() -> (Identifier, CausalDotStore<MvReg>) {
    dson::enable_determinism();

    let omni_id = Identifier::new(1, 0);
    let omni = CausalDotStore::<MvReg>::default();
    let write = api::register::write(MvRegValue::Bool(false))(&omni.store, &omni.context, omni_id);
    (omni_id, omni.join(write, &mut DummySentinel).unwrap())
}

#[library_benchmark]
#[bench::bool(setup_register())]
fn register_write((id, omni): (Identifier, CausalDotStore<MvReg>)) {
    let omni = black_box(omni);
    let write = api::register::write(MvRegValue::Bool(true))(&omni.store, &omni.context, id);
    black_box(write);
}

#[library_benchmark]
#[bench::bool(setup_register())]
fn register_clear((id, omni): (Identifier, CausalDotStore<MvReg>)) {
    let omni = black_box(omni);
    let clear = api::register::clear()(&omni.store, &omni.context, id);
    black_box(clear);
}

struct Ccs {
    big1: CausalContext,
    big2: CausalContext,
    small1: CausalContext,
    #[allow(dead_code)]
    small2: CausalContext,
}

fn setup_cc() -> Ccs {
    dson::enable_determinism();

    let big1 = CausalContext::from_iter(BIG1.iter().copied());
    let big2 = CausalContext::from_iter(BIG2.iter().copied());
    let small1 = CausalContext::from_iter(SMALL1.iter().copied());
    let small2 = CausalContext::from_iter(SMALL2.iter().copied());
    Ccs {
        big1,
        big2,
        small1,
        small2,
    }
}

#[library_benchmark]
#[bench::id(setup_cc())]
fn cc_join_big_small(ccs: Ccs) {
    let mut ccs = black_box(ccs);
    ccs.big1.union(&ccs.small1);
    black_box(ccs);
}

#[library_benchmark]
#[bench::id(setup_cc())]
fn cc_join_big_big(ccs: Ccs) {
    let mut ccs = black_box(ccs);
    ccs.big1.union(&ccs.big2);
    black_box(ccs);
}

library_benchmark_group!(
    name = arrays;
    benchmarks = array_unshift, array_delete, array_update, array_insert
);
library_benchmark_group!(
    name = maps;
    benchmarks = map_insert, map_remove, map_update
);
library_benchmark_group!(
    name = direct_crdt_maps;
    benchmarks = direct_crdt_map_insert, direct_crdt_map_remove, direct_crdt_map_update
);
library_benchmark_group!(
    name = registers;
    benchmarks = register_write, register_clear
);
library_benchmark_group!(
    name = causal_contexts;
    benchmarks = cc_join_big_small, cc_join_big_big
);

#[cfg(target_os = "linux")]
main!(
    library_benchmark_groups = arrays,
    maps,
    direct_crdt_maps,
    registers,
    causal_contexts
);

#[cfg(not(target_os = "linux"))]
fn main() {}
