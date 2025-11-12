// (c) Copyright 2025 Helsing GmbH. All rights reserved.
// because we need this below to retain 'static on the borrows of omni
#![allow(clippy::borrow_deref_ref)]

use dson::{
    CausalContext, CausalDotStore, Dot, Identifier, MvReg, OrArray, OrMap, api,
    crdts::{NoExtensionTypes, mvreg::MvRegValue},
    sentinel::DummySentinel,
};
use std::hint::black_box;
use tango_bench::{IntoBenchmarks, benchmark_fn, tango_benchmarks, tango_main};

include!(concat!(env!("OUT_DIR"), "/random_dots.rs"));

fn array_benchmarks() -> impl IntoBenchmarks {
    dson::enable_determinism();

    let omni_id = Identifier::new(1, 0);
    let mut omni = CausalDotStore::<OrArray<NoExtensionTypes>>::default();
    for i in 0..255 {
        let add = api::array::insert_register(
            |cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
            i,
        )(&omni.store, &omni.context, omni_id);
        omni.consume(add, &mut DummySentinel).unwrap();
    }

    let omni: &'static _ = Box::leak(Box::new(omni));
    [
        benchmark_fn("array::unshift", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                api::array::insert_register(
                    |cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    0,
                )(&omni.store, &omni.context, omni_id)
            })
        }),
        benchmark_fn("array::delete", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                api::array::delete(128)(&omni.store, &omni.context, omni_id)
            })
        }),
        benchmark_fn("array::update", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                api::array::apply_to_register(
                    |old, cc, id| old.write(MvRegValue::Bool(false), cc, id),
                    128,
                )(&omni.store, &omni.context, omni_id)
            })
        }),
        benchmark_fn("array::insert", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                api::array::insert_register(
                    |cc, id| MvReg::default().write(MvRegValue::Bool(false), cc, id),
                    128,
                )(&omni.store, &omni.context, omni_id)
            })
        }),
        benchmark_fn("array::push", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                api::array::insert_register(
                    |cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    omni.store.len(),
                )(&omni.store, &omni.context, omni_id)
            })
        }),
    ]
}

fn map_benchmarks() -> impl IntoBenchmarks {
    dson::enable_determinism();

    let omni_id = Identifier::new(1, 0);
    let mut omni = CausalDotStore::<OrMap<String, NoExtensionTypes>>::default();
    for i in 0..255 {
        let add = api::map::apply_to_register(
            |_, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
            i.to_string(),
        )(&omni.store, &omni.context, omni_id);
        omni.consume(add, &mut DummySentinel).unwrap();
    }

    let omni: &'static _ = Box::leak(Box::new(omni));
    [
        benchmark_fn("map::insert", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                api::map::apply_to_register(
                    |_, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    "duck".into(),
                )(&omni.store, &omni.context, omni_id)
            })
        }),
        benchmark_fn("map::remove", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                api::map::remove("128")(&omni.store, &omni.context, omni_id)
            })
        }),
        benchmark_fn("map::update", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                api::map::apply_to_register(
                    |old, cc, id| old.write(MvRegValue::Bool(true), cc, id),
                    "128".into(),
                )(&omni.store, &omni.context, omni_id)
            })
        }),
    ]
}

fn direct_crdt_map_benchmarks() -> impl IntoBenchmarks {
    dson::enable_determinism();

    let omni_id = Identifier::new(1, 0);
    let mut omni = CausalDotStore::<OrMap<String, NoExtensionTypes>>::default();
    for i in 0..255 {
        let delta = omni.store.apply_to_register(
            |reg, ctx, id| reg.write(MvRegValue::Bool(true), ctx, id),
            i.to_string(),
            &omni.context,
            omni_id,
        );
        omni.consume(delta, &mut DummySentinel).unwrap();
    }

    let omni: &'static _ = Box::leak(Box::new(omni));
    [
        benchmark_fn("direct-crdt::map::insert", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                omni.store.apply_to_register(
                    |reg, ctx, id| reg.write(MvRegValue::Bool(true), ctx, id),
                    "duck".to_string(),
                    &omni.context,
                    omni_id,
                )
            })
        }),
        benchmark_fn("direct-crdt::map::remove", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                omni.store
                    .remove(&"128".to_string(), &omni.context, omni_id)
            })
        }),
        benchmark_fn("direct-crdt::map::update", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                omni.store.apply_to_register(
                    |reg, ctx, id| reg.write(MvRegValue::Bool(true), ctx, id),
                    "128".to_string(),
                    &omni.context,
                    omni_id,
                )
            })
        }),
    ]
}

fn register_benchmarks() -> impl IntoBenchmarks {
    dson::enable_determinism();

    let omni_id = Identifier::new(1, 0);
    let mut omni = CausalDotStore::<MvReg>::default();
    let write = api::register::write(MvRegValue::Bool(false))(&omni.store, &omni.context, omni_id);
    omni.consume(write, &mut DummySentinel).unwrap();

    let omni: &'static _ = Box::leak(Box::new(omni));
    [
        benchmark_fn("register::write", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                api::register::write(MvRegValue::Bool(true))(&omni.store, &omni.context, omni_id)
            })
        }),
        benchmark_fn("register::clear", move |b| {
            b.iter(move || {
                let omni = black_box(&*omni);
                api::register::clear()(&omni.store, &omni.context, omni_id)
            })
        }),
    ]
}

fn transaction_map_benchmarks() -> impl IntoBenchmarks {
    dson::enable_determinism();
    let omni_id = Identifier::new(1, 0);

    // Setup for single-op benchmarks (no pre-population, isolate transaction overhead)
    [
        benchmark_fn("transaction::map::insert-empty", move |b| {
            b.iter(move || {
                let mut omni = CausalDotStore::<OrMap<String, NoExtensionTypes>>::default();
                let mut tx = black_box(&mut omni).transact(omni_id);
                tx.write_register("duck".to_string(), MvRegValue::Bool(true));
                black_box(tx.commit())
            })
        }),
        benchmark_fn("transaction::map::insert-with-setup", move |b| {
            b.iter(move || {
                let mut omni = CausalDotStore::<OrMap<String, NoExtensionTypes>>::default();
                for i in 0..255 {
                    let delta = omni.store.apply_to_register(
                        |reg, ctx, id| reg.write(MvRegValue::Bool(true), ctx, id),
                        i.to_string(),
                        &omni.context,
                        omni_id,
                    );
                    omni.consume(delta, &mut DummySentinel).unwrap();
                }
                let mut tx = black_box(&mut omni).transact(omni_id);
                tx.write_register("duck".to_string(), MvRegValue::Bool(true));
                black_box(tx.commit())
            })
        }),
        benchmark_fn("transaction::map::remove-empty", move |b| {
            b.iter(move || {
                let mut omni = CausalDotStore::<OrMap<String, NoExtensionTypes>>::default();
                let mut tx = black_box(&mut omni).transact(omni_id);
                tx.remove("128".to_string());
                black_box(tx.commit())
            })
        }),
        benchmark_fn("transaction::map::update-empty", move |b| {
            b.iter(move || {
                let mut omni = CausalDotStore::<OrMap<String, NoExtensionTypes>>::default();
                let mut tx = black_box(&mut omni).transact(omni_id);
                tx.write_register("128".to_string(), MvRegValue::Bool(true));
                black_box(tx.commit())
            })
        }),
    ]
}

fn cc_benchmarks() -> impl IntoBenchmarks {
    dson::enable_determinism();

    let big1 = CausalContext::from_iter(BIG1.iter().copied());
    let big2 = CausalContext::from_iter(BIG2.iter().copied());
    let small1 = CausalContext::from_iter(SMALL1.iter().copied());
    let small2 = CausalContext::from_iter(SMALL2.iter().copied());

    let big1: &'static _ = Box::leak(Box::new(big1));
    let big2: &'static _ = Box::leak(Box::new(big2));
    let small1: &'static _ = Box::leak(Box::new(small1));
    let small2: &'static _ = Box::leak(Box::new(small2));
    [
        benchmark_fn("causal-context::join::both_same_small", move |b| {
            b.iter(|| {
                let mut left = black_box(small1.clone());
                let right = black_box(&*small1);
                left.union(right)
            })
        }),
        benchmark_fn("causal-context::join::both_small", move |b| {
            b.iter(|| {
                let mut left = black_box(small1.clone());
                let right = black_box(&*small2);
                left.union(right)
            })
        }),
        benchmark_fn("causal-context::join::left_big", move |b| {
            b.iter(|| {
                let mut left = black_box(big1.clone());
                let right = black_box(&*small1);
                left.union(right)
            })
        }),
        benchmark_fn("causal-context::join::right_big", move |b| {
            b.iter(|| {
                let mut left = black_box(small1.clone());
                let right = black_box(&*big1);
                left.union(right)
            })
        }),
        benchmark_fn("causal-context::join::both_big", move |b| {
            b.iter(|| {
                let mut left = black_box(big1.clone());
                let right = black_box(&*big2);
                left.union(right)
            })
        }),
        benchmark_fn("causal-context::join::both_same_big", move |b| {
            b.iter(|| {
                let mut left = black_box(big1.clone());
                let right = black_box(&*big1);
                left.union(right)
            })
        }),
    ]
}

tango_benchmarks!(
    array_benchmarks(),
    map_benchmarks(),
    direct_crdt_map_benchmarks(),
    transaction_map_benchmarks(),
    register_benchmarks(),
    cc_benchmarks()
);
tango_main!();
