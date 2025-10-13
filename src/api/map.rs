// (c) Copyright 2025 Helsing GmbH. All rights reserved.
use crate::{
    CausalContext, CausalDotStore, ExtensionType, Identifier, MvReg, OrArray, OrMap,
    crdts::{
        TypeVariantValue, Value,
        snapshot::{self, ToValue},
    },
};
use std::{borrow::Borrow, fmt, hash::Hash};

/// Returns the values of this map without collapsing conflicts.
pub fn values<K, C>(
    m: &OrMap<K, C>,
) -> snapshot::OrMap<'_, K, snapshot::AllValues<'_, C::ValueRef<'_>>>
where
    K: Hash + Eq + fmt::Display,
    C: ExtensionType,
{
    m.values()
}

/// Returns the values of this map assuming (and asserting) no conflicts on element values.
// NOTE: A type alias won't help much here :melt:.
#[allow(clippy::type_complexity)]
pub fn value<K, C>(
    m: &OrMap<K, C>,
) -> Result<
    snapshot::OrMap<'_, K, snapshot::CollapsedValue<'_, C::ValueRef<'_>>>,
    Box<snapshot::SingleValueError<<&OrMap<K, C> as ToValue>::LeafValue>>,
>
where
    K: Hash + Eq + fmt::Debug + fmt::Display + Clone,
    C: ExtensionType,
{
    m.value()
}

/// Creates a new map.
pub fn create<K, C>()
-> impl Fn(&OrMap<K, C>, &CausalContext, Identifier) -> CausalDotStore<OrMap<K, C>>
where
    K: Hash + Eq + fmt::Debug + Clone,
    C: ExtensionType,
{
    move |m, cc, id| m.create(cc, id)
}

/// Applies a function to the value at the given key.
pub fn apply<K, C, O>(
    o: O,
    k: K,
) -> impl FnOnce(&OrMap<K, C>, &CausalContext, Identifier) -> CausalDotStore<OrMap<K, C>>
where
    K: Hash + Eq + fmt::Debug + Clone,
    O: FnOnce(&TypeVariantValue<C>, &CausalContext, Identifier) -> CausalDotStore<Value<C>>,
    C: ExtensionType,
{
    move |m, cc, id| m.apply(o, k.clone(), cc, id)
}

/// Applies a function to the map at the given key.
pub fn apply_to_map<K, C, O>(
    o: O,
    k: K,
) -> impl FnOnce(&OrMap<K, C>, &CausalContext, Identifier) -> CausalDotStore<OrMap<K, C>>
where
    K: Hash + Eq + fmt::Debug + Clone,
    O: FnOnce(&OrMap<String, C>, &CausalContext, Identifier) -> CausalDotStore<OrMap<String, C>>,
    C: ExtensionType,
{
    move |m, cc, id| m.apply_to_map(o, k.clone(), cc, id)
}

/// Applies a function to the array at the given key.
pub fn apply_to_array<K, C, O>(
    o: O,
    k: K,
) -> impl FnOnce(&OrMap<K, C>, &CausalContext, Identifier) -> CausalDotStore<OrMap<K, C>>
where
    K: Hash + Eq + fmt::Debug + Clone,
    O: FnOnce(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>,
    C: ExtensionType,
{
    move |m, cc, id| m.apply_to_array(o, k.clone(), cc, id)
}

/// Applies a function to the register at the given key.
pub fn apply_to_register<K, C, O>(
    o: O,
    k: K,
) -> impl FnOnce(&OrMap<K, C>, &CausalContext, Identifier) -> CausalDotStore<OrMap<K, C>>
where
    K: Hash + Eq + fmt::Debug + Clone,
    O: FnOnce(&MvReg, &CausalContext, Identifier) -> CausalDotStore<MvReg>,
    C: ExtensionType,
{
    move |m, cc, id| m.apply_to_register(o, k.clone(), cc, id)
}

/// Removes a key from the map.
pub fn remove<Q, K, C>(
    k: &Q,
) -> impl Fn(&OrMap<K, C>, &CausalContext, Identifier) -> CausalDotStore<OrMap<K, C>> + '_
where
    K: Hash + Eq + fmt::Debug + Clone + Borrow<Q>,
    Q: Hash + Eq + ?Sized,
    C: ExtensionType,
{
    move |m, cc, id| m.remove(k, cc, id)
}

/// Clears the map.
pub fn clear<K, C>()
-> impl Fn(&OrMap<K, C>, &CausalContext, Identifier) -> CausalDotStore<OrMap<K, C>>
where
    K: Hash + Eq + fmt::Debug + Clone,
    C: ExtensionType,
{
    move |m, cc, id| m.clear(cc, id)
}
