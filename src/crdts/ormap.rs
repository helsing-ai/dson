// (c) Copyright 2025 Helsing GmbH. All rights reserved.
use super::{
    Either, NoExtensionTypes, TypeVariantValue, Value, ValueRef,
    mvreg::MvRegValue,
    orarray::Uid,
    snapshot::{self, AllValues, CollapsedValue, SingleValueError, SingleValueIssue, ToValue},
};
use crate::{
    CausalContext, CausalDotStore, DotMap, DotStoreJoin, ExtensionType, Identifier, MvReg, OrArray,
    dotstores::{DotChange, DotStore, DryJoinOutput},
    sentinel::{KeySentinel, TypeSentinel, ValueSentinel, Visit},
};
use std::{borrow::Borrow, fmt, hash::Hash, ops::Index};

/// An **Observed-Remove Map**, a map-like CRDT that allows for concurrent creation, updates, and
/// removals of key-value pairs.
///
/// `OrMap` is one of the three core CRDT primitives provided by this crate, alongside [`OrArray`]
/// and [`MvReg`]. It is the most common choice for a top-level CRDT, as it can hold other CRDTs as
/// values, allowing for the creation of nested, JSON-like data structures.
///
/// ## Usage
///
/// An `OrMap` is typically wrapped in a [`CausalDotStore`], which tracks the causal history of
/// operations. Modifications are performed by creating a "delta" CRDT, which is then merged back
/// into the original `CausalDotStore`.
///
/// ```rust
/// # use dson::{CausalDotStore, OrMap, MvReg, crdts::{mvreg::MvRegValue, snapshot::{ToValue, CollapsedValue}}, Identifier, sentinel::DummySentinel};
/// // Create a new CausalDotStore containing an OrMap.
/// let mut doc: CausalDotStore<OrMap<String>> = CausalDotStore::new();
/// let id = Identifier::new(0, 0);
///
/// // Create a delta to insert a value.
/// let delta = doc.store.apply_to_register(
///     |reg, cc, id| reg.write(MvRegValue::U64(42), cc, id),
///     "key".into(),
///     &doc.context,
///     id,
/// );
///
/// // Merge the delta into the document.
/// doc = doc.join(delta, &mut DummySentinel).unwrap();
///
/// // The value can now be read from the map.
/// let val = doc.store.get("key").unwrap();
/// assert_eq!(val.reg.value().unwrap(), &MvRegValue::U64(42));
/// ```
///
/// You can find more convenient, higher-level APIs for manipulating `OrMap` in the
/// [`api::map`](crate::api::map) module. The methods on `OrMap` itself are low-level and
/// intended for use when implementing custom CRDTs or when you need fine-grained control over
/// delta creation.
///
/// This type is a composable mapping of keys (`K`) to an arbitrary 𝛿-based CRDT such as an
/// [`OrArray`], a [`MvReg`], or a nested [`OrMap`] (all represented via [`TypeVariantValue`]).
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct OrMap<K: Hash + Eq, C = NoExtensionTypes>(pub(super) DotMap<K, TypeVariantValue<C>>);

impl<K, C> Default for OrMap<K, C>
where
    K: Hash + Eq,
{
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<K, C> std::fmt::Debug for OrMap<K, C>
where
    K: Hash + Eq + std::fmt::Debug,
    C: fmt::Debug + ExtensionType,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<K, C> FromIterator<(K, TypeVariantValue<C>)> for OrMap<K, C>
where
    K: Eq + Hash,
{
    fn from_iter<T: IntoIterator<Item = (K, TypeVariantValue<C>)>>(iter: T) -> Self {
        Self(DotMap::from_iter(iter))
    }
}

impl<K, Q, C> Index<&Q> for OrMap<K, C>
where
    K: Eq + Hash + Borrow<Q>,
    Q: Eq + Hash + ?Sized,
{
    type Output = TypeVariantValue<C>;

    fn index(&self, index: &Q) -> &Self::Output {
        self.0.index(index)
    }
}

impl<K, C> DotStore for OrMap<K, C>
where
    K: Hash + Eq + fmt::Debug + Clone,
    C: ExtensionType,
{
    fn dots(&self) -> CausalContext {
        self.0.dots()
    }

    fn add_dots_to(&self, other: &mut CausalContext) {
        self.0.add_dots_to(other);
    }

    fn is_bottom(&self) -> bool {
        self.0.is_bottom()
    }

    fn subset_for_inflation_from(&self, frontier: &CausalContext) -> Self {
        Self(DotMap::subset_for_inflation_from(&self.0, frontier))
    }
}

impl<K, C, S> DotStoreJoin<S> for OrMap<K, C>
where
    K: Hash + Eq + fmt::Debug + Clone,
    C: ExtensionType + DotStoreJoin<S> + fmt::Debug + Clone + PartialEq,
    S: Visit<K>
        + Visit<String>
        + Visit<Uid>
        + KeySentinel
        + TypeSentinel<C::ValueKind>
        + ValueSentinel<MvRegValue>,
{
    fn join(
        (m1, cc1): (Self, &CausalContext),
        (m2, cc2): (Self, &CausalContext),
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<Self, S::Error>
    where
        Self: Sized,
        S: KeySentinel,
    {
        // NOTE! When making changes to this method, consider if corresponding
        // changes need to be done to ::dry_join as well!

        Ok(Self(DotMap::join(
            (m1.0, cc1),
            (m2.0, cc2),
            on_dot_change,
            sentinel,
        )?))
    }
    fn dry_join(
        (m1, cc1): (&Self, &CausalContext),
        (m2, cc2): (&Self, &CausalContext),
        sentinel: &mut S,
    ) -> Result<DryJoinOutput, S::Error>
    where
        Self: Sized,
        S: KeySentinel,
    {
        DotMap::dry_join((&m1.0, cc1), (&m2.0, cc2), sentinel)
    }
}

impl<'doc, K, C> ToValue for &'doc OrMap<K, C>
where
    K: Hash + Eq + fmt::Display,
    C: ExtensionType,
{
    type Values = snapshot::OrMap<'doc, K, AllValues<'doc, C::ValueRef<'doc>>>;
    type Value = snapshot::OrMap<'doc, K, CollapsedValue<'doc, C::ValueRef<'doc>>>;
    type LeafValue = Either<MvRegValue, <C::ValueRef<'doc> as ToValue>::LeafValue>;

    fn values(self) -> Self::Values {
        let mut ret_map = snapshot::OrMap::default();
        for (key, inner_map) in self.0.iter() {
            let v = match inner_map.coerce_to_value_ref() {
                ValueRef::Map(m) => AllValues::Map(m.values()),
                ValueRef::Array(a) => AllValues::Array(a.values()),
                ValueRef::Register(r) => AllValues::Register(r.values()),
                ValueRef::Custom(c) => AllValues::Custom(c.values()),
            };
            ret_map.map.insert(key.borrow(), v);
        }
        ret_map
    }

    fn value(self) -> Result<Self::Value, Box<SingleValueError<Self::LeafValue>>> {
        let mut ret_map = snapshot::OrMap::default();
        for (key, inner_map) in self.0.iter() {
            let v = match inner_map.coerce_to_value_ref() {
                ValueRef::Map(m) => m.value().map(CollapsedValue::Map).map(Some),
                ValueRef::Array(a) => a.value().map(CollapsedValue::Array).map(Some),
                ValueRef::Register(r) => {
                    match r.value() {
                        Ok(v) => Ok(Some(CollapsedValue::Register(v))),

                        // don't include empty values in the map
                        //
                        // NOTE: this means that clearing an `MvReg` that's held in a map
                        // effectively removes the element, but does *not* actually remove it from the
                        // map (ie, its `InnerMap` is still there). is that a problem?
                        Err(e) if e.issue == SingleValueIssue::Cleared => Ok(None),

                        Err(mut e) => {
                            // make errors more helpful by including the path to the MvReg with conflicts
                            e.path.push(key.to_string());
                            Err(e.map_values(Either::Left))
                        }
                    }
                }
                ValueRef::Custom(c) => c
                    .value()
                    .map(CollapsedValue::Custom)
                    .map(Some)
                    .map_err(|v| v.map_values(Either::Right)),
            }?;

            if let Some(v) = v {
                ret_map.map.insert(key.borrow(), v);
            }
        }
        Ok(ret_map)
    }
}

impl<K, C> OrMap<K, C>
where
    K: Hash + Eq,
{
    /// Returns a reference to the element at the given key, if any.
    pub fn get<Q>(&self, key: &Q) -> Option<&TypeVariantValue<C>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.get(key)
    }

    /// Returns a mutable reference to the element at the given key, if any.
    ///
    /// Invalidates the dots cache for the given map entry, so calling `.dots()` on this collection
    /// after invoking this method may be slower as it has to call `.dots()` on this entry to
    /// re-compute.
    pub fn get_mut_and_invalidate<Q>(&mut self, key: &Q) -> Option<&mut TypeVariantValue<C>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.get_mut_and_invalidate(key)
    }

    /// Returns the number of elements in this map.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns true if this map has no elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    // Insert an element into the map.
    //
    // Note, this is a low level operation. CRDT types should generally
    // not be manipulated directly by user code. For one thing, you'd normally
    // want to also modify a CausalContext every time an OrMap is modified.
    #[doc(hidden)]
    pub fn insert(&mut self, key: K, value: TypeVariantValue<C>) {
        self.0.insert(key, value);
    }

    /// Iterates over key-value pairs in this CRDT, mutably, in arbitrary order.
    ///
    /// Invalidates the dots cache for all the map's entries, so calling `.dots()` on this
    /// collection after invoking this method may be quite slow (it has to call `.dots()` on all
    /// the entries).
    pub fn iter_mut_and_invalidate(
        &mut self,
    ) -> impl ExactSizeIterator<Item = (&K, &mut TypeVariantValue<C>)> {
        self.0.iter_mut_and_invalidate()
    }

    /// Retain only the entries for which a predicate is true.
    ///
    /// Invalidates the dots cache for all the map's entries, so calling `.dots()` on this
    /// collection after invoking this method may be quite slow (it has to call `.dots()` on all
    /// the entries).
    pub fn retain_and_invalidate(&mut self, f: impl FnMut(&K, &mut TypeVariantValue<C>) -> bool) {
        self.0.retain_and_invalidate(f)
    }

    pub fn inner(&self) -> &DotMap<K, TypeVariantValue<C>> {
        &self.0
    }
}

macro_rules! apply_to_X {
    ($name:ident, $frag:literal, $field:ident, [$($others:ident),*], $innerType:ty) => {
        /// Updates the value at key `k` to be
        #[doc = $frag]
        /// using `o`.
        ///
        /// This is mostly a convenience wrapper around [`OrMap::apply`].
        /// See that method for more details.
        ///
        /// # Multiple Operations
        ///
        /// Multiple operations within the closure `o` require manual context management.
        /// Each operation needs a context containing dots from previous operations.
        /// Call this method multiple times to avoid manual context handling.
        pub fn $name<'data, O>(&'data self, o: O, k: K, cc: &'_ CausalContext, id: Identifier) -> CausalDotStore<Self>
        where
            O: for<'cc, 'v> FnOnce(
                &'v $innerType,
                &'cc CausalContext,
                Identifier,
            ) -> CausalDotStore<$innerType>,
        {
            let CausalDotStore {
                store: ret_map,
                context: mut ret_cc,
            } = self.apply(
                move |m, cc, id| {
                    // NOTE: the original code calls ORMap.apply again here because everything
                    // is just weakly-typed stringly maps. we use structured types, so can't easily
                    // call ORMap.apply recursively here. that mostly shouldn't be a problem,
                    // though there is one difference that I don't _think_ matters: ORMap.apply
                    // _always_ injects an ALIVE key into the map it generates, which means that it
                    // injects an ALIVE field into the equivalent of InnerMap as well! That extra
                    // ALIVE is, as far as I can tell, not used or relevant, but I wanted to call
                    // it out nonetheless.
                    o(&m.$field, cc, id).map_store(Value::from)
                },
                k.clone(),
                cc,
                id
            );
            // recommitted value of type $field, delete the other two ($others).
            if let Some(inner) = self.0.get(&k) {
                $( inner.$others.add_dots_to(&mut ret_cc); )*
            }
            CausalDotStore {
                store: ret_map,
                context: ret_cc,
            }
        }
    };
}

impl<K, C> OrMap<K, C>
where
    K: Hash + Eq + fmt::Debug + Clone,
    C: ExtensionType,
{
    /// Creates a CRDT for the creation of a new empty [`OrMap`].
    pub fn create(&self, _cc: &CausalContext, _id: Identifier) -> CausalDotStore<Self> {
        // NOTE: the original OrMap implementation also sets an `.alive` field here.
        // see the YADR in `mod crdts` for why we don't do that.
        CausalDotStore {
            store: Self(Default::default()),
            context: CausalContext::default(),
        }
    }

    apply_to_X!(
        apply_to_map,
        "an [`OrMap`]",
        map,
        [array, reg, custom],
        OrMap<String, C>
    );
    apply_to_X!(
        apply_to_array,
        "an [`OrArray`]",
        array,
        [map, reg, custom],
        OrArray<C>
    );
    apply_to_X!(
        apply_to_register,
        "an [`MvReg`]",
        reg,
        [map, array, custom],
        MvReg
    );

    /// Updates the value at key `k` to be a custom type using `o`.
    ///
    /// This is mostly a convenience wrapper around [`OrMap::apply`].
    /// See that method for more details.
    ///
    /// # Multiple Operations
    ///
    /// Multiple operations within the closure `o` require manual context management.
    /// Each operation needs a context containing dots from previous operations.
    /// Call this method multiple times to avoid manual context handling.
    // NOTE(ow): Can't use the `apply_to_X` macro above, as `O` goes from
    // `C` to `C::Value`.
    pub fn apply_to_custom<'data, O>(
        &'data self,
        o: O,
        k: K,
        cc: &'_ CausalContext,
        id: Identifier,
    ) -> CausalDotStore<Self>
    where
        O: for<'cc, 'v> FnOnce(&'v C, &'cc CausalContext, Identifier) -> CausalDotStore<C::Value>,
    {
        let CausalDotStore {
            store: ret_map,
            context: mut ret_cc,
        } = self.apply(
            move |m, cc, id| {
                let y = o(&m.custom, cc, id);
                y.map_store(Value::Custom)
            },
            k.clone(),
            cc,
            id,
        );
        if let Some(inner) = self.0.get(&k) {
            inner.map.add_dots_to(&mut ret_cc);
            inner.array.add_dots_to(&mut ret_cc);
            inner.reg.add_dots_to(&mut ret_cc);
        }
        CausalDotStore {
            store: ret_map,
            context: ret_cc,
        }
    }

    /// Creates a CRDT that represents `O` applied to the [`Value`] of the element with key `key`,
    /// if any, and written back to that same key in the map.
    ///
    /// `O` will be passed `None` if there is currently no value with key `key`, such as when apply
    /// is used on an empty map or on an [`OrMap`] CRDT that doesn't _change_ the value at `key`.
    ///
    /// # Multiple Operations
    ///
    /// Multiple operations within the closure require manual context management. Each operation
    /// needs a context containing dots from previous operations. Call `apply` multiple times or
    /// use the transaction API to avoid manual context handling.
    pub fn apply<'data, O>(
        &'data self,
        o: O,
        key: K,
        cc: &'_ CausalContext,
        id: Identifier,
    ) -> CausalDotStore<Self>
    where
        O: for<'cc, 'v> FnOnce(
            &'v TypeVariantValue<C>,
            &'cc CausalContext,
            Identifier,
        ) -> CausalDotStore<Value<C>>,
    {
        let mut ret_dot_map = Self::default();
        let v = if let Some(v) = self.get(&key) {
            v
        } else {
            &TypeVariantValue::default()
        };

        // NOTE: the original OrArray implementation also updates an `.alive` field here.
        // see the YADR in `mod crdts` for why we don't do that.

        // ask `O` to generate the new value for this key,
        // remembering to incorporate the `alive` change into what o receives
        let CausalDotStore {
            store: new_v,
            context: ret_cc,
        } = o(v, cc, id);
        ret_dot_map.0.set(key, new_v.into());

        CausalDotStore {
            store: ret_dot_map,
            context: ret_cc,
        }
    }

    /// Creates a CRDT that represents the removal of the element with key `k`.
    ///
    /// A removed element, if there was one, is represented by a store only including the bottom
    /// element ⊥ and the embedded dots of the removed value.
    pub fn remove<Q>(&self, k: &Q, _cc: &CausalContext, _id: Identifier) -> CausalDotStore<Self>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let Some(inner_map) = self.0.get(k) else {
            // If there's no inner map, there's nothing to change,
            // and an empty dot store is sufficient.
            return CausalDotStore::new();
        };

        // NOTE: the original implementation does not write alive here, but that means
        // deleting from an array can make it bottom, which doesn't align with the behavior seen
        // when clearing. one of the authors of the original DSON paper confirmed by email on
        // 2023-08-25 that the right thing to do here is likely to write alive in remove as well.
        // _but_ since we don't use `.alive` in this implementation (see YADR in `mod crdts`), we
        // do nothing.

        // mark all dots as seen so that joining with this CRDT will erase all other entries.
        let ret_cc = inner_map.dots();

        CausalDotStore {
            store: Self(Default::default()),
            context: ret_cc,
        }
    }

    /// Creates a CRDT that represents the erasure of all elements values of this array.
    ///
    /// A cleared map is represented by a store only including the alive field. This means that it
    /// is not equal to the bottom element ⊥, and thus signals an empty map. It also includes all
    /// embedded dots of the map to make it clear that it has seen those writes and did not want
    /// them to continue existing.
    pub fn clear(&self, _cc: &CausalContext, _id: Identifier) -> CausalDotStore<Self> {
        // NOTE: the original OrArray implementation also updates an `.alive` field here.
        // see the YADR in `mod crdts` for why we don't do that.

        // mark all dots as seen so that joining with this CRDT will erase all other entries.
        let ret_cc = self.dots();

        CausalDotStore {
            store: Self(Default::default()),
            context: ret_cc,
        }
    }

    /// Removes the entry for key `k` from this CRDT directly.
    ///
    /// This change is not represented as a delta-CRDT, meaning this may cause unintended
    /// consequences if `self` is later distributed along with an unmodified [`CausalContext`]. You
    /// almost certainly don't want to use this method and want [`OrMap::remove`] instead.
    pub fn remove_immediately<Q>(&mut self, k: &Q) -> Option<TypeVariantValue<C>>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.0.remove(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        crdts::{
            NoExtensionTypes,
            test_util::{Ops, join_harness},
        },
        sentinel::{DummySentinel, test::ValueCountingValidator},
    };
    use std::collections::BTreeMap;

    type OrMap<K> = super::OrMap<K, NoExtensionTypes>;

    #[test]
    fn empty() {
        let cds = CausalDotStore::<OrMap<String>>::default();
        assert!(cds.store.is_bottom());
        assert!(cds.store.value().unwrap().is_empty());
        assert_eq!(cds.store.values().len(), 0);
    }

    #[test]
    fn created_is_bottom() {
        let map = OrMap::<String>::default();
        let cc = CausalContext::new();
        let id = Identifier::new(0, 0);

        let m = map.create(&cc, id);
        assert!(m.store.is_bottom());
        assert_eq!(map, m.store);
    }

    #[test]
    fn cleared_is_bottom() {
        let map = OrMap::<String>::default();
        let cc = CausalContext::new();
        let id = Identifier::new(0, 0);

        let m = map.create(&cc, id);
        let m = m.store.clear(&m.context, id);
        assert!(m.store.is_bottom());
    }

    #[test]
    fn set_get_remove() {
        let map = OrMap::<String>::default();
        let cc = CausalContext::new();
        let id = Identifier::new(0, 0);

        let m = map.apply_to_register(
            |_old, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
            "foo".into(),
            &cc,
            id,
        );
        assert!(!m.store.is_bottom());
        assert_eq!(
            m.store.value().unwrap().get(&String::from("foo")).cloned(),
            Some(CollapsedValue::Register(&MvRegValue::Bool(true)))
        );
        assert_eq!(m.store.len(), 1);
        // count the number of dots we generated under `id`
        assert_eq!(
            m.context.next_dot_for(id).sequence().get() - 1,
            1 /* mvreg.write */
        );

        let m = m.store.remove("foo", &cc, id);
        assert!(m.store.is_bottom()); // empty maps become bottom
        assert_eq!(m.store.value().unwrap().get(&String::from("foo")), None);
        assert_eq!(m.store.len(), 0);

        assert_eq!(m.context.next_dot_for(id).sequence().get() - 1, 1);
    }

    #[test]
    fn set_one_key_then_another() {
        let map = CausalDotStore::<OrMap<String>>::new();
        let id = Identifier::new(0, 0);

        let delta = map.store.apply_to_register(
            |_old, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
            "true".into(),
            &map.context,
            id,
        );
        assert!(!delta.store.is_bottom());
        assert_eq!(
            delta
                .store
                .value()
                .unwrap()
                .get(&String::from("true"))
                .cloned(),
            Some(CollapsedValue::Register(&MvRegValue::Bool(true)))
        );
        assert_eq!(delta.store.len(), 1);

        let map = map.join(delta, &mut DummySentinel).unwrap();

        let delta = map.store.apply_to_register(
            |_old, cc, id| MvReg::default().write(MvRegValue::Bool(false), cc, id),
            "false".into(),
            &map.context,
            id,
        );
        assert!(!delta.store.is_bottom());
        assert_eq!(
            delta
                .store
                .value()
                .unwrap()
                .get(&String::from("false"))
                .cloned(),
            Some(CollapsedValue::Register(&MvRegValue::Bool(false)))
        );
        assert_eq!(delta.store.len(), 1);

        let map = map.join(delta, &mut DummySentinel).unwrap();
        assert!(!map.store.is_bottom());
        assert_eq!(
            map.store
                .value()
                .unwrap()
                .get(&String::from("true"))
                .cloned(),
            Some(CollapsedValue::Register(&MvRegValue::Bool(true)))
        );
        assert_eq!(
            map.store
                .value()
                .unwrap()
                .get(&String::from("false"))
                .cloned(),
            Some(CollapsedValue::Register(&MvRegValue::Bool(false)))
        );
        assert_eq!(map.store.len(), 2);
    }

    #[test]
    fn independent_keys() {
        join_harness(
            OrMap::<String>::default(),
            |cds, _| cds,
            |m, cc, id| {
                m.apply_to_register(
                    |_old, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    "foo".into(),
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                m.apply_to_register(
                    |_old, cc, id| MvReg::default().write(MvRegValue::U64(42), cc, id),
                    "bar".into(),
                    &cc,
                    id,
                )
            },
            DummySentinel,
            |CausalDotStore { store: m, .. }, _| {
                assert!(!m.is_bottom());
                assert_eq!(
                    m.value().unwrap().get(&String::from("foo")).cloned(),
                    Some(CollapsedValue::Register(&MvRegValue::Bool(true)))
                );
                assert_eq!(
                    m.value().unwrap().get(&String::from("bar")).cloned(),
                    Some(CollapsedValue::Register(&MvRegValue::U64(42)))
                );
            },
        );
    }

    #[test]
    fn conflicting_reg_value() {
        join_harness(
            OrMap::<String>::default(),
            |cds, _| cds,
            |m, cc, id| {
                m.apply_to_register(
                    |_old, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    "foo".into(),
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                m.apply_to_register(
                    |_old, cc, id| MvReg::default().write(MvRegValue::U64(42), cc, id),
                    "foo".into(),
                    &cc,
                    id,
                )
            },
            ValueCountingValidator::default(),
            |CausalDotStore { store: m, .. }, sentinel| {
                assert!(!m.is_bottom());
                let values = m.values();
                let AllValues::Register(v) = values.get(&String::from("foo")).unwrap() else {
                    panic!("foo isn't a register even though we only wrote registers");
                };
                assert_eq!(v.len(), 2);
                assert!(v.contains(&MvRegValue::Bool(true)));
                assert!(v.contains(&MvRegValue::U64(42)));
                // we end up with two values, but only added 1 in the join
                assert_eq!(sentinel.added, BTreeMap::from([(MvRegValue::U64(42), 1)]));
                assert!(sentinel.removed.is_empty());
            },
        );
    }

    #[test]
    fn concurrent_clear() {
        join_harness(
            OrMap::<String>::default(),
            |CausalDotStore {
                 store: m,
                 context: cc,
             },
             id| {
                m.apply_to_register(
                    |_old, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    "foo".into(),
                    &cc,
                    id,
                )
            },
            |m, cc, id| m.clear(&cc, id),
            |m, cc, id| m.clear(&cc, id),
            DummySentinel,
            |CausalDotStore { store: m, .. }, _| {
                // empty maps become bottom
                assert!(m.is_bottom());
                let values = m.values();
                assert_eq!(values.len(), 0);
            },
        );
    }

    #[test]
    fn remove_reg_value() {
        join_harness(
            OrMap::<String>::default(),
            |CausalDotStore {
                 store: m,
                 context: cc,
             },
             id| {
                m.apply_to_register(
                    |_old, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    "foo".into(),
                    &cc,
                    id,
                )
            },
            |m, cc, id| m.clear(&cc, id),
            |m, cc, _| CausalDotStore {
                store: m.clone(),
                context: cc,
            },
            ValueCountingValidator::new(true),
            |CausalDotStore { store: m, .. }, sentinel| {
                // empty maps become bottom
                assert!(m.is_bottom());
                let values = m.values();
                assert_eq!(values.get(&String::from("foo")), None);
                assert!(sentinel.added.is_empty());
                // conventionally the left side is the base state and the right side is the delta,
                // so this join semantically just discards the delta since the start and end states
                // are empty
                assert!(sentinel.removed.is_empty());
            },
        );

        join_harness(
            OrMap::<String>::default(),
            |CausalDotStore {
                 store: m,
                 context: cc,
             },
             id| {
                m.apply_to_register(
                    |_old, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    "foo".into(),
                    &cc,
                    id,
                )
            },
            |m, cc, _| CausalDotStore {
                store: m.clone(),
                context: cc,
            },
            |m, cc, id| m.clear(&cc, id),
            ValueCountingValidator::new(true),
            |CausalDotStore { store: m, .. }, sentinel| {
                // empty maps become bottom
                assert!(m.is_bottom());
                let values = m.values();
                assert_eq!(values.get(&String::from("foo")), None);
                assert!(sentinel.added.is_empty());
                // now we start with a non-empty value, so a change is observed
                assert_eq!(
                    sentinel.removed,
                    BTreeMap::from([(MvRegValue::Bool(true), 1)])
                );
            },
        );
    }

    #[test]
    fn update_vs_remove() {
        join_harness(
            OrMap::<String>::default(),
            |CausalDotStore {
                 store: m,
                 context: cc,
             },
             id| {
                // start out with a map with the "foo" key set
                m.apply_to_register(
                    |_old, cc, id| MvReg::default().write(MvRegValue::U64(42), cc, id),
                    "foo".into(),
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                // one writer updates foo
                m.apply_to_register(
                    |_old, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    "foo".into(),
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                // the other writer removes foo
                m.remove("foo", &cc, id)
            },
            ValueCountingValidator::default(),
            |CausalDotStore { store: m, .. }, sentinel| {
                // _not_ bottom since the map isn't empty
                assert!(!m.is_bottom());
                // the semantics of observed-remove (remember "*OR*map")
                // is that updates concurrent with removes leave the updates intact
                let values = m.values();
                let AllValues::Register(v) = values.get(&String::from("foo")).unwrap() else {
                    panic!("foo isn't a register even though we only wrote registers");
                };
                assert_eq!(v, [MvRegValue::Bool(true)]);
                assert!(sentinel.added.is_empty());
                assert!(sentinel.removed.is_empty());
            },
        );
    }

    #[test]
    fn nested_update_vs_remove() {
        join_harness(
            OrMap::<String>::default(),
            |CausalDotStore {
                 store: m,
                 context: cc,
             },
             id| {
                // start out with a map like {foo: {bar: 42}}
                m.apply_to_map(
                    |_old, cc, id| {
                        OrMap::default().apply_to_register(
                            |_old, cc, id| MvReg::default().write(MvRegValue::U64(42), cc, id),
                            "bar".into(),
                            cc,
                            id,
                        )
                    },
                    "foo".into(),
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                // one writer adds a field (baz) to the inner map
                m.apply_to_map(
                    |old, cc, id| {
                        old.apply_to_register(
                            |_old, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                            "baz".into(),
                            cc,
                            id,
                        )
                    },
                    "foo".into(),
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                // the other writer removes foo
                m.remove("foo", &cc, id)
            },
            ValueCountingValidator::default(),
            |CausalDotStore { store: m, .. }, sentinel| {
                // _not_ bottom since the map isn't empty
                assert!(!m.is_bottom());
                // the semantics of observed-remove (remember "*OR*map")
                // is that updates concurrent with removes leave the updates intact,
                // so we'll end up with {foo: {baz: true}}
                // as counter-intuitive as that may seem
                let values = m.values();
                let AllValues::Map(m) = values.get(&String::from("foo")).unwrap() else {
                    panic!("foo isn't a map even though we only wrote map");
                };
                assert_eq!(values.len(), 1);
                let AllValues::Register(r) = m
                    .get(&String::from("baz"))
                    .expect("baz key isn't preserved")
                else {
                    panic!("baz isn't a register though we only wrote a register ")
                };
                assert_eq!(m.len(), 1);
                assert_eq!(r, [MvRegValue::Bool(true)]);
                assert!(sentinel.added.is_empty());
                assert!(sentinel.removed.is_empty());
            },
        );
    }

    #[quickcheck]
    fn order_invariant(ops: Ops<OrMap<String>>, seed: u64) -> quickcheck::TestResult {
        ops.check_order_invariance(seed)
    }
}
