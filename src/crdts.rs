// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! # Composable CRDTs for JSON-like Data
//!
//! This module provides a set of composable, conflict-free replicated data types (CRDTs)
//! that can be used to build complex, JSON-like data structures. These CRDTs are the
//! building blocks of the DSON library and are designed to be nested
//! to create arbitrarily complex documents.
//!
//! ## Core CRDTs
//!
//! The fundamental CRDTs provided are:
//!
//! - **[`OrMap`]**: An **Observed-Remove Map**, which maps arbitrary keys to other CRDT
//!   values. It allows for the creation of nested objects.
//!
//! - **[`OrArray`]**: An **Observed-Remove Array**, which provides a list-like structure
//!   that can hold other CRDTs.
//!
//! - **[`MvReg`]**: A **Multi-Value Register**, used for storing primitive values. When
//!   concurrent writes occur, the register holds all conflicting values, allowing the
//!   application to resolve them.
//!
//! ## Type-Safe Composition and Extensibility
//!
//! The CRDTs in this module are designed to be composable. The
//! [`TypeVariantValue`] can hold any of the core CRDTs, as well as custom types
//! defined through the [`ExtensionType`] trait.
//!
//! ### Type Conflicts
//!
//! [`TypeVariantValue`]s can also represent **type conflicts**.
//! If one replica updates a field to be a map while another concurrently updates it to
//! be an array, the [`TypeVariantValue`] will hold both the map and the array.
//! This preserves all concurrent updates.
use self::{mvreg::MvRegValue, orarray::Uid, snapshot::ToValue};
use crate::{
    CausalContext, DotStoreJoin, ExtensionType, MvReg, OrArray, OrMap,
    dotstores::{DotChange, DotStore, DryJoinOutput},
    sentinel::{KeySentinel, Sentinel, TypeSentinel, ValueSentinel, Visit},
};
use std::{fmt, hash::Hash};

/*
 * YADR: 2024-05-06 Removal of `.alive` tracking
 *
 * In the context of DSON's ability to represent empty collections (maps and arrays), we faced
 * a question of whether to accept the significant number of extra updates supporting such
 * collections cause to _all_ map and array operations due to the requisite write of the `.alive`
 * field.
 *
 * We decided for removing the ability to represent empty collections by removing all use of the
 * `.alive` field, and neglected keeping the DSON implementation in line with the original paper
 * (which does support such collections).
 *
 * We did this to achieve (significantly) smaller deltas when updating fields in deeply nested
 * documents, specifically by avoiding the extra `DotFun`s that need to be transmitted for the
 * `.alive` of each nesting level, accepting the inability to distinguish between an unset
 * array/map and a set-but-empty one.
 *
 * We think this is the right trade-off because in practice we suspect most users of this crate
 * will use some kind of schema mechanism (even if it's just `serde_json`) that will be able to
 * interpret missing collection types as empty ones.
 *
 * For some added context, the original OrArray and OrMap implementations carry an `.alive` field
 * to distinguish an empty array from an undefined array. This field must be updated on every
 * insert/apply to maintain observe-removed semantics; if `.alive` wasn't written to, then a
 * concurrent unsetting of the array would lead to `.alive` being unset, which would in turn lead
 * to the array being in an inconsistent state (has elements but not `.alive`). Unfortunately, that
 * extra update for each OrArray or OrMap operation requires sending extra dots and bools with
 * _every_ update, one for each level of nesting, which adds significant overhead to even trivial
 * updates if they are deeply nested in a document.
 *
 * There was a brief discussion about the need to update this field with the original DSON paper
 * authors over email on 2023-08-25; they wrote:
 *
 * > This is intentional so the alive fields exists passed a concurrent delete. The field
 * > is a MVREG, so if an inner insert is concurrent with a delete and we don't update the
 * > alive field, then it gets deleted. If we then delete the value they we may delete the
 * > array rather than leaving it as empty. There may be an optimization that allows for
 * > forgoing this.
 *
 * We have not found such an optimization, and so have decided to opt for the weakened semantics
 * instead.
 *
 * It's worth noting that, prior to its removal, this crate's code had some bug fixes for the
 * handling of `.alive` compared to the research code published by the original authors. These are
 * still represented by NOTE comments at the time of writing, but interested parties may want to
 * inspect the commit prior to this YADR's introduction to also see the relevant code.
 */

pub mod mvreg;
pub mod orarray;
pub mod ormap;
pub mod snapshot;

// TODO: should we also provide more handy register types like counters?

#[cfg(any(test, feature = "arbitrary"))]
mod test_util;

/// Indicator that only the basic DSON types should be supported.
///
/// For use as the type value of the `<Custom>` type parameter to many of DSON's types.
#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct NoExtensionTypes;

/// Always-uninhabited instance of [`NoExtensionTypes`].
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
// TODO: potentially replace with ! when https://github.com/rust-lang/rust/issues/35121 lands.
pub enum NoExtensionTypesType {}

#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "
        Custom: ::serde::Serialize,
        for<'dea> Custom: ::serde::Deserialize<'dea>,
        <Custom as ExtensionType>::Value: ::serde::Serialize,
        for<'deb> <Custom as ExtensionType>::Value: ::serde::Deserialize<'deb>,
    ")
)]
pub enum Value<Custom>
where
    Custom: ExtensionType,
{
    Map(OrMap<String, Custom>),
    Array(OrArray<Custom>),
    Register(MvReg),
    Custom(<Custom as ExtensionType>::Value),
}

impl<C> fmt::Debug for Value<C>
where
    C: fmt::Debug + ExtensionType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Map(m) => f.debug_tuple("Value::Map").field(m).finish(),
            Value::Array(a) => f.debug_tuple("Value::Array").field(a).finish(),
            Value::Register(r) => f.debug_tuple("Value::Register").field(r).finish(),
            Value::Custom(c) => f.debug_tuple("Value::Custom").field(c).finish(),
        }
    }
}

impl<C> Value<C>
where
    C: ExtensionType,
{
    fn is_bottom(&self) -> bool {
        match self {
            Value::Map(m) => m.is_bottom(),
            Value::Array(a) => a.is_bottom(),
            Value::Register(r) => r.is_bottom(),
            Value::Custom(c) => c.is_bottom(),
        }
    }

    #[cfg(any(test, feature = "arbitrary"))]
    fn dots(&self) -> CausalContext {
        match self {
            Value::Map(m) => m.dots(),
            Value::Array(a) => a.dots(),
            Value::Register(r) => r.dots(),
            Value::Custom(c) => c.dots(),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Map(_) => "Map",
            Self::Array(_) => "Array",
            Self::Register(_) => "Register",
            Self::Custom(c) => C::type_name(&C::ValueRef::from(c)),
        }
    }
}

pub enum ValueRef<'a, Custom>
where
    Custom: ExtensionType,
    Custom::ValueRef<'a>: Copy,
{
    Map(&'a OrMap<String, Custom>),
    Array(&'a OrArray<Custom>),
    Register(&'a MvReg),
    Custom(Custom::ValueRef<'a>),
}

// NOTE: the Clone, Copy, PartialEq, and Debug impls must be manual (ie, they can't be
// derived) so we get the right bounds; ref https://github.com/rust-lang/rust/issues/26925.
impl<C> Clone for ValueRef<'_, C>
where
    C: ExtensionType,
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<C> Copy for ValueRef<'_, C> where C: ExtensionType {}

impl<C> PartialEq for ValueRef<'_, C>
where
    C: ExtensionType + PartialEq,
    for<'doc> C::ValueRef<'doc>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (ValueRef::Map(m1), ValueRef::Map(m2)) => m1.eq(m2),
            (ValueRef::Array(a1), ValueRef::Array(a2)) => a1.eq(a2),
            (ValueRef::Register(r1), ValueRef::Register(r2)) => r1.eq(r2),
            (ValueRef::Custom(c1), ValueRef::Custom(c2)) => c1.eq(&c2),
            _ => false,
        }
    }
}

macro_rules! impl_partial_eq {
    ({$($t:ty),+}) => {
        $(impl_partial_eq!($t);)+
    };

    ($t:ty) => {
        impl<C> PartialEq<$t> for ValueRef<'_, C>
        where
            C: ExtensionType,
        {
            fn eq(&self, other: &$t) -> bool {
                matches!(*self, ValueRef::Register(r1) if r1 == other)
            }
        }
    };
}
impl_partial_eq!({[u8], &[u8], str, &str, bool, f64, u64, i64});
// i32 because it's the "default" inference integer type
impl_partial_eq!(i32);
// byte literals
impl<C, const N: usize> PartialEq<&[u8; N]> for ValueRef<'_, C>
where
    C: ExtensionType,
{
    fn eq(&self, other: &&[u8; N]) -> bool {
        matches!(*self, ValueRef::Register(r1) if r1 == other)
    }
}

impl<C> fmt::Debug for ValueRef<'_, C>
where
    C: fmt::Debug + ExtensionType,
    for<'a> C::ValueRef<'a>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueRef::Map(m) => f.debug_tuple("ValueRef::Map").field(m).finish(),
            ValueRef::Array(a) => f.debug_tuple("ValueRef::Array").field(a).finish(),
            ValueRef::Register(r) => f.debug_tuple("ValueRef::Register").field(r).finish(),
            ValueRef::Custom(c) => f.debug_tuple("ValueRef::Custom").field(c).finish(),
        }
    }
}

impl<C> ValueRef<'_, C>
where
    C: ExtensionType,
{
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Map(_) => "Map",
            Self::Array(_) => "Array",
            Self::Register(_) => "Register",
            Self::Custom(c) => C::type_name(c),
        }
    }
}

impl<C> From<ValueRef<'_, C>> for Value<C>
where
    C: ExtensionType + Clone,
{
    fn from(val: ValueRef<'_, C>) -> Self {
        match val {
            ValueRef::Map(m) => Value::Map(m.clone()),
            ValueRef::Array(a) => Value::Array(a.clone()),
            ValueRef::Register(r) => Value::Register(r.clone()),
            ValueRef::Custom(c) => Value::Custom(c.into()),
        }
    }
}

impl<'a, C> From<&'a Value<C>> for ValueRef<'a, C>
where
    C: ExtensionType,
{
    fn from(val: &'a Value<C>) -> Self {
        match val {
            Value::Map(m) => ValueRef::Map(m),
            Value::Array(a) => ValueRef::Array(a),
            Value::Register(r) => ValueRef::Register(r),
            Value::Custom(c) => ValueRef::Custom(c.into()),
        }
    }
}

impl<'a, C> From<&'a OrMap<String, C>> for ValueRef<'a, C>
where
    C: ExtensionType,
{
    fn from(value: &'a OrMap<String, C>) -> Self {
        Self::Map(value)
    }
}

impl<'a, C> From<&'a OrArray<C>> for ValueRef<'a, C>
where
    C: ExtensionType,
{
    fn from(value: &'a OrArray<C>) -> Self {
        Self::Array(value)
    }
}

impl<'a, C> From<&'a MvReg> for ValueRef<'a, C>
where
    C: ExtensionType,
{
    fn from(value: &'a MvReg) -> Self {
        Self::Register(value)
    }
}

// we can't impl From<C> for ValueRef<C> since those are overlap hazards with the above impls, so
// we instead provide a convenience variant for use with customs in place of `ValueRef::from`.
impl<'a, C> ValueRef<'a, C>
where
    C: ExtensionType,
{
    pub fn custom(value: impl Into<C::ValueRef<'a>>) -> Self {
        Self::Custom(value.into())
    }
}

impl<C> From<OrMap<String, C>> for Value<C>
where
    C: ExtensionType,
{
    fn from(value: OrMap<String, C>) -> Self {
        Self::Map(value)
    }
}

impl<C> From<OrArray<C>> for Value<C>
where
    C: ExtensionType,
{
    fn from(value: OrArray<C>) -> Self {
        Self::Array(value)
    }
}

impl<C> From<MvReg> for Value<C>
where
    C: ExtensionType,
{
    fn from(value: MvReg) -> Self {
        Self::Register(value)
    }
}

// ditto as for ValueRef::custom
impl<C> Value<C>
where
    C: ExtensionType,
{
    pub fn custom(value: impl Into<C::Value>) -> Self {
        Self::Custom(value.into())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ValueType<C> {
    Map,
    Array,
    Register,
    Custom(C),
}

impl<C> From<ValueRef<'_, C>> for ValueType<C::ValueKind>
where
    C: ExtensionType,
{
    fn from(value: ValueRef<'_, C>) -> Self {
        match value {
            ValueRef::Map(_) => Self::Map,
            ValueRef::Array(_) => Self::Array,
            ValueRef::Register(_) => Self::Register,
            ValueRef::Custom(c) => Self::Custom(c.into()),
        }
    }
}

/// A container for a value that can be one of several types.
///
/// # Concurrent Mutations and Type Conflicts
///
/// It is possible for different actors to concurrently modify the same piece of
/// data. This can lead to situations where an actor updates a given value as a map,
/// and another one updates it concurrently as an array. DSON
/// is designed to represent these conflicts.
///
/// `TypeVariantValue` is a struct rather than an enum precisely to manage these type conflicts.
/// If it were an enum, one variant would have to be chosen over the other, potentially
/// losing the concurrent update. Instead, `TypeVariantValue` can hold multiple types
/// simultaneously. For instance, if one actor writes a map and another concurrently writes an
/// array to the same logical field, the resulting `TypeVariantValue` will contain both the map
/// and the array.
///
/// This approach preserves all concurrent writes -- both on the value as well as on the type
/// level -- allowing the application layer to decide how to resolve the conflict.
#[derive(Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct TypeVariantValue<Custom> {
    // NOTE: We decided for the OrMap to be not generic over the key type. If you require an
    // `OrMap` with another key type, consider using implementing an [`ExtensionType`].
    pub map: OrMap<String, Custom>,
    pub array: OrArray<Custom>,
    pub reg: MvReg,
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub custom: Custom,
}

impl<C> fmt::Debug for TypeVariantValue<C>
where
    C: ExtensionType + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (
            self.map.is_bottom(),
            self.array.is_bottom(),
            self.reg.is_bottom(),
            self.custom.is_bottom(),
        ) {
            (false, true, true, true) => self.map.fmt(f),
            (true, false, true, true) => self.array.fmt(f),
            (true, true, false, true) => self.reg.fmt(f),
            (true, true, true, false) => self.custom.fmt(f),
            _ => {
                let mut w = f.debug_struct("TypeVariantValue");
                if !self.map.is_bottom() {
                    w.field("map", &self.map);
                }
                if !self.array.is_bottom() {
                    w.field("array", &self.array);
                }
                if !self.reg.is_bottom() {
                    w.field("reg", &self.reg);
                }
                if !self.custom.is_bottom() {
                    w.field("custom", &self.custom);
                }
                w.finish_non_exhaustive()
            }
        }
    }
}

impl<C> TypeVariantValue<C> {
    pub fn coerce_to_value_ref(&self) -> ValueRef<'_, C>
    where
        C: ExtensionType,
    {
        if !self.custom.is_bottom() {
            ValueRef::Custom(self.custom.coerce_to_value_ref())
        } else if !self.map.is_bottom() {
            ValueRef::Map(&self.map)
        } else if !self.array.is_bottom() {
            ValueRef::Array(&self.array)
        } else if !self.reg.is_bottom() {
            ValueRef::Register(&self.reg)
        } else {
            // TODO: how is this possible? empty InnerMaps should not be left in the
            // map. it can perhaps happen if someone tries to read out of a CRDT that
            // represents a removal CRDT.
            panic!("attempt to coerce empty TypeVariantValue to ValueRef");
        }
    }
}

impl<C> DotStore for TypeVariantValue<C>
where
    C: ExtensionType,
{
    fn add_dots_to(&self, other: &mut CausalContext) {
        self.map.add_dots_to(other);
        self.array.add_dots_to(other);
        self.reg.add_dots_to(other);
        self.custom.add_dots_to(other);
    }

    fn is_bottom(&self) -> bool {
        self.map.is_bottom()
            && self.array.is_bottom()
            && self.reg.is_bottom()
            && self.custom.is_bottom()
    }

    fn subset_for_inflation_from(&self, frontier: &CausalContext) -> Self {
        Self {
            map: self.map.subset_for_inflation_from(frontier),
            array: self.array.subset_for_inflation_from(frontier),
            reg: self.reg.subset_for_inflation_from(frontier),
            custom: self.custom.subset_for_inflation_from(frontier),
        }
    }
}

impl<C, S> DotStoreJoin<S> for TypeVariantValue<C>
where
    S: Visit<String>
        + Visit<Uid>
        + KeySentinel
        + ValueSentinel<MvRegValue>
        + TypeSentinel<C::ValueKind>,
    C: ExtensionType + DotStoreJoin<S> + Clone + PartialEq + fmt::Debug,
{
    fn join(
        ds1: (Self, &CausalContext),
        ds2: (Self, &CausalContext),
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<Self, S::Error>
    where
        Self: Sized,
        S: Sentinel,
    {
        // NOTE! When making changes to this method, consider if corresponding
        // changes need to be done to ::dry_join as well!

        let (m1, cc1) = ds1;
        let (m2, cc2) = ds2;

        let types_before = [
            !m1.map.is_bottom(),
            !m1.array.is_bottom(),
            !m1.reg.is_bottom(),
        ];

        let map = OrMap::join((m1.map, cc1), (m2.map, cc2), on_dot_change, sentinel)?;
        let array = OrArray::join((m1.array, cc1), (m2.array, cc2), on_dot_change, sentinel)?;
        let reg = MvReg::join((m1.reg, cc1), (m2.reg, cc2), on_dot_change, sentinel)?;
        let custom = C::join((m1.custom, cc1), (m2.custom, cc2), on_dot_change, sentinel)?;

        let types_after = [!map.is_bottom(), !array.is_bottom(), !reg.is_bottom()];

        // Normally we either go from no type to 1 type (join bottom with non-bottom) or 1 type to 1
        // type. But in case of conflicts we may end up with multiple types being set or unset.
        // NOTE: The following loop does not call set_type when transiting to Custom(C) and does not
        // call unset_type when transiting away from Custom(C). Those calls are emitted from the
        // C::join further up.
        for (ty, (before, after)) in [ValueType::Map, ValueType::Array, ValueType::Register]
            .into_iter()
            .zip(types_before.into_iter().zip(types_after))
        {
            match (before, after) {
                (true, false) => sentinel.unset_type(ty)?,
                (false, true) => sentinel.set_type(ty)?,
                _ => (),
            }
        }

        Ok(TypeVariantValue {
            map,
            array,
            reg,
            custom,
        })
    }

    fn dry_join(
        ds1: (&Self, &CausalContext),
        ds2: (&Self, &CausalContext),
        sentinel: &mut S,
    ) -> Result<DryJoinOutput, S::Error>
    where
        Self: Sized,
        S: Sentinel,
    {
        let (m1, cc1) = ds1;
        let (m2, cc2) = ds2;

        let types_before = [
            !m1.map.is_bottom(),
            !m1.array.is_bottom(),
            !m1.reg.is_bottom(),
        ];

        let map = OrMap::dry_join((&m1.map, cc1), (&m2.map, cc2), sentinel)?;
        let array = OrArray::dry_join((&m1.array, cc1), (&m2.array, cc2), sentinel)?;
        let reg = MvReg::dry_join((&m1.reg, cc1), (&m2.reg, cc2), sentinel)?;
        let custom = C::dry_join((&m1.custom, cc1), (&m2.custom, cc2), sentinel)?;

        let types_after = [!map.is_bottom(), !array.is_bottom(), !reg.is_bottom()];
        // Normally we either go from no type to 1 type (join bottom with non-bottom) or 1 type to 1
        // type. But in case of conflicts we may end up with multiple types being set or unset.
        // NOTE: The following loop does not call set_type when transiting to Custom(C) and does not
        // call unset_type when transiting away from Custom(C). Those calls are emitted from the
        // C::join further up.
        for (ty, (before, after)) in [ValueType::Map, ValueType::Array, ValueType::Register]
            .into_iter()
            .zip(types_before.into_iter().zip(types_after))
        {
            match (before, after) {
                (true, false) => sentinel.unset_type(ty)?,
                (false, true) => sentinel.set_type(ty)?,
                _ => (),
            }
        }
        let result_is_non_bottom = types_after.iter().any(|x| *x) || !custom.is_bottom();
        Ok(DryJoinOutput::new(!result_is_non_bottom))
    }
}

impl<C> From<Value<C>> for TypeVariantValue<C>
where
    C: ExtensionType,
{
    fn from(value: Value<C>) -> Self {
        match value {
            Value::Map(m) => TypeVariantValue {
                map: m,
                array: Default::default(),
                reg: Default::default(),
                custom: Default::default(),
            },
            Value::Array(a) => TypeVariantValue {
                map: Default::default(),
                array: a,
                reg: Default::default(),
                custom: Default::default(),
            },
            Value::Register(r) => TypeVariantValue {
                map: Default::default(),
                array: Default::default(),
                reg: r,
                custom: Default::default(),
            },
            Value::Custom(c) => TypeVariantValue {
                map: Default::default(),
                array: Default::default(),
                reg: Default::default(),
                custom: c.into(),
            },
        }
    }
}

impl DotStore for NoExtensionTypes {
    fn add_dots_to(&self, _: &mut CausalContext) {}
    fn is_bottom(&self) -> bool {
        true
    }
    fn subset_for_inflation_from(&self, _: &CausalContext) -> Self {
        Self
    }
}

impl DotStore for NoExtensionTypesType {
    fn add_dots_to(&self, _: &mut CausalContext) {
        match *self {}
    }
    fn is_bottom(&self) -> bool {
        match *self {}
    }
    fn subset_for_inflation_from(&self, _: &CausalContext) -> Self {
        match *self {}
    }
}

impl DotStore for () {
    fn add_dots_to(&self, _: &mut CausalContext) {}
    fn is_bottom(&self) -> bool {
        true
    }
    fn subset_for_inflation_from(&self, _: &CausalContext) -> Self {}
}

impl<S> DotStoreJoin<S> for NoExtensionTypes {
    fn join(
        _: (Self, &CausalContext),
        _: (Self, &CausalContext),
        _: &mut dyn FnMut(DotChange),
        _: &mut S,
    ) -> Result<Self, <S>::Error>
    where
        Self: Sized,
        S: Sentinel,
    {
        // NOTE! When making changes to this method, consider if corresponding
        // changes need to be done to ::dry_join as well!

        Ok(Self)
    }

    fn dry_join(
        _ds1: (&Self, &CausalContext),
        _ds2: (&Self, &CausalContext),
        _sentinel: &mut S,
    ) -> Result<DryJoinOutput, S::Error>
    where
        Self: Sized,
        S: Sentinel,
    {
        Ok(DryJoinOutput::bottom())
    }
}

#[cfg(feature = "serde")]
impl From<&'_ NoExtensionTypes> for serde_json::Value {
    fn from(_: &'_ NoExtensionTypes) -> Self {
        serde_json::Value::Null
    }
}

#[cfg(feature = "serde")]
impl From<NoExtensionTypes> for serde_json::Value {
    fn from(_: NoExtensionTypes) -> Self {
        serde_json::Value::Null
    }
}

impl From<NoExtensionTypesType> for NoExtensionTypes {
    fn from(v: NoExtensionTypesType) -> Self {
        match v {}
    }
}

impl From<&Self> for NoExtensionTypesType {
    fn from(v: &Self) -> Self {
        match *v {}
    }
}

impl From<NoExtensionTypes> for () {
    fn from(_: NoExtensionTypes) -> Self {
        Self::default()
    }
}

impl ToValue for NoExtensionTypesType {
    type Values = ();
    type Value = ();

    fn values(self) -> Self::Values {
        match self {}
    }

    fn value(self) -> Result<Self::Value, Box<snapshot::SingleValueError>> {
        match self {}
    }
}

impl ExtensionType for NoExtensionTypes {
    type ValueKind = NoExtensionTypesType;
    type Value = NoExtensionTypesType;
    type ValueRef<'doc> = NoExtensionTypesType;

    fn coerce_to_value_ref(&self) -> Self::ValueRef<'_> {
        panic!("NoExtensionTypes is always bottom, and cannot be coerced into a ValueRef");
    }

    fn type_name(value: &Self::ValueRef<'_>) -> &'static str {
        match *value {}
    }

    fn bottom() -> Self {
        Self
    }
}
