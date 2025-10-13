// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! Provides snapshots of CRDTs for inspection.
//!
//! A snapshot is a read-only, immutable view of the state of a CRDT at a
//! particular point in time.
//!
//! This module provides two ways to get a snapshot of a CRDT, exposed via the
//! [`ToValue`] trait:
//!
//! 1. **Conflict-preserving snapshots** via [`ToValue::values()`]: This method returns an
//!    [`AllValues`] snapshot. In this representation, any multi-value registers
//!    ([`MvReg`]) will contain all of their concurrently written values. This is useful when you
//!    expect conflicts and want to handle them in your application logic (e.g., by merging,
//!    presenting them to the user, or picking one).
//!
//! 2. **Conflict-collapsing snapshots** via [`ToValue::value()`]: This method returns a
//!    [`CollapsedValue`] snapshot. It assumes that there are no conflicts in the CRDT. If any
//!    multi-value register holds more than one value, this method will return an error. This
//!    provides a more convenient, "normal" data view when you don't expect conflicts or have a
//!    logic in place to resolve them before reading.
//!
//! The snapshot types like [`OrMap`], [`OrArray`], and [`MvReg`] mirror the structure of the
//! actual CRDTs but are read-only.
use super::{Either, ValueRef, mvreg::MvRegValue};
use crate::{DsonRandomState, ExtensionType, create_map, dotstores::DotFunValueIter};
use std::{
    collections::HashMap,
    error, fmt,
    hash::Hash,
    ops::{Deref, Index},
};

/// A type that holds values that may or may not feature conflicts.
pub trait ToValue {
    /// The conflict-preserving value type.
    type Values;

    /// The conflict-less value type.
    type Value;

    /// The (owned) leaf value type.
    ///
    /// This is the value type of the outmost nested CRDT, like a [`MvRegValue`].
    type LeafValue;

    /// Returns the values of this type without collapsing conflicts.
    ///
    /// That is, any [`MvReg`](crate::MvReg) nested below this value will produce the full _set_ of
    /// possible values, not just a single (arbitrarily chosen) value.
    fn values(self) -> Self::Values;

    /// Returns the values of this type assuming (and asserting) no conflicts on element values.
    ///
    /// That is, for any [`MvReg`](crate::MvReg) nested below this value, this method asserts that
    /// the `MvReg` has only a single possible value (ie, it ultimately calls [`ToValue::value`] on
    /// each such `MvReg`), and returns just that one value.
    ///
    /// This makes for a more ergonomic API than [`ToValue::values`], but comes at the cost of
    /// erroring when conflicts are found.
    ///
    /// If a contained [`MvReg`](crate::MvReg) has conflicting values, this method returns an `Err`
    /// with [`SingleValueIssue::HasConflict`].
    fn value(self) -> Result<Self::Value, Box<SingleValueError<Self::LeafValue>>>;
}

impl<'doc, C> ToValue for ValueRef<'doc, C>
where
    C: ExtensionType,
    C::ValueRef<'doc>: ToValue,
{
    type Values = AllValues<'doc, C::ValueRef<'doc>>;
    type Value = CollapsedValue<'doc, C::ValueRef<'doc>>;
    type LeafValue = Either<MvRegValue, <C::ValueRef<'doc> as ToValue>::LeafValue>;

    fn values(self) -> Self::Values {
        match self {
            ValueRef::Map(map) => AllValues::Map(map.values()),
            ValueRef::Array(arr) => AllValues::Array(arr.values()),
            ValueRef::Register(reg) => AllValues::Register(reg.values()),
            ValueRef::Custom(custom) => AllValues::Custom(custom.values()),
        }
    }

    fn value(self) -> Result<Self::Value, Box<SingleValueError<Self::LeafValue>>> {
        Ok(match self {
            ValueRef::Map(map) => CollapsedValue::Map(map.value()?),
            ValueRef::Array(arr) => CollapsedValue::Array(arr.value()?),
            ValueRef::Register(reg) => {
                CollapsedValue::Register(reg.value().map_err(|v| v.map_values(Either::Left))?)
            }
            ValueRef::Custom(custom) => {
                CollapsedValue::Custom(custom.value().map_err(|v| v.map_values(Either::Right))?)
            }
        })
    }
}

macro_rules! impl_partial_eq {
    ($on:ty, *; {$($t:ty),+}) => {
        $(impl_partial_eq!($on, *; $t);)+
    };
    ($on:ty; {$($t:ty),+}) => {
        $(impl_partial_eq!($on; $t);)+
    };

    ($on:ty$(, $map:tt)?; $t:ty) => {
        impl<C> PartialEq<$t> for $on
        where
            C: ToValue,
        {
            fn eq(&self, other: &$t) -> bool {
                matches!($($map)? self, Self::Register(r1) if r1 == other)
            }
        }
    };
}

/// A representation of all values in a CRDT while preserving any potentially conflicting leaf
/// values.
///
/// This is a snapshot of the CRDT that can be used to inspect the current state of the CRDT.
/// However, it is not a CRDT, and cannot be modified.
///
/// See [`ToValue::values`].
#[derive(Debug, Clone, PartialEq)]
pub enum AllValues<'doc, Custom>
where
    Custom: ToValue,
{
    /// A multi-value register, which can hold multiple values at the same time.
    Register(MvReg<'doc>),
    /// An observed-remove map, which is a map that supports concurrent removal of keys.
    Map(OrMap<'doc, String, AllValues<'doc, Custom>>),
    /// An observed-remove array, which is an array that supports concurrent removal of elements.
    Array(OrArray<AllValues<'doc, Custom>>),
    /// A custom CRDT type.
    Custom(Custom::Values),
}

impl_partial_eq!(AllValues<'_, C>; {[u8], &[u8], str, &str, bool, f64, u64, i64});
// i32 because it's the "default" inference integer type
impl_partial_eq!(AllValues<'_, C>; i32);
// byte literals
impl<C, const N: usize> PartialEq<&[u8; N]> for AllValues<'_, C>
where
    C: ToValue,
{
    fn eq(&self, other: &&[u8; N]) -> bool {
        matches!(self, Self::Register(r1) if r1 == other)
    }
}

/// A representation of all values in a CRDT where there are no conflicting values at the leaves.
///
/// This is a snapshot of the CRDT that can be used to inspect the current state of the CRDT.
/// However, it is not a CRDT, and cannot be modified.
///
/// See [`ToValue::value`].
#[derive(Debug, Clone, PartialEq)]
pub enum CollapsedValue<'doc, Custom>
where
    Custom: ToValue,
{
    /// A multi-value register, which can hold multiple values at the same time.
    Register(&'doc MvRegValue),
    /// An observed-remove map, which is a map that supports concurrent removal of keys.
    Map(OrMap<'doc, String, CollapsedValue<'doc, Custom>>),
    /// An observed-remove array, which is an array that supports concurrent removal of elements.
    Array(OrArray<CollapsedValue<'doc, Custom>>),
    /// A custom CRDT type.
    Custom(Custom::Value),
}

impl_partial_eq!(CollapsedValue<'_, C>, *; {[u8], &[u8], str, &str, bool, f64, u64, i64});
// i32 because it's the "default" inference integer type
impl_partial_eq!(CollapsedValue<'_, C>, *; i32);
// byte literals
impl<C, const N: usize> PartialEq<&[u8; N]> for CollapsedValue<'_, C>
where
    C: ToValue,
{
    fn eq(&self, other: &&[u8; N]) -> bool {
        matches!(*self, Self::Register(r1) if r1 == other)
    }
}

/// An error that occurs when trying to collapse a CRDT with conflicting values.
#[derive(Debug, Clone)]
pub struct SingleValueError<T> {
    /// The path to the value that has an issue.
    pub path: Vec<String>,
    /// The issue that occurred.
    pub issue: SingleValueIssue<T>,
}

// We can't derive `PartialEq` because `SingleValueIssue` implements it manually
// to provide order-insensitive equality for conflicts. For that we need the
// additional `Ord` bound.
impl<V: PartialEq + Ord> PartialEq for SingleValueError<V> {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.issue == other.issue
    }
}

impl<T> SingleValueError<T> {
    /// Maps the values within the error to a different type.
    pub fn map_values<Other>(self, f: impl Fn(T) -> Other) -> Box<SingleValueError<Other>> {
        let Self { path, issue } = self;
        let issue = match issue {
            SingleValueIssue::HasConflict(conflicts) => {
                SingleValueIssue::HasConflict(conflicts.into_iter().map(f).collect())
            }
            SingleValueIssue::Cleared => SingleValueIssue::Cleared,
        };

        Box::new(SingleValueError { path, issue })
    }
}

impl<T> fmt::Display for SingleValueError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at <self>")?;
        // reverse since paths are appended as the error bubbles outwards
        for p in self.path.iter().rev() {
            write!(f, ".{p}")?;
        }
        Ok(())
    }
}

impl<V: fmt::Debug + 'static> error::Error for SingleValueError<V> {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        Some(&self.issue)
    }
}

/// An issue that can occur when trying to collapse a CRDT with conflicting values.
#[derive(Debug, Clone)]
pub enum SingleValueIssue<V> {
    // NOTE: Contrary to `DotFun`, there are _NO_ ordering guarantees on the
    // conflicted values
    HasConflict(smallvec::SmallVec<[V; 2]>),
    Cleared,
}

impl<V: PartialEq + Ord> PartialEq for SingleValueIssue<V> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::HasConflict(l0), Self::HasConflict(r0)) => {
                let mut l0 = l0.iter().collect::<Vec<_>>();
                let mut r0 = r0.iter().collect::<Vec<_>>();
                l0.sort_unstable();
                r0.sort_unstable();
                l0 == r0
            }
            (Self::Cleared, Self::Cleared) => true,
            _ => false,
        }
    }
}
impl<V: Eq + Ord> Eq for SingleValueIssue<V> {}

impl<T> fmt::Display for SingleValueIssue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SingleValueIssue::HasConflict(set) => write!(f, "has {} possible values", set.len()),
            SingleValueIssue::Cleared => write!(f, "has been cleared"),
        }
    }
}

impl<T: fmt::Debug> error::Error for SingleValueIssue<T> {}

// NOTE: This does not expose the HashMap so we can change its inner type later.
/// A snapshot of an [`OrMap`](crate::OrMap) that can be used to inspect the current state of the
/// CRDT.
#[derive(Debug, PartialEq, Eq)]
pub struct OrMap<'doc, K, V>
where
    K: Hash + Eq + ?Sized + 'doc,
    V: 'doc,
{
    // TODO: is it worthwhile to keep &K here instead of K?
    // it ends up making the ergonomics for accessing this map worse since &String doesn't
    // implement Borrow<str> (only String does).
    pub(crate) map: HashMap<&'doc K, V, DsonRandomState>,
}

impl<K, V> Clone for OrMap<'_, K, V>
where
    K: Hash + Eq + ?Sized,
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            map: self.map.clone(),
        }
    }
}

impl<K, V> Default for OrMap<'_, K, V>
where
    K: Hash + Eq + ?Sized,
{
    fn default() -> Self {
        Self { map: create_map() }
    }
}

impl<K, V> Index<&K> for OrMap<'_, K, V>
where
    K: Eq + Hash,
{
    type Output = V;

    fn index(&self, index: &K) -> &Self::Output {
        self.map.index(index)
    }
}

impl<K, V> OrMap<'_, K, V>
where
    K: Hash + Eq + ?Sized,
{
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&K, &V)> {
        self.map.iter().map(|(&k, v)| (k, v))
    }

    pub fn keys(&self) -> impl ExactSizeIterator<Item = &K> {
        self.map.keys().copied()
    }

    pub fn values(&self) -> impl ExactSizeIterator<Item = &V> {
        self.map.values()
    }
}

impl<'doc, K, V> IntoIterator for OrMap<'doc, K, V>
where
    K: Hash + Eq + ?Sized,
{
    type Item = (&'doc K, V);
    type IntoIter = std::collections::hash_map::IntoIter<&'doc K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

/// A snapshot of an [`OrArray`](crate::OrArray) that can be used to inspect the current state of
/// the CRDT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrArray<V> {
    pub(crate) list: Vec<V>,
}

impl<V> Default for OrArray<V> {
    fn default() -> Self {
        Self {
            list: Default::default(),
        }
    }
}

impl<V> Deref for OrArray<V> {
    type Target = [V];

    fn deref(&self) -> &Self::Target {
        &self.list[..]
    }
}

impl<V> AsRef<[V]> for OrArray<V> {
    fn as_ref(&self) -> &[V] {
        &self.list[..]
    }
}

impl<V> OrArray<V> {
    pub fn len(&self) -> usize {
        self.list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }

    pub fn get(&self, i: usize) -> Option<&V> {
        self.list.get(i)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &V> {
        self.list.iter()
    }
}

impl<V> IntoIterator for OrArray<V> {
    type Item = V;
    type IntoIter = std::vec::IntoIter<V>;

    fn into_iter(self) -> Self::IntoIter {
        self.list.into_iter()
    }
}

/// A snapshot of an [`MvReg`](crate::MvReg) that can be used to inspect the current state of the
/// CRDT.
#[derive(Debug, Clone)]
pub struct MvReg<'doc> {
    // NOTE: DotFunValueIter is basically a std::slice::Iter, so is cheap to Clone
    pub(crate) values: DotFunValueIter<'doc, MvRegValue>,
}

impl<'doc> MvReg<'doc> {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.len() == 0
    }

    pub fn get(&self, i: usize) -> Option<&'doc MvRegValue> {
        self.values.clone().nth(i)
    }

    pub fn contains(&self, x: &'_ MvRegValue) -> bool {
        self.values.clone().any(|v| v == x)
    }
}

impl PartialEq for MvReg<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.values.clone().eq(other.clone())
    }
}
impl Eq for MvReg<'_> {}

macro_rules! impl_partial_eq {
    ({$($t:ty),+}) => {
        $(impl_partial_eq!($t);)+
    };

    ($t:ty) => {
        impl PartialEq<$t> for MvReg<'_> {
            fn eq(&self, other: &$t) -> bool {
                self.values.clone().any(|v| v == other)
            }
        }
    };
}
impl_partial_eq!({[u8], &[u8], str, &str, bool, f64, u64, i64});
// i32 because it's the "default" inference integer type
impl_partial_eq!(i32);
// byte literals
impl<const N: usize> PartialEq<&[u8; N]> for MvReg<'_> {
    fn eq(&self, other: &&[u8; N]) -> bool {
        self.values.clone().any(|v| v == other)
    }
}

impl PartialEq<[MvRegValue]> for MvReg<'_> {
    fn eq(&self, other: &[MvRegValue]) -> bool {
        (self.values.clone()).eq(other.iter())
    }
}

impl PartialEq<[MvRegValue]> for &'_ MvReg<'_> {
    fn eq(&self, other: &[MvRegValue]) -> bool {
        (self.values.clone()).eq(other.iter())
    }
}

impl<const N: usize> PartialEq<[MvRegValue; N]> for MvReg<'_> {
    fn eq(&self, other: &[MvRegValue; N]) -> bool {
        (self.values.clone()).eq(other.iter())
    }
}

impl<const N: usize> PartialEq<[MvRegValue; N]> for &'_ MvReg<'_> {
    fn eq(&self, other: &[MvRegValue; N]) -> bool {
        (self.values.clone()).eq(other.iter())
    }
}

impl<'doc> IntoIterator for MvReg<'doc> {
    type Item = &'doc MvRegValue;
    type IntoIter = DotFunValueIter<'doc, MvRegValue>;

    fn into_iter(self) -> Self::IntoIter {
        self.values
    }
}
