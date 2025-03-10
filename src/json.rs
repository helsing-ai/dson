// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! JSON representation
//!
//! Value-level conflicts, which can only occur in [`MvReg`s](crate::crdts::MvReg), are
//! represented as a JSON array of the conflicting values in an **arbitrary but deterministic order**.
//!
//! # Examples
//!
//! ## A simple document without conflicts
//!
//! ```json
//! {
//!   "name": "John Doe",
//!   "age": 43,
//!   "phones": [
//!     "+44 1234567",
//!     "+44 2345678"
//!   ]
//! }
//! ```
//!
//! ## A document with a value conflict
//!
//! If two users concurrently edit the "name" field, the conflict is preserved.
//!
//! ```json
//! {
//!   "name": ["John Doe", "Jon Dough"],
//!   "age": 43,
//!   "phones": [
//!     "+44 1234567",
//!     "+44 2345678"
//!   ]
//! }
//! ```
use crate::{
    ExtensionType,
    api::timestamp,
    crdts::{
        ValueRef,
        mvreg::MvRegValue,
        snapshot::{self, ToValue},
    },
};
use serde_json::Value;
use std::{fmt, hash::Hash};

/// Converts a [`MvRegValue`] to a [`serde_json::Value`].
impl From<MvRegValue> for Value {
    fn from(val: MvRegValue) -> Self {
        match val {
            MvRegValue::Bytes(v) => {
                base64::Engine::encode(&base64::engine::general_purpose::STANDARD, v).into()
            }
            MvRegValue::String(v) => v.into(),
            MvRegValue::Float(v) => v.into(),
            MvRegValue::Double(v) => v.into(),
            MvRegValue::U64(v) => v.into(),
            MvRegValue::I64(v) => v.into(),
            MvRegValue::Bool(v) => v.into(),
            MvRegValue::Timestamp(v) => timestamp_to_json(v),
            #[cfg(feature = "ulid")]
            MvRegValue::Ulid(v) => serde_json::to_value(v).expect("ULID is JSON serializable"),
        }
    }
}

#[cfg(feature = "chrono")]
fn timestamp_to_json(v: timestamp::Timestamp) -> Value {
    v.into()
}

#[cfg(not(feature = "chrono"))]
fn timestamp_to_json(v: timestamp::Timestamp) -> Value {
    v.as_millis().into()
}

/// Converts a [`snapshot::AllValues`] to a [`serde_json::Value`].
impl<C> From<snapshot::AllValues<'_, C>> for Value
where
    C: ToValue,
    serde_json::Value: From<C::Values>,
{
    fn from(value: snapshot::AllValues<'_, C>) -> Self {
        match value {
            snapshot::AllValues::Register(reg) => reg.into(),
            snapshot::AllValues::Map(map) => map.into(),
            snapshot::AllValues::Array(arr) => arr.into(),
            snapshot::AllValues::Custom(c) => c.into(),
        }
    }
}

/// Converts a [`ValueRef`] to a `serde_json::Value`.
impl<C> From<ValueRef<'_, C>> for Value
where
    C: ExtensionType,
    for<'doc> serde_json::Value: From<<C::ValueRef<'doc> as ToValue>::Values>,
{
    fn from(value: ValueRef<'_, C>) -> Self {
        value.values().into()
    }
}

/// Converts a [`snapshot::OrMap`] to a `serde_json::Value`.
impl<K, V> From<snapshot::OrMap<'_, K, V>> for serde_json::Value
where
    K: Hash + Eq + fmt::Display,
    V: Into<serde_json::Value>,
{
    fn from(value: snapshot::OrMap<'_, K, V>) -> Self {
        let obj = value
            .map
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.into()))
            .collect();
        serde_json::Value::Object(obj)
    }
}

/// Converts a [`snapshot::OrArray`] to a `serde_json::Value`.
impl<V> From<snapshot::OrArray<V>> for serde_json::Value
where
    V: Into<serde_json::Value>,
{
    fn from(value: snapshot::OrArray<V>) -> Self {
        // NOTE: items are sorted by the dot, which we need for handling
        // single-writer (temporary) conflicts client-side.
        let arr = value.list.into_iter().map(Into::into).collect();
        serde_json::Value::Array(arr)
    }
}

/// Converts a [`snapshot::MvReg`] to a `serde_json::Value`.
///
/// * If the register is empty, it returns `Null`.
/// * If the register has one value, it returns that value.
/// * If the register has multiple values, it returns an array of those values.
impl From<snapshot::MvReg<'_>> for serde_json::Value {
    fn from(reg: snapshot::MvReg<'_>) -> Self {
        match reg.values.len() {
            0 => serde_json::Value::Null,
            1 => (reg.get(0).expect("len > 0")).clone().into(),
            _ => serde_json::Value::Array(reg.into_iter().map(|x| (*x).clone().into()).collect()),
        }
    }
}
