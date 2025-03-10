// (c) Copyright 2025 Helsing GmbH. All rights reserved.
pub(crate) mod mvreg;
pub(crate) mod orarray;
pub(crate) mod ormap;

pub(crate) use mvreg::RegisterOp;
pub(crate) use orarray::ArrayOp;
pub(crate) use ormap::MapOp;

/// A type that holds a [`Delta`] for one of the known CRDT [`Delta`] types.
///
/// This exists so that [`MapOp`] and [`ArrayOp`] don't need a separate operation type for each
/// type of inner value they may want to insert or update at a given key.
///
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[derive(Debug, Clone)]
pub(crate) enum ValueDelta {
    Map(MapOp),
    Array(ArrayOp),
    Register(RegisterOp),
}
