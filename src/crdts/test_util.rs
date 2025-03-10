// (c) Copyright 2025 Helsing GmbH. All rights reserved.
use super::Value;
use crate::{
    CausalContext, CausalDotStore, DotStoreJoin, Identifier,
    dotstores::recording_sentinel::RecordingSentinel,
    sentinel::{DummySentinel, Sentinel},
};
use quickcheck::Gen;
use std::{fmt, ops::RangeBounds};

mod arbitrary_delta_impls;
mod qc_arbitrary_impls;
mod qc_arbitrary_ops;
#[cfg_attr(feature = "arbitrary", allow(dead_code))]
pub(crate) fn join_harness<DS, Init, W1, W2, S, C>(
    zero: DS,
    init: Init,
    w1: W1,
    w2: W2,
    mut sentinel: S,
    check: C,
) where
    DS: DotStoreJoin<S> + DotStoreJoin<RecordingSentinel> + Default + Clone,
    S: Sentinel,
    S::Error: fmt::Debug,
    Init: FnOnce(CausalDotStore<DS>, Identifier) -> CausalDotStore<DS>,
    W1: FnOnce(&DS, CausalContext, Identifier) -> CausalDotStore<DS>,
    W2: FnOnce(&DS, CausalContext, Identifier) -> CausalDotStore<DS>,
    C: FnOnce(CausalDotStore<DS>, S),
{
    let v = zero;
    let init_id = Identifier::new(9, 0);
    let v = init(
        CausalDotStore {
            store: v,
            context: CausalContext::new(),
        },
        init_id,
    );
    let w1_id = Identifier::new(0, 0);
    let mut w1_v = w1(&v.store, v.context.clone(), w1_id);
    let w2_id = Identifier::new(1, 0);
    let w2_v = w2(&v.store, v.context.clone(), w2_id);
    w1_v.test_join_with_and_track(w2_v.store, &w2_v.context, &mut |_| {}, &mut sentinel)
        .unwrap();
    check(w1_v, sentinel)
}

/// Types that can construct descriptors of an arbitrary modification to themselves.
pub(crate) trait ArbitraryDelta: Sized {
    #[cfg(not(feature = "serde"))]
    /// The type of the descriptor.
    type Delta: Delta<DS = Self>;
    #[cfg(feature = "serde")]
    type Delta: Delta<DS = Self> + ::serde::Serialize + ::serde::de::DeserializeOwned;

    /// Produces a descriptor for an arbitrary modification to `&self`.
    ///
    /// If the descriptor produces a new key in `self`, it should represent that key as a `usize`
    /// as returned by the `add_*_key` methods on [`KeyTracker`]. Any deltas to inner collections
    /// should be passed `&mut keys.inner_keys[keyi]` so they can also track their collections.
    ///
    /// `depth` is used solely to produce visual guides (eg, indents) so that nested calls to
    /// `arbitrary_delta` are easier to distinguish.
    fn arbitrary_delta(
        &self,
        cc: &CausalContext,
        id: Identifier,
        keys: &mut KeyTracker,
        g: &mut Gen,
        depth: usize,
    ) -> (Self::Delta, CausalDotStore<Self>);
}

/// Types that describe a modification to an instance of [`Delta::DS`].
pub(crate) trait Delta: Sized + fmt::Display {
    /// The [`DotStore`] type that this delta applies to.
    type DS: DotStoreJoin<DummySentinel>;

    /// Returns true if this delta specifically depends on a key in the given keyi range.
    ///
    /// Some examples:
    ///
    /// - an `Update(keyi = 42)` should return `true` when passed a range `16..`.
    /// - an `Update(keyi = 4)` should return `false` when passed a range `16..`.
    /// - an `Insert(keyi = 42)` should return `true` when passed a range `16..`.
    /// - a `Clear` should return `false` when passed a range `16..`.
    fn depends_on_keyi_in<R: RangeBounds<usize>>(&self, range: R) -> bool;

    /// Turns this modification description into a CRDT over `ds` that, when joined with `ds`, will
    /// produce the desired modification.
    ///
    /// `keys` tracks the sequence of keys produced so far. See [`ArbitraryDelta::arbitrary_delta`]
    /// for details.
    #[cfg_attr(feature = "arbitrary", allow(dead_code))]
    fn into_crdt(
        self,
        ds: &Self::DS,
        cc: &CausalContext,
        id: Identifier,
        keys: &mut KeyTracker,
    ) -> CausalDotStore<Self::DS>;
}

pub(crate) use qc_arbitrary_ops::KeyTracker;
#[cfg(test)]
pub(crate) use qc_arbitrary_ops::Ops;
