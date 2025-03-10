// (c) Copyright 2025 Helsing GmbH. All rights reserved.
use crate::{
    CausalContext, CausalDotStore, Identifier, MvReg,
    crdts::{
        mvreg::MvRegValue,
        test_util::{ArbitraryDelta, Delta, KeyTracker},
    },
};
use quickcheck::{Arbitrary, Gen};
use std::{fmt, ops::RangeBounds};

#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[derive(Debug, Clone)]
pub(crate) struct RegisterOp(pub(crate) Option<MvRegValue>);

impl fmt::Display for RegisterOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_some() {
            write!(f, "writes a value to the register")
        } else {
            write!(f, "clears the register")
        }
    }
}

impl ArbitraryDelta for MvReg {
    type Delta = RegisterOp;

    fn arbitrary_delta(
        &self,
        cc: &CausalContext,
        id: Identifier,
        _keys: &mut KeyTracker,
        g: &mut Gen,
        depth: usize,
    ) -> (Self::Delta, CausalDotStore<Self>) {
        // NOTE: it's tempting to assert that keys.inner_keys.is_empty(), but since we
        // generate traces where values change _type_, inner_keys may actually hold things for
        // "when this value is an array".
        let indent = "  ".repeat(depth);
        // TODO: we currently do not generate clear()s as they do _really_ weird things to
        // registers. see the OrArray push_bottom test.
        #[expect(clippy::overly_complex_bool_expr)]
        if false && bool::arbitrary(g) {
            eprintln!("{indent} -> clearing register");
            (RegisterOp(None), self.clear())
        } else {
            let v = MvRegValue::arbitrary(g);
            eprintln!("{indent} -> writing to register ({v:?})");
            (RegisterOp(Some(v.clone())), self.write(v, cc, id))
        }
    }
}

impl Delta for RegisterOp {
    type DS = MvReg;

    fn depends_on_keyi_in<R: RangeBounds<usize>>(&self, _range: R) -> bool {
        // TODO: how can we support shrinking MvRegs given they don't have keys?
        false
    }

    fn into_crdt(
        self,
        ds: &Self::DS,
        cc: &CausalContext,
        id: Identifier,
        _keys: &mut KeyTracker,
    ) -> CausalDotStore<Self::DS> {
        // NOTE: same as in arbitrary_delta, we cannot assert that keys.inner_keys.is_empty()
        if let Some(v) = self.0 {
            ds.write(v, cc, id)
        } else {
            ds.clear()
        }
    }
}
