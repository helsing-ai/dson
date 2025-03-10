// (c) Copyright 2025 Helsing GmbH. All rights reserved.
use crate::{
    CausalContext, CausalDotStore, Identifier, MvReg,
    crdts::{
        mvreg::MvRegValue,
        snapshot::{SingleValueError, ToValue},
    },
};

/// Returns the values of this register without collapsing conflicts.
pub fn values(m: &MvReg) -> impl ExactSizeIterator<Item = &MvRegValue> {
    m.values().into_iter()
}

/// Returns the value of this register assuming (and asserting) no conflicts on element values.
pub fn value(m: &MvReg) -> Result<&MvRegValue, Box<SingleValueError>> {
    m.value()
}

/// Writes a value to the register.
pub fn write(
    v: MvRegValue,
) -> impl FnMut(&MvReg, &CausalContext, Identifier) -> CausalDotStore<MvReg> {
    move |m, cc, id| m.write(v.clone(), cc, id)
}

/// Clears the register.
pub fn clear() -> impl Fn(&MvReg, &CausalContext, Identifier) -> CausalDotStore<MvReg> {
    move |m, _cc, _id| m.clear()
}
