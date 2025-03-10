// (c) Copyright 2025 Helsing GmbH. All rights reserved.
use super::ValueDelta;
use crate::{
    CausalContext, CausalDotStore, DotStore, Identifier, MvReg, OrArray, OrMap,
    crdts::{
        NoExtensionTypes, Value,
        orarray::Position,
        test_util::{ArbitraryDelta, Delta, KeyTracker},
    },
};
use quickcheck::{Arbitrary, Gen};
use std::{fmt, ops::RangeBounds};

// NOTE: Box is needed here to allow arbitrary nesting, otherwise the type isn't Sized.
// This is because `ValueDelta` itself contains `ArrayOp`.
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[derive(Debug, Clone)]
pub(crate) enum ArrayOp {
    Insert(usize, Position, Box<ValueDelta>),
    Update(usize, Position, Box<ValueDelta>),
    Delete(usize),
    Move(usize, Position),
    Clear,
}

impl fmt::Display for ArrayOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Insert(keyi, _, _) => write!(f, "insert key #{keyi}"),
            Self::Update(keyi, _, _) => write!(f, "updates key #{keyi}"),
            Self::Delete(keyi) => write!(f, "deletes key #{keyi}"),
            Self::Move(keyi, _) => write!(f, "moves key #{keyi}"),
            Self::Clear => write!(f, "clears the map"),
        }
    }
}

impl Delta for ArrayOp {
    type DS = OrArray<NoExtensionTypes>;

    fn depends_on_keyi_in<R: RangeBounds<usize>>(&self, range: R) -> bool {
        match *self {
            Self::Insert(keyi, _, _)
            | Self::Update(keyi, _, _)
            | Self::Delete(keyi)
            | Self::Move(keyi, _) => range.contains(&keyi),
            Self::Clear => false,
        }
    }

    fn into_crdt(
        self,
        ds: &Self::DS,
        cc: &CausalContext,
        id: Identifier,
        keys: &mut KeyTracker,
    ) -> CausalDotStore<Self::DS> {
        match self {
            Self::Insert(expected_keyi, p, v) => {
                assert_eq!(expected_keyi, keys.len());
                let keyi = expected_keyi;
                let cc = cc.clone();
                let uid = cc.next_dot_for(id).into();
                let mut inner_keys = KeyTracker::default();
                let crdt = ds.insert(
                    uid,
                    |cc, id| match *v {
                        ValueDelta::Map(m) => m
                            .into_crdt(&<_>::default(), cc, id, &mut inner_keys)
                            .map_store(Value::Map),
                        ValueDelta::Array(a) => a
                            .into_crdt(&<_>::default(), cc, id, &mut inner_keys)
                            .map_store(Value::Array),
                        ValueDelta::Register(r) => r
                            .into_crdt(&<_>::default(), cc, id, &mut inner_keys)
                            .map_store(Value::Register),
                    },
                    p,
                    &cc,
                    id,
                );
                keys.inner_keys.push(inner_keys);
                keys.array_keys.insert(uid, keyi);
                crdt
            }
            Self::Update(keyi, p, v) => {
                let inner_keys = &mut keys.inner_keys[keyi];
                let uid = *keys.array_keys.get_by_right(&keyi).unwrap();
                ds.apply(
                    uid,
                    |old, cc, id| match *v {
                        ValueDelta::Map(m) => m
                            .into_crdt(&old.map, cc, id, inner_keys)
                            .map_store(Value::Map),
                        ValueDelta::Array(a) => a
                            .into_crdt(&old.array, cc, id, inner_keys)
                            .map_store(Value::Array),
                        ValueDelta::Register(r) => r
                            .into_crdt(&old.reg, cc, id, inner_keys)
                            .map_store(Value::Register),
                    },
                    p,
                    cc,
                    id,
                )
            }
            Self::Delete(keyi) => {
                let uid = *keys.array_keys.get_by_right(&keyi).unwrap();
                ds.delete(uid, cc, id)
            }
            Self::Move(keyi, p) => {
                let uid = *keys.array_keys.get_by_right(&keyi).unwrap();
                ds.mv(uid, p, cc, id)
            }
            Self::Clear => ds.clear(cc, id),
        }
    }
}

impl ArbitraryDelta for OrArray<NoExtensionTypes> {
    type Delta = ArrayOp;

    fn arbitrary_delta(
        &self,
        cc: &CausalContext,
        id: Identifier,
        keys: &mut KeyTracker,
        g: &mut Gen,
        depth: usize,
    ) -> (Self::Delta, CausalDotStore<Self>) {
        // NOTE: see the outer_remove_vs_inner_mv test for why we need this
        let valid_keys: Vec<_> = self
            .0
            .iter()
            .filter_map(|(k, v)| (!v.value.is_bottom()).then_some(k))
            .collect();

        let op = if valid_keys.is_empty() && self.0.is_empty() {
            g.choose(&["insert", "clear"])
        } else if valid_keys.is_empty() {
            g.choose(&["insert", "delete", "clear"])
        } else {
            g.choose(&["insert", "update", "delete", "move", "clear"])
        };
        let indent = "  ".repeat(depth);

        match op.copied().unwrap() {
            "insert" => {
                let uid = cc.next_dot_for(id).into();
                let kind = if g.size() <= 1 {
                    "register"
                } else {
                    g.choose(&["map", "array", "register"]).copied().unwrap()
                };
                let keyi = keys.add_array_key(uid);
                eprintln!("{indent} -> inserting #{keyi} ({uid:?})");
                let inner_keys = &mut keys.inner_keys[keyi];
                let p = Position::arbitrary(g);
                let mut value_delta = None;
                let crdt = self.insert(
                    uid,
                    |cc, id| {
                        eprintln!("{indent} -> generating inner {kind} operation");
                        let (vd, value_crdt) = match kind {
                            "map" => {
                                let mut g = Gen::new(g.size() / 2);
                                let g = &mut g;
                                let (delta, crdt) = OrMap::arbitrary_delta(
                                    &<_>::default(),
                                    cc,
                                    id,
                                    inner_keys,
                                    g,
                                    depth + 1,
                                );
                                (ValueDelta::Map(delta), crdt.map_store(Value::Map))
                            }
                            "array" => {
                                let mut g = Gen::new(g.size() / 2);
                                let g = &mut g;
                                let (delta, crdt) = OrArray::arbitrary_delta(
                                    &<_>::default(),
                                    cc,
                                    id,
                                    inner_keys,
                                    g,
                                    depth + 1,
                                );
                                (ValueDelta::Array(delta), crdt.map_store(Value::Array))
                            }
                            "register" => {
                                let (delta, crdt) = MvReg::arbitrary_delta(
                                    &<_>::default(),
                                    cc,
                                    id,
                                    inner_keys,
                                    g,
                                    depth + 1,
                                );
                                (ValueDelta::Register(delta), crdt.map_store(Value::Register))
                            }
                            kind => unreachable!("need match arm for '{kind}'"),
                        };
                        value_delta = Some(vd);
                        value_crdt
                    },
                    p,
                    cc,
                    id,
                );
                (
                    ArrayOp::Insert(
                        keyi,
                        p,
                        Box::new(value_delta.expect("insert closure is always called")),
                    ),
                    crdt,
                )
            }
            "update" => {
                let uid = **g
                    .choose(&valid_keys)
                    .expect("this arm is only taken if non-empty");
                // TODO: how should this handle the case of concurrent inserts of the same
                //            key, which will imply that a single key has _multiple_ keyi.
                let keyi = *keys.array_keys.get_by_left(&uid).unwrap();
                eprintln!("{indent} -> updating #{keyi} ({uid:?})");
                let inner_keys = &mut keys.inner_keys[keyi];
                let p = Position::arbitrary(g);

                // NOTE: this _may_ change the type -- that is intentional! test thoroughly.
                let mut value_delta = None;
                let crdt = self.apply(
                    uid,
                    |old, cc, id| {
                        let kind = if g.size() <= 1 {
                            "register"
                        } else {
                            g.choose(&["map", "array", "register"]).copied().unwrap()
                        };
                        eprintln!("{indent} -> generating inner {kind} operation");
                        let (vd, value_crdt) = match kind {
                            "map" => {
                                let mut g = Gen::new(g.size() / 2);
                                let g = &mut g;
                                let (delta, crdt) = OrMap::arbitrary_delta(
                                    &old.map,
                                    cc,
                                    id,
                                    inner_keys,
                                    g,
                                    depth + 1,
                                );
                                (ValueDelta::Map(delta), crdt.map_store(Value::Map))
                            }
                            "array" => {
                                let mut g = Gen::new(g.size() / 2);
                                let g = &mut g;
                                let (delta, crdt) = OrArray::arbitrary_delta(
                                    &old.array,
                                    cc,
                                    id,
                                    inner_keys,
                                    g,
                                    depth + 1,
                                );
                                (ValueDelta::Array(delta), crdt.map_store(Value::Array))
                            }
                            "register" => {
                                let (delta, crdt) = MvReg::arbitrary_delta(
                                    &old.reg,
                                    cc,
                                    id,
                                    inner_keys,
                                    g,
                                    depth + 1,
                                );
                                (ValueDelta::Register(delta), crdt.map_store(Value::Register))
                            }
                            kind => unreachable!("need match arm for '{kind}'"),
                        };
                        value_delta = Some(vd);
                        value_crdt
                    },
                    p,
                    cc,
                    id,
                );
                (
                    ArrayOp::Update(
                        keyi,
                        p,
                        Box::new(value_delta.expect("apply closure is always called")),
                    ),
                    crdt,
                )
            }
            "delete" => {
                // NOTE: we specifically use the whole range of keys here, not just
                // "valid_keys", since we want to test what happens if a bottom-value element is
                // deleted.
                let mut uids = self.0.keys();
                let uidi = usize::arbitrary(g) % uids.len();
                let uid = *uids
                    .nth(uidi)
                    .expect("this arm is only taken if non-empty, and n is % len");
                let keyi = *keys.array_keys.get_by_left(&uid).unwrap();
                eprintln!("{indent} -> deleting #{keyi} ({uid:?})");
                (ArrayOp::Delete(keyi), self.delete(uid, cc, id))
            }
            "move" => {
                let uid = **g
                    .choose(&valid_keys)
                    .expect("this arm is only taken if non-empty");
                let keyi = *keys.array_keys.get_by_left(&uid).unwrap();
                eprintln!("{indent} -> moving #{keyi} ({uid:?})");
                let p = Position::arbitrary(g);
                (ArrayOp::Move(keyi, p), self.mv(uid, p, cc, id))
            }
            "clear" => {
                eprintln!("{indent} -> clearing array");
                (ArrayOp::Clear, self.clear(cc, id))
            }
            op => unreachable!("need match arm for '{op}'"),
        }
    }
}
