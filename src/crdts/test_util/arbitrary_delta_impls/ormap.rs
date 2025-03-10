// (c) Copyright 2025 Helsing GmbH. All rights reserved.
use super::ValueDelta;
use crate::{
    CausalContext, CausalDotStore, DotStore, Identifier, MvReg, OrArray, OrMap,
    crdts::{
        NoExtensionTypes, Value,
        test_util::{ArbitraryDelta, Delta, KeyTracker},
    },
};
use quickcheck::{Arbitrary, Gen};
use std::{fmt, ops::RangeBounds};

// NOTE: Box is needed here to allow arbitrary nesting, otherwise the type isn't Sized.
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[derive(Debug, Clone)]
pub(crate) enum MapOp {
    Apply(usize, Option<String>, Box<ValueDelta>),
    Remove(usize),
    Clear,
}

impl fmt::Display for MapOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Apply(keyi, Some(key), _) => write!(f, "inserts key #{keyi} ({key})"),
            Self::Apply(keyi, None, _) => write!(f, "updates key #{keyi}"),
            Self::Remove(keyi) => write!(f, "deletes key #{keyi}"),
            Self::Clear => write!(f, "clears the map"),
        }
    }
}

impl Delta for MapOp {
    type DS = OrMap<String, NoExtensionTypes>;

    fn depends_on_keyi_in<R: RangeBounds<usize>>(&self, range: R) -> bool {
        match *self {
            Self::Apply(keyi, _, _) | Self::Remove(keyi) => range.contains(&keyi),
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
            Self::Apply(expected_keyi, insert_key, v) => {
                let keyi = expected_keyi;
                // apply is both insert and update -- figure out which one this is
                let should_exist = if let Some(insert_key) = insert_key {
                    if let Some(&keyi) = keys.map_keys.get_by_left(&insert_key) {
                        // this means two actors simulteneously inserted with the same key
                        // that's totally possible! in that case we want to make sure they share
                        // the same keyi and thus the same KeyTracker so that we accurately track
                        // inner keys and dependent operations (for shrinking).
                        assert_eq!(keyi, expected_keyi);
                    } else {
                        // this is the actually-first insert of this key, so set up the necessary
                        // state for tracking inner keys.
                        let keyi = keys.add_map_key(insert_key);
                        assert_eq!(keyi, expected_keyi);
                    }
                    // NOTE: even for a simultaneous insert, _this_ node should not see the
                    // key as already existing at the time of this op.
                    false
                } else {
                    true
                };
                let inner_keys = &mut keys.inner_keys[keyi];
                let key = keys.map_keys.get_by_right(&keyi).unwrap();
                ds.apply(
                    |old, cc, id| {
                        if should_exist {
                            assert!(!old.is_bottom());
                        } else {
                            assert!(old.is_bottom());
                        }
                        match *v {
                            ValueDelta::Map(m) => m
                                .into_crdt(&old.map, cc, id, inner_keys)
                                .map_store(Value::Map),
                            ValueDelta::Array(a) => a
                                .into_crdt(&old.array, cc, id, inner_keys)
                                .map_store(Value::Array),
                            ValueDelta::Register(r) => r
                                .into_crdt(&old.reg, cc, id, inner_keys)
                                .map_store(Value::Register),
                        }
                    },
                    key.clone(),
                    cc,
                    id,
                )
            }
            Self::Remove(keyi) => {
                let key = keys.map_keys.get_by_right(&keyi).unwrap();
                ds.remove(key, cc, id)
            }
            Self::Clear => ds.clear(cc, id),
        }
    }
}

impl ArbitraryDelta for OrMap<String, NoExtensionTypes> {
    type Delta = MapOp;

    fn arbitrary_delta(
        &self,
        cc: &CausalContext,
        id: Identifier,
        keys: &mut KeyTracker,
        g: &mut Gen,
        depth: usize,
    ) -> (Self::Delta, CausalDotStore<Self>) {
        let op = if self.0.is_empty() {
            g.choose(&["insert", "clear"])
        } else {
            g.choose(&["insert", "update", "remove", "clear"])
        };
        let indent = "  ".repeat(depth);

        match op.copied().unwrap() {
            "insert" => {
                let (key, keyi) = {
                    if self.0.len() != keys.map_keys.len() && bool::arbitrary(g) {
                        // generate an insert of the same key as another node just inserted (but
                        // that we haven't observed yet). that is, a simultaneous same-key insert.
                        let candidates: Vec<_> = keys
                            .map_keys
                            .iter()
                            .filter(|&(k, _)| !self.0.has(k))
                            .collect();
                        let (key, keyi) = *g
                            .choose(&candidates)
                            .expect("if means we only get here if there's at least one candidate");
                        (key.clone(), *keyi)
                    } else {
                        // generate an insert with a key that not only doesn't exist in this node's
                        // map, but also isn't used by any _other_ node.
                        let mut tries = 0;
                        let key = loop {
                            let candidate = format!("key{}", u8::arbitrary(g));
                            if !keys.map_keys.contains_left(&candidate) {
                                break candidate;
                            }
                            tries += 1;
                            if tries >= 10 {
                                panic!("could not generate a distinct map key for insert");
                            }
                        };
                        let keyi = keys.add_map_key(key.clone());
                        (key, keyi)
                    }
                };
                eprintln!("{indent} -> insert #{keyi} ({key})");
                let inner_keys = &mut keys.inner_keys[keyi];
                let key = keys.map_keys.get_by_right(&keyi).unwrap().clone();
                let mut value_delta = None;
                let crdt = self.apply(
                    |old, cc, id| {
                        assert!(old.is_bottom());
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
                    key.clone(),
                    cc,
                    id,
                );
                (
                    MapOp::Apply(
                        keyi,
                        Some(key),
                        Box::new(value_delta.expect("insert closure is always called")),
                    ),
                    crdt,
                )
            }
            "update" => {
                let mut keyset = self.0.keys();
                let keyi = usize::arbitrary(g) % keyset.len();
                let key = keyset
                    .nth(keyi)
                    .expect("this arm is only taken if non-empty, and n is % len")
                    .clone();
                let keyi = *keys.map_keys.get_by_left(&key).unwrap();
                eprintln!("{indent} -> updating #{keyi} ({key})");
                let inner_keys = &mut keys.inner_keys[keyi];
                // NOTE: this _may_ change the type -- that is intentional! test thoroughly.
                let mut value_delta = None;
                let crdt = self.apply(
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
                    key,
                    cc,
                    id,
                );
                (
                    MapOp::Apply(
                        keyi,
                        None,
                        Box::new(value_delta.expect("apply closure is always called")),
                    ),
                    crdt,
                )
            }
            "remove" => {
                let mut keyset = self.0.keys();
                let keyi = usize::arbitrary(g) % keyset.len();
                let key = keyset
                    .nth(keyi)
                    .expect("this arm is only taken if non-empty, and n is % len")
                    .clone();
                let keyi = *keys.map_keys.get_by_left(&key).unwrap();
                eprintln!("{indent} -> removing #{keyi} ({key})");
                (MapOp::Remove(keyi), self.remove(&key, cc, id))
            }
            "clear" => {
                eprintln!("{indent} -> clearing map");
                (MapOp::Clear, self.clear(cc, id))
            }
            op => unreachable!("need match arm for '{op}'"),
        }
    }
}
