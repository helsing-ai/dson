// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! This module implements Arbitrary for sequences of operations.

use super::ArbitraryDelta;
use crate::{
    CausalDotStore, ComputeDeletionsArg, DotStoreJoin, Identifier, compute_deletions_unknown_to,
    crdts::{orarray::Uid, test_util::Delta},
    dotstores::recording_sentinel::RecordingSentinel,
    sentinel::DummySentinel,
};
use bimap::BiHashMap;
use quickcheck::Gen;
use std::{collections::HashMap, fmt};

/// A type that tracks the keys in a collection, and in any inner collections.
///
/// This type exists to break the constraint that the exact same keys need to be used when
/// _generating_ a [`Ops`] and when _executing_ one. This is desirable to enable shrinking -- we
/// may want to eliminate some operations from the trace to produce more minimal reproducing
/// examples, but this may in turn change what keys are generated (eg, for [`OrArray`] which use
/// whatever the next [`Dot`] is as the key). By using an index, a [`Delta`] can store that it
/// updates "the nth created key", whatever that happens to be in the current execution.
#[derive(Debug, Clone, Default)]
pub(crate) struct KeyTracker {
    /// Bijective (ie, 1:1) mapping between OrMap String keys and their keyi.
    pub(crate) map_keys: BiHashMap<String, usize>,
    /// Bijective (ie, 1:1) mapping between OrArray Uid keys and their keyi.
    pub(crate) array_keys: BiHashMap<Uid, usize>,
    /// Mapping from keyi to the KeyTracker for any inner collection.
    pub(crate) inner_keys: Vec<KeyTracker>,
}

impl KeyTracker {
    /// Returns the number of keys currently tracked.
    pub(crate) fn len(&self) -> usize {
        self.inner_keys.len()
    }

    /// Tracks a new map (ie, [`String`]-based) key, and returns its keyi.
    pub(crate) fn add_map_key(&mut self, key: String) -> usize {
        let keyi = self.len();
        self.map_keys.insert_no_overwrite(key, keyi).unwrap();
        self.inner_keys.push(Default::default());
        keyi
    }

    /// Tracks a new array (ie, [`Uid`]-based) key, and returns its keyi.
    pub(crate) fn add_array_key(&mut self, key: Uid) -> usize {
        let keyi = self.len();
        self.array_keys.insert_no_overwrite(key, keyi).unwrap();
        self.inner_keys.push(Default::default());
        keyi
    }
}

/// A single operation that a node may perform during a distributed systems trace.
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[derive(Debug, Clone)]
struct Op<Delta> {
    /// The identifier of the node that should perform this action.
    by: Identifier,
    /// The action the node should take.
    action: Action<Delta>,
}

/// An action a distributed node can take with respects to its current state.
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[derive(Debug, Clone)]
enum Action<Delta> {
    /// Synchronize with (specifically *from*) the node with the given identifier.
    Sync(Identifier),
    /// Apply the given modification to the current state.
    Data(Delta),
}

/// A sequence of legal per-node operations over a `DS: DotStore`.
///
/// Specifically, this type constains a sequence of per-node operations over a [`DotStore`] in such
/// a way that each node takes legal-but-arbitrary actions at each step based on its own current
/// view of the world. Synchronization between nodes are also modeled explicitly to effectively
/// produce a distributed systems trace.
///
/// Since this type implements [`quickcheck::Arbitrary`], it can be used to fuzz-test the
/// distributed operation of any [`DotStore`]. The most basic test to perform over any such
/// sequence is the order-invariance of the produced CRDTs, which is done by
/// [`Ops::check_order_invariance`].
///
/// The operations also model nested [`DotStore`]s (eg, an array of maps of registers), though
/// avoids very deeply nested or large sub-elements as bugs tend to only require one level of
/// nesting to present.
///
/// ## A note about correctness
///
/// The "simulator" that `impl Arbitrary for Ops` implements to determine what the set of legal
/// actions are for each node at each point in time makes use of the [`DotStore`] to track the
/// current state. For example, it uses the [`OrMap`] CRDT to determine which keys are valid to
/// then generate operations against the OrMap CRDT. This is a circular assumption -- if there's a
/// bug in the CRDT logic, it may cause us to then generate invalid operations (or not a full
/// subset of legal ones). This is done for the sake of our collective sanity. Trust me on this
/// one; modeling the set of keys that a node "should" know about _without_ using CRDTs in the face
/// of arbitrary sync operations is _very_ painful. You basically end up re-inventing CRDTs one
/// observed corner case at a time.
///
/// Despite this limitation, this kind of circular-assumption simulation is still useful. It
/// explores a wide range of operation sequences that _may_ end up surfacing bugs (often when
/// _generating_ the sequence in the first place), and it will still produce a set of CRDTs that we
/// can then check the order-invariance of.
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[cfg_attr(not(feature = "serde"), derive(::core::fmt::Debug))]
pub(crate) struct Ops<DS>
where
    DS: ArbitraryDelta,
{
    /// The sequence of operations to perform.
    ops: Vec<Op<DS::Delta>>,

    /// The number of distinct top-level keys used in self.ops.
    ///
    /// This value is stored just so that [`OpsShrinker`] knows when it is including all keys.
    nkeys: usize,
}

#[cfg(feature = "json")]
impl<DS> fmt::Debug for Ops<DS>
where
    DS: ArbitraryDelta,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(&self).unwrap())
    }
}

// NOTE: manual impl so we don't require that DS: Clone (which derive(Clone) would)
impl<DS> Clone for Ops<DS>
where
    DS: ArbitraryDelta,
    DS::Delta: Clone,
{
    fn clone(&self) -> Self {
        Self {
            ops: self.ops.clone(),
            nkeys: self.nkeys,
        }
    }
}

impl<DS> quickcheck::Arbitrary for Ops<DS>
where
    DS: ArbitraryDelta + DotStoreJoin<RecordingSentinel> + Clone + Default + 'static,
    DS::Delta: Clone,
{
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        eprintln!("\n:: Generate a new test case");

        // rather than storing the keys over and over, we use indices
        let mut key_tracker = KeyTracker::default();

        // no need to have more than 4 nodes for _most_ distributed systems problems
        let nodes = 0..((u8::arbitrary(g) % 4) + 1/* at least one */);
        let mut ccs: Vec<(Identifier, CausalDotStore<DS>)> = nodes
            .map(|i| {
                let id = Identifier::new(1, i as u16);
                (id, CausalDotStore::new())
            })
            .collect();
        assert!(!ccs.is_empty());

        // restrict the amount of state generated per element of the ops vector.
        // we also avoid generating ops sequences that are _too_ long. very very few bugs
        // require 256+ operations to reproduce, and if they do, the chance of finding them by
        // fuzzing is quite small. the main reason to not make this number even smaller is that
        // a longer sequence means we'll explore more possible operation interleavings in one
        // execution of the test (at the cost of longer test times). 256 means the test takes
        // ~12s, which feels about right.
        let n = g.size().min(256) as u8;
        let mut g = Gen::new(g.size() / ccs.len());
        let g = &mut g;

        let mut ops = Vec::with_capacity(usize::from(n));
        for _ in 0..n {
            // choose which node should perform the next operation
            let id = {
                let &(id, _) = g.choose(&ccs).unwrap();
                id
            };

            // sometimes, nodes should sync with other nodes:
            // this is u8::arbitrary not bool::arbitrary so that we can make it less than 50%
            if u8::arbitrary(g) < 64 {
                // sync means we'll be joining the other node's causal context.
                // it's unidirectional, so if a syncs from b, it does _not_
                // mean that b sees a's state.

                // choose who we'll synchronize with
                let (wid, with) = g.choose(&ccs).unwrap().clone();

                // sync with self is a no-op
                if wid == id {
                    continue;
                }

                eprintln!("==> {id:?} syncs from {wid:?}");

                // make sure the executor also knows to sync
                ops.push(Op {
                    by: id,
                    action: Action::Sync(wid),
                });

                // sync is then just to absorb the other node's state
                let (_, cc) = ccs.iter_mut().find(|&&mut (ccid, _)| ccid == id).unwrap();
                cc.test_join_with(with.store.clone(), &with.context);

                continue;
            };

            // determine the node's current view of the world
            let (_, cc) = ccs.iter_mut().find(|&&mut (ccid, _)| ccid == id).unwrap();

            // okay, pick a random (legal) operation to perform
            eprintln!("==> {id:?} generates data delta");
            let (op, crdt) =
                DS::arbitrary_delta(&cc.store, &cc.context, id, &mut key_tracker, g, 1);

            // merge the associated CRDT to update this node's view of the world.
            cc.test_join_with(crdt.store.clone(), &crdt.context);

            ops.push(Op {
                by: id,
                action: Action::Data(op),
            });
        }

        let mut s = Self {
            ops,
            nkeys: key_tracker.inner_keys.len(),
        };
        s.prune_unnecessary();
        s
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        if self.ops.is_empty() {
            return quickcheck::empty_shrinker();
        }

        Box::new(OpsShrinker {
            seed: self.clone(),
            key_subset: 1,
            size: 0,
        })
    }
}

impl<DS> Ops<DS>
where
    DS: ArbitraryDelta,
    DS::Delta: Clone,
{
    /// Removes sub-sequences of operations that have no observable effect.
    fn prune_unnecessary(&mut self) {
        // keep track of what a node last did to see if sync is useful
        let mut previous_for_node = HashMap::new();

        let mut i = 0;
        self.ops.retain(|op| {
            i += 1;
            let id = op.by;
            let mut keep = true;
            match op.action {
                // TODO: some repeated data operations can be pruned or combined, such as
                // clears that happen immediately after each other (on a given node) or a delete
                // that is followed by a clear. but leave that for future work.
                Action::Data(_) => {}

                // sync is unecessary if the other node has done nothing since the last time we
                // synced with them.
                //
                // TODO: two syncs with the same node right after each other is also
                // unnecessary.
                Action::Sync(oid) if id == oid => {
                    // we _shouldn't_ be generating self-syncs, but handle them just in case.
                    keep = false;
                }
                Action::Sync(oid) => {
                    let other_last = previous_for_node.get(&oid).cloned();
                    // if the last thing _we_ did was a sync
                    if let Some(&(synci, Action::Sync(loid))) = previous_for_node.get(&id) {
                        // and that sync was with the same node
                        if loid == oid {
                            // and _that_ node didn't do anything in between
                            if other_last.is_none_or(|(lasti, _)| lasti < synci) {
                                keep = false;
                            }
                        }
                    }
                }
            }
            if keep {
                previous_for_node.insert(id, (i, op.action.clone()));
            }
            keep
        });
    }
}

impl<DS> fmt::Display for Ops<DS>
where
    DS: ArbitraryDelta,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, op) in self.ops.iter().enumerate() {
            if i != 0 && i % 10 == 0 {
                writeln!(f, "--- {i} ---")?;
            }
            let id = op.by;
            match &op.action {
                Action::Data(op) => {
                    writeln!(f, " -> {id:?} {op}")?;
                }
                Action::Sync(i2) => writeln!(f, " -> {id:?} syncs from {i2:?}")?,
            }
        }

        Ok(())
    }
}

impl<DS> Ops<DS>
where
    DS: DotStoreJoin<RecordingSentinel>,
    DS: ArbitraryDelta + fmt::Debug,
{
    /// Produces a sequence of CRDTs for the operations in `self` and checks that the CRDTs resolve
    /// to the same final state no matter how they are combined.
    #[cfg_attr(feature = "arbitrary", allow(dead_code))]
    pub fn check_order_invariance(self, seed: u64) -> quickcheck::TestResult
    where
        DS: DotStoreJoin<DummySentinel> + Default + Clone + PartialEq,
    {
        eprintln!("\n:: Running test case:");
        eprint!("{self}");
        eprintln!("==> Executing steps");

        if self.ops.is_empty() {
            return quickcheck::TestResult::passed();
        }

        // we need to construct a new, fresh KeyTracker since shrinking may have happened between
        // when this [`Ops`] was generated and when this function is called. if it has, different
        // keys may end up being generated, and we need to allow for that. note also that we use a
        // single KeyTracker across all actors. this is needed so that node A can resolve a keyi
        // in a Delta for a key that was inserted by node B after A has synced from B.
        let mut keys = KeyTracker::default();

        // set up a new, empty state for a node that hasn't seen anything
        let fresh = CausalDotStore::<DS>::new();

        // apply all ops to synthesize a final state with local modifications and all deltas
        let (final_state, ordered_deltas) = self.ops.into_iter().enumerate().fold(
            (HashMap::new(), vec![]),
            |(mut causals, mut deltas), (opi, op)| {
                eprintln!(" -> executing #{opi}");

                // grab the executing node's internal state
                let id = op.by;
                let mut causal = causals
                    .entry(id)
                    .or_insert_with(|| CausalDotStore::<DS>::new());

                // generate the crdt for the indicated operation
                match op.action {
                    Action::Data(op) => {
                        let modded: CausalDotStore<DS> =
                            op.into_crdt(&causal.store, &causal.context, id, &mut keys);

                        // merge the crdt into the joined state _and_ keep track of the delta
                        causal
                            .join_with(modded.store.clone(), &modded.context, &mut DummySentinel)
                            .unwrap();

                        deltas.push(modded);
                    }
                    Action::Sync(other) => {
                        let other = causals
                            .entry(other)
                            .or_insert_with(|| CausalDotStore::<DS>::new())
                            .clone();

                        // re-borrow causal so we get to temporarily borrow causals above
                        causal = causals.get_mut(&id).unwrap();

                        // pull out the delta to inflate this node to match the other's state
                        let mut inflate = other.subset_for_inflation_from(&causal.context);
                        inflate
                            .context
                            .union(&compute_deletions_unknown_to(ComputeDeletionsArg {
                                known_dots: &other.context,
                                live_dots: &other.store.dots(),
                                ignorant: &causal.store.dots(),
                            }));

                        // while we're at it, check that inflation subset has the same effect as
                        // merging with the full state
                        let mut full_sync = causal.clone();
                        full_sync
                            .join_with(other.store.clone(), &other.context, &mut DummySentinel)
                            .unwrap();
                        let mut partial_sync = causal.clone();
                        partial_sync
                            .join_with(inflate.store.clone(), &inflate.context, &mut DummySentinel)
                            .unwrap();
                        assert_eq!(full_sync, partial_sync, "inflating with {inflate:?}");
                        *causal = partial_sync;

                        // TODO: test that if a node syncs an "overheard" ::Update, it doesn't
                        // botch their state (esp. around deletions).

                        deltas.push(inflate);
                    }
                }

                (causals, deltas)
            },
        );

        // compute a final state that joins the final state of all the nodes
        let final_state = final_state
            .into_values()
            .reduce(|acc, delta| acc.join(delta, &mut DummySentinel).unwrap())
            .unwrap();

        // apply all CRDTs one by one to the initial state in-order
        let state_ordered = ordered_deltas
            .clone()
            .into_iter()
            .fold(fresh.clone(), |acc, delta| {
                acc.join(delta, &mut DummySentinel).unwrap()
            });

        // merge all CRDTs into a single one in-order, and then apply that to the initial state
        let merged_delta = ordered_deltas
            .clone()
            .into_iter()
            .reduce(|acc, delta| acc.join(delta, &mut DummySentinel).unwrap())
            .unwrap();
        let state_ordered_merged = {
            let mut fresh = fresh.clone();
            fresh
                .join_with(
                    merged_delta.store,
                    &merged_delta.context,
                    &mut DummySentinel,
                )
                .unwrap();
            fresh
        };

        // apply all CRDTs one by one to the initial state in random order
        let shuffled_deltas = {
            use rand::{SeedableRng, seq::SliceRandom};
            let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

            let mut tmp = ordered_deltas;
            tmp.shuffle(&mut rng);
            tmp
        };
        let state_shuffled = shuffled_deltas.into_iter().fold(fresh, |acc, delta| {
            acc.join(delta, &mut DummySentinel).unwrap()
        });

        // moment of truth -- are they all the same (ie, eventually consistent)?
        quickcheck::TestResult::from_bool(
            dbg!(final_state == state_ordered_merged)
                && dbg!(final_state == state_ordered)
                && dbg!(final_state == state_shuffled),
        )
    }
}

/// An iterator that produces smaller-but-still-legal versions of an [`Ops`].
///
/// The "but still legal" part makes shrinking tricky since we're working with a distributed
/// systems trace. This shrinker currently explores two shrinking dimensions of [`Ops`]:
///
/// - the length of the operational sequence, since every prefix is valid by construction.
/// - the set of keys that are used, since every trace remains valid if you take out all operations
///   pertaining to a particular key (ie, there are no cross-key dependencies).
struct OpsShrinker<DS>
where
    DS: ArbitraryDelta,
{
    /// The original `Ops` that we're shrinking.
    seed: Ops<DS>,
    /// The subset of keys ([..key_subset]) we're currently including.
    key_subset: usize,
    /// The subset of ops ([..size]) we're currently including.
    size: usize,
}

impl<DS> Iterator for OpsShrinker<DS>
where
    DS: ArbitraryDelta,
    DS::Delta: Clone,
{
    type Item = Ops<DS>;

    fn next(&mut self) -> Option<Self::Item> {
        // the general guidance for quickcheck shrinking appears to be to create "smaller" inputs
        // first. my understanding is that quickcheck walks the iterator until it first replicates
        // a failure. then it discards the rest of that iterator and instead shrinks starting from
        // that replication point. naturally it follows that it's best to produce the "most
        // reduced" candidates first.
        //
        // NOTE: we only generate items that are strictly smaller along at least one
        // dimension. that means we'll never here yield `seed` again, as that would just lead to
        // infinite recursion.

        // try to see how an empty set of ops does first
        if self.size == 0 {
            self.size = 1;
            return Some(Ops {
                ops: Vec::new(),
                nkeys: 0,
            });
        }

        // first, try to yield a short (and then longer and longer) prefix of ops, as fewer ops
        // means simpler traces. it has to be a prefix since only prefixes are guaranteed to still
        // hold only legal operations.
        if self.size < self.seed.ops.len() {
            let ops = Vec::from(&self.seed.ops[..self.size]);
            // NOTE: grow by 2x to avoid very slow shrinking
            self.size *= 2;
            return Some(Ops {
                ops,
                nkeys: self.seed.nkeys,
            });
        }

        // if we get here it means no shorter prefix reproduces the problem. so, instead of
        // reducing the trace length, we reduce its "breadth". specifically, we keep only ops
        // related to a subset of the keys, starting with just a single key. this works because
        // there is no dependence between the state of different keys, so erasing all ops for a
        // particular key leaves the ops on other keys still in a legal state.
        //
        // NOTE: there is one caveat to this, which is that removing the insertion of a key
        // would alter the observed keyi for all subsequent. we solve for that by eliminating keys
        // by index. ie, we will remove all the keys with keyi >= n, which ensures that there are
        // no later keyis that can be affected by the lack of an insert.
        if self.key_subset < self.seed.nkeys {
            let nkeys = self.key_subset;
            // NOTE: grow by 2x to avoid very slow shrinking
            self.key_subset *= 2;
            let ops = self
                .seed
                .ops
                .iter()
                .filter(|op| {
                    if let Action::Data(d) = &op.action {
                        !d.depends_on_keyi_in(nkeys..)
                    } else {
                        true
                    }
                })
                .cloned()
                .collect();

            // re-prune since removing key ops may cause clears/syncs to now appear right after
            // each other, and thus become redundant.
            let mut s = Ops { ops, nkeys };
            s.prune_unnecessary();
            return Some(s);
        }

        // TODO: other prune opportunities:
        // - all operations each node does after it is last synced _from_

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        OrMap,
        crdts::{
            NoExtensionTypes,
            mvreg::MvRegValue,
            test_util::arbitrary_delta_impls::{MapOp, RegisterOp, ValueDelta},
        },
    };
    use quickcheck::Arbitrary;

    #[test]
    fn shrink_empty() {
        let ops = Ops::<OrMap<String, NoExtensionTypes>> {
            ops: vec![],
            nkeys: 0,
        };
        assert_eq!(ops.shrink().count(), 0);
    }

    #[test]
    fn shrink_single_key() {
        let ops = Ops::<OrMap<String, NoExtensionTypes>> {
            ops: vec![
                Op {
                    by: (0, 0).into(),
                    action: Action::Data(MapOp::Apply(
                        0,
                        Some(String::from("key0")),
                        Box::new(ValueDelta::Register(RegisterOp(Some(MvRegValue::Bool(
                            true,
                        ))))),
                    )),
                },
                Op {
                    by: (0, 0).into(),
                    action: Action::Data(MapOp::Apply(
                        0,
                        None,
                        Box::new(ValueDelta::Register(RegisterOp(Some(MvRegValue::Bool(
                            false,
                        ))))),
                    )),
                },
            ],
            nkeys: 1,
        };

        // we should see 2 shrinks:
        let shrinks: Vec<_> = ops.shrink().collect();
        // first, an empty set of ops
        assert_eq!(shrinks[0].ops.len(), 0);
        // then, only 1 of the 2 ops
        assert_eq!(shrinks[1].ops.len(), 1);
        assert!(matches!(
            shrinks[1].ops[0].action,
            Action::Data(MapOp::Apply(_, Some(_), _))
        ));
        // then, no more (2/2 ops with all the keys present wouldn't be a shrink)
        assert_eq!(shrinks.len(), 2);
    }

    #[test]
    fn shrink_keys() {
        let ops = Ops::<OrMap<String, NoExtensionTypes>> {
            ops: vec![
                Op {
                    by: (0, 0).into(),
                    action: Action::Data(MapOp::Apply(
                        0,
                        Some(String::from("key0")),
                        Box::new(ValueDelta::Register(RegisterOp(Some(MvRegValue::Bool(
                            true,
                        ))))),
                    )),
                },
                Op {
                    by: (0, 0).into(),
                    action: Action::Data(MapOp::Apply(
                        1,
                        Some(String::from("key1")),
                        Box::new(ValueDelta::Register(RegisterOp(Some(MvRegValue::Bool(
                            true,
                        ))))),
                    )),
                },
                Op {
                    by: (0, 0).into(),
                    action: Action::Data(MapOp::Apply(
                        0,
                        None,
                        Box::new(ValueDelta::Register(RegisterOp(Some(MvRegValue::Bool(
                            false,
                        ))))),
                    )),
                },
            ],
            nkeys: 2,
        };

        // we should see 4 shrinks:
        let shrinks: Vec<_> = ops.shrink().collect();
        // first, an empty set of ops
        assert_eq!(shrinks[0].ops.len(), 0);
        // then, only 1 of the 3 ops
        assert_eq!(shrinks[1].ops.len(), 1);
        assert!(matches!(
            shrinks[1].ops[0].action,
            Action::Data(MapOp::Apply(0, Some(_), _))
        ));
        // then, only 2 of the 3 ops
        assert_eq!(shrinks[2].ops.len(), 2);
        assert!(matches!(
            shrinks[2].ops[0].action,
            Action::Data(MapOp::Apply(0, Some(_), _))
        ));
        assert!(matches!(
            shrinks[2].ops[1].action,
            Action::Data(MapOp::Apply(1, Some(_), _))
        ));
        // then, one with half the keys pruned (so only op[0] and op[2])
        assert_eq!(shrinks[3].ops.len(), 2);
        assert!(matches!(
            shrinks[3].ops[0].action,
            Action::Data(MapOp::Apply(0, Some(_), _))
        ));
        assert!(matches!(
            shrinks[3].ops[1].action,
            Action::Data(MapOp::Apply(0, None, _))
        ));
        // then, no more
        assert_eq!(shrinks.len(), 4);
    }
}
