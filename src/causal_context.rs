// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! # Causal Context
//!
//! This module provides the core data structures for tracking causality in DSON.
//! Causal consistency is maintained by tracking the history of operations using
//! `Dot`s, which are globally unique identifiers for each operation. The set of
//! all observed dots forms a `CausalContext`.
//!
//! - **[`Identifier`]**: A unique identifier for an actor in the system. It consists of
//!   of a `NodeId` and an `ApplicationId` to distinguish between different
//!   applications running on the same node.
//!
//! - **[`Dot`]**: A globally unique identifier for a single operation (for example, an insert
//!   or update). It consists of an `Identifier` and a sequence number, which is
//!   monotonically increasing for that specific actor.
//!
//! - **[`CausalContext`]**: A data structure that stores the set of all `Dot`s that a
//!   replica has observed. It represents the replica's knowledge of the system's
//!   history. By comparing `CausalContext`s, replicas can determine which
//!   operations are new, concurrent, or have already been seen, enabling correct
//!   merging of states.
//!
//! The `CausalContext` is implemented using a `BTreeMap` of `Identifier`s to
//! `IntervalSet`s, which efficiently stores contiguous ranges of sequence numbers.
//! This allows for a compact representation of the causal history.
use self::interval::{Interval, IntervalSet};
use interval::IntervalError;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, btree_map::Entry},
    fmt,
    num::NonZeroU64,
    ops::{BitAnd, Sub},
};

mod interval;

/// Maximum representable application id.
///
/// Note that id 0 is reserved, so the number of applications that can be
/// registered is one lower.
pub const MAX_APPLICATION_ID: u16 = (1 << 12) - 1;

/// Error returned when attempting to create an [`Identifier`] from bits with an invalid [`Priority`]
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct InvalidPriority(pub u8);

impl fmt::Display for InvalidPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid priority {}", self.0)
    }
}

impl std::error::Error for InvalidPriority {}

/// Error returned when attempting to create an [`Identifier`] from an invalid bits sequence
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum IdentifierError {
    /// Invalid priority
    Priority(InvalidPriority),

    /// Bits extracted from the value are invalid
    InvalidBits { field: &'static str, value: u32 },
}

impl fmt::Display for IdentifierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdentifierError::Priority(err) => write!(f, "{err}"),
            IdentifierError::InvalidBits { field, value } => {
                write!(f, "invalid value {value} for field {field}")
            }
        }
    }
}

impl std::error::Error for IdentifierError {}

impl From<InvalidPriority> for IdentifierError {
    fn from(value: InvalidPriority) -> Self {
        Self::Priority(value)
    }
}

/// Indicates the priority level a given CRDT update is associated with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[repr(u8)] // really 3 bits, so max value is 7
pub enum Priority {
    /// Update that should not leave the current node.
    Local = 0,
    /// Update that should be synchronized after all others.
    Low = 2,
    /// Update that should be synchronized as necessary.
    Medium = 4,
    /// Update that should be synchronized ahead of all others.
    High = 6,
}
pub const PRIORITY_MAX: Priority = Priority::High;

impl TryFrom<u8> for Priority {
    type Error = InvalidPriority;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Ok(match value {
            0 => Priority::Local,
            2 => Priority::Low,
            4 => Priority::Medium,
            6 => Priority::High,
            _ => return Err(InvalidPriority(value)),
        })
    }
}

/// The application-id used for the root document.
///
/// When [`Identifier`] instances are used to identify a node, they use this application value.
pub const ROOT_APP_ID: u16 = 0;

/// The identifier we choose to use for actors in the system.
///
/// It is space-efficient and is passed around _everywhere_.
///
/// This identifier is composed of a node identifier, and an application identifier.
/// This is so that all the applications running _on_ a node can be modeled as
/// independent actors as far as the CRDT is concerned.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
#[repr(transparent)]
pub struct Identifier {
    /// ```text
    ///  0      7 8         19   21  24       32
    /// +--------+------------+-+---+--------+
    /// |   N    |     A      |R| P | unused |
    /// +--------+------------+-+---+--------+
    /// ```
    ///
    /// - N: node identifier (8 bits, so 256 nodes)
    /// - A: application identifier (12 bits, so 4096 applications per node)
    /// - R: reserved bit (should be 0)
    /// - P: priority (3 bits, so 8 priority levels)
    ///
    /// Note, bit 0 is the most significant bit in the diagram above.
    /*
     * YADR: 2024-04-19 Allocation of identifier bits
     *
     * In the context of reducing bandwidth footprint, we faced the question of how we should
     * represent participant identifiers in memory.
     *
     * We decided for making them stored as a packed 24-bit integer in a 32-bit integer with 8 bits
     * for node IDs, 12 bits for application IDs, and 3 bits for priority levels, and neglected
     * using structured types, tuples, differently-sized integer types, or other allocations of
     * bits.
     *
     * We did this to achieve a balance between the size of the identifier space (ie, how many
     * unique identifiers we can have) and the amount of bytes we need to send over the wire,
     * accepting a relatively small but realistic hard limit on the number of nodes, applications,
     * and priority levels.
     *
     * We think this is the right trade-off because identifiers are included _everywhere_ and thus
     * even a single excess byte translates into potentially kB included in CRDT deltas.
     *
     * We expect that there should never be more than 256 nodes (8 bits) in a single network --
     * beyond that size the network should instead be multiple smaller networks that selectively
     * exchange (and potentially aggregate) data.
     *
     * We expect that there should never be more than 4096 applications (12 bits) connected to a
     * single instance. If there is, it suggests a broken use pattern where applications are
     * constantly establishing new connections, which comes with its own set of problems and should
     * be discouraged. We know that browsers sometime recycle connections, but even with occasional
     * recycling, 4k seems like a reasonable limit between "acceptable" and "something needs to
     * change".
     *
     * We expect that 8 priority levels (3 bits) should be sufficient for most applications. Even 8
     * is potentially excessive, as it already captures high, medium, low, and local with four
     * levels to spare. It seems unlikely to us that finer-grained priorities (like 15 being
     * different to 16) are worth the extra bits. We did not limit to 2 bits so that we _do_ have
     * some room for growing requirements.
     *
     * We left a single reserved bit for unforeseen future uses, rather than allocating them to any
     * of the aforementioned limits (which we think are all reasonable already).
     *
     * We stayed within 24 bits so that identifiers can be sent as just 3 bytes on the wire rather
     * than 4, which potentially saves a significant amount of bandwidth if hundreds are sent in a
     * delta.
     */
    bits: u32,
}

/// Custom implementation that renders the virtual components of this struct.
impl std::fmt::Debug for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.priority() {
            Priority::High => write!(f, "@{}.{}h", self.node(), self.app()),
            Priority::Medium => write!(f, "@{}.{}", self.node(), self.app()),
            Priority::Low => write!(f, "@{}.{}l", self.node(), self.app()),
            Priority::Local => write!(f, "@{}.{}-", self.node(), self.app()),
        }
    }
}

impl From<(u8, u16)> for Identifier {
    fn from((node, application): (u8, u16)) -> Self {
        Identifier::new(node, application)
    }
}

impl PartialEq<(u8, u16)> for Identifier {
    fn eq(&self, &(node, application): &(u8, u16)) -> bool {
        self == &Identifier::new(node, application)
    }
}

impl TryFrom<u32> for Identifier {
    type Error = IdentifierError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        const BIT_FIELDS: [(u32, u32); 5] = [(0, 7), (8, 19), (20, 20), (21, 23), (24, 31)];

        let [_node_id, _app_id, reserved, priority, unused] = BIT_FIELDS.map(|(start, end)| {
            let bits_count = end - start + 1;
            let mask = (!0u32) >> (u32::BITS - bits_count);
            let shift = u32::BITS - end - 1;
            (value >> shift) & mask
        });

        let _priority = Priority::try_from(priority as u8)?;
        if reserved != 0 {
            return Err(IdentifierError::InvalidBits {
                field: "reserved",
                value: reserved,
            });
        }

        if unused != 0 {
            return Err(IdentifierError::InvalidBits {
                field: "unused",
                value: unused,
            });
        }

        Ok(Self { bits: value })
    }
}

impl Identifier {
    /// Constructs a new Identifier for the given node-application pair.
    ///
    /// Application must be a valid u12 (meaning a u16 with the high four bits unset), or the
    /// function will panic.
    pub const fn new(node: u8, application: u16) -> Self {
        if application > MAX_APPLICATION_ID {
            // NOTE: cannot print the value since we're in a const fn
            panic!("application exceeds u12");
        }
        Self {
            bits: ((node as u32) << (12 + 1 + 3 + 8)) | ((application as u32) << (1 + 3 + 8)),
        }
        .with_priority(Priority::Medium)
    }

    /// Get the representable 'next larger' Identifier.
    /// Returns None if no such identifier exists.
    /// This does not take priority into account.
    pub fn checked_successor(self) -> Option<Identifier> {
        if self.app() != MAX_APPLICATION_ID {
            return Some(Identifier::new(self.node().value(), self.app() + 1));
        }
        if self.node() != NodeId::MAX {
            return Some(Identifier::new(self.node().value() + 1, self.app()));
        }
        None
    }

    pub const fn node(&self) -> NodeId {
        NodeId {
            node_id: (self.bits >> (12 + 1 + 3 + 8)) as u8,
        }
    }

    pub const fn app(&self) -> u16 {
        ((self.bits >> (1 + 3 + 8)) & 0xfff) as u16
    }

    pub const fn priority(&self) -> Priority {
        let bits = ((self.bits >> 8) & 0b111) as u8;
        match bits {
            0 if Priority::Local as u8 == 0 => Priority::Local,
            2 if Priority::Low as u8 == 2 => Priority::Low,
            4 if Priority::Medium as u8 == 4 => Priority::Medium,
            6 if Priority::High as u8 == 6 => Priority::High,
            _ if cfg!(debug_assertions) => {
                // NOTE: cannot print bits since we're in a const fn
                panic!("illegal priority")
            }
            // SAFETY: it's only possible to set the priority bits using methods on `Identifier`,
            // and those all take `Priority`, whose values we check for above. should we have
            // missed any (ie, because `Priority` was modified), that'll be caught at test time
            // with the branch above (+ quickcheck). at release time, this shouldn't be reachable.
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    // TODO: have this be more like a `resolve_priority` method that decides whether to use
    // the provided priority or the already-set one. we could have a number of heuristics for how
    // to decide (probably: last wins). we'll probably want to also include a `bool` argument here
    // to indicate "was overridden", which should take precedence over something from the schema,
    // and we can maybe tuck that flag into the one remaining bit we have in `Identifier.0`.
    pub const fn with_priority(self, priority: Priority) -> Self {
        Self {
            bits: (self.bits & !(0b111 << 8)) | ((priority as u32) << 8),
        }
    }

    pub const fn bits(&self) -> u32 {
        self.bits
    }
}

/// A unique identifier for a single node in the network.
///
/// All applications in a single node have the same NodeId,
/// even though they have different [`Identifier`]s.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct NodeId {
    node_id: u8,
}

impl NodeId {
    pub const MAX: NodeId = NodeId::new(u8::MAX);

    pub const fn value(self) -> u8 {
        self.node_id
    }

    pub const fn new(node_id: u8) -> Self {
        NodeId { node_id }
    }

    /// Returns the main-identifier for this node (that is, for application 0)
    pub fn identifier(self) -> Identifier {
        Identifier::new(self.node_id, ROOT_APP_ID)
    }
}

impl std::fmt::Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.node_id)
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.node_id, f)
    }
}

/// A unique identifier for an operation.
///
/// Every DSON operation is assigned a unique operation in the form of a `Dot`. These are a
/// combination of a unique node identifier and an ever-increasing sequence number.
///
/// Dots are ordered by the sequence number _first_ and _then_ the actor identifier.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct Dot(Identifier, NonZeroU64);

impl std::fmt::Debug for Dot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?}, {})", self.0, self.1)
    }
}

impl<I> From<(I, NonZeroU64)> for Dot
where
    I: Into<Identifier>,
{
    fn from((id, seq): (I, NonZeroU64)) -> Self {
        Self(id.into(), seq)
    }
}

impl Dot {
    /// Get the 'next' larger dot.
    /// This method never changes the identifier, it only increments the sequence number.
    /// This wraps around, keeping the same id, in case of overflow.
    pub fn successor(&self) -> Dot {
        Dot::mint(self.0, self.1.get().wrapping_add(1).max(1))
    }

    /// Creates a new [`Dot`] out of thin air.
    ///
    /// All real dots should be made through the use of a [`CausalContext`].
    /// This constructor is mainly useful for tests and documentation examples.
    ///
    /// # Panics
    ///
    /// If `seq == 0`.
    pub const fn mint(id: Identifier, seq: u64) -> Self {
        Self(
            id,
            if let Some(seq) = NonZeroU64::new(seq) {
                seq
            } else {
                panic!("attempted to construct Dot for 0th sequence number");
            },
        )
    }

    pub const fn with_priority(self, priority: Priority) -> Self {
        Self(self.0.with_priority(priority), self.1)
    }

    /// Returns the [`Identifier`] of the actor that produced this [`Dot`].
    pub fn actor(&self) -> Identifier {
        self.0
    }

    /// Returns the sequence number (ie, per-actor operation index) of this [`Dot`].
    pub fn sequence(&self) -> NonZeroU64 {
        self.1
    }
}

impl PartialEq<(Identifier, u64)> for Dot {
    fn eq(&self, other: &(Identifier, u64)) -> bool {
        self.0 == other.0 && self.1.get() == other.1
    }
}

impl PartialEq<((u8, u16), u64)> for Dot {
    fn eq(&self, other: &((u8, u16), u64)) -> bool {
        self.0 == Identifier::from(other.0) && self.1.get() == other.1
    }
}

/// Tracks the set of sequence numbers observed from each actor in the system.
///
/// This type can be used both to track observed causal context, and to produce new `Dot`s. If only
/// needed for the former, construct using [`CausalContext::default()`]. To produce new
/// `Dot`s as well, use the [`CausalContext::new`] constructor to also supply the
/// current actor's identifier.
///
/// # Examples
///
/// ## Producing [`Dot`]s
///
/// ```rust
/// # use dson::{CausalContext, Dot, Identifier};
/// let id = Identifier::new(0, 0);
/// let mut cause = CausalContext::new();
///
/// // The causal context can be used to produce new dots:
/// let dot1 = cause.next_dot_for(id);
/// // New dots are not implicitly absorbed:
/// assert_eq!(cause.next_dot_for(id), dot1);
/// // You must explicitly add them to generate newer dots:
/// cause.insert_next_dot(dot1);
/// let dot2 = cause.next_dot_for(id);
/// assert_ne!(dot1, dot2);
/// cause.insert_next_dot(dot2);
///
/// // The first dot produced will have sequence number 1:
/// assert_eq!(dot1, ((0, 0), 1));
///
/// // If one dot is produced after another, it is also ordered after:
/// assert!(dot2 > dot1);
///
/// // The causal context considers any dot produced as observed:
/// assert!(cause.dot_in(dot1));
/// assert!(cause.dot_in(dot2));
/// ```
///
/// ## Tracking causal context
///
/// ```rust
/// # use dson::{CausalContext, Dot};
/// let mut cause = CausalContext::default();
///
/// // With no observed causal relationships, no dots are in the context:
/// assert!(!cause.dot_in(Dot::mint((0, 0).into(), 1)));
///
/// // Once a dot is observed, that dot is in the causal context, but no others:
/// cause.extend([Dot::mint((0, 0).into(), 1)]);
/// assert!(cause.dot_in(Dot::mint((0, 0).into(), 1)));
/// assert!(!cause.dot_in(Dot::mint((0, 0).into(), 2)));
/// assert!(!cause.dot_in(Dot::mint((1, 0).into(), 1)));
///
/// // The context can track causal context across multiple nodes:
/// cause.extend([Dot::mint((1, 0).into(), 1)]);
/// assert!(cause.dot_in(Dot::mint((0, 0).into(), 1)));
/// assert!(cause.dot_in(Dot::mint((1, 0).into(), 1)));
///
/// // and the context can track out-of-order dots:
/// cause.extend([Dot::mint((0, 0).into(), 10)]);
/// assert!(cause.dot_in(Dot::mint((0, 0).into(), 1)));
/// assert!(cause.dot_in(Dot::mint((0, 0).into(), 10)));
/// assert!(!cause.dot_in(Dot::mint((0, 0).into(), 2)));
/// assert!(!cause.dot_in(Dot::mint((0, 0).into(), 9)));
/// assert!(!cause.dot_in(Dot::mint((0, 0).into(), 11)));
///
/// // If more consecutive dots from the same actor are observed, they are stored compactly:
/// let before = cause.size();
/// cause.extend([
///     Dot::mint((0, 0).into(), 2),
///     Dot::mint((0, 0).into(), 3)
/// ]);
/// assert_eq!(before, cause.size());
/// assert!(cause.dot_in(Dot::mint((0, 0).into(), 1)));
/// assert!(cause.dot_in(Dot::mint((0, 0).into(), 2)));
/// assert!(cause.dot_in(Dot::mint((0, 0).into(), 3)));
/// assert!(cause.dot_in(Dot::mint((1, 0).into(), 1)));
/// assert!(!cause.dot_in(Dot::mint((0, 0).into(), 4)));
/// assert!(!cause.dot_in(Dot::mint((1, 0).into(), 2)));
/// ```
// TODO: here we spend a lot of time accessing map entries in batch operations. we could probably do
// a lot better with btree cursors, which unfortunately is only on nightly right now. see
// https://github.com/rust-lang/rust/issues/107540.
// Also, we can add a temporary dot cloud as a way to buffer inserts before
// applying them as a batch operation. This would require significant changes in
// many places, including `partial_cmp_dots`.
// TODO: check if `self.dots` is compacted on serialization.
#[derive(Default, Clone)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct CausalContext {
    dots: BTreeMap<Identifier, IntervalSet>,
}

impl std::fmt::Debug for CausalContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CausalContext").field(&self.dots).finish()
    }
}

impl CausalContext {
    /// Constructs a new [`CausalContext`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a [`CausalContext`] from a sequence of raw intervals
    pub fn from_intervals<I, RI>(iter: I) -> Result<Self, IntervalError>
    where
        I: IntoIterator<Item = (Identifier, RI)>,
        RI: IntoIterator<Item = (NonZeroU64, Option<NonZeroU64>)>,
    {
        let mut dots = BTreeMap::new();
        for (id, bounds) in iter {
            dots.insert(id, IntervalSet::from_intervals(bounds)?);
        }

        Ok(Self { dots })
    }

    /// Produces the next unused [`Dot`] for this node.
    pub fn next_dot_for(&self, id: Identifier) -> Dot {
        self.next_n_dots_for(1, id)
            .next()
            .expect("yields as many as are indicated")
    }

    /// Attempt to find an Identifier that is not present in this causal context.
    /// If one exists, it is returned. Otherwise, None is returned.
    pub fn unused_identifier(&self) -> Option<Identifier> {
        let mut candidate = Identifier::new(0, 0);
        for key in self.dots.keys() {
            if candidate < *key {
                // candidate is always set to the next representable Identifier. If
                // this is not the next key in the map, we have found an unused identifier.
                // It works for the first element, since we initialize candidate to (0,0),
                // and if the first key in the map isn't (0,0), then (0,0) is obviosly unused.
                return Some(candidate);
            }
            candidate = key.checked_successor()?;
        }
        Some(candidate)
    }

    /// Returns the largest sequence number that exists in the causal context
    /// for each identifier belonging to a single `node`.
    pub fn largest_for_node(&self, node: u8) -> impl Iterator<Item = Dot> + '_ {
        self.dots
            .range(Identifier::new(node, 0)..=Identifier::new(node, MAX_APPLICATION_ID))
            .filter_map(|(k, v)| v.last().map(|last_element| Dot(*k, last_element.end())))
    }

    /// Produces `n` unused [`Dot`]s for this node.
    pub fn next_n_dots_for(
        &self,
        n: u8,
        id: Identifier,
    ) -> impl Iterator<Item = Dot> + Clone + use<> {
        let spans = self.dots.get(&id);

        // NOTE: this method and its use implicitly assumes that a node's sequence numbers are
        // always produced in sequence _and_ always compacted. let's make sure that that's actually
        // the case:
        debug_assert!(
            spans
                .map(|seqs| seqs.len() == 1
                    && seqs.first().expect("not empty").start() == NonZeroU64::MIN)
                .unwrap_or(true),
            "dots for self.id are not sequential and compacted in {:?}",
            self.dots
        );

        let next_seq = spans
            .map(IntervalSet::next_after)
            .unwrap_or(NonZeroU64::MIN);

        // TODO: avoid the extra NonZero wrapping once we get
        // https://github.com/rust-lang/libs-team/issues/130
        (next_seq.get()..next_seq.get() + u64::from(n)).map(move |seq| {
            // SAFETY: the start of the range was a NonZeroU64, and we're adding unsigned u8s
            let seq = unsafe { NonZeroU64::new_unchecked(seq) };
            Dot(id, seq)
        })
    }

    #[cfg(test)]
    pub fn dots_for(&self, id: Identifier) -> impl Iterator<Item = Dot> + '_ {
        let dots: Vec<_> = self
            .dots
            .get(&id)
            .iter()
            .flat_map(|ivals| ivals.seqs().map(|seq| Dot::mint(id, seq.get())))
            .collect();
        dots.into_iter()
    }

    /// Iterator over all the dots that the context holds
    pub fn dots(&self) -> impl Iterator<Item = Dot> + '_ {
        self.dots
            .iter()
            .flat_map(|(id, ivals)| ivals.seqs().map(|seq| (*id, seq).into()))
    }

    /// True if there are no dots in this causal context.
    pub fn is_empty(&self) -> bool {
        debug_assert!(
            self.dots.values().all(|v| !v.is_empty()),
            "should not retain empty interval sets"
        );
        self.dots.is_empty()
    }

    /// The approximate size of this causal context including compaction.
    pub fn size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.dots.len()
                * (std::mem::size_of::<Identifier>() + std::mem::size_of::<IntervalSet>())
            + self
                .dots
                .values()
                .map(|ivals| ivals.len() * std::mem::size_of::<Interval>())
                .sum::<usize>()
    }

    /// Return the total number of dots.
    #[must_use]
    pub fn dot_count(&self) -> u64 {
        self.dots
            .values()
            .map(|ivals| ivals.total_interval_length())
            .sum()
    }

    /// Determines if the given `dot` is in the current causal context.
    #[must_use]
    pub fn dot_in(&self, dot: Dot) -> bool {
        self.dots
            .get(&dot.actor())
            .is_some_and(|s| s.contains(dot.sequence()))
    }

    /// Returns an arbitrary [`Dot`] among those in this context.
    ///
    /// No guarantee is given about which [`Dot`] is returned if there are multiple.
    pub fn one(&self) -> Option<Dot> {
        self.dots
            .iter()
            .flat_map(|(id, ivals)| ivals.first().map(|ival| Dot::from((*id, ival.start()))))
            .next()
    }

    pub fn is_compact_for_node(&self, node: u8) -> bool {
        self.dots
            .range(
                Identifier::new(node, 0)
                    ..=Identifier::new(node, MAX_APPLICATION_ID).with_priority(PRIORITY_MAX),
            )
            .all(|(_, spans)| {
                spans.len() == 1 && spans.first().expect("not empty").start() == NonZeroU64::MIN
            })
    }

    /// Records a new observed [`Dot`] in the causal context.
    ///
    /// Will not compact automatically.
    pub fn insert_dot(&mut self, dot: Dot) {
        self.dots
            .entry(dot.actor())
            .or_insert_with(IntervalSet::new)
            .insert(dot.sequence());
    }

    /// Records a newly generated [`Dot`] in the causal context.
    pub fn insert_next_dot(&mut self, dot: Dot) {
        match self.dots.entry(dot.actor()) {
            Entry::Vacant(v) => {
                assert_eq!(dot.sequence(), NonZeroU64::MIN);
                v.insert(IntervalSet::single(dot.sequence()));
            }
            Entry::Occupied(mut o) => {
                let next = o.get().next_after();
                assert_eq!(dot.sequence(), next);
                o.get_mut().extend_end_by_one();
            }
        }
    }

    /// Records multiple observed [`Dot`] in the causal context.
    ///
    /// Will not compact automatically.
    pub(crate) fn insert_dots(&mut self, dots: impl IntoIterator<Item = Dot>) {
        // TODO: batching would really help here, since the `Entry` API is a
        // huge bottleneck. we could just temporarily stash dots in a dot cloud
        // using something like a `BTreeSet` and then later make a single call
        // to `.entry` per actor when compacting (since dots will be sorted by
        // actor).
        for dot in dots {
            match self.dots.entry(dot.actor()) {
                Entry::Vacant(v) => {
                    v.insert(IntervalSet::single(dot.sequence()));
                }
                Entry::Occupied(mut o) => {
                    o.get_mut().insert(dot.sequence());
                }
            }
        }
    }

    /// Removes the given `dot` from the causal context.
    ///
    /// Returns `true` if the `dot` was in the causal context.
    pub fn remove_dot(&mut self, dot: Dot) -> bool {
        let Some(ivals) = self.dots.get_mut(&dot.actor()) else {
            return false;
        };
        let removed = ivals.remove(dot.sequence());
        if ivals.is_empty() {
            self.dots.remove(&dot.actor());
        }
        removed
    }

    /// Removes all the dots in the given causal context from this causal context.
    pub fn remove_dots_in(&mut self, remove: &CausalContext) {
        self.dots.retain(|k, v1| {
            if let Some(v2) = remove.dots.get(k) {
                *v1 = v1.difference(v2);
                !v1.is_empty()
            } else {
                true
            }
        })
    }

    /// Incorporates the observations from another causal context into this one.
    ///
    /// After the `union`, all [`Dot`]s known to `other` will be considered observed by `self`.
    ///
    /// ```rust
    /// # use dson::{CausalContext, Dot};
    /// let mut cause1 = CausalContext::default();
    /// let mut cause2 = CausalContext::default();
    ///
    /// cause1.extend([
    ///     Dot::mint((0, 0).into(), 1),
    ///     Dot::mint((0, 0).into(), 2),
    ///     Dot::mint((0, 0).into(), 4),
    ///     Dot::mint((1, 0).into(), 2),
    /// ]);
    ///
    /// cause2.extend([
    ///     Dot::mint((0, 0).into(), 1),
    ///     Dot::mint((0, 0).into(), 5),
    ///     Dot::mint((1, 0).into(), 1),
    /// ]);
    ///
    /// cause1.union(&cause2);
    /// let cause = cause1;
    ///
    /// assert!(cause.dot_in(Dot::mint((0, 0).into(), 1)));
    /// assert!(cause.dot_in(Dot::mint((0, 0).into(), 2)));
    /// assert!(!cause.dot_in(Dot::mint((0, 0).into(), 3)));
    /// assert!(cause.dot_in(Dot::mint((0, 0).into(), 4)));
    /// assert!(cause.dot_in(Dot::mint((0, 0).into(), 5)));
    /// assert!(!cause.dot_in(Dot::mint((0, 0).into(), 6)));
    /// assert!(cause.dot_in(Dot::mint((1, 0).into(), 1)));
    /// assert!(cause.dot_in(Dot::mint((1, 0).into(), 2)));
    /// assert!(!cause.dot_in(Dot::mint((1, 0).into(), 3)));
    /// ```
    pub fn union(&mut self, other: &CausalContext) {
        for (k, v1) in &mut self.dots {
            if let Some(v2) = other.dots.get(k) {
                // see note on union as to why we don't unite in-place
                *v1 = v1.union(v2);
            }
        }

        for (k, v2) in &other.dots {
            if !self.dots.contains_key(k) {
                self.dots.insert(*k, v2.clone());
            }
        }
    }

    /// Retains only dots whose [`Identifier`] the provided closure returns `true` for.
    ///
    /// ```rust
    /// # use dson::{CausalContext, Dot};
    /// let mut cause = CausalContext::default();
    ///
    /// cause.extend([
    ///     Dot::mint((0, 0).into(), 1),
    ///     Dot::mint((0, 0).into(), 2),
    ///     Dot::mint((1, 0).into(), 2),
    ///     Dot::mint((2, 0).into(), 1),
    /// ]);
    ///
    /// assert!(cause.dot_in(Dot::mint((0, 0).into(), 1)));
    /// assert!(cause.dot_in(Dot::mint((0, 0).into(), 2)));
    /// assert!(cause.dot_in(Dot::mint((1, 0).into(), 2)));
    /// assert!(cause.dot_in(Dot::mint((2, 0).into(), 1)));
    ///
    /// cause.retain_from(|id| id != (1, 0));
    ///
    /// assert!(cause.dot_in(Dot::mint((0, 0).into(), 1)));
    /// assert!(cause.dot_in(Dot::mint((0, 0).into(), 2)));
    /// assert!(!cause.dot_in(Dot::mint((1, 0).into(), 2)));
    /// assert!(cause.dot_in(Dot::mint((2, 0).into(), 1)));
    /// ```
    pub fn retain_from(&mut self, mut f: impl FnMut(Identifier) -> bool) {
        self.dots.retain(|&id, _| f(id));
    }

    /// Returns true if the provided closure returns `true` for the [`Identifier`]s of any [`Dot`]
    /// in this context.
    ///
    /// ```rust
    /// # use dson::{CausalContext, Dot};
    /// let mut cause = CausalContext::default();
    ///
    /// cause.extend([
    ///     Dot::mint((0, 0).into(), 1),
    ///     Dot::mint((0, 0).into(), 2),
    ///     Dot::mint((2, 0).into(), 1),
    /// ]);
    ///
    /// assert!(cause.any_dot_id_with(|id| id == (0, 0)));
    /// assert!(!cause.any_dot_id_with(|id| id == (1, 0)));
    /// assert!(cause.any_dot_id_with(|id| id == (2, 0)));
    /// ```
    pub fn any_dot_id_with(&self, mut f: impl FnMut(Identifier) -> bool) -> bool {
        self.dots.keys().any(|&id| f(id))
    }

    /// Returns true if the provided context contains at least one [`Dot`] that also exists in this
    /// context.
    ///
    /// ```rust
    /// # use dson::{CausalContext, Dot};
    /// let mut cc1 = CausalContext::default();
    /// let mut cc2 = CausalContext::default();
    ///
    /// cc1.extend([
    ///     Dot::mint((0, 0).into(), 1),
    ///     Dot::mint((0, 0).into(), 2),
    ///     Dot::mint((2, 0).into(), 1),
    /// ]);
    /// cc2.extend([
    ///     Dot::mint((0, 0).into(), 2),
    /// ]);
    ///
    /// assert!(cc1.any_dot_in(&cc1));
    /// assert!(cc1.any_dot_in(&cc2));
    /// assert!(cc2.any_dot_in(&cc1));
    /// assert!(cc2.any_dot_in(&cc2));
    ///
    /// cc2 = CausalContext::default();
    /// cc2.extend([
    ///     Dot::mint((0, 0).into(), 3),
    ///     Dot::mint((2, 0).into(), 2),
    ///     Dot::mint((3, 0).into(), 1),
    /// ]);
    ///
    /// assert!(!cc1.any_dot_in(&cc2));
    /// assert!(!cc2.any_dot_in(&cc1));
    /// ```
    pub fn any_dot_in(&self, other: &Self) -> bool {
        let (smaller_dots, larger_dots) = if self.dots.len() > other.dots.len() {
            (&other.dots, &self.dots)
        } else {
            (&self.dots, &other.dots)
        };
        for (k, ivals1) in smaller_dots {
            if let Some(ivals2) = larger_dots.get(k) {
                if ivals1.intersects(ivals2) {
                    return true;
                }
            }
        }
        false
    }

    /// Iterator over the raw intervals that this context holds
    pub fn intervals(
        &self,
    ) -> impl ExactSizeIterator<
        Item = (
            Identifier,
            impl ExactSizeIterator<Item = (NonZeroU64, Option<NonZeroU64>)> + '_,
        ),
    > + '_ {
        self.dots.iter().map(|(id, set)| (*id, set.intervals()))
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub enum ComparisonMode {
    /// Considers all dots from either causal context.
    #[default]
    Full,
    /// Ignores any dots tagged as local via the priority attribute.
    IgnoreLocal,
}

impl CausalContext {
    /// Returns the happens-after ordering between `self` and `other`.
    ///
    /// Specifically, returns:
    ///
    /// - `Some(Ordering::Equal)` if `self` == `other`
    /// - `Some(Ordering::Less)` if `other` happens-after `self`
    /// - `Some(Ordering::Greater)` if `self` happens-after `other`
    /// - `None` if there is no happens-after relationship between `self` and `other`
    ///
    /// `a` happens-after `b` if `a` has observed everything that `b` has (that is, all dots), and at
    /// least one other dot, meaning that `a` causally happens strictly after `b`.
    ///
    /// The `mode` argument controls some filtering options, specifically so that local fields can
    /// be excluded from the comparison when comparing with a remote state. By default all dots
    /// are considered.
    pub fn partial_cmp_dots(
        &self,
        other: &CausalContext,
        mode: ComparisonMode,
    ) -> Option<Ordering> {
        // for vector clocks:
        // x -> y if VC(y) > VC(x)
        // VC(y) > VC(x) if all components of y are >= the component of x && >=1 component is >
        //
        // we can translate this for causal contexts by observing that the check is essentially
        // "are all dots in x in y" && "does y have at least one dot that x does not".

        let remove_local = |&(id, _): &(&Identifier, _)| {
            mode == ComparisonMode::Full || id.priority() != Priority::Local
        };

        let mut ours = self.dots.iter().filter(remove_local).peekable();
        let mut theirs = other.dots.iter().filter(remove_local).peekable();
        let (mut o_unique, mut t_unique) = (false, false);
        loop {
            // early exit if both sides have dots that don't appear in the other
            // - this means neither happened-before the other, so the ordering
            // is undefined.
            if o_unique && t_unique {
                return None;
            }
            match (ours.peek(), theirs.peek()) {
                (None, None) => {
                    break;
                }
                (None, Some(_)) => {
                    t_unique = true;
                    // from this point on, every iteration will hit this case,
                    // until `theirs` is also exhausted, so the `o_unique` value
                    // won't change anymore and it's safe to break early
                    break;
                }
                (Some(_), None) => {
                    // symmetrical of the arm above
                    o_unique = true;
                    break;
                }
                (Some((o_id, o_ival)), Some((t_id, t_ival))) => {
                    match o_id.cmp(t_id) {
                        Ordering::Equal => {
                            match o_ival.partial_cmp(t_ival) {
                                // identical interval sets
                                Some(Ordering::Equal) => (),
                                // theirs has dots ours doesn't
                                Some(Ordering::Less) => t_unique = true,
                                // ours has dots theirs doesn't
                                Some(Ordering::Greater) => o_unique = true,
                                // partial overlap, some unique dots in either side
                                None => return None,
                            }
                            ours.next();
                            theirs.next();
                        }
                        Ordering::Less => {
                            // we have dots for an identifier they don't
                            o_unique = true;
                            ours.next();
                        }
                        Ordering::Greater => {
                            // they have dots for an identifier we don't
                            t_unique = true;
                            theirs.next();
                        }
                    }
                }
            }
        }
        match (o_unique, t_unique) {
            (true, true) => None,
            (true, false) => Some(Ordering::Greater),
            (false, true) => Some(Ordering::Less),
            (false, false) => Some(Ordering::Equal),
        }
    }

    /// Returns true if `self` _happens-after_ `other`.
    ///
    /// In particular, this function returns true if `self` has observed everything that `other`
    /// has (i.e., all dots), and at least one other dot, meaning that `self` causally happens
    /// strictly after `other`.
    pub fn after(&self, other: &CausalContext) -> bool {
        self.partial_cmp_dots(other, ComparisonMode::default()) == Some(Ordering::Greater)
    }

    /// Returns true if `self` _happens-before_ `other`.
    ///
    /// In particular, this function returns true if `other` has observed everything that `self`
    /// has (i.e., all dots), and at least one other dot, meaning that `other` causally happens
    /// strictly after `self`.
    pub fn happened_before(&self, other: &CausalContext) -> bool {
        other.partial_cmp_dots(self, ComparisonMode::default()) == Some(Ordering::Greater)
    }
}

impl PartialEq for CausalContext {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp_dots(other, ComparisonMode::default()) == Some(Ordering::Equal)
    }
}

impl PartialOrd for CausalContext {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp_dots(other, ComparisonMode::default())
    }
}

impl Sub<&CausalContext> for &CausalContext {
    type Output = CausalContext;

    fn sub(self, rhs: &CausalContext) -> Self::Output {
        let mut out = CausalContext::new();
        for (id, left) in &self.dots {
            if let Some(right) = rhs.dots.get(id) {
                let diff = left.difference(right);
                if !diff.is_empty() {
                    out.dots.insert(*id, left.difference(right));
                }
            } else {
                out.dots.insert(*id, left.clone());
            }
        }
        out
    }
}

impl BitAnd<&CausalContext> for &CausalContext {
    type Output = CausalContext;

    fn bitand(self, rhs: &CausalContext) -> Self::Output {
        let mut out = CausalContext::new();
        for (id, v1) in &self.dots {
            if let Some(v2) = rhs.dots.get(id) {
                let intersection = v1.intersection(v2);
                if !intersection.is_empty() {
                    out.dots.insert(*id, intersection);
                }
            }
        }
        out
    }
}

impl FromIterator<Dot> for CausalContext {
    fn from_iter<T: IntoIterator<Item = Dot>>(iter: T) -> Self {
        let mut cc = CausalContext::default();
        cc.insert_dots(iter);
        cc
    }
}

impl Extend<Dot> for CausalContext {
    fn extend<T: IntoIterator<Item = Dot>>(&mut self, iter: T) {
        self.insert_dots(iter);
    }
}

impl Extend<CausalContext> for CausalContext {
    fn extend<T: IntoIterator<Item = CausalContext>>(&mut self, iter: T) {
        for cc in iter {
            self.union(&cc);
        }
    }
}

impl<'cc> Extend<&'cc CausalContext> for CausalContext {
    fn extend<T: IntoIterator<Item = &'cc CausalContext>>(&mut self, iter: T) {
        for cc in iter {
            self.union(cc);
        }
    }
}

/// This macro parses the debug-representation of a CausalContext, giving an easy way
/// to instantiate a cc with a given set of dots:
/// ```
/// use dson::{CausalContext};
/// use dson::causal_context;
/// let remote_ctx: CausalContext =
///            causal_context!({@0.1: {6}, @1.1: {1..=3}, @2.1: {1..=4}, @3.1: {1..=2}, @4.1: {1..=4, 6, 8, 10}, @5.1: {1..=4, 6}});
/// ```
#[macro_export]
macro_rules! causal_context(
    ( { $( @$frac_id:literal : {$( $start:literal $(..=$end:literal)? ),*} ),* } ) => {
        {
            let tracks = vec![
            $(
                {
                    // Unfortunately the source text 0.4 is lexed by rust as the floating point value 0.4.
                    // But we can stringify the floating point literal, and then parse it as a string.
                    let mut frac_id_str = stringify!($frac_id).splitn(2,'.');
                    let node = frac_id_str.next().unwrap().parse().expect("node number must be numerical");
                    let app = frac_id_str.next().expect("missing '.' after node number").parse().expect("app must be numerical");
                    let id = $crate::Identifier::new(node, app);
                    let track = vec![
                    $(
                        {
                            #[allow(unused_mut)]
                            let mut temp = ($start.try_into().unwrap(), None);
                            $(
                                temp.1 = Some(
                                    $end.try_into().unwrap()
                                );
                            )*
                            temp
                        },
                    )*
                    ];
                    (id, track)
                },
            )*
            ];

            $crate::CausalContext::from_intervals(tracks).expect("invalid interval encountered")
        }
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_context;
    use std::collections::HashSet;

    impl quickcheck::Arbitrary for Priority {
        fn arbitrary(g: &mut quickcheck::Gen) -> Self {
            *g.choose(&[
                Priority::Local,
                Priority::Low,
                Priority::Medium,
                Priority::High,
            ])
            .unwrap()
        }
    }

    #[quickcheck]
    fn identifier_bits(node: u8, application: u8, priority: Priority) {
        let application = u16::from(application);
        let id = Identifier::new(node, application).with_priority(priority);
        assert_eq!(
            (node, application, priority),
            (id.node().value(), id.app(), id.priority())
        );

        let id_2 = Identifier::try_from(id.bits);
        assert_eq!(id_2, Ok(id));
    }

    // TODO(https://github.com/BurntSushi/quickcheck/issues/187): limit number of other dots
    #[quickcheck]
    fn compaction(dots: Vec<Dot>, other_dots: Vec<Dot>) -> bool {
        let cc = CausalContext::from_iter(dots.iter().copied());

        // NOTE: we need the extra set operation here since there _could_ be dots repeated between
        // `dots` and `other_dots`, in which case they _will_ be in the CC.
        let has: HashSet<_> = dots.into_iter().collect();
        let mut doesnt_have = other_dots.into_iter().filter(|dot| !has.contains(dot));

        has.iter().all(|&dot| cc.dot_in(dot)) && doesnt_have.all(|dot| !cc.dot_in(dot))
    }

    #[test]
    fn check_compaction_range() {
        let cc = CausalContext::from_iter([
            // Compact range for medium priority
            Dot(
                Identifier::new(0, MAX_APPLICATION_ID).with_priority(Priority::Medium),
                NonZeroU64::new(1).unwrap(),
            ),
            // non-compact range for high priority
            Dot(
                Identifier::new(0, MAX_APPLICATION_ID).with_priority(PRIORITY_MAX),
                NonZeroU64::new(2).unwrap(),
            ),
        ]);

        assert!(!cc.is_compact_for_node(0));
    }

    #[test]
    #[expect(clippy::neg_cmp_op_on_partial_ord)]
    fn happened_before() {
        let id = Identifier::new(0, 0);
        let cc1 = CausalContext::from_iter([Dot::mint((0, 0).into(), 1)]);
        let mut cc2 = cc1.clone();
        cc2.insert_next_dot(cc1.next_dot_for(id));
        assert!(cc1.happened_before(&cc2));
        assert!(!cc2.happened_before(&cc1));
        assert!(!cc1.happened_before(&cc1));
        assert!(cc2.after(&cc1));
        assert!(!cc1.after(&cc2));
        assert!(!cc1.after(&cc1));
        assert!(cc2 > cc1);
        assert!(!(cc2 < cc1));
    }

    #[quickcheck]
    fn qc_happened_before(dots_both: HashSet<Dot>, mut dots_other: HashSet<Dot>) {
        dots_other.retain(|dot| !dots_both.contains(dot));

        if dots_other.is_empty() {
            // this test assumes that we have one cc that _did_ happen after the other
            return;
        }

        let mut before = CausalContext::new();
        let mut after = before.clone();

        for dot in dots_both {
            before.insert_dot(dot);
            after.insert_dot(dot);
        }
        for dot in dots_other {
            after.insert_dot(dot);
        }

        assert!(before != after);
        assert!(!(before == after));
        assert!(before.happened_before(&after));
        assert!(!after.happened_before(&before));
        assert!(after.after(&before));
        assert!(!before.after(&after));
        assert!(!before.happened_before(&before));
        assert!(!after.happened_before(&after));
        assert!(!before.after(&before));
        assert!(!after.after(&after));
    }

    #[quickcheck]
    fn qc_equal(dots_both: HashSet<Dot>) {
        let mut before = CausalContext::new();
        let mut after = before.clone();

        for dot in dots_both {
            before.insert_dot(dot);
            after.insert_dot(dot);
        }

        assert!(!(before != after));
        assert!(before == after);

        assert!(!before.happened_before(&after));
        assert!(!after.happened_before(&before));
        assert!(!after.after(&before));
        assert!(!before.after(&after));
        assert!(!before.happened_before(&before));
        assert!(!after.happened_before(&after));
        assert!(!before.after(&before));
        assert!(!after.after(&after));
    }

    #[quickcheck]
    // (u8, u8) to make it more likely to get consecutive sequences
    fn removal(dots: Vec<(u8, u8)>) {
        let dots: HashSet<Dot> = dots
            .into_iter()
            .map(|(id, seq)| Dot::mint((id, 0).into(), u64::from(seq) + 1))
            .collect();
        let mut cc = CausalContext::from_iter(dots.iter().copied());
        for dot in dots {
            assert!(cc.dot_in(dot));
            assert!(cc.remove_dot(dot));
            assert!(!cc.dot_in(dot));
        }
    }

    #[quickcheck]
    // (bool, u8) to make it more likely to get non-zero intersection
    fn any_dot_in(a_dots: Vec<(bool, u8)>, b_dots: Vec<(bool, u8)>) {
        let a_dots: HashSet<Dot> = a_dots
            .into_iter()
            .map(|(id, seq)| Dot::mint((<_>::from(id), 0).into(), u64::from(seq) + 1))
            .collect();
        let b_dots: HashSet<Dot> = b_dots
            .into_iter()
            .map(|(id, seq)| Dot::mint((<_>::from(id), 0).into(), u64::from(seq) + 1))
            .collect();
        let a_cc = CausalContext::from_iter(a_dots.iter().copied());
        let b_cc = CausalContext::from_iter(b_dots.iter().copied());
        if !a_dots.is_empty() {
            assert!(a_cc.any_dot_in(&a_cc));
        }
        if !b_dots.is_empty() {
            assert!(b_cc.any_dot_in(&b_cc));
        }
        if a_dots.is_disjoint(&b_dots) {
            assert!(!a_cc.any_dot_in(&b_cc));
            assert!(!b_cc.any_dot_in(&a_cc));
        } else {
            assert!(a_cc.any_dot_in(&b_cc));
            assert!(b_cc.any_dot_in(&a_cc));
        }
    }

    #[quickcheck]
    // (bool, u8) to make it more likely to get non-zero intersection
    fn intersection(a_dots: Vec<(bool, u8)>, b_dots: Vec<(bool, u8)>) {
        let a_dots: HashSet<Dot> = a_dots
            .into_iter()
            .map(|(id, seq)| Dot::mint((<_>::from(id), 0).into(), u64::from(seq) + 1))
            .collect();
        let b_dots: HashSet<Dot> = b_dots
            .into_iter()
            .map(|(id, seq)| Dot::mint((<_>::from(id), 0).into(), u64::from(seq) + 1))
            .collect();
        let a_cc = CausalContext::from_iter(a_dots.iter().copied());
        let b_cc = CausalContext::from_iter(b_dots.iter().copied());
        let isect1 = &a_cc & &b_cc;
        let isect2 = &b_cc & &a_cc;
        for &dot in a_dots.intersection(&b_dots) {
            assert!(isect1.dot_in(dot), "a & b does not have {dot:?}");
            assert!(isect2.dot_in(dot), "b & a does not have {dot:?}");
        }
        for &dot in a_dots.symmetric_difference(&b_dots) {
            assert!(!isect1.dot_in(dot), "a & b should not have {dot:?}");
            assert!(!isect2.dot_in(dot), "b & a should not have {dot:?}");
        }
    }

    #[quickcheck]
    fn difference(a_dots: Vec<(bool, u8)>, b_dots: Vec<(bool, u8)>) {
        let a_dots: HashSet<Dot> = a_dots
            .into_iter()
            .map(|(id, seq)| Dot::mint((<_>::from(id), 0).into(), u64::from(seq) + 1))
            .collect();
        let b_dots: HashSet<Dot> = b_dots
            .into_iter()
            .map(|(id, seq)| Dot::mint((<_>::from(id), 0).into(), u64::from(seq) + 1))
            .collect();
        let a_cc = CausalContext::from_iter(a_dots.iter().copied());
        let b_cc = CausalContext::from_iter(b_dots.iter().copied());
        let diff = &a_cc - &b_cc;
        for &dot in a_dots.difference(&b_dots) {
            assert!(diff.dot_in(dot), "a - b does not have {dot:?}");
        }
        for &dot in a_dots.intersection(&b_dots) {
            assert!(
                !diff.dot_in(dot),
                "a - b should not have {dot:?} which is in both"
            );
        }
        for &dot in b_dots.difference(&a_dots) {
            assert!(
                !diff.dot_in(dot),
                "a - b should not have {dot:?} which is only in b"
            );
        }
    }

    #[quickcheck]
    // (u8, u8) to make it more likely to get consecutive sequences
    fn remove_dots_in(dots: HashSet<(u8, u8)>) {
        let dots: HashSet<Dot> = dots
            .into_iter()
            .map(|(id, seq)| Dot::mint((id, 0).into(), u64::from(seq) + 1))
            .collect();
        let mut initial = CausalContext::from_iter(dots.iter().copied());
        let remove = CausalContext::from_iter(dots.iter().take(dots.len() / 2).copied());
        initial.remove_dots_in(&remove);
        for dot in dots.iter().take(dots.len() / 2).copied() {
            assert!(!initial.dot_in(dot));
        }
        for dot in dots.iter().skip(dots.len() / 2).copied() {
            assert!(initial.dot_in(dot));
        }
    }

    #[test]
    fn remove_in_consecutive() {
        let id = (0, 0).into();
        let dot1 = Dot::mint(id, 1);
        let dot2 = Dot::mint(id, 2);
        let dot3 = Dot::mint(id, 3);
        let dot7 = Dot::mint(id, 7);
        let mut cc = CausalContext::from_iter(vec![dot1, dot2, dot3, dot7]);
        for dot in [dot1, dot2, dot3] {
            assert!(cc.dot_in(dot));
        }

        // removing from the end of consecutive should be fine
        assert!(cc.remove_dot(dot3));
        assert!(!cc.dot_in(dot3));
        assert!(cc.dot_in(dot2));
        assert!(cc.dot_in(dot1));
        assert!(cc.dot_in(dot7));
        for dot in [dot1, dot2] {
            assert!(cc.dot_in(dot));
        }

        // removing from beginning should also be fine
        assert!(cc.remove_dot(dot1));
        assert!(!cc.dot_in(dot1));
        assert!(cc.dot_in(dot2));
        assert!(cc.dot_in(dot7));

        // removing after emptying consecutive should also be fine
        assert!(cc.remove_dot(dot2));
        assert!(!cc.dot_in(dot2));
        assert!(cc.dot_in(dot7));
    }

    #[test]
    fn try_identifier_from_u32_rejects_invalid_bits() {
        let reserved_bit_used = (1u32 << (u32::BITS - 8))
            | (1u32 << (u32::BITS - 20))
            | (1u32 << (u32::BITS - 21))
            | ((Priority::Medium as u32) << (u32::BITS - 24));

        assert_eq!(
            Identifier::try_from(reserved_bit_used),
            Err(IdentifierError::InvalidBits {
                field: "reserved",
                value: 1
            })
        );

        let unused_bits_used = (1u32 << (u32::BITS - 8))
            | (1u32 << (u32::BITS - 20))
            | (0u32 << (u32::BITS - 21))
            | ((Priority::Medium as u32) << (u32::BITS - 24))
            | 0b101;

        assert_eq!(
            Identifier::try_from(unused_bits_used),
            Err(IdentifierError::InvalidBits {
                field: "unused",
                value: 0b101
            })
        );

        let invalid_priority = (1u32 << (u32::BITS - 8))
            | (1u32 << (u32::BITS - 20))
            | (0u32 << (u32::BITS - 21))
            | (5u32 << (u32::BITS - 24));

        assert_eq!(
            Identifier::try_from(invalid_priority),
            Err(IdentifierError::Priority(InvalidPriority(5)))
        );
    }

    #[test]
    fn causal_context_macro_works() {
        let cc = causal_context!({@5.1: {1..=4, 6}});
        let dots = cc.dots().collect::<Vec<_>>();
        let id = Identifier::new(5, 1);
        assert_eq!(
            dots,
            vec![
                Dot::mint(id, 1),
                Dot::mint(id, 2),
                Dot::mint(id, 3),
                Dot::mint(id, 4),
                Dot::mint(id, 6),
            ]
        )
    }

    #[test]
    fn causal_context_macro_roundtrips_debug_repr() {
        let cc: CausalContext = causal_context!({@0.1: {1, 3..=4}, @255.4095: {1, 3..=4}});
        assert_eq!(
            format!("{cc:?}"),
            "CausalContext({@0.1: {1, 3..=4}, @255.4095: {1, 3..=4}})"
        );
    }

    #[expect(clippy::neg_cmp_op_on_partial_ord)]
    #[test]
    fn repro_miscompare() {
        // This is a specific case that triggered a bug.
        // Preserve it here, so that this exact particular case will never
        // fail again.
        let remote_ctx = causal_context!({@0.1: {6}, @1.1: {1..=3}, @2.1: {1..=4}, @3.1: {1..=2}, @4.1: {1..=4, 6, 8, 10}, @5.1: {1..=4, 6}});
        let own_ctx = causal_context!({@0.1: {1..=4, 6}, @1.1: {1..=3}, @2.1: {1..=4}, @3.1: {1..=2}, @4.1: {1..=4, 6, 8, 10}, @5.1: {1..=4, 6}});

        assert!(remote_ctx < own_ctx);
        assert!(!(remote_ctx > own_ctx));
        assert!(!(remote_ctx == own_ctx));
        assert!(remote_ctx != own_ctx);
    }

    #[expect(clippy::neg_cmp_op_on_partial_ord)]
    #[test]
    fn repro_miscompare_smaller() {
        let remote_ctx = causal_context!({@0.1: {6}});
        let own_ctx = causal_context!({@0.1: {1..=4, 6}});

        assert!(remote_ctx < own_ctx);
        assert!(!(remote_ctx > own_ctx));
        assert!(!(remote_ctx == own_ctx));
    }

    #[expect(clippy::neg_cmp_op_on_partial_ord)]
    #[test]
    fn repro_miscompare_minimal() {
        let a = causal_context!({@0.1: {2}});
        let b = causal_context!({@0.1: {1}});

        assert!(!(a < b));
        assert!(!(a > b));
        assert!(!(a == b));
    }

    // This quickcheck-test reveals the specific bug showcased by repro_miscompare
    #[quickcheck]
    fn cc_compare(input_lhs: Vec<(u8, u8, u8)>, input_rhs: Vec<(u8, u8, u8)>) {
        let create_cc_dots = |mut input: Vec<(u8, u8, u8)>| {
            input.truncate(5);
            let mut cc_dots = HashSet::new();
            for (node, start, count) in input {
                let node = node % 2;
                for i in 1..count % 4 {
                    cc_dots.insert(Dot::mint(
                        Identifier::new(node, 0),
                        (start % 4).saturating_add(i).into(),
                    ));
                }
            }
            cc_dots
        };
        let lhs = create_cc_dots(input_lhs);
        let rhs = create_cc_dots(input_rhs);
        let cc_lhs = CausalContext::from_iter(lhs.iter().copied());
        let cc_rhs = CausalContext::from_iter(rhs.iter().copied());

        let correct_ord = if lhs.is_subset(&rhs) && rhs.is_subset(&lhs) {
            Some(Ordering::Equal)
        } else if lhs.is_subset(&rhs) {
            Some(Ordering::Less)
        } else if rhs.is_subset(&lhs) {
            Some(Ordering::Greater)
        } else {
            None
        };

        let cc_ord = cc_lhs.partial_cmp(&cc_rhs);
        assert_eq!(
            cc_ord,
            correct_ord,
            "{}",
            &format!("failed: {cc_lhs:?} cmp {cc_rhs:?}")
        );
    }
}
