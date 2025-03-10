// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! # Interval and IntervalSet
//!
//! This module provides the `Interval` and `IntervalSet` data structures, which are
//! used to efficiently represent sets of sequence numbers in a `CausalContext`.
//!
//! ## `Interval`
//!
//! An `Interval` represents a contiguous range of non-zero unsigned 64-bit integers.
//! It is a space-efficient way to store a sequence of dots from a single actor.
//!
//! ## `IntervalSet`
//!
//! An `IntervalSet` is a collection of `Interval`s, which together represent the
//! set of all dots observed from a single actor. It is implemented as a sorted
//! vector of non-overlapping intervals, which allows for efficient storage and
//! retrieval of sequence numbers.

use std::{cmp::Ordering, num::NonZeroU64, ops::RangeInclusive};

/// Represents an interval of non-zero numbers.
///
/// If end is unset, the interval contains only the starting point.
///
/// We intentionally don't use an enum here so as to minimise space usage
/// as much as possible (this is particularly important for serialization).
#[derive(Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub(super) struct Interval {
    /// Start of the interval (inclusive)
    start: NonZeroU64,
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    /// End of the interval (inclusive)
    end: Option<NonZeroU64>,
}

impl std::fmt::Debug for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.start)?;
        if let Some(end) = &self.end {
            write!(f, "..={end}")?;
        }
        Ok(())
    }
}

impl PartialOrd for Interval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.end() < other.start() {
            Some(Ordering::Less)
        } else if self.start() > other.end() {
            Some(Ordering::Greater)
        } else if self.start() == other.start() && self.end() == other.end() {
            // total overlap
            Some(Ordering::Equal)
        } else {
            // partial overlap
            None
        }
    }
}

impl PartialEq<NonZeroU64> for Interval {
    fn eq(&self, other: &NonZeroU64) -> bool {
        *self == Self::point(*other)
    }
}

impl PartialOrd<NonZeroU64> for Interval {
    fn partial_cmp(&self, other: &NonZeroU64) -> Option<Ordering> {
        self.partial_cmp(&Self::point(*other))
    }
}

impl From<NonZeroU64> for Interval {
    fn from(value: NonZeroU64) -> Self {
        Self::point(value)
    }
}

impl TryFrom<(NonZeroU64, Option<NonZeroU64>)> for Interval {
    type Error = IntervalError;

    fn try_from((start, end): (NonZeroU64, Option<NonZeroU64>)) -> Result<Self, Self::Error> {
        if let Some(end) = end {
            (end > start)
                .then_some(Self {
                    start,
                    end: Some(end),
                })
                .ok_or(IntervalError("end must be > start"))
        } else {
            Ok(Self { start, end })
        }
    }
}

#[derive(Debug)]
pub struct IntervalError(&'static str);

impl std::fmt::Display for IntervalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for IntervalError {}

impl TryFrom<u64> for Interval {
    type Error = IntervalError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Ok(Self::point(
            NonZeroU64::new(value).ok_or(IntervalError("value must be > 0"))?,
        ))
    }
}

impl TryFrom<RangeInclusive<u64>> for Interval {
    type Error = IntervalError;

    fn try_from(value: RangeInclusive<u64>) -> Result<Self, Self::Error> {
        assert!(
            value.start() < value.end(),
            "start must be < end: {value:?}"
        );
        Ok(Self::span(
            NonZeroU64::new(*value.start()).ok_or(IntervalError("start must be > 0"))?,
            NonZeroU64::new(*value.end()).ok_or(IntervalError("end must be > 0"))?,
        ))
    }
}

#[derive(Debug)]
pub enum IntervalDifference {
    Empty,
    Single(Interval),
    Split(Interval, Interval),
}

impl Interval {
    /// Creates a new [`Interval`] containing a single point.
    #[must_use]
    pub fn point(seq: NonZeroU64) -> Self {
        Self {
            start: seq,
            end: None,
        }
    }

    /// Creates a new [`Interval`] spanning more than one point.
    ///
    /// # Panics
    /// The given `start` must be strictly less than `end`, otherwise
    /// this function panics. If you must have `start` == `end`, use the
    /// [`Self::point`] constructor instead (or just pass `None` as the `end`
    /// argument).
    #[must_use]
    pub fn span(start: NonZeroU64, end: impl Into<Option<NonZeroU64>>) -> Self {
        let end = end.into();
        if let Some(end) = end {
            assert!(start < end, "{start} < {end}");
        }
        Self { start, end }
    }

    #[must_use]
    pub fn next_after(&self) -> NonZeroU64 {
        self.end.unwrap_or(self.start).saturating_add(1)
    }

    #[must_use]
    pub fn end(&self) -> NonZeroU64 {
        self.end.unwrap_or(self.start)
    }

    #[must_use]
    pub fn start(&self) -> NonZeroU64 {
        self.start
    }

    pub fn interval(&self) -> (NonZeroU64, Option<NonZeroU64>) {
        (self.start, self.end)
    }

    #[must_use]
    pub fn contains(&self, seq: NonZeroU64) -> bool {
        if let Some(end) = self.end {
            seq >= self.start && seq <= end
        } else {
            seq == self.start
        }
    }

    /// Returns whether `self` is a superset (A ⊇ B) of `other`.
    ///
    /// Note that if they are equal, this returns true. To test for proper
    /// superset (A ⊃ B) use [`Self::partial_set_cmp`] instead.
    #[must_use]
    pub fn is_superset(&self, other: &Self) -> bool {
        self.start() <= other.start() && other.end() <= self.end()
    }

    /// Iterator over all the sequence number that this interval holds
    pub fn seqs(&self) -> impl Iterator<Item = NonZeroU64> {
        // TODO: can simplify once https://github.com/rust-lang/rust/pull/127534 is stable
        (self.start.get()..=self.end.unwrap_or(self.start).get())
            // SAFETY: start and end are non-zero, so all numbers in-between must be as well
            .map(|s| unsafe { NonZeroU64::new_unchecked(s) })
    }

    /// Returns the partial ordering with respect to set comparison.
    ///
    /// - If `self` is a proper subset of `other` (A ⊂ B), the result is `Less`.
    /// - If `self` is a proper superset of `other` (A ⊃ B), the result is `Greater.`
    /// - If both intervals are the same (A = B), the result is `Equal`.
    ///
    /// Otherwise the result is `None`, indicating there isn't a well defined
    /// set hierarchy between them. This could mean there's no overlap, or that
    /// the overlap is partial.
    #[must_use]
    pub fn partial_set_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.start() == other.start() && self.end() == other.end() {
            Some(Ordering::Equal)
        } else if self.start() <= other.start() && self.end() >= other.end() {
            Some(Ordering::Greater)
        } else if other.start() <= self.start() && other.end() >= self.end() {
            Some(Ordering::Less)
        } else {
            None
        }
    }

    /// Combines two intervals together, if they overlap or are adjacent.
    ///
    /// If the intervals overlap or are adjacent, the result is a single
    /// interval representing the union of all seqs in either one.
    ///
    /// If the intervals are disjoint (ie, have a gap in between them), this
    /// method returns `None`.
    #[must_use]
    pub fn merge(&self, other: &Self) -> Option<Self> {
        // this is fundamentally doing a traditional interval overlap check: if the later start is
        // before the earlier end, then the two intervals overlap. we only spice things up by adding
        // one to the earlier end so that we cover the case where they _just about_ touch together
        // (since our intervals are inclusive and our elements discrete integers, this means they
        // can also be merged in that case).
        if self.end().min(other.end()).saturating_add(1) >= self.start().max(other.start()) {
            let start = self.start().min(other.start());
            let end = self.end().max(other.end());
            if start == end {
                Some(Self::point(start))
            } else {
                Some(Self::span(start, end))
            }
        } else {
            None
        }
    }

    #[must_use]
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let start = self.start().max(other.start());
        let end = self.end().min(other.end());
        match start.cmp(&end) {
            Ordering::Less => Some(Self::span(start, end)),
            Ordering::Equal => Some(Self::point(start)),
            Ordering::Greater => None,
        }
    }

    #[must_use]
    pub fn difference(&self, other: &Self) -> IntervalDifference {
        // if two intervals overlap, these will indicate the start and end of the overlapping range.
        let later_start = self.start().max(other.start());
        let earlier_end = self.end().min(other.end());
        match later_start.cmp(&earlier_end) {
            // overlap
            Ordering::Less | Ordering::Equal => {
                let left = (self.start() < later_start).then(|| {
                    // SAFETY: we know later_start - 1 > 0 because later_start >= self.start() and self.start() > 0
                    let new_end = unsafe { NonZeroU64::new_unchecked(later_start.get() - 1) };
                    if new_end == self.start() {
                        Self::point(new_end)
                    } else {
                        Self::span(self.start(), new_end)
                    }
                });
                let right = (earlier_end < self.end()).then(|| {
                    let new_start = earlier_end.saturating_add(1);
                    if new_start == self.end() {
                        Self::point(new_start)
                    } else {
                        Self::span(new_start, self.end())
                    }
                });
                match (left, right) {
                    // total overlap
                    (None, None) => IntervalDifference::Empty,
                    // start removed
                    (None, Some(right)) => IntervalDifference::Single(right),
                    // end removed
                    (Some(left), None) => IntervalDifference::Single(left),
                    // split in the middle
                    (Some(left), Some(right)) => IntervalDifference::Split(left, right),
                }
            }
            // no overlap
            Ordering::Greater => IntervalDifference::Single(*self),
        }
    }

    #[cfg(test)]
    pub fn is_point(&self) -> bool {
        self.end.is_none()
    }

    #[cfg(test)]
    pub fn is_span(&self) -> bool {
        self.end.is_some()
    }

    /// The length of this interval.
    ///
    /// Alternatively, the number of individual integer values it contains.
    pub fn interval_length(&self) -> u64 {
        if let Some(end) = self.end {
            end.get() - self.start().get() + 1
        } else {
            1
        }
    }
}

// TODO: would it be worth using a BTree here and a more traditional interval set? We chose a
// vector-based interval set because we don't expect this to grow very large, but only benchmarks
// will tell what is actually better.
#[derive(Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub(super) struct IntervalSet(Vec<Interval>);

impl std::fmt::Debug for IntervalSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.0.iter()).finish()
    }
}

impl IntervalSet {
    #[must_use]
    pub fn new() -> Self {
        Self(Vec::new())
    }

    #[must_use]
    pub fn single(seq: NonZeroU64) -> Self {
        Self(Vec::from([Interval::point(seq)]))
    }

    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self(Vec::with_capacity(n))
    }

    pub fn from_intervals(
        iter: impl IntoIterator<Item = (NonZeroU64, Option<NonZeroU64>)>,
    ) -> Result<Self, IntervalError> {
        Ok(Self(
            iter.into_iter()
                .map(Interval::try_from)
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[must_use]
    pub fn first(&self) -> Option<Interval> {
        self.0.first().copied()
    }

    #[must_use]
    pub fn last(&self) -> Option<Interval> {
        self.0.last().copied()
    }

    /// The total length of all intervals, summed
    #[must_use]
    pub fn total_interval_length(&self) -> u64 {
        self.0.iter().map(|i| i.interval_length()).sum()
    }

    #[must_use]
    pub fn next_after(&self) -> NonZeroU64 {
        self.last()
            .map(|ival| ival.next_after())
            .unwrap_or(NonZeroU64::MIN)
    }

    /// Iterator over all the sequence numbers of this set
    pub fn seqs(&self) -> impl Iterator<Item = NonZeroU64> + '_ {
        self.0.iter().flat_map(|ival| ival.seqs())
    }

    /// Iterator over the raw interval ranges (start, end) of this set
    pub fn intervals(
        &self,
    ) -> impl ExactSizeIterator<Item = (NonZeroU64, Option<NonZeroU64>)> + '_ {
        self.0.iter().map(Interval::interval)
    }

    pub fn insert(&mut self, value: impl Into<Interval>) {
        let ival = value.into();

        // find the first interval that does not strictly precede `ival`. this
        // means it could be adjacent to `ival`, overlap with it or be strictly
        // after. we cover each case below.
        let i = self
            .0
            .partition_point(|s| s.end().saturating_add(1) < ival.start());
        if i == self.0.len() {
            // all elements strictly before `ival` (not even adjacent), so
            // just add the new interval to the end
            self.0.push(ival);
        } else if let Some(merged) = self.0[i].merge(&ival) {
            // this means the intervals overlap or are adjacent
            self.0[i] = merged;
            // now the next interval could still overlap or be adjacent, so we
            // run compaction for the remaining vector - note this is "free",
            // because we'd need to shift things over anyway if we just inserted
            // the point without merging
            self.normalize_starting_at(i);
        } else {
            // this means there is a gap between d and the values before and after
            self.0.insert(i, ival);
        }
    }

    pub fn remove(&mut self, seq: NonZeroU64) -> bool {
        let p = self.0.partition_point(|s| *s < seq);
        let Some(ival) = self.0.get(p) else {
            // this happens if seq is _after_ end
            return false;
        };
        if ival.contains(seq) {
            if let Some(end) = ival.end {
                let start = ival.start;
                if seq == start {
                    let new_start = start.saturating_add(1);
                    if new_start == end {
                        self.0[p] = Interval::point(new_start);
                    } else {
                        self.0[p] = Interval::span(new_start, end);
                    }
                } else {
                    // SAFETY: we know end - 1 > 0 because end > start and start > 0
                    let new_end = unsafe { NonZeroU64::new_unchecked(seq.get() - 1) };
                    if new_end == start {
                        self.0[p] = Interval::point(start);
                    } else {
                        self.0[p] = Interval::span(start, new_end);
                    }

                    if seq != end {
                        if seq.saturating_add(1) == end {
                            self.0.insert(p + 1, Interval::point(end));
                        } else {
                            self.0
                                .insert(p + 1, Interval::span(seq.saturating_add(1), end));
                        }
                        // no need to normalize now as only growth can lead to
                        // intervals becoming mergeable, and we've only shrunk
                    }
                }
            } else {
                self.0.remove(p);
            }
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn contains(&self, seq: NonZeroU64) -> bool {
        let p = self.0.partition_point(|s| *s < seq);
        if p == self.0.len() {
            false
        } else {
            self.0[p].contains(seq)
        }
    }

    /// Extends the last interval in this set by one seq.
    ///
    /// If the interval set is empty, it will be extended from 0, so
    // it results in {1}.
    pub fn extend_end_by_one(&mut self) {
        match self.0.last_mut() {
            Some(ival) => {
                if let Some(end) = ival.end {
                    *ival = Interval::span(ival.start, end.saturating_add(1));
                } else {
                    *ival = Interval::span(ival.start, ival.start.saturating_add(1));
                }
            }
            None => self.0.push(Interval::span(NonZeroU64::MIN, None)),
        }
    }

    fn normalize_starting_at(&mut self, i: usize) {
        let right_start = i + 1;
        for j in right_start..self.0.len() {
            if let Some(merged) = self.0[i].merge(&self.0[j]) {
                self.0[i] = merged;
            } else {
                if j != right_start {
                    // this means there's a segment of merged intervals from
                    // `i+1..j` of length n that needs to be shifted over to the
                    // left. we achieve that shifting by "swapping" the range
                    // from `j..` and `i+1..j` (via a rotation) and truncating.
                    let n = j - i - 1;
                    self.0[right_start..].rotate_left(n);
                    self.0.truncate(self.0.len() - n);
                } else {
                    // if the _first_ merge fails, there will be no chain reaction,
                    // so we can just return. note that this assumes the function is
                    // called when _only_ the starting interval has been modified.
                }
                return;
            }
        }
        // if we get here it means that everything collapsed into a single interval. nice!
        self.0.truncate(right_start);
    }

    // NOTE: doing this inplace turns out to be a bad idea, as it has a O(n^2) worst case due to
    // shifting elements when right < left. Benchmarks show degradation of up to 90%. We'll have to
    // live with the extra allocation (unless we switch to a different data struct maybe?).
    #[must_use]
    pub fn union(&self, other: &Self) -> Self {
        let mut ours = self.0.iter().peekable();
        let mut theirs = other.0.iter().peekable();
        let mut result = Self::with_capacity(self.0.len().max(other.0.len()));
        loop {
            let next = match (ours.peek(), theirs.peek()) {
                (Some(&o_ival), Some(&t_ival)) => match o_ival.partial_cmp(t_ival) {
                    Some(Ordering::Less) => {
                        ours.next();
                        *o_ival
                    }
                    Some(Ordering::Greater) => {
                        theirs.next();
                        *t_ival
                    }
                    Some(Ordering::Equal) => {
                        ours.next();
                        theirs.next();
                        *o_ival
                    }
                    None => match o_ival.merge(t_ival) {
                        Some(merged) => {
                            ours.next();
                            theirs.next();
                            merged
                        }
                        None => {
                            unreachable!(
                                "if there is no ordering defined, there must be an overlap"
                            )
                        }
                    },
                },
                (Some(&&next), None) => {
                    ours.next();
                    next
                }
                (None, Some(&&next)) => {
                    theirs.next();
                    next
                }
                (None, None) => break,
            };
            if let Some(last) = result.0.last_mut() {
                if let Some(merged) = last.merge(&next) {
                    *last = merged;
                    continue;
                }
            }
            result.0.push(next);
        }
        result
    }

    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        let (mut lhs, mut rhs) = (self.0.iter().peekable(), other.0.iter().peekable());
        while let (Some(&left), Some(&right)) = (lhs.peek(), rhs.peek()) {
            match left.partial_cmp(right) {
                Some(Ordering::Less) => {
                    lhs.next();
                }
                Some(Ordering::Greater) => {
                    rhs.next();
                }
                Some(Ordering::Equal) | None => return true,
            }
        }
        false
    }

    #[must_use]
    pub fn intersection(&self, other: &Self) -> Self {
        let mut intersection = Self::with_capacity(self.len().min(other.len()));
        let (mut lhs, mut rhs) = (self.0.iter().peekable(), other.0.iter().peekable());
        while let (Some(&left), Some(&right)) = (lhs.peek(), rhs.peek()) {
            match left.partial_cmp(right) {
                Some(Ordering::Less) => {
                    lhs.next();
                }
                Some(Ordering::Greater) => {
                    rhs.next();
                }
                Some(Ordering::Equal) => {
                    intersection.insert(*left);
                    lhs.next();
                    rhs.next();
                }
                None => {
                    intersection.insert(left.intersect(right).expect("no ordering => overlaps"));
                    // we advance only the interval that ends earlier, as the
                    // next one might still overlap
                    if left.end() > right.end() {
                        rhs.next();
                    } else {
                        lhs.next();
                    }
                }
            }
        }
        intersection
    }

    #[must_use]
    pub fn difference(&self, other: &Self) -> Self {
        let mut diff = Self::with_capacity(self.len());
        let mut ours = self.0.iter().peekable();
        let mut theirs = other.0.iter().peekable();
        // copied once so we don't hold a borrow of the peeked reference, which is ephemeral
        // copied again so we can hold ownership of `second` below with the same variable
        let mut curr = ours.peek().copied().copied();
        loop {
            match (curr, theirs.peek()) {
                (None, None) => break,
                (None, Some(_)) => {
                    theirs.next();
                }
                (Some(o_ival), None) => {
                    diff.insert(o_ival);
                    ours.next();
                    curr = ours.peek().copied().copied();
                }
                (Some(o_ival), Some(&t_ival)) => match o_ival.partial_cmp(t_ival) {
                    Some(Ordering::Greater) => {
                        theirs.next();
                    }
                    Some(Ordering::Less) => {
                        diff.insert(o_ival);
                        ours.next();
                        curr = ours.peek().copied().copied();
                    }
                    Some(Ordering::Equal) => {
                        ours.next();
                        theirs.next();
                        curr = ours.peek().copied().copied();
                    }
                    // overlap
                    None => {
                        match o_ival.difference(t_ival) {
                            // `t_ival` is a (proper) superset of `o_ival`, so nothing remains
                            // of `o_ival`. `t_ival` may still overlap with the next from `ours`,
                            // so we don't advance that side yet, just `ours`.
                            IntervalDifference::Empty => {
                                ours.next();
                                curr = ours.peek().copied().copied();
                            }
                            // one interval remains - this could be the left side of `o_ival`, or
                            // the right side. if it's the right side, we can't insert yet, as the
                            // next interval from `theirs` may still overlap with it. since we don't
                            // know, we just set `curr` and don't advance `ours`.
                            IntervalDifference::Single(common) => {
                                curr = Some(common);
                            }
                            // left gets split into two smaller intervals
                            IntervalDifference::Split(first, second) => {
                                // `first` interval is < `t_ival`, so we can insert now because we
                                // know no other interval from `theirs` will overlap with it.
                                diff.insert(first);
                                // but `second` may still overlap with the next from `theirs`, so we
                                // keep it as `curr` instead of advancing `ours`.
                                curr = Some(second);
                                // this also means it's safe to advance `theirs`, because `t_ival`
                                // is known to end before `second` (that is, we know we'll hit the
                                // Greater case in the next iteration, so may as well skip that
                                // step)
                                theirs.next();
                            }
                        }
                        // NOTE: may be tempting to advance `theirs` now, since we used the interval
                        // to subtract, but it's possible the right interval extends past left, in
                        // which case it may still overlap with the next left. the one exception is
                        // in the split case above (see the comment).
                    }
                },
            }
        }
        diff
    }
}

impl Extend<NonZeroU64> for IntervalSet {
    fn extend<T: IntoIterator<Item = NonZeroU64>>(&mut self, iter: T) {
        for seq in iter {
            self.insert(seq);
        }
    }
}

impl FromIterator<NonZeroU64> for IntervalSet {
    fn from_iter<T: IntoIterator<Item = NonZeroU64>>(iter: T) -> Self {
        let mut new = Self::new();
        new.extend(iter);
        new
    }
}

impl PartialOrd for IntervalSet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let mut ours = self.0.iter().peekable();
        let mut theirs = other.0.iter().peekable();
        let (mut o_unique, mut t_unique) = (false, false);
        loop {
            // early exit if both sides have unique seqs
            if o_unique && t_unique {
                return None;
            }
            match (ours.peek(), theirs.peek()) {
                (None, None) => break,
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
                (Some(&o_ival), Some(&t_ival)) => {
                    match o_ival.partial_cmp(t_ival) {
                        Some(Ordering::Less) => {
                            ours.next();
                            o_unique = true;
                        }
                        Some(Ordering::Greater) => {
                            theirs.next();
                            t_unique = true;
                        }
                        Some(Ordering::Equal) => {
                            ours.next();
                            theirs.next();
                        }
                        None => {
                            match o_ival.partial_set_cmp(t_ival) {
                                Some(Ordering::Equal) => unreachable!(
                                    "covered by outer equal arm - equality definitions must match"
                                ),
                                // `t_ival` is a superset of `o_ival`
                                // TODO: use a macro to ensure symmetry between Less and Greater?
                                Some(Ordering::Less) => {
                                    if o_unique {
                                        // we know here that `t_ival` contains unique dots not in
                                        // `o_val` (because it's a superset), so if `o_unique` is
                                        // already true, it means both sides overlap partially, and
                                        // the ordering is undefined. since we know that already, we
                                        // can just return early.
                                        return None;
                                    }
                                    t_unique = true;
                                    ours.next();
                                    // make sure we advance the lhs to the next non-overlapping item
                                    while let Some(&left) = ours.peek() {
                                        if !t_ival.is_superset(left) {
                                            break;
                                        }
                                        ours.next();
                                    }
                                    theirs.next();
                                }
                                // `o_ival` is a superset of `t_ival`
                                Some(Ordering::Greater) => {
                                    if t_unique {
                                        // see `Less` branch for why we can return early here
                                        return None;
                                    }
                                    o_unique = true;
                                    theirs.next();
                                    // make sure we advance the rhs to the next non-overlapping item
                                    while let Some(&right) = theirs.peek() {
                                        if !o_ival.is_superset(right) {
                                            break;
                                        }
                                        theirs.next();
                                    }
                                    ours.next();
                                }
                                None => return None,
                            }
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
}

#[cfg(test)]
#[allow(clippy::undocumented_unsafe_blocks)]
mod tests {

    use ahash::HashSet;

    use super::*;

    impl IntervalSet {
        fn count_points(&self) -> usize {
            self.0.iter().filter(|ival| ival.is_point()).count()
        }

        fn count_spans(&self) -> usize {
            self.0.iter().filter(|ival| ival.is_span()).count()
        }

        fn assert_normalized(&self) {
            let mut compacted = self.clone();
            compacted.normalize_starting_at(0);
            assert_eq!(self, &compacted);
        }
    }

    macro_rules! seqs {
        ($($span:expr),* $(,)?) => {{
            let mut vec = Vec::new();
            $(
                for i in $span {
                    vec.push(NonZeroU64::new(i).expect("> 0"));
                }
            )*
            vec
        }};
    }

    fn nz(n: u64) -> NonZeroU64 {
        match n {
            1 => NonZeroU64::MIN,
            n if n > 1 => NonZeroU64::MIN.saturating_add(n - 1),
            _ => panic!("{n} must be >= 1"),
        }
    }

    #[test]
    fn seqs() {
        let ival = Interval::from(nz(10));
        assert_eq!(ival.seqs().collect::<Vec<_>>(), [nz(10)]);

        let ival = Interval::try_from(1..=3).unwrap();
        assert_eq!(ival.seqs().collect::<Vec<_>>(), seqs!(1..=3));

        let mut ivals = IntervalSet::new();
        assert_eq!(ivals.seqs().next(), None);
        ivals.insert(nz(1));
        assert_eq!(ivals.seqs().collect::<Vec<_>>(), [nz(1)]);
        ivals.extend([nz(2), nz(3)]);
        assert_eq!(ivals.seqs().collect::<Vec<_>>(), seqs!(1..=3));

        ivals.assert_normalized();
    }

    #[test]
    fn sequential_add() {
        let mut ivals = IntervalSet::new();
        assert!(ivals.is_empty());
        assert_eq!(ivals.len(), 0);
        assert_eq!(ivals.first(), None);
        assert_eq!(ivals.last(), None);
        assert_eq!(ivals.next_after(), NonZeroU64::MIN);
        assert_eq!(ivals.seqs().next(), None);

        let seqs = seqs!(1..=3);
        ivals.insert(seqs[0]);
        ivals.insert(seqs[1]);
        ivals.insert(seqs[2]);

        let combined = Interval::span(seqs[0], seqs[2]);

        assert!(!ivals.is_empty());
        assert_eq!(ivals.len(), 1);
        assert_eq!(ivals.first(), Some(combined));
        assert_eq!(ivals.last(), Some(combined));
        assert_eq!(ivals.next_after(), seqs[2].saturating_add(1));
        assert_eq!(ivals.seqs().collect::<Vec<_>>(), seqs);
        assert!(ivals.contains(seqs[0]));
        assert!(ivals.contains(seqs[1]));
        assert!(ivals.contains(seqs[2]));
        assert!(!ivals.contains(seqs[2].saturating_add(1)));

        assert!(ivals.remove(seqs[0]));
        assert!(ivals.remove(seqs[1]));
        assert!(ivals.remove(seqs[2]));

        assert!(ivals.is_empty());
        assert_eq!(ivals.len(), 0);
        assert_eq!(ivals.first(), None);
        assert_eq!(ivals.last(), None);
        assert_eq!(ivals.next_after(), NonZeroU64::MIN);
        assert_eq!(ivals.seqs().next(), None);

        ivals.assert_normalized();
    }

    #[test]
    fn with_gaps() {
        let seqs = seqs!([1], 3..=4, 6..=8);
        let mut ivals = IntervalSet::from_iter(seqs.iter().copied());
        for seq in seqs.iter().copied() {
            assert!(ivals.contains(seq));
        }

        assert_eq!(ivals.len(), 3);
        assert_eq!(ivals.first(), Some(Interval::point(seqs[0])));
        assert_eq!(ivals.last(), Some(Interval::span(seqs[3], seqs[5])));
        assert_eq!(ivals.next_after(), seqs[5].saturating_add(1));
        assert_eq!(ivals.seqs().collect::<Vec<_>>(), seqs);

        // remove middle
        assert!(ivals.remove(seqs[4])); // 7
        assert!(!ivals.contains(seqs[4]));
        assert_eq!(ivals.len(), 4);
        assert_eq!(ivals.count_points(), 3);
        assert_eq!(ivals.count_spans(), 1);

        // remove end
        assert!(ivals.remove(seqs[1])); // 3
        assert!(!ivals.contains(seqs[1]));
        assert_eq!(ivals.len(), 4);
        assert_eq!(ivals.count_points(), 4);
        assert_eq!(ivals.count_spans(), 0);

        // bring back a span
        ivals.insert(seqs[4]); // 7
        assert_eq!(ivals.len(), 3);
        assert_eq!(ivals.count_points(), 2);
        assert_eq!(ivals.count_spans(), 1);

        // remove start
        assert!(ivals.remove(seqs[3]));
        assert!(!ivals.contains(seqs[3]));
        assert_eq!(ivals.len(), 3);
        assert_eq!(ivals.count_points(), 2);
        assert_eq!(ivals.count_spans(), 1);

        // remove point
        assert!(ivals.remove(seqs[0]));
        assert!(!ivals.contains(seqs[0]));
        assert_eq!(ivals.len(), 2);
        assert_eq!(ivals.count_points(), 1);
        assert_eq!(ivals.count_spans(), 1);

        // final sanity check
        assert_eq!(
            ivals.seqs().collect::<Vec<_>>(),
            [seqs[2], seqs[4], seqs[5]]
        );

        ivals.assert_normalized();
    }

    #[test]
    fn extend_one() {
        let mut ivals = IntervalSet::from_iter(seqs!([1], [3, 4]));

        ivals.extend_end_by_one();
        assert!(ivals.contains(nz(5)));

        assert!(ivals.remove(nz(4)));
        ivals.extend_end_by_one();
        assert!(ivals.contains(nz(6)));

        assert_eq!(ivals.count_points(), 2);
        assert_eq!(ivals.count_spans(), 1);

        ivals.assert_normalized();
    }

    #[test]
    fn insert() {
        let mut ivals = IntervalSet::new();
        ivals.insert(Interval::point(nz(10)));
        assert_eq!(ivals.seqs().collect::<Vec<_>>(), [nz(10)]);

        let mut ivals = IntervalSet::new();
        ivals.insert(Interval::span(nz(10), nz(11)));
        assert_eq!(ivals.seqs().collect::<Vec<_>>(), seqs!(10..=11));
        // insert again the same ival to make sure nothing changes
        ivals.insert(Interval::span(nz(10), nz(11)));
        assert_eq!(ivals.seqs().collect::<Vec<_>>(), seqs!(10..=11));

        ivals.insert(Interval::span(nz(1), nz(3)));
        ivals.insert(Interval::span(nz(9), nz(12)));
        ivals.insert(Interval::span(nz(30), nz(31)));
        assert_eq!(
            ivals.seqs().collect::<Vec<_>>(),
            seqs!(1..=3, 9..=12, 30..=31)
        );
        ivals.insert(Interval::span(nz(13), nz(15)));
        assert_eq!(
            ivals.seqs().collect::<Vec<_>>(),
            seqs!(1..=3, 9..=15, 30..=31)
        );
        ivals.insert(Interval::point(nz(10)));
        assert_eq!(
            ivals.seqs().collect::<Vec<_>>(),
            seqs!(1..=3, 9..=15, 30..=31)
        );
        ivals.insert(Interval::span(nz(8), nz(14)));
        assert_eq!(
            ivals.seqs().collect::<Vec<_>>(),
            seqs!(1..=3, 8..=15, 30..=31)
        );
        ivals.insert(Interval::span(nz(14), nz(17)));
        assert_eq!(
            ivals.seqs().collect::<Vec<_>>(),
            seqs!(1..=3, 8..=17, 30..=31)
        );
        ivals.insert(Interval::span(nz(28), nz(29)));
        assert_eq!(
            ivals.seqs().collect::<Vec<_>>(),
            seqs!(1..=3, 8..=17, 28..=31)
        );
        ivals.insert(Interval::span(nz(5), nz(6)));
        assert_eq!(
            ivals.seqs().collect::<Vec<_>>(),
            seqs!(1..=3, 5..=6, 8..=17, 28..=31)
        );
        ivals.assert_normalized();

        let mut ivals = IntervalSet::new();
        ivals.insert(nz(9));
        ivals.insert(nz(1));
        ivals.insert(nz(2));
        ivals.assert_normalized();

        let mut ivals = IntervalSet::new();
        ivals.insert(nz(231));
        ivals.insert(nz(229));
        ivals.insert(nz(227));
        ivals.insert(nz(228));
        assert_eq!(
            ivals.0,
            [
                Interval::span(
                    NonZeroU64::MIN.saturating_add(226),
                    NonZeroU64::MIN.saturating_add(228)
                ),
                Interval::point(NonZeroU64::MIN.saturating_add(230))
            ]
        );
    }

    #[test]
    fn union() {
        let left = IntervalSet::from_iter(seqs!([2]));
        let right = IntervalSet::from_iter(seqs!([5]));
        let union = left.union(&right);
        let union_seqs = seqs!([2], [5]);
        assert_eq!(union.seqs().collect::<Vec<_>>(), union_seqs);
        assert_eq!(union.count_points(), 2);
        assert_eq!(union.count_spans(), 0);
        union.assert_normalized();

        let left = IntervalSet::from_iter(seqs!(14..=21));
        let right = IntervalSet::from_iter(seqs!(6..=8, 10..=12));
        let union = left.union(&right);
        let union_seqs = seqs!(6..=8, 10..=12, 14..=21);
        assert_eq!(union.seqs().collect::<Vec<_>>(), union_seqs);
        assert_eq!(union.count_points(), 0);
        assert_eq!(union.count_spans(), 3);
        union.assert_normalized();

        let left = IntervalSet::from_iter(seqs!([1], [3, 4], 6..=8));
        let right = IntervalSet::from_iter(seqs!([2], [5]));

        let union = left.union(&right);
        let union_seqs = seqs!(1..=8);
        assert_eq!(union.seqs().collect::<Vec<_>>(), union_seqs);
        assert_eq!(union.count_points(), 0);
        assert_eq!(union.count_spans(), 1);
        union.assert_normalized();

        let left = IntervalSet::from_iter(seqs!([1], [3, 4], 6..=8));
        let right = IntervalSet::from_iter(seqs!([2], [5]));

        // reverse should yield the same
        let union = right.union(&left);
        assert_eq!(union.seqs().collect::<Vec<_>>(), union_seqs);
        assert_eq!(union.count_points(), 0);
        assert_eq!(union.count_spans(), 1);
        assert_eq!(union, union);

        let right = IntervalSet::from_iter(seqs!([3], [9]));
        let union = left.union(&right);
        assert_eq!(union.seqs().collect::<Vec<_>>(), seqs!([1], [3, 4], 6..=9));
        assert_eq!(union.count_points(), 1);
        assert_eq!(union.count_spans(), 2);
        union.assert_normalized();

        let left = IntervalSet::new();
        let right = IntervalSet::from_iter(seqs!([228], [230], [232]));
        let union = left.union(&right);
        assert_eq!(union.seqs().collect::<Vec<_>>(), seqs!([228], [230], [232]));
        assert_eq!(union.count_points(), 3);
        assert_eq!(union.count_spans(), 0);
        union.assert_normalized();
    }

    #[test]
    fn intersects() {
        let seqs = seqs!([1], 3..=4, 6..=8);
        let left = IntervalSet::from_iter(seqs.iter().copied());

        for seq in seqs {
            assert!(left.intersects(&IntervalSet::single(seq)));
            assert!(IntervalSet::single(seq).intersects(&left));
        }

        // complement
        assert!(!left.intersects(&IntervalSet::single(nz(2))));
        assert!(!IntervalSet::single(nz(2)).intersects(&left));
        assert!(!left.intersects(&IntervalSet::single(nz(5))));
        assert!(!IntervalSet::single(nz(5)).intersects(&left));
        assert!(!left.intersects(&IntervalSet::single(nz(9))));
        assert!(!IntervalSet::single(nz(9)).intersects(&left));

        // mixed bag
        assert!(left.intersects(&IntervalSet::from_iter([nz(1), nz(11)])));
        assert!(IntervalSet::from_iter([nz(1), nz(11)]).intersects(&left));
        assert!(left.intersects(&IntervalSet::from_iter([nz(3), nz(4), nz(5)])));
        assert!(IntervalSet::from_iter([nz(3), nz(4), nz(5)]).intersects(&left));
    }

    #[test]
    fn intersection() {
        let left = IntervalSet::from_iter(seqs!(1..=10));
        let right = IntervalSet::from_iter(seqs!(4..=5));
        let common = left.intersection(&right);
        assert_eq!(common, right);
        common.assert_normalized();

        let right = IntervalSet::single(nz(1));
        let common = left.intersection(&right);
        assert_eq!(common, right);
        common.assert_normalized();

        let right = IntervalSet::from_iter(seqs!(5..=15));
        let common = left.intersection(&right);
        assert_eq!(common, IntervalSet::from_iter(seqs!(5..=10)));
        common.assert_normalized();

        let left = IntervalSet::from_iter(seqs!(1..=5, 10..=20, 25..=30));
        let right = IntervalSet::from_iter(seqs!(1..=2, [4], 6..=10, 12..=15, 18..=40));
        let common = left.intersection(&right);
        assert_eq!(
            common,
            IntervalSet::from_iter(seqs!(1..=2, [4], [10], 12..=15, 18..=20, 25..=30))
        );
        common.assert_normalized();
    }

    #[test]
    fn difference() {
        let left = IntervalSet::single(nz(1));
        let right = IntervalSet::from_iter(seqs!([1, 2]));
        let diff = left.difference(&right);
        assert!(diff.is_empty());

        let left = IntervalSet::from_iter(seqs!([1], [10]));
        let right = IntervalSet::single(nz(10));
        let diff = left.difference(&right);
        assert_eq!(diff.seqs().collect::<Vec<_>>(), seqs!([1]));

        let left = IntervalSet::from_iter(seqs!([1, 2]));
        let right = IntervalSet::from_iter(seqs!([1, 2], [10]));
        let diff = left.difference(&right);
        assert!(diff.is_empty());

        let left = IntervalSet::from_iter(seqs!(1..=3));
        let right = IntervalSet::from_iter(seqs!([1, 3]));
        let diff = left.difference(&right);
        assert_eq!(diff.seqs().collect::<Vec<_>>(), seqs!([2]));

        let left = IntervalSet::from_iter(seqs!([1], [3], [5]));
        let right = IntervalSet::from_iter(seqs!(1..=2, [4]));
        let diff = left.difference(&right);
        assert_eq!(diff.seqs().collect::<Vec<_>>(), seqs!([3], [5]));

        let left = IntervalSet::from_iter(seqs!([10, 11]));
        let right = IntervalSet::from_iter(seqs!([10]));
        let diff = left.difference(&right);
        assert_eq!(diff.seqs().collect::<Vec<_>>(), seqs!([11]));
    }

    #[test]
    fn set_compare() {
        let left = IntervalSet::from_iter(seqs!([1], 3..=4, 6..=8));

        let right = IntervalSet::from_iter([nz(1)]);
        assert_eq!(left.partial_cmp(&right), Some(Ordering::Greater));

        let right = IntervalSet::from_iter(seqs!([1], [3], 6..=8));
        assert_eq!(left.partial_cmp(&right), Some(Ordering::Greater));

        let right = IntervalSet::from_iter(seqs!([1], 3..=4, 6..=8));
        assert_eq!(left.partial_cmp(&right), Some(Ordering::Equal));

        let right = IntervalSet::from_iter(seqs!([1], 3..=8));
        assert_eq!(left.partial_cmp(&right), Some(Ordering::Less));

        let right = IntervalSet::from_iter(seqs!([1], 3..=4, 6..=9, 11..=12));
        assert_eq!(left.partial_cmp(&right), Some(Ordering::Less));

        let right = IntervalSet::from_iter(seqs!([1], 3..=7));
        assert_eq!(left.partial_cmp(&right), None);
    }

    #[quickcheck]
    fn qc_contains(seqs: Vec<u8>) {
        let seqs: Vec<_> = seqs
            .into_iter()
            .map(|s| unsafe { NonZeroU64::new_unchecked(s as u64 + 1) })
            .collect();
        let ival_set = IntervalSet::from_iter(seqs.iter().copied());
        let set = HashSet::from_iter(seqs.into_iter());
        for seq in 0..=u8::MAX {
            let seq = unsafe { NonZeroU64::new_unchecked(seq as u64 + 1) };
            if set.contains(&seq) {
                assert!(ival_set.contains(seq));
            } else {
                assert!(!ival_set.contains(seq));
            }
        }
    }

    #[quickcheck]
    fn qc_union(left: Vec<u8>, right: Vec<u8>) {
        let left: Vec<_> = left
            .into_iter()
            .map(|s| unsafe { NonZeroU64::new_unchecked(s as u64 + 1) })
            .collect();
        let left_ival_set = IntervalSet::from_iter(left.iter().copied());
        let left_set = HashSet::from_iter(left.into_iter());

        let right: Vec<_> = right
            .into_iter()
            .map(|s| unsafe { NonZeroU64::new_unchecked(s as u64 + 1) })
            .collect();
        let right_ival_set = IntervalSet::from_iter(right.iter().copied());
        let right_set = HashSet::from_iter(right.into_iter());

        let ival_union = left_ival_set.union(&right_ival_set);
        let set_union = left_set.union(&right_set);
        assert_eq!(
            ival_union.seqs().collect::<HashSet<_>>(),
            set_union.into_iter().copied().collect::<HashSet<_>>()
        );
    }

    #[quickcheck]
    fn qc_intersection(left: Vec<u8>, right: Vec<u8>) {
        let left: Vec<_> = left
            .into_iter()
            .map(|s| unsafe { NonZeroU64::new_unchecked(s as u64 + 1) })
            .collect();
        let left_ival_set = IntervalSet::from_iter(left.iter().copied());
        let left_set = HashSet::from_iter(left.into_iter());

        let right: Vec<_> = right
            .into_iter()
            .map(|s| unsafe { NonZeroU64::new_unchecked(s as u64 + 1) })
            .collect();
        let right_ival_set = IntervalSet::from_iter(right.iter().copied());
        let right_set = HashSet::from_iter(right.into_iter());

        let ival_union = left_ival_set.intersection(&right_ival_set);
        let set_union = left_set.intersection(&right_set);
        assert_eq!(
            ival_union.seqs().collect::<HashSet<_>>(),
            set_union.into_iter().copied().collect::<HashSet<_>>()
        );
    }

    #[quickcheck]
    fn qc_difference(left: Vec<u8>, right: Vec<u8>) {
        let left: Vec<_> = left
            .into_iter()
            .map(|s| unsafe { NonZeroU64::new_unchecked(s as u64 + 1) })
            .collect();
        let left_ival_set = IntervalSet::from_iter(left.iter().copied());
        let left_set = HashSet::from_iter(left.into_iter());

        let right: Vec<_> = right
            .into_iter()
            .map(|s| unsafe { NonZeroU64::new_unchecked(s as u64 + 1) })
            .collect();
        let right_ival_set = IntervalSet::from_iter(right.iter().copied());
        let right_set = HashSet::from_iter(right.into_iter());

        let ival_union = left_ival_set.difference(&right_ival_set);
        let set_union = left_set.difference(&right_set);
        assert_eq!(
            ival_union.seqs().collect::<HashSet<_>>(),
            set_union.into_iter().copied().collect::<HashSet<_>>()
        );
    }
}
