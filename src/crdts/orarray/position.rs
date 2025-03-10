// (c) Copyright 2025 Helsing GmbH. All rights reserved.
use std::cmp::Ordering;

// NOTE: the original implementation has an atoms field with node identifiers in them stored
// inside each Position, but none of that is actually _used_ anywhere, so it's been left over. one
// of the original DSON paper authors confirmed by email on 2023-08-25 that the atoms/nodeid bits
// are leftover from an earlier algorithm they used.
//
// TODO: in the same email, the author suggests that even `f64` may not be an ideal choice
// here since the algorithm assumes that between every two points there exist a third, which is
// true for real numbers, but only kind of true for `f64`. One option is to use a SmallVec<u8; 8>
// so that for the happy case (fewer than 64 pushes) we use no more space, and with more we
// seamlessly transition to a bigger type.
/// A position in an [`OrArray`](super::OrArray).
///
/// This is a wrapper around an `f64` that represents a position in an ordered sequence. The
/// positions are used to determine the order of elements in the array.
// TODO: Consider replacing `Position(f64)` with an unbounded rational
// identifier such as `Fraction`, which stores each coordinate as a growable
// vector of 31-bit digits (base = 2^31).  A 64-bit float yields only 2^52
// distinct values in our interval, so after roughly fifty “insert-the-average”
// operations in the same gap the two neighbours become bit-identical and
// `between()` can no longer create a fresh position, forcing an expensive
// renumbering of the entire list.  By contrast, a vector-based representation can
// always append another digit to refine the interval, ensuring that a new
// position can be generated.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct Position(pub(in super::super) f64);

impl std::fmt::Debug for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq for Position {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Position {}

impl PartialOrd for Position {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Position {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Position {
    pub(crate) const LOWER: f64 = 0.0;
    pub(crate) const UPPER: f64 = 32767.0;

    /// Returns a new position between two existing positions.
    pub fn between(left: Option<Position>, right: Option<Position>) -> Self {
        // NOTE: the original implementation also takes a node id (ie, `Identifier`), but then
        // never does anything with it, so we leave it off here.
        Self(
            (left.map(|p| p.0).unwrap_or(Position::LOWER)
                + right.map(|p| p.0).unwrap_or(Position::UPPER))
                / 2.0,
        )
    }

    /// Creates a `Position` from a raw `f64` value.
    ///
    /// Returns `None` if the value is outside the valid range.
    pub fn from_raw(value: f64) -> Option<Position> {
        (Position::LOWER..=Position::UPPER)
            .contains(&value)
            .then_some(Self(value))
    }

    /// Returns the raw `f64` value of the position.
    pub fn as_raw(&self) -> f64 {
        self.0
    }
}
