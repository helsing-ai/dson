// (c) Copyright 2025 Helsing GmbH. All rights reserved.
use super::snapshot::{self, SingleValueError, SingleValueIssue, ToValue};
use crate::{
    CausalContext, CausalDotStore, Dot, DotFun, DotStoreJoin, Identifier, api,
    dotstores::{DotChange, DotStore, DryJoinOutput},
    sentinel::{Sentinel, ValueSentinel},
};
use std::cmp::Ordering;

/// A **Multi-Value Register**, a CRDT for storing a single, atomic value.
///
/// `MvReg` is one of the three core CRDT primitives provided by this crate, alongside [`crate::OrMap`] and
/// [`crate::OrArray`]. It is used to hold primitive values like integers, strings, or booleans.
///
/// ## Conflict Handling
///
/// When two replicas concurrently write different values to the same `MvReg`, the register will
/// hold both values simultaneously. This is the "multi-value" aspect. A subsequent read will return
/// all conflicting values, allowing the application to resolve the conflict in a way that makes
/// sense for its use case. A subsequent write will overwrite all conflicting values, resolving the
/// conflict by establishing a new, single value.
///
/// If a value is concurrently cleared and overwritten, the written value "wins" and the register
/// will contain the new value.
///
/// ## Usage
///
/// An `MvReg` is typically used as a value within an [`crate::OrMap`] or [`crate::OrArray`].
/// It is not usually used as a top-level CRDT.
///
/// ```rust
/// # use dson::{CausalDotStore, MvReg, crdts::{mvreg::MvRegValue, snapshot::ToValue}, Identifier, sentinel::DummySentinel};
/// // Create a new CausalDotStore containing an MvReg.
/// let mut doc: CausalDotStore<MvReg> = CausalDotStore::new();
/// let id = Identifier::new(0, 0);
///
/// // Create a delta to write a value.
/// let delta = doc.store.write(MvRegValue::U64(42), &doc.context, id);
///
/// // Merge the delta into the document.
/// doc = doc.join(delta, &mut DummySentinel).unwrap();
///
/// // The value can now be read from the register.
/// assert_eq!(*doc.store.value().unwrap(), MvRegValue::U64(42));
/// ```
///
/// You can find more convenient, higher-level APIs for manipulating `MvReg` in the
/// [`api::register`] module.
#[derive(Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct MvReg(pub DotFun<MvRegValue>);

impl std::fmt::Debug for MvReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${:?}", self.0)
    }
}

macro_rules! impl_partial_eq {
    ({$($t:ty),+}) => {
        $(impl_partial_eq!($t);)+
    };

    ($t:ty) => {
        impl PartialEq<$t> for MvReg {
            fn eq(&self, other: &$t) -> bool {
                self.values().into_iter().any(|v| v == other)
            }
        }
    };
}
impl_partial_eq!({[u8], &[u8], str, &str, bool, f64, u64, i64});
// i32 because it's the "default" inference integer type
impl_partial_eq!(i32);
// byte literals
impl<const N: usize> PartialEq<&[u8; N]> for MvReg {
    fn eq(&self, other: &&[u8; N]) -> bool {
        self.values().into_iter().any(|v| v.eq(other))
    }
}

impl DotStore for MvReg {
    fn dots(&self) -> CausalContext {
        self.0.dots()
    }

    fn add_dots_to(&self, other: &mut CausalContext) {
        self.0.add_dots_to(other);
    }

    fn is_bottom(&self) -> bool {
        self.0.is_bottom()
    }

    fn subset_for_inflation_from(&self, frontier: &CausalContext) -> Self {
        Self(DotFun::subset_for_inflation_from(&self.0, frontier))
    }
}

impl<S> DotStoreJoin<S> for MvReg
where
    S: ValueSentinel<MvRegValue>,
{
    fn join(
        (m1, cc1): (Self, &CausalContext),
        (m2, cc2): (Self, &CausalContext),
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<Self, S::Error>
    where
        Self: Sized,
        S: Sentinel,
    {
        // NOTE! When making changes to this method, consider if corresponding
        // changes need to be done to ::dry_join as well!

        Ok(Self(DotFun::join(
            (m1.0, cc1),
            (m2.0, cc2),
            on_dot_change,
            sentinel,
        )?))
    }
    fn dry_join(
        (m1, cc1): (&Self, &CausalContext),
        (m2, cc2): (&Self, &CausalContext),
        sentinel: &mut S,
    ) -> Result<DryJoinOutput, S::Error>
    where
        Self: Sized,
        S: Sentinel,
    {
        DotFun::dry_join((&m1.0, cc1), (&m2.0, cc2), sentinel)
    }
}

/// The value stored in a [`MvReg`].
///
/// This enum represents the different types of values that can be stored in a multi-value
/// register.
// NOTE: Why no U32 or I32? Make this a serialization concern.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
// TODO(jon): should we make this #[non_exhaustive] so we can add to it without breaking?
pub enum MvRegValue {
    // NOTE: the #[serde] here is needed to get efficient encoding of byte-arrays for
    // protocols that support it (like msgpack):
    // <https://docs.rs/rmp-serde/1/rmp_serde/index.html#efficient-storage-of-u8-types>
    Bytes(#[cfg_attr(feature = "serde", serde(with = "serde_bytes"))] Vec<u8>),
    String(String),
    Float(f32),
    Double(f64),
    U64(u64),
    I64(i64),
    Bool(bool),
    Timestamp(api::timestamp::Timestamp),
    #[cfg(feature = "ulid")]
    Ulid(ulid::Ulid),
}

impl MvRegValue {
    /// When ordering MvRegValue instances of different types, we order them
    /// according to this order.
    const fn comparison_order(&self) -> usize {
        // Desired order: Bytes > String > Ulid > Timestamp > Double > U64 > I64 > Bool
        match self {
            MvRegValue::Bytes(_) => 8,
            MvRegValue::String(_) => 7,
            #[cfg(feature = "ulid")]
            MvRegValue::Ulid(_) => 6,
            MvRegValue::Timestamp(_) => 5,
            MvRegValue::Double(_) => 4,
            MvRegValue::Float(_) => 3,
            MvRegValue::U64(_) => 2,
            MvRegValue::I64(_) => 1,
            MvRegValue::Bool(_) => 0,
        }
    }
}

macro_rules! impl_from {
(
    $(
        $source:ty => $target:ident $(with $conv:ident)?
    ),* $(,)?
    ) => {
        $(
            impl From<$source> for MvRegValue {
                fn from(value: $source) -> Self {
                    Self::$target(impl_from!(value$(, $conv)?))
                }
            }
        )*
    };

    ($value:ident, $conv:ident) => {
        $value.$conv()
    };

    ($value:ident) => {
        $value
    };
}

impl_from!(
    &[u8]      => Bytes with into,
    Vec<u8>    => Bytes,
    String     => String,
    &str       => String with to_string,
    f64        => Double,
    u8         => U64 with into,
    u16        => U64 with into,
    u32        => U64 with into,
    u64        => U64,
    i8         => I64 with into,
    i16        => I64 with into,
    i32        => I64 with into,
    i64        => I64,
    bool       => Bool,
);

#[cfg(feature = "ulid")]
impl From<ulid::Ulid> for MvRegValue {
    fn from(value: ulid::Ulid) -> Self {
        Self::Ulid(value)
    }
}

impl std::fmt::Debug for MvRegValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes(inner) => write!(f, "{inner:02X?}"),
            Self::String(inner) => inner.fmt(f),
            Self::Bool(inner) => inner.fmt(f),
            // Make sure to always print at least 1 decimal, so we can non-ambiguously
            // tell apart I64 and floats (this is achieved by {:?} instead of {}).
            Self::Float(inner) => write!(f, "{inner:?}f"),
            Self::Double(inner) => write!(f, "{inner:?}d"),
            Self::U64(inner) => write!(f, "{inner}u"),
            Self::I64(inner) => write!(f, "{inner}"),
            Self::Timestamp(inner) => inner.fmt(f),
            #[cfg(feature = "ulid")]
            Self::Ulid(inner) => inner.fmt(f),
        }
    }
}

impl PartialEq for MvRegValue {
    fn eq(&self, other: &Self) -> bool {
        use MvRegValue::*;
        match (self, other) {
            (Bytes(b1), Bytes(b2)) => b1.eq(b2),
            (String(s1), String(s2)) => s1.eq(s2),
            (Double(d1), Double(d2)) => d1.total_cmp(d2).is_eq(),
            (Float(d1), Float(d2)) => d1.total_cmp(d2).is_eq(),
            (U64(u1), U64(u2)) => u1.eq(u2),
            (I64(i1), I64(i2)) => i1.eq(i2),
            (Bool(b1), Bool(b2)) => b1.eq(b2),
            (Timestamp(t1), Timestamp(t2)) => t1.eq(t2),
            #[cfg(feature = "ulid")]
            (Ulid(ulid1), Ulid(ulid2)) => ulid1.eq(ulid2),
            _ => false,
        }
    }
}
impl Eq for MvRegValue {}

impl PartialEq<[u8]> for MvRegValue {
    fn eq(&self, other: &[u8]) -> bool {
        matches!(self, Self::Bytes(b) if b == other)
    }
}
impl PartialEq<&[u8]> for MvRegValue {
    fn eq(&self, other: &&[u8]) -> bool {
        matches!(self, Self::Bytes(b) if b == other)
    }
}
impl PartialEq<str> for MvRegValue {
    fn eq(&self, other: &str) -> bool {
        matches!(self, Self::String(s) if s == other)
    }
}
impl PartialEq<&str> for MvRegValue {
    fn eq(&self, other: &&str) -> bool {
        matches!(self, Self::String(s) if s == other)
    }
}
impl PartialEq<bool> for MvRegValue {
    fn eq(&self, other: &bool) -> bool {
        matches!(self, Self::Bool(b) if b == other)
    }
}
impl PartialEq<f64> for MvRegValue {
    fn eq(&self, other: &f64) -> bool {
        matches!(self, Self::Double(f) if f == other)
    }
}
impl PartialEq<u64> for MvRegValue {
    fn eq(&self, other: &u64) -> bool {
        match self {
            Self::U64(u) => u == other,
            Self::I64(i) => u64::try_from(*i).is_ok_and(|u| &u == other),
            Self::Bytes(_)
            | Self::String(_)
            | Self::Double(_)
            | Self::Float(_)
            | Self::Bool(_)
            | Self::Timestamp(_) => false,
            #[cfg(feature = "ulid")]
            Self::Ulid(_) => false,
        }
    }
}
impl PartialEq<i64> for MvRegValue {
    fn eq(&self, other: &i64) -> bool {
        match self {
            Self::U64(u) => i64::try_from(*u).is_ok_and(|i| &i == other),
            Self::I64(i) => i == other,
            Self::Bytes(_)
            | Self::String(_)
            | Self::Float(_)
            | Self::Double(_)
            | Self::Bool(_)
            | Self::Timestamp(_) => false,
            #[cfg(feature = "ulid")]
            Self::Ulid(_) => false,
        }
    }
}
// i32 because it's the "default" inference integer type
impl PartialEq<i32> for MvRegValue {
    fn eq(&self, other: &i32) -> bool {
        match self {
            Self::U64(u) => i32::try_from(*u).is_ok_and(|i| &i == other),
            Self::I64(i) => i32::try_from(*i).is_ok_and(|i| &i == other),
            Self::Bytes(_)
            | Self::String(_)
            | Self::Double(_)
            | Self::Float(_)
            | Self::Bool(_)
            | Self::Timestamp(_) => false,
            #[cfg(feature = "ulid")]
            Self::Ulid(_) => false,
        }
    }
}
// byte literals
impl<const N: usize> PartialEq<&[u8; N]> for MvRegValue {
    fn eq(&self, other: &&[u8; N]) -> bool {
        matches!(self, Self::Bytes(b) if b == other)
    }
}

impl PartialOrd for MvRegValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MvRegValue {
    fn cmp(&self, other: &Self) -> Ordering {
        use MvRegValue::*;
        // For order of cross-variant comparisons, see:
        // [`MvRegValue::comparison_order`]
        match (self, other) {
            (Bytes(b1), Bytes(b2)) => b1.cmp(b2),
            (String(s1), String(s2)) => s1.cmp(s2),
            (Double(d1), Double(d2)) => d1.total_cmp(d2),
            (Float(d1), Float(d2)) => d1.total_cmp(d2),
            (U64(u1), U64(u2)) => u1.cmp(u2),
            (I64(i1), I64(i2)) => i1.cmp(i2),
            (Bool(b1), Bool(b2)) => b1.cmp(b2),
            (Timestamp(t1), Timestamp(t2)) => t1.cmp(t2),
            #[cfg(feature = "ulid")]
            (Ulid(ulid1), Ulid(ulid2)) => ulid1.cmp(ulid2),
            (a, b) => {
                let a_order = a.comparison_order();
                let b_order = b.comparison_order();
                debug_assert_ne!(
                    a_order, b_order,
                    "match must handle all comparisons between similar variants"
                );
                a_order.cmp(&b_order)
            }
        }
    }
}

impl<'doc> ToValue for &'doc MvReg {
    type Values = snapshot::MvReg<'doc>;
    type Value = &'doc MvRegValue;
    type LeafValue = MvRegValue;

    /// Returns the set of all possible values for this register in an arbitrary
    /// order.
    ///
    /// NOTE: values are ordered by the sequence number of their associated dot.
    fn values(self) -> Self::Values {
        snapshot::MvReg {
            values: self.0.values(),
        }
    }

    /// Returns the single value of the MvReg.
    ///
    /// If the value has been cleared, [`SingleValueIssue::Cleared`] is returned as `Err`.
    ///
    /// If there are multiple (ie, conflicting) values, [`SingleValueIssue::HasConflict`] is
    /// returned as `Err`.
    fn value(self) -> Result<Self::Value, Box<SingleValueError<Self::LeafValue>>> {
        match self.0.len() {
            0 => Err(Box::new(SingleValueError {
                path: Vec::new(),
                issue: SingleValueIssue::Cleared,
            })),
            1 => {
                let a_dot = self
                    .dots()
                    .one()
                    .expect("if we have values, we should also have dots");
                Ok(self.0.get(&a_dot).expect(
                    ".dots is the keys of the map, so if we get a Dot back, it must be present",
                ))
            }
            _ => {
                let conflicts = self.0.values().cloned().collect();
                Err(Box::new(SingleValueError {
                    path: Vec::new(),
                    issue: SingleValueIssue::HasConflict(conflicts),
                }))
            }
        }
    }
}

impl MvReg {
    #[doc(hidden)]
    pub fn push(&mut self, dot: Dot, value: impl Into<MvRegValue>) {
        self.0.set(dot, value.into());
    }

    /// Creates a CRDT that represents the overwrite of all past values of this
    /// register with the value in `self`.
    pub fn write(&self, v: MvRegValue, cc: &CausalContext, id: Identifier) -> CausalDotStore<Self> {
        let dot = cc.next_dot_for(id);

        // write collapses the state of the value
        let mut new_state = DotFun::default();
        new_state.set(dot, v);

        let mut new_cc = CausalContext::new();
        new_cc.insert_dot(dot);
        self.add_dots_to(&mut new_cc);

        CausalDotStore {
            store: Self(new_state),
            context: new_cc,
        }
    }

    /// Creates a CRDT that represents the erasure of all past values of this register.
    pub fn clear(&self) -> CausalDotStore<Self> {
        CausalDotStore {
            store: Self::default(),
            context: self.dots(),
        }
    }

    /// Directly retains only the values for which a predicate is true.
    ///
    /// This change is not represented as a delta-CRDT, meaning this may cause unintended
    /// consequences if `self` is later distributed along with an unmodified [`CausalContext`]. Only
    /// use this method if you know what you are doing.
    pub fn retain_immediately(&mut self, f: impl FnMut(&Dot, &mut MvRegValue) -> bool) {
        self.0.retain(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Dot,
        crdts::test_util::join_harness,
        sentinel::{DummySentinel, test::NoChangeValidator},
    };

    #[test]
    fn empty() {
        let cds = CausalDotStore::<MvReg>::default();
        assert_eq!(
            cds.store.value().unwrap_err().issue,
            SingleValueIssue::Cleared
        );
        assert_eq!(cds.store.values().len(), 0);
        assert_eq!(cds.store.values().get(0), None);
    }

    #[test]
    fn clear_and_write() {
        join_harness(
            MvReg::default(),
            |CausalDotStore {
                 store: m,
                 context: cc,
             },
             id| { m.write(MvRegValue::Bool(false), &cc, id) },
            |m, cc, id| m.write(MvRegValue::Bool(true), &cc, id),
            |m, _cc, _id| m.clear(),
            NoChangeValidator,
            |CausalDotStore { store: m, .. }, _| {
                // for a concurrent clear and write, only the written value should remain
                assert!(!m.is_bottom());
                let values = m.values();
                assert_eq!(values.len(), 1);
                assert!(values.into_iter().any(|v| v == &MvRegValue::Bool(true)));
            },
        );
    }

    #[quickcheck]
    fn values(vs: Vec<(Dot, MvRegValue)>) {
        // We shouldn't have the same dot for multiple values
        let mut dedup_dots = std::collections::HashSet::new();
        let vs: Vec<_> = vs.into_iter().filter(|x| dedup_dots.insert(x.0)).collect();
        let mut cds = CausalDotStore::<MvReg>::default();
        let mut possible_values = Vec::<MvRegValue>::default();
        for (dot, v) in vs.clone() {
            cds.store.0.set(dot, v.clone());
            possible_values.push(v);
        }
        {
            let mut a = possible_values.clone();
            let mut values_in_store = cds.store.values().into_iter().cloned().collect::<Vec<_>>();
            a.sort_unstable();
            values_in_store.sort_unstable();
            assert_eq!(a, values_in_store);
        }
        let expected_value = if possible_values.len() == 1 {
            Ok(possible_values.first().unwrap())
        } else if !possible_values.is_empty() {
            Err(Box::new(SingleValueError {
                path: Vec::new(),
                issue: SingleValueIssue::HasConflict(possible_values.into_iter().collect()),
            }))
        } else {
            Err(Box::new(SingleValueError {
                path: Vec::new(),
                issue: SingleValueIssue::Cleared,
            }))
        };
        assert_eq!(cds.store.value(), expected_value, "input: {vs:?}");
    }

    #[quickcheck]
    fn write(vs: Vec<(Dot, MvRegValue)>, new: MvRegValue) {
        let mut dedup_dots = std::collections::HashSet::new();
        let vs: Vec<_> = vs.into_iter().filter(|x| dedup_dots.insert(x.0)).collect();

        let mut cds = CausalDotStore::<MvReg>::new();

        for &(dot, ref v) in &vs {
            cds.store.0.set(dot, v.clone());
            cds.store.add_dots_to(&mut cds.context);
        }

        // Find an unused causal track, which we can use as 'our' id, so
        // we're sure to have a compact track (otherwise we will trigger asserts).
        let id = cds
            .context
            .unused_identifier()
            .expect("test case is not large enough to have used all identifiers");

        // write a new value that dominates all the past writes
        let delta = cds.store.write(new, &cds.context, id);
        assert_eq!(delta.store.0.len(), 1);
        let new_dot = delta.store.0.keys().next().unwrap();
        for &(dot, _) in &vs {
            assert!(delta.context.dot_in(dot));
        }
        // check that the delta takes effect when joined into the original state
        let CausalDotStore { store, context } = cds.join(delta, &mut DummySentinel).unwrap();
        assert_eq!(store.values().len(), 1);

        // clear the map, which will include clearing the new value
        let clear = store.clear();
        assert_eq!(clear.store.0.len(), 0);
        // NOTE: one might expect that clear.context would also contain all the dots in vs,
        // but it does not. instead, it only contains the dot that it _observed_, which is the dot
        // from the write which in turn dominates all the dots in vs.
        for &(dot, _) in &vs {
            assert!(!clear.context.dot_in(dot));
        }
        assert!(clear.context.dot_in(new_dot));
        // check that the delta takes effect when joined into the original state
        let store = MvReg::join(
            (store, &context),
            (clear.store, &clear.context),
            &mut |_| {},
            &mut DummySentinel,
        )
        .unwrap();
        assert_eq!(store.values().len(), 0);
        assert_eq!(store.value().unwrap_err().issue, SingleValueIssue::Cleared);
    }

    #[quickcheck]
    fn partial_cmp_is_involutive(v1: MvRegValue, v2: MvRegValue) {
        assert_eq!(v1.cmp(&v2), v2.cmp(&v1).reverse());
    }
}
