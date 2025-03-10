// (c) Copyright 2025 Helsing GmbH. All rights reserved.
use super::{
    NoExtensionTypes, TypeVariantValue, Value, ValueRef,
    mvreg::MvRegValue,
    snapshot::{self, AllValues, CollapsedValue, SingleValueError, SingleValueIssue, ToValue},
};
use crate::{
    CausalContext, CausalDotStore, DETERMINISTIC_HASHER, Dot, DotFun, DotFunMap, DotMap,
    DotStoreJoin, ExtensionType, Identifier, MvReg, OrMap,
    dotstores::{DotChange, DotStore, DryJoinOutput},
    sentinel::{DummySentinel, KeySentinel, Sentinel, TypeSentinel, ValueSentinel, Visit},
};
pub use position::Position;
use std::{convert::Infallible, fmt};

pub(super) mod position;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct Uid(Dot);

impl From<Dot> for Uid {
    fn from(value: Dot) -> Self {
        Self(value)
    }
}

impl fmt::Debug for Uid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Uid")
            .field(&self.0.actor())
            .field(&self.0.sequence())
            .finish()
    }
}

impl Uid {
    // needs to be public for integration tests
    #[doc(hidden)]
    pub fn dot(&self) -> Dot {
        self.0
    }
}

/// An **Observed-Remove Array**, a list-like CRDT that supports concurrent insertions, updates,
/// removals, and moves of elements.
///
/// `OrArray` is one of the three core CRDT primitives provided by this crate, alongside [`OrMap`]
/// and [`MvReg`]. It can be nested within other CRDTs to build complex, JSON-like data structures.
///
/// ## Stable Positioning
///
/// This type does not use integer indices directly as the position would require updating multiple
/// element positions following an insertion or deletion, which isn't feasible to do in a CRDT
/// context. Therefore, the implementation uses _stable identifiers_ (Martin Kleppmann; Moving
/// elements in list CRDTs (2020)) in the form of real numbers (`f64`), which allow insertion and
/// deletion without updating other elements. In this model, an insertion of an element between an
/// element at position 𝑝1 and one at 𝑝2, inserts an element at position (𝑝1 + 𝑝2)/2 (see
/// [`Position::between`]. The array is then the sequence of values sorted by the corresponding
/// position in ascending order. For example, the array [𝑎, 𝑏, 𝑐] can be represented by the set
/// {(𝑎, 1.0), (𝑏, 7.5), (𝑐, 42.7)}.
///
/// This type allows and reconciles concurrent operations (as all CRDTs) do. As a result, a given
/// position may be ambiguous, similarly to the value of an [`MvReg`] under concurrent writes. To
/// this end, the position is actually a set of _possible positions_. For example, if element 𝑏 is
/// concurrently moved before 𝑎 and after 𝑐, it’s position may be {0.3, 54}. To sort the array,
/// replicas deterministically choose an element from the position set (eg, the maximum) to use for
/// ordering the values. Since all mutations explicitly also set the position of an element, the
/// set of positions is collapsed any time a mutation happens.
///
/// ## Unique Identifiers
///
/// To keep track of values as they change positions under concurrent operations, we identify
/// each array element by a unique identifier. Concretely, we use the dot with which the element
/// was created as the creation point is unique. The representation of an [`OrArray`] is therefore
/// a map from unique identifiers to value-positions pairs. For example, the aforementioned array
/// can be represented as the map:
///   {(𝑖, 1) ↦ (𝑎, {1.0}),
///    (𝑗, 1) ↦ (𝑏, {0.3, 54}),
///    (𝑖, 2) ↦ (𝑐, {42.7})
///   }
///
/// where (𝑖, 1) is the dot marking the creation even of element 𝑎, (𝑗, 1) is the creation event of
/// 𝑏, and (𝑖, 2) is the creation event of 𝑐.
///
/// When an element is inserted for the first time, the API receives the unique identifier that
/// identifies the inserted element until its deletion.
///
/// ## Usage
///
/// Like `OrMap`, an `OrArray` is typically wrapped in a [`CausalDotStore`]. Modifications are
/// performed by creating a "delta" CRDT, which is then merged back into the original.
///
/// ```rust
/// # use dson::{CausalDotStore, OrArray, MvReg, crdts::{Value, mvreg::MvRegValue, orarray::{Position, Uid}, snapshot::{CollapsedValue, ToValue}}, Identifier, sentinel::DummySentinel};
/// // Create a new CausalDotStore containing an OrArray.
/// let mut doc: CausalDotStore<OrArray> = CausalDotStore::new();
/// let id = Identifier::new(0, 0);
///
/// // Create a delta to insert a value at the beginning of the array.
/// let delta = dson::api::array::insert(
///     |cc, id| MvReg::default().write(MvRegValue::U64(42), cc, id).map_store(Value::Register),
///     doc.store.len(),
/// )(&doc.store, &doc.context, id);
///
/// // Merge the delta into the document.
/// doc = doc.join(delta, &mut DummySentinel).unwrap();
///
/// // The value can now be read from the array.
/// let array = doc.store.value().unwrap();
/// let val = doc.store.get(0).unwrap();
/// assert_eq!(val.reg.value().unwrap(), &MvRegValue::U64(42));
/// ```
///
/// You can find more convenient, higher-level APIs for manipulating `OrArray` in the
/// [`api::array`](crate::api::array) module. The methods on `OrArray` itself are low-level and
/// useful when implementing custom CRDTs or when you need fine-grained control over
/// delta creation.
#[derive(Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct OrArray<C = NoExtensionTypes>(pub(super) DotMap<Uid, PairMap<C>>);

impl<C> fmt::Debug for OrArray<C>
where
    C: fmt::Debug + ExtensionType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[]{:?}", self.0)
    }
}

impl<C> DotStore for OrArray<C>
where
    C: ExtensionType,
{
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
        Self(DotMap::subset_for_inflation_from(&self.0, frontier))
    }
}

impl<C, S> DotStoreJoin<S> for OrArray<C>
where
    C: ExtensionType + DotStoreJoin<S> + Default + fmt::Debug + Clone + PartialEq,
    S: Visit<Uid>
        + Visit<String>
        + KeySentinel
        + TypeSentinel<C::ValueKind>
        + ValueSentinel<MvRegValue>,
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

        Ok(Self(DotMap::join(
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
        DotMap::dry_join((&m1.0, cc1), (&m2.0, cc2), sentinel)
    }
}

/// The position and value information for a given entry in an [`OrArray`].
#[derive(Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub(super) struct PairMap<Custom> {
    /// The value assigned to this element.
    ///
    /// For reference, this field is called `first` in the original DSON paper and implementation.
    #[doc(alias = "first")]
    pub(super) value: TypeVariantValue<Custom>,

    /// The set of positions assigned to this element.
    ///
    /// For reference, this field is called `second` in the original DSON paper and implementation.
    #[doc(alias = "second")]
    pub(super) positions: DotFunMap<DotFun<Position>>,
}

impl<C> fmt::Debug for PairMap<C>
where
    C: ExtensionType + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("")
            .field(&format_args!("{:?}", self.value))
            .field(&format_args!("pos={:?}", self.positions))
            .finish()
    }
}

impl<C> DotStore for PairMap<C>
where
    C: ExtensionType,
{
    fn add_dots_to(&self, other: &mut CausalContext) {
        self.value.add_dots_to(other);
        self.positions.add_dots_to(other);
    }

    fn is_bottom(&self) -> bool {
        self.value.is_bottom() && self.positions.is_bottom()
    }

    fn subset_for_inflation_from(&self, frontier: &CausalContext) -> Self {
        Self {
            value: self.value.subset_for_inflation_from(frontier),
            positions: self.positions.subset_for_inflation_from(frontier),
        }
    }
}

impl<C, S> DotStoreJoin<S> for PairMap<C>
where
    C: ExtensionType + DotStoreJoin<S> + fmt::Debug + Clone + PartialEq,
    S: Visit<String>
        + Visit<Uid>
        + KeySentinel
        + TypeSentinel<C::ValueKind>
        + ValueSentinel<MvRegValue>,
{
    fn join(
        ds1: (Self, &CausalContext),
        ds2: (Self, &CausalContext),
        on_dot_change: &mut dyn FnMut(DotChange),
        sentinel: &mut S,
    ) -> Result<Self, S::Error>
    where
        Self: Sized,
        S: Sentinel,
    {
        // NOTE! When making changes to this method, consider if corresponding
        // changes need to be done to ::dry_join as well!

        let (m1, cc1) = ds1;
        let (m2, cc2) = ds2;

        let value = DotStoreJoin::join((m1.value, cc1), (m2.value, cc2), on_dot_change, sentinel)?;
        let positions = DotStoreJoin::join(
            (m1.positions, cc1),
            (m2.positions, cc2),
            on_dot_change,
            // We don't consider the Position as a value, even though it works much like a MvReg so
            // changes are not observed by the sentinel
            &mut DummySentinel,
        )
        .expect("DummySentinel is Infallible");

        Ok(PairMap { value, positions })
    }

    fn dry_join(
        ds1: (&Self, &CausalContext),
        ds2: (&Self, &CausalContext),
        sentinel: &mut S,
    ) -> Result<DryJoinOutput, S::Error>
    where
        Self: Sized,
        S: Sentinel,
    {
        let (m1, cc1) = ds1;
        let (m2, cc2) = ds2;

        let value = DotStoreJoin::dry_join((&m1.value, cc1), (&m2.value, cc2), sentinel)?;
        let positions = DotStoreJoin::dry_join(
            (&m1.positions, cc1),
            (&m2.positions, cc2),
            // We don't consider the Position as a value, even though it works much like a MvReg so
            // changes are not observed by the sentinel
            &mut DummySentinel,
        )
        .expect("DummySentinel is Infallible");

        Ok(positions.union(value))
    }
}

impl<C> OrArray<C> {
    /// Create an array from raw entries
    pub fn from_entries<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (Uid, TypeVariantValue<C>, DotFunMap<DotFun<Position>>)>,
    {
        let iter = iter
            .into_iter()
            .map(|(uid, value, positions)| (uid, PairMap { value, positions }));

        Self(DotMap::from_iter(iter))
    }

    #[doc(hidden)]
    pub fn insert_raw(
        &mut self,
        uid: Uid,
        pos: impl Iterator<Item = (Dot, Dot, f64)>,
        value: TypeVariantValue<C>,
    ) {
        let mut dotfunmap = DotFunMap::<DotFun<Position>>::default();
        for (dot1, dot2, ordering_f64) in pos {
            let mut dotfun = DotFun::<Position>::default();
            dotfun.set(dot2, Position(ordering_f64));
            dotfunmap.set(dot1, dotfun);
        }

        self.0.insert(
            uid,
            PairMap {
                value,
                positions: dotfunmap,
            },
        )
    }

    /// Yields the array's elements in random order.
    ///
    /// This deterministically resolves any ambiguities around each element's position (eg, if it
    /// was concurrently moved by multiple actors) and returns the elements along with those
    /// resolved positions. Given the inherently concurrent nature of this data structure, calling
    /// `iter_as_is` after joining with another [`OrArray`] may yield significantly different
    /// positions.
    pub fn iter_as_is(&self) -> impl Iterator<Item = (&TypeVariantValue<C>, Uid, Position)> {
        self.0.iter().map(|(uid, pair)| {
            let inner_map = &pair.value;

            // choose which position among the possible positions to use for this element.
            // there may be more than one in the presence of concurrent moves. this operation
            // needs to happen identically across all replicas that have the same `OrArray`
            // (ie, it must be deterministic), so we always pick the max.
            let positions = &pair.positions;

            // positions is a Dot => { Dot => Position }.
            // the outer Dot is "someone created a position set"
            // the inner Dot is "someone updated a position set"
            // so if we have:
            //
            // {d1 => {d2 => 0.5, d3 => 1.5}, d4 => {100}}
            //
            // it implies that d2 and d3 were two simultaneous `mv`s of the element when its
            // value was the one created at d1. d4, meanwhile, was a concurrent apply to the
            // element (that may or may not have changed its original index).
            //
            // in this case, we're going to (arbitrarily but deterministically) pick the max
            // position of the max outer dot, so (assuming d4 > d3 > d2 > d1): 100.
            //
            // NOTE: we cannot rely on the position always being available.
            //
            // Consider the following sequence of events:
            //
            // 1. A composite array element is created by node A:
            //
            //    store = {Uid(1) => (
            //             value => {"foo": {2 => true}},
            //             pos => {3 => {4 => 100.0}})}
            //    cc = {1..=4}
            //
            // 2. An internal mutation is made to this element:
            //
            //    store = {Uid(1) => (
            //             value => {"foo": {2 => true}, "bar": {5 => true}},
            //             pos => {6 => {7 => 50.0}} )}
            //    cc = {1..=7}
            //
            // 3. Now node B receives only this second update.
            //
            //    delta_store = {Uid(1) => (
            //                   value => {"bar": {5 => true}}, pos => {6 => {7 => 50.0}})}
            //    delta_cc = {3..=7}
            //
            //    >> Results in state:
            //
            //    store = {Uid(1) => (value => {"bar": {5 => true}}, pos => {6 => {7 => 50.0}})}
            //    cc = {3..=7}
            //
            // 4. Node B now deletes the element:
            //
            //    store = {}
            //    cc = {3..=7}
            //
            // 5. And then sends an update to A:
            //
            //    delta_store = {}
            //    delta_cc = {5..=7}
            //
            //    >> Results in state
            //
            //    store = {Uid(1) => (value => {"foo": {2 => true}}, pos => {})}
            //    cc = {1..=7}
            //
            // This is an example of a scenario where we end up with an array element which contains
            // a non-bottom value, yet doesn't have a position defined. Importantly, this is a
            // permanent situation: node A can later send a catch-up delta to B, which will achieve
            // global consistency, but not resolve the incomplete state.
            //
            // To get out of this situation, a node must actively modify the element by generating a
            // new APPLY operation. This will create a new position root, so the element will have a
            // well defined index again. But if no action is taken, the best we can do is synthesize
            // a position from the uid, so we have an arbitrary but deterministic ordering. That's
            // what we do here.
            //
            // The rationale for this choice is detailed in the following YADR.
            //
            // YADR: 2024-06-18 Array elements without a position defined
            //
            // In the context of dealing with array elements which don't have a position defined,
            // we faced a decision of how to expose these elements through the public OrArray API.
            //
            // We decided for assigning these elements an arbitrary but deterministic position, as
            // a function of their uid, and neglected to attempt to map them to the start or end of
            // the array, or provide a separate API for access to position-less elements, when their
            // uid is not yet known.
            //
            // We did this to achieve minimal impact to the user-facing API, to avoid increasing the
            // cognitive burden of using this crate, and to ensure that every node has a consistent
            // view of the array when they share the same state, accepting that users may be
            // surprised to find that a non-move operation (like a delete) can result in an element
            // being assigned a different position.
            //
            // We think this is the right trade-off because this is a rare edge case, and placing
            // the burden of handling it on users (by providing a separate access interface)
            // would've been unreasonable. Additionally, non-deterministic views of the array would
            // have violated a core assumption that nodes in sync with each other have the same view
            // of the state.
            let p = if let Some(max_root) = positions.keys().max() {
                let at_max_root = positions
                    .get(&max_root)
                    .expect("this is the max key from just above, so must exist");
                let max_dot = at_max_root
                    .keys()
                    .max()
                    .expect("every position set has at least one position (from its creator)");
                at_max_root
                    .get(&max_dot)
                    .expect("this is the max key from just above, so must exist")
            } else {
                const MASK: u64 = u64::MAX >> (u64::BITS - f64::MANTISSA_DIGITS + 1);
                const SCALE_FACTOR: f64 = Position::UPPER / MASK as f64;

                let value = (DETERMINISTIC_HASHER.hash_one(uid) & MASK) as f64;
                &Position::from_raw(value * SCALE_FACTOR).expect("within range")
            };

            (inner_map, *uid, *p)
        })
    }

    /// Iterates over the raw entries of this array
    pub fn iter_entries(
        &self,
    ) -> impl Iterator<Item = (Uid, &TypeVariantValue<C>, &DotFunMap<DotFun<Position>>)> {
        self.0.iter().map(|(uid, pair)| {
            let value = &pair.value;
            let positions = &pair.positions;

            (*uid, value, positions)
        })
    }

    /// Iterates over array elements in an arbitrary order with mutable access.
    ///
    /// This is similar to `iter_as_is`, but does not resolve ambiguities around each element's
    /// position, thus it is faster.
    ///
    /// Invalidates the dots cache for all the array's entries, so calling `.dots()` on this
    /// collection after invoking this method may be quite slow (it has to call `.dots()` on all
    /// the entries).
    pub fn iter_mut_and_invalidate(
        &mut self,
    ) -> impl ExactSizeIterator<Item = &mut TypeVariantValue<C>> {
        self.0
            .iter_mut_and_invalidate()
            .map(|(_, pair)| &mut pair.value)
    }

    /// Keeps only the values for which a predicate is true.
    ///
    /// Iteration is done in some arbitrary order.
    ///
    /// Invalidates the dots cache for all the array's entries, so calling `.dots()` on this
    /// collection after invoking this method may be quite slow (it has to call `.dots()` on all
    /// the entries).
    pub fn retain_values_and_invalidate(
        &mut self,
        mut f: impl FnMut(&mut TypeVariantValue<C>) -> bool,
    ) {
        self.0.retain_and_invalidate(|_, pair| f(&mut pair.value));
    }

    /// Generates the array's by-index representation.
    ///
    /// This deterministically resolves any ambiguities around each element's position (eg, if it
    /// was concurrently moved by multiple actors) and returns the elements according to those
    /// resolved positions. Given the inherently concurrent nature of this data structure, calling
    /// `with_list` after joining with another [`OrArray`] may yield significantly different
    /// orderings.
    ///
    /// The provided `map` function allows skipping over elements when the entire list is not
    /// needed (eg, when filtering) and propagating error cases (eg, for single-value collapsed
    /// reads in the presence of conflicts).
    // TODO: doctest once we have a reasonably-easy constructor
    // TODO: keep an internal materialization of the numerical index -> uid mapping so that we
    //            don't need to re-compute this unnecessarily. maybe that materialization can even
    //            be maintained incrementally through join!
    pub fn with_list<'ds, F, R, E>(&'ds self, mut map: F) -> Result<Vec<(R, Uid, Position)>, E>
    where
        F: FnMut(&'ds TypeVariantValue<C>, Uid, Position) -> Result<Option<R>, E>,
    {
        let mut result: Vec<_> = self
            .iter_as_is()
            .filter_map(|(inner_map, uid, p)| -> Option<Result<_, E>> {
                // NOTE: the transpose here is so that we get the `Option` on the _outside_
                // and can use `?` at least once. i wanted `O` to have a return value of
                // `Result<Option>` so that it's easier to use `?` in the definition of `O` to
                // handle errors, but that has an impedance mismatch with the signature required by
                // `filter_map` which we're in the context of here.
                let v = match (map)(inner_map, uid, p).transpose()? {
                    Ok(v) => v,
                    Err(e) => return Some(Err(e)),
                };

                Some(Ok((v, uid, p)))
            })
            .collect::<Result<_, E>>()?;

        // NOTE: the original implementation sorts _only_ by Position here, but i belive that
        // to be wrong -- Position is not guaranteed to be unique (eg, if two nodes concurrently
        // push), which would mean a non-deterministic sort. so, we make it determinstic by also
        // sorting by the uid.
        result.sort_unstable_by_key(|&(_, uid, p)| (p, uid));

        Ok(result)
    }

    /// Returns the element at the given index.
    ///
    /// Since the array is not actually stored in index order, this requires processing and sorting
    /// the entire array, which can be quite slow!
    pub fn get(&self, idx: usize) -> Option<&TypeVariantValue<C>> {
        self.get_entry(idx).map(|(_, v)| v)
    }

    /// Returns the element at the given index and its [`Uid`].
    ///
    /// Since the array is not actually stored in index order, this requires processing and sorting
    /// the entire array, which can be quite slow!
    pub fn get_entry(&self, idx: usize) -> Option<(Uid, &TypeVariantValue<C>)> {
        if idx >= self.0.len() {
            return None;
        }

        if idx == 0 {
            // short-circuit head access which doesn't require collecting entire list
            let first = self
                .iter_as_is()
                .min_by_key(|&(_, _, p)| p)
                .expect("0 >= len, so len > 0");
            return Some((first.1, first.0));
        }
        if idx == self.len() - 1 {
            // short-circuit last element access which doesn't require collecting entire list
            let last = self
                .iter_as_is()
                .max_by_key(|&(_, _, p)| p)
                .expect("0 >= len, so len > 0");
            return Some((last.1, last.0));
        }

        // TODO(https://github.com/rust-lang/rust/issues/61695): use into_ok
        let mut result = self
            .with_list(|v, u, _| Ok::<_, Infallible>(Some((u, v))))
            .expect("E == Infallible");

        // NOTE: swap_remove is okay here since we're throwing away the array anyway
        Some(result.swap_remove(idx).0)
    }

    /// Returns a reference to the element at the given [`Uid`], if any.
    pub fn get_by_uid(&self, uid: Uid) -> Option<&TypeVariantValue<C>> {
        self.0.get(&uid).map(|pm| &pm.value)
    }

    /// Returns a mutable reference to the element at the given [`Uid`], if any.
    ///
    /// Invalidates the dots cache for the given array entry, so calling `.dots()` on this
    /// collection after invoking this method may be slower as it has to call `.dots()` on this
    /// entry to re-compute.
    pub fn get_by_uid_mut_and_invalidate(&mut self, uid: Uid) -> Option<&mut TypeVariantValue<C>> {
        self.0.get_mut_and_invalidate(&uid).map(|pm| &mut pm.value)
    }

    /// Returns the number of elements in this array.
    pub fn len(&self) -> usize {
        // NOTE: the original has to walk the fields to filter out alive, we don't \o/
        self.0.len()
    }

    /// Returns true if this array has no elements.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<'doc, C> ToValue for &'doc OrArray<C>
where
    C: ExtensionType,
{
    type Values = snapshot::OrArray<AllValues<'doc, C::ValueRef<'doc>>>;
    type Value = snapshot::OrArray<CollapsedValue<'doc, C::ValueRef<'doc>>>;

    fn values(self) -> Self::Values {
        let result = self.with_list(|v, _, _| match v.coerce_to_value_ref() {
            ValueRef::Map(m) => Ok::<_, Infallible>(Some(AllValues::Map(m.values()))),
            ValueRef::Array(a) => Ok(Some(AllValues::Array(a.values()))),
            ValueRef::Register(r) => Ok(Some(AllValues::Register(r.values()))),
            ValueRef::Custom(c) => Ok(Some(AllValues::Custom(c.values()))),
        });

        // TODO(https://github.com/rust-lang/rust/issues/61695): use into_ok
        let list = result.unwrap().into_iter().map(|(v, _, _)| v).collect();

        snapshot::OrArray { list }
    }

    fn value(self) -> Result<Self::Value, Box<SingleValueError>> {
        let result = self.with_list(|v, uid, p| match v.coerce_to_value_ref() {
            ValueRef::Map(m) => Ok(Some(CollapsedValue::Map(m.value()?))),
            ValueRef::Array(a) => Ok(Some(CollapsedValue::Array(a.value()?))),
            ValueRef::Custom(c) => Ok(Some(CollapsedValue::Custom(c.value()?))),
            ValueRef::Register(r) => match r.value() {
                Ok(v) => Ok(Some(CollapsedValue::Register(v))),

                // don't include empty values in the array
                //
                // NOTE: this means that clearing an `MvReg` that's held in an array
                // effectively removes the element, but does *not* actually remove it from the
                // array (ie, its `PairMap` is still there). is that a problem?
                Err(e) if e.issue == SingleValueIssue::Cleared => Ok(None),

                Err(mut e) => {
                    // make errors more helpful by including the path to the MvReg with conflicts
                    e.path.push(format!("[{uid:?}@{}]", p.0));
                    Err(e)
                }
            },
        });

        let list = result?.into_iter().map(|(v, _, _)| v).collect();

        Ok(snapshot::OrArray { list })
    }
}

macro_rules! apply_to_X {
    ($name:ident, $frag:literal, $field:ident, [$($others:ident),*], $innerType:ty) => {
        /// Updates the value at position `p` to be
        #[doc = $frag]
        /// using `o`.
        ///
        /// This is mostly a convenience wrapper around [`OrArray::apply`].
        /// See that method for more details.
        pub fn $name<'data, O>(
            &'data self,
            uid: Uid,
            o: O,
            p: Position,
            cc: &'_ CausalContext,
            id: Identifier,
        ) -> CausalDotStore<Self>
        where
            O: for<'cc> FnOnce(
                &'data $innerType,
                &'cc CausalContext,
                Identifier
            ) -> CausalDotStore<$innerType>,
        {
            let CausalDotStore {
                store: ret_map,
                context: mut ret_cc,
            } = self.apply(
                uid,
                move |m, cc, id| {
                    // NOTE: in the original code, this calls ORMap.apply since there `store`
                    // is just a DotMap with the keys MAP, ARRAY, and VALUE. In our case, we have a
                    // more strongly typed variant where we can write the fields directly. Like for
                    // apply_to_X in ormap.rs, we just apply to the indicated field directly
                    // instead and have all the other fields be `None`.
                    o(&m.$field, cc, id).map_store(Value::from)
                },
                p,
                cc,
                id
            );
            // recommitted value of type $field, delete the other two ($others).
            if let Some(inner) = self.0.get(&uid).map(|pm| &pm.value) {
                $( inner.$others.add_dots_to(&mut ret_cc); )*
            }
            CausalDotStore {
                store: ret_map,
                context: ret_cc,
            }
        }
    };
}

macro_rules! insert_X {
    ($name:ident, $frag:literal, $field:ident, [$($others:ident),*], $innerType:ty) => {
        /// Inserts
        #[doc = $frag]
        /// value produced by `O` at position `p`.
        ///
        /// This is mostly a convenience wrapper around [`OrArray::insert`].
        /// See that method for more details.
        pub fn $name<O>(
            &self,
            uid: Uid,
            o: O,
            p: Position,
            cc: &'_ CausalContext,
            id: Identifier,
        ) -> CausalDotStore<Self>
        where
            O: for<'cc> FnOnce(&'cc CausalContext, Identifier) -> CausalDotStore<$innerType>,
        {
            self.insert(
                uid,
                move |cc, id| {
                    // NOTE: see comment in apply_to_X about ORMap::apply
                    // NOTE: the original code is provided with the old `PairMap` (which we
                    //            assume is always `None`; see comment in `fn insert`) which it
                    //            then passes directly to `o`. however, it never calls .get(FIRST)
                    //            to project out the value, so any `o`s that were passed in would
                    //            have a hard time access the value of the current element if it
                    //            _did_ ever exist. which is all to say i believe the original
                    //            implementation of `insert` never said the `PairMap` be `Some` for
                    //            `insert`, which supports our assumption in `fn insert`.
                    o(cc, id).map_store(Value::from)
                },
                p,
                cc,
                id,
            )
        }
    };
}

impl<C> OrArray<C>
where
    C: ExtensionType + fmt::Debug + PartialEq,
{
    /// Creates a CRDT for the creation of a new empty [`OrArray`].
    pub fn create(&self, _cc: &CausalContext, _id: Identifier) -> CausalDotStore<Self> {
        // NOTE: the original OrArray implementation also sets an `.alive` field here.
        // see the YADR in `mod crdts` for why we don't do that.
        CausalDotStore {
            store: Self(Default::default()),
            context: CausalContext::default(),
        }
    }

    apply_to_X!(
        apply_to_map,
        "an [`OrMap`]",
        map,
        [array, reg],
        OrMap<String, C>
    );
    apply_to_X!(apply_to_array, "an [`OrArray`]", array, [map, reg], Self);
    apply_to_X!(apply_to_register, "an [`MvReg`]", reg, [map, array], MvReg);

    insert_X!(insert_map, "an [`OrMap`]", map, [array, reg], OrMap<String, C>);
    insert_X!(insert_array, "an [`OrArray`]", array, [map, reg], Self);
    insert_X!(insert_register, "an [`MvReg`]", reg, [map, array], MvReg);

    /// Creates a CRDT that represents the insertion of the [`Value`] produced by `O` at stable
    /// position `p` in the array.
    ///
    /// You will generally want to use [`Position::between`] to generate `p` so as to place the
    /// newly-inserted element at the numerical index you desire in the array.
    ///
    /// The provided `uid` should uniquely identify the newly-inserted element, and so should be a
    /// freshly-generated [`Uid`] (ie, [`Dot`] produced by [`CausalContext::next_dot_for`]). It
    /// will be incorporated into the returned delta. It is passed in rather than created on your
    /// behalf so that you can choose to store the `Uid` into the inserted element.
    pub fn insert<O>(
        &self,
        uid: Uid,
        o: O,
        p: Position,
        cc: &'_ CausalContext,
        id: Identifier,
    ) -> CausalDotStore<Self>
    where
        O: for<'cc> FnOnce(&'cc CausalContext, Identifier) -> CausalDotStore<Value<C>>,
    {
        let mut cc = cc.clone();

        // join in the uid dot so we guarantee a different uid will be generated next insertion even
        // if we're inserting a bottom
        cc.insert_dot(uid.dot());

        // NOTE: the original OrArray implementation also updates an `.alive` field here.
        // see the YADR in `mod crdts` for why we don't do that.
        let mut ret_dot_map = Self::default();

        let existing_pair = self.0.get(&uid);
        // NOTE: the original implementation passes the existing pair, if any, to `O`.
        // however, i don't think there ever _can_ be an existing pair, since `Uid` here is a
        // `Dot`, and a fresh `Dot` is supposed to be generated for every call to `insert` (eg, see
        // `crate::dson::array::insert*`)! let's test that hypothesis as it makes the interface
        // nicer *and* lets us make `PairMap` a private type:
        debug_assert_eq!(existing_pair, None);

        // ask `O` to generate the value we are to insert
        let CausalDotStore {
            store: v,
            context: mut ret_cc,
        } = o(&cc, id);

        ret_cc.insert_dot(uid.dot());

        if v.is_bottom() {
            // the semantics of inserting a bottom are _super_ weird (see the `push_bottom` test),
            // so if that's requested we simply propagate bottom (and don't generate a position).
            // if the caller wishes to remove an element, they should be using `remove`, not
            // inserting bottom.
            //
            // however, we still want to _allow_ this because callers may not realize that an
            // operation will produce bottom ahead of time. for instance, since empty collections
            // count as bottom (since the removal of the `.alive` field), an insert of `[{}, [{}]]`
            // into an array would actually be an insert of a bottom.
            //
            // NOTE: one awkard implication of this is that if someone does, say
            //
            //     array.insert(0, [])
            //
            // they might expect that all the elements get shifted over by 1 (to make place for an
            // element at [0]), but that won't happen since we basically ignore the request to
            // insert.
            return CausalDotStore {
                store: Default::default(),
                // we _do_ still propagate any new dots from nested operations though, just to
                // reduce confusion
                context: ret_cc,
            };
        }

        // NOTE: there's a possible opportunity for optimisation here. if `o` doesn't represent
        // a DELETE operation (which _should_ be the case, since it didn't produce a bottom store)
        // then it must contain new dots minted from the base state represented by `cc`. in theory,
        // this should guarantee that the highest dot is in `ret_cc`, which would allow us to mint
        // the next dot from it, without merging with `cc` first. we need to consider this carefully
        // though, as generally we need a fully compact cc (i.e. the base state) to mint dots (and
        // we enforce that in `next_dot_for`).
        cc.union(&ret_cc);
        // produce a unique dot to identify the position and position-set of this new element
        let mut dots = cc.next_n_dots_for(2, id);
        let position_set_dot = dots.next().expect("should be 2 left");
        let position_dot = dots.next().expect("should be 1 left");
        ret_cc.insert_dots([position_set_dot, position_dot]);

        // produce the initial position set for the new element
        let mut dot_fun = DotFun::default();
        dot_fun.set(position_dot, p);
        let mut dot_fun_map = DotFunMap::default();
        dot_fun_map.set(position_set_dot, dot_fun);
        let pair = PairMap {
            value: v.into(),
            positions: dot_fun_map,
        };
        ret_dot_map.0.set(uid, pair);

        CausalDotStore {
            store: ret_dot_map,
            context: ret_cc,
        }
    }

    /// Creates a CRDT that represents `O` applied to the [`Value`] of the element identified by
    /// `uid`, and stores the updated element at stable position `p` in the array.
    ///
    /// `O` will be passed `None` when apply is used on an [`OrArray`] CRDT that contains no value
    /// for the element at `uid` (eg, one generated by [`OrArray::mv`]).
    ///
    /// # Panics
    ///
    /// Panics if there is no element at `uid`.
    pub fn apply<'data, O>(
        &'data self,
        uid: Uid,
        o: O,
        p: Position,
        cc: &'_ CausalContext,
        id: Identifier,
    ) -> CausalDotStore<Self>
    where
        O: for<'cc> FnOnce(
            &'data TypeVariantValue<C>,
            &'cc CausalContext,
            Identifier,
        ) -> CausalDotStore<Value<C>>,
    {
        // NOTE: the original OrArray implementation also updates an `.alive` field here.
        // see the YADR in `mod crdts` for why we don't do that.
        let mut ret_dot_map = Self::default();

        // extract the current value at `uid` and apply `O`
        let Some(current) = self.0.get(&uid) else {
            // NOTE: the original code doesn't panic here, but it _does_ implicitly assume
            // that map[uid] isn't `undefined` when it computes `roots` further down, which means
            // the original implementors also expected `apply` to only ever be used on elements
            // that _do_ exist.
            panic!("no element in array with uid {uid:?}");
        };

        let CausalDotStore {
            store: v,
            context: mut ret_cc,
        } = o(&current.value, cc, id);

        if v.is_bottom() {
            // if the inner operation is a delete, we may end up with a bottom value here, in which
            // case the array apply effectively becomes a delete. this is what happens, for example,
            // if the last remaining keys in a nested map are removed (a legit use-case for apply),
            // since empty maps are bottom values.

            // make sure we delete the position (not just the value), by including its dot in the cc
            current.add_dots_to(&mut ret_cc);

            return CausalDotStore {
                store: Default::default(),
                context: ret_cc,
            };
        }

        // produce a unique dot to identify the (potentially new) position of the element
        let mut dot_gen = cc.clone();
        dot_gen.union(&ret_cc);
        let mut dots = dot_gen.next_n_dots_for(2, id);
        let position_set_dot = dots.next().expect("should be 2 left");
        let position_dot = dots.next().expect("should be 1 left");

        // produce the updated position set for the updated element
        let mut dot_fun = DotFun::default();
        dot_fun.set(position_dot, p);
        let mut dot_fun_map = DotFunMap::default();
        dot_fun_map.set(position_set_dot, dot_fun);
        let pair = PairMap {
            value: v.into(),
            positions: dot_fun_map,
        };
        ret_dot_map.0.set(uid, pair);

        let mut cc = ret_cc;
        cc.insert_dots([position_set_dot, position_dot]);

        // the apply doesn't just overwrite the value, it also collapses the position set to the
        // single position that was passed in.
        let roots = current.positions.keys();
        cc.insert_dots(roots);

        CausalDotStore {
            store: ret_dot_map,
            context: cc,
        }
    }

    /// Creates a CRDT that represents the element identified by `uid` being moved to stable
    /// position `p` in the array.
    ///
    /// # Panics
    ///
    /// Panics if there is no element at `uid`.
    pub fn mv(
        &self,
        uid: Uid,
        p: Position,
        cc: &CausalContext,
        id: Identifier,
    ) -> CausalDotStore<Self> {
        // extract the current set of positions (and their dots)
        let Some(current) = self.0.get(&uid) else {
            // NOTE: the original code doesn't explicitly error here, but it _does_ implicitly
            // assume that map[uid] isn't `undefined` since it calls .get(SECOND).dots() on it.
            panic!("no element in array with uid {uid:?}");
        };
        let positions = &current.positions;

        // generate a dot for each of the new positions we're about to insert
        // NOTE: the original implementation uses the same dot for all the collapsed position
        // set positions. we genereate distinct dots instead so that each dot only appears at most
        // once in any given structure. this is necessary so that `on_dot_change` doesn't get
        // confused if one position gets removed in a join but another does not.
        let dots = cc.next_n_dots_for(
            u8::try_from(positions.len()).expect("never more than 255 position sets"),
            id,
        );
        let also_dots = dots.clone();

        // overwrite (and collapse) all current position sets so they point at `p`.
        // we need to update all of them because we don't know which of them may end up being the
        // "winner" when all conflicts are resolved.
        let mut ps = DotFunMap::default();
        for (r, d) in positions.keys().zip(dots) {
            let mut dot_fun = DotFun::default();
            dot_fun.set(d, p);
            ps.set(r, dot_fun);
        }

        // the CRDT representing this update is one with no value (ie, the value is "bottom"),
        // which ensures that when we join this move CRDT with any other operation we don't
        // conflict with whatever the value might be.
        let pair = PairMap {
            value: Default::default(),
            positions: ps,
        };
        let mut ret_dot_map = Self::default();
        // NOTE: we don't need to set the alive field here (unlike insert/apply) since it must
        // already have been set for `uid` to exist in the first place. we also don't have to set
        // it since we've decided to forgo using `.alive` in the first place. see the YADR in `mod
        // crdts` for the reasoning about why.
        ret_dot_map.0.set(uid, pair);

        let mut cc = CausalContext::default();
        cc.insert_dots(also_dots);
        positions.add_dots_to(&mut cc);

        CausalDotStore {
            store: ret_dot_map,
            context: cc,
        }
    }

    /// Creates a CRDT that represents the removal of the element identified by `uid`.
    ///
    /// # Panics
    ///
    /// Panics if there is no element at `uid`.
    pub fn delete(&self, uid: Uid, _cc: &CausalContext, _id: Identifier) -> CausalDotStore<Self> {
        let Some(pair_map) = self.0.get(&uid) else {
            // NOTE: the original code doesn't explicitly error here, but it _does_ implicitly
            // assume that map[uid] isn't `undefined` since it calls .dots() on it.
            panic!("no element in array with uid {uid:?}");
        };

        // NOTE: the original implementation does not write alive here, but that means
        // deleting from an array can make it bottom rather than empty. one of the authors of the
        // original DSON paper confirmed by email on 2023-08-25 that the right thing to do is
        // likely to write ALIVE here (as in OrMap::delete). _but_ since we don't use `.alive` in
        // this implementation (see YADR in `mod crdts`), we do nothing.

        // mark all dots in pair_map as seen so joining with this CRDT erases all other entries.
        let mut ret_cc = CausalContext::new();
        pair_map.add_dots_to(&mut ret_cc);

        CausalDotStore {
            store: Self(Default::default()),
            context: ret_cc,
        }
    }

    /// Creates a CRDT that represents the erasure of all elements values of this array.
    pub fn clear(&self, _cc: &CausalContext, _id: Identifier) -> CausalDotStore<Self> {
        // NOTE: the original implementation does not write alive here, but that means
        // clearing an array makes it bottom rather than empty. one of the authors of the original
        // DSON paper confirmed by email on 2023-08-25 that the right thing to do is likely to
        // write ALIVE here (as in OrMap::clear). _but_ since we don't use `.alive` in this
        // implementation (see YADR in `mod crdts`), we do nothing.

        // mark all dots as seen so that joining with this CRDT will erase all other elements.
        let ret_cc = self.dots();

        CausalDotStore {
            store: Self(Default::default()),
            context: ret_cc,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        crdts::{
            NoExtensionTypes,
            test_util::{Ops, join_harness},
        },
        sentinel::test::{KeyCountingValidator, ValueCountingValidator},
    };
    use std::collections::BTreeMap;

    type OrArray = super::OrArray<NoExtensionTypes>;

    #[test]
    fn empty() {
        let cds = CausalDotStore::<OrArray>::default();
        assert!(cds.store.is_bottom());
        assert!(cds.store.value().unwrap().is_empty());
        assert_eq!(cds.store.values().len(), 0);
    }

    #[test]
    fn created_is_bottom() {
        let list = OrArray::default();
        let cc = CausalContext::new();

        let m = list.create(&cc, Identifier::new(0, 0));
        assert!(m.store.is_bottom());
    }

    #[test]
    fn cleared_is_bottom() {
        let list = OrArray::default();
        let cc = CausalContext::new();
        let id = Identifier::new(0, 0);

        let m = list.create(&cc, id);
        let m = m.store.clear(&m.context, id);
        assert!(m.store.is_bottom());
    }

    #[test]
    fn push_get_remove() {
        let list = OrArray::default();
        let cc = CausalContext::new();
        let id = Identifier::new(0, 0);

        let uid = cc.next_dot_for(id).into();

        let m = list.insert_register(
            uid,
            |cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
            Position::between(None, None),
            &cc,
            id,
        );
        assert!(!m.store.is_bottom());
        assert_eq!(
            m.store.value().unwrap().get(0).cloned(),
            Some(CollapsedValue::Register(&MvRegValue::Bool(true)))
        );
        assert_eq!(m.store.len(), 1);

        let m = m.store.delete(uid, &cc, id);
        assert!(m.store.is_bottom()); // empty arrays become bottom
        assert_eq!(m.store.value().unwrap().get(0), None);
        assert_eq!(m.store.len(), 0);
    }

    #[test]
    // NOTE: we disallow this now, but the semantics are still weird so keeping
    #[ignore = "no longer relevant when inserted bottom values are discarded"]
    #[should_panic = "asked to insert bottom element Register(MvReg(DotFun { state: {} }))"]
    fn push_bottom() {
        let list = OrArray::default();
        let cc = CausalContext::new();
        let id = Identifier::new(0, 0);

        let uid = cc.next_dot_for(id).into();

        let m = list.insert_register(
            uid,
            |_, _| MvReg::default().clear(),
            Position::between(None, None),
            &cc,
            id,
        );

        // this is a _weird_ state.
        // the map _isn't_ bottom since there's a null in there
        assert!(!m.store.is_bottom());
        // but there is nothing at index 0
        assert_eq!(m.store.value().unwrap().get(0).cloned(), None);
        // yet the length is 1!
        assert_eq!(m.store.len(), 1);

        // and that's not all!
        // if you try to apply to the value, the old value will show as Some
        let _ = m.store.apply(
            uid,
            |old, _, _| {
                assert!(old.is_bottom());
                // not important what goes here as we discard the apply crdt
                MvReg::default().clear().map_store(Value::Register)
            },
            Position::between(None, None),
            &cc,
            id,
        );

        // but if we join the state, the old state is suddenly None (since we're joining bottoms)
        // this is the same thing you'd see if an element is concurrently moved and deleted.
        let m = m.clone().join(m, &mut DummySentinel).unwrap();
        m.store.apply(
            uid,
            |old, _, _| {
                assert!(old.is_bottom());
                // not important what goes here either
                MvReg::default().clear().map_store(Value::Register)
            },
            Position::between(None, None),
            &cc,
            id,
        );
    }

    #[test]
    fn delete_overrides_move() {
        let list = OrArray::default();
        let cc1 = CausalContext::new();
        let id1 = Identifier::new(1, 0);

        let uid = cc1.next_dot_for(id1).into();
        let m = list.insert_register(
            uid,
            |cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
            Position::between(None, None),
            &cc1,
            id1,
        );
        let cc1 = m.context;
        let cc2 = cc1.clone();
        let id2 = Identifier::new(2, 0);
        let list = m.store;

        // one crdt that deletes $dot:
        let m1 = list.delete(uid, &cc1, id1);
        // and one crdt that moves $dot:
        let m2 = list.mv(uid, Position::between(None, None), &cc2, id2);
        // together, what do they do?
        let mm = m1.join(m2, &mut DummySentinel).unwrap();
        // the delete will override the move since the position set that move updated is still
        // keyed by the dot that the delete also removed. so, the move is effectively voided.
        //
        // NOTE: this situation is likely why the original code passed the equivalent of
        // PairMap to O, not _just_ the value, precisely so that position conflicts are also
        // surfaced. that said, when the value has been removed, what is there really left
        // to do with _just_ knowledge of the position conflict?
        assert!(!mm.store.0.has(&uid));
        // and just in case; that should remain true after a self-join:
        let mm = mm.clone().join(mm, &mut DummySentinel).unwrap();
        assert!(!mm.store.0.has(&uid));
    }

    #[test]
    fn outer_remove_vs_inner_mv() {
        let id1 = Identifier::new(1, 0);
        let id2 = Identifier::new(2, 0);
        let mut cc1 = CausalDotStore::<OrArray>::new();
        let mut cc2 = CausalDotStore::<OrArray>::new();

        // node 1 inserts
        let uid: Uid = cc1.context.next_dot_for(id1).into();
        let mut inner_uid = None;
        cc1.context.extend([uid.dot()]);
        let crdt = cc1.store.insert_array(
            uid,
            |cc, id| {
                let mut cc = cc.clone();
                let uid: Uid = cc.next_dot_for(id).into();
                cc.extend([uid.dot()]);
                let mut crdt = OrArray::default().insert_register(
                    uid,
                    |cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    Position::between(None, None),
                    &cc,
                    id,
                );
                crdt.context.extend([uid.dot()]);
                inner_uid = Some(uid);
                crdt
            },
            Position::between(None, None),
            &cc1.context,
            id1,
        );
        cc1 = cc1.join(crdt, &mut DummySentinel).unwrap();
        let inner_uid = inner_uid.expect("insert closure is always called");

        // node 2 syncs from node 1
        cc2 = cc1.clone().join(cc2, &mut DummySentinel).unwrap();

        // node 2 updates
        let crdt = cc2.store.apply_to_array(
            uid,
            |old, cc, id| old.mv(inner_uid, Position::between(None, None), cc, id),
            Position::between(None, None),
            &cc2.context,
            id2,
        );
        eprintln!("mv: {crdt:?}");
        cc2 = cc2.join(crdt, &mut DummySentinel).unwrap();

        // node 1 clears (delete $dot would have the same effect)
        let crdt = cc1.store.clear(&cc1.context, id1);
        eprintln!("clear: {crdt:?}");
        cc1 = cc1.join(crdt, &mut DummySentinel).unwrap();

        // node 2 syncs from node 1 again
        cc2 = cc1.clone().join(cc2, &mut DummySentinel).unwrap();

        // node 1 syncs from node 2
        cc1 = cc2.clone().join(cc1, &mut DummySentinel).unwrap();

        // node 1 should have the key because of node 2's update (which beats node 1's clear)
        assert!(
            cc1.store.0.keys().any(|k| k == &uid),
            "{uid:?} wasn't in cc1 keys:\n{:?}",
            cc1.store.0
        );

        // node 2 should have the key because its update beats node 1's clear
        assert!(
            cc2.store.0.keys().any(|k| k == &uid),
            "{uid:?} wasn't in cc2 keys:\n{:?}",
            cc2.store.0
        );

        eprintln!("{cc1:?}");

        // NOTE: now this is where it gets weird.
        //
        // node 2 updated the key, so the key survived the clear. however, it only survived in the
        // sense that the join of the key's updates did not produce bottom. in practice, the join
        // of the outer clear (which eliminates the value) and the inner move that node 2 did as
        // its update, is an array with _only_ position information. normally, the resolution of a
        // delete joined with a move results in bottom (see the delete_overrides_move test), but
        // apparently an _outer_ clear ends up different from a delete/inner clear, and thus
        // doesn't "erase" the move-only op.
        //
        // i've emailed the DSON paper authors (on 2023-08-25) to see if this is intentional.
        // in the meantime, this test ensure that the behavior remains constant.
        // specifically, it checks that you _can_ apply to $dot (which requires that it exists),
        // but that it's _value_ is bottom.
        cc1.store.apply_to_array(
            uid,
            |old, cc, id| {
                assert!(
                    old.is_bottom(),
                    "dot exists due to node 2's update, but value was erased by node 1's clear"
                );
                old.clone().clear(cc, id)
            },
            Position::between(None, None),
            &cc1.context,
            id1,
        );
        cc2.store.apply_to_array(
            uid,
            |old, cc, id| {
                assert!(
                    old.is_bottom(),
                    "dot exists due to node 2's update, but value was erased by node 1's clear"
                );
                old.clone().clear(cc, id)
            },
            Position::between(None, None),
            &cc2.context,
            id2,
        );
    }

    // TODO: test that concurrent pushes both end up coming _after_ initial push
    #[test]
    fn concurrent_push() {
        join_harness(
            OrArray::default(),
            |cds, _| cds,
            |m, cc, id| {
                m.insert_register(
                    cc.next_dot_for(id).into(),
                    |cc, id| MvReg::default().write(MvRegValue::U64(2), cc, id),
                    Position::between(None, None),
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                m.insert_register(
                    cc.next_dot_for(id).into(),
                    |cc, id| MvReg::default().write(MvRegValue::U64(3), cc, id),
                    Position::between(None, None),
                    &cc,
                    id,
                )
            },
            // we observe a value type change to register (from no type)
            ValueCountingValidator::new(true),
            |CausalDotStore { store: m, .. }, sentinel| {
                assert!(!m.is_bottom());
                let list = m.value().unwrap();
                assert_eq!(list.len(), 2);
                assert!(
                    list.iter()
                        .any(|v| v == &CollapsedValue::Register(&MvRegValue::U64(2)))
                );
                assert!(
                    list.iter()
                        .any(|v| v == &CollapsedValue::Register(&MvRegValue::U64(3)))
                );
                assert_eq!(sentinel.added, BTreeMap::from([(MvRegValue::U64(3), 1)]));
                assert!(sentinel.removed.is_empty());
            },
        );
    }

    #[test]
    fn concurrent_move_delete() {
        // `join_harness` uses `(9, 0)` as the Identifier when making the init state.
        let shared_uid = Dot::mint((9, 0).into(), 1).into();
        let p = Position::between(None, None);

        join_harness(
            OrArray::default(),
            |cds, id| {
                let uid = cds.context.next_dot_for(id).into();
                assert_eq!(uid, shared_uid);
                cds.store.insert_register(
                    uid,
                    |cc, id| MvReg::default().write(MvRegValue::U64(1), cc, id),
                    p,
                    &cds.context,
                    id,
                )
            },
            |m, cc, id| m.mv(shared_uid, p, &cc, id),
            |m, cc, id| m.delete(shared_uid, &cc, id),
            // we delete a key here
            KeyCountingValidator::default(),
            |CausalDotStore { store: m, .. }, sentinel| {
                // a concurrent move and delete should result in the element's removal:
                assert_eq!(m.len(), 0);
                let list = m.values();
                assert_eq!(list.len(), 0);
                assert_eq!(list.get(0), None);
                // empty arrays become bottom
                assert!(m.is_bottom());
                assert_eq!(sentinel.added, 0);
                assert_eq!(sentinel.removed, 1);
            },
        );
    }

    #[test]
    fn concurrent_update() {
        let shared_uid = Dot::mint((9, 0).into(), 1).into();
        let p = Position::between(None, None);

        join_harness(
            OrArray::default(),
            |cds, id| {
                let uid = cds.context.next_dot_for(id).into();
                assert_eq!(uid, shared_uid);
                cds.store.insert_register(
                    uid,
                    |cc, id| MvReg::default().write(MvRegValue::U64(1), cc, id),
                    p,
                    &cds.context,
                    id,
                )
            },
            |m, cc, id| {
                m.apply_to_register(
                    shared_uid,
                    |old, cc, id| {
                        assert_eq!(old.value().unwrap(), &MvRegValue::U64(1));
                        MvReg::default().write(MvRegValue::U64(2), cc, id)
                    },
                    p,
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                m.apply_to_register(
                    shared_uid,
                    |old, cc, id| {
                        assert_eq!(old.value().unwrap(), &MvRegValue::U64(1));
                        MvReg::default().write(MvRegValue::U64(3), cc, id)
                    },
                    p,
                    &cc,
                    id,
                )
            },
            ValueCountingValidator::default(),
            |CausalDotStore { store: m, .. }, sentinel| {
                assert!(!m.is_bottom());
                let list = m.values();
                assert_eq!(list.len(), 1);
                let AllValues::Register(v) = list.get(0).unwrap() else {
                    panic!("[0] isn't a register though we only wrote a register")
                };
                assert!(v.contains(&MvRegValue::U64(2)));
                assert!(v.contains(&MvRegValue::U64(3)));
                // the RHS of the join (the "delta") contributes just one value
                // even though we end up with 2
                assert_eq!(sentinel.added, BTreeMap::from([(MvRegValue::U64(3), 1)]));
                assert!(sentinel.removed.is_empty());
            },
        );
    }

    #[test]
    fn concurrent_clear() {
        let shared_uid = Dot::mint((9, 0).into(), 1).into();
        let p = Position::between(None, None);

        join_harness(
            OrArray::default(),
            |CausalDotStore {
                 store: m,
                 context: cc,
             },
             id| {
                m.insert_register(
                    shared_uid,
                    |cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    p,
                    &cc,
                    id,
                )
            },
            |m, cc, id| m.clear(&cc, id),
            |m, cc, id| m.clear(&cc, id),
            ValueCountingValidator::default(),
            |CausalDotStore { store: m, .. }, sentinel| {
                // empty arrays become bottom
                assert!(m.is_bottom());
                let values = m.values();
                assert_eq!(values.len(), 0);
                // where's the update? from the perspective of the sentinel
                // we're joining a delta onto a base that is already empty, so
                // nothing is changing
                assert!(sentinel.added.is_empty());
                assert!(sentinel.removed.is_empty());
            },
        );
    }

    #[test]
    fn update_vs_remove() {
        let shared_uid = Dot::mint((9, 0).into(), 1).into();
        let p = Position::between(None, None);

        join_harness(
            OrArray::default(),
            |CausalDotStore {
                 store: m,
                 context: cc,
             },
             id| {
                // start out with an array with an element
                m.insert_register(
                    shared_uid,
                    |cc, id| MvReg::default().write(MvRegValue::U64(42), cc, id),
                    p,
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                // one writer updates [shared_dot]
                m.apply_to_register(
                    shared_uid,
                    |_old, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                    p,
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                // the other writer removes [shared_dot]
                m.delete(shared_uid, &cc, id)
            },
            ValueCountingValidator::default(),
            |CausalDotStore { store: m, .. }, sentinel| {
                // _not_ bottom because this is still an active map
                assert!(!m.is_bottom());
                // the semantics of observed-remove (remember "*OR*array")
                // is that updates concurrent with removes leave the updates intact
                let values = m.values();
                let AllValues::Register(v) = values.get(0).unwrap() else {
                    panic!("[0] isn't a register even though we only wrote registers");
                };
                assert_eq!(v, [MvRegValue::Bool(true)]);
                // where's the update? from the perspective of the sentinel
                // we're joining a delta that is attempting to remove a value
                // that hasn't been observed... so nothing happens
                assert!(sentinel.added.is_empty());
                assert!(sentinel.removed.is_empty());
            },
        );
    }

    #[test]
    fn nested_update_vs_remove() {
        let shared_uid = Dot::mint((9, 0).into(), 1).into();
        let p = Position::between(None, None);

        join_harness(
            OrArray::default(),
            |CausalDotStore {
                 store: m,
                 context: cc,
             },
             id| {
                // start out with an array like [{bar: 42}]
                m.insert_map(
                    shared_uid,
                    |cc, id| {
                        OrMap::default().apply_to_register(
                            |_old, cc, id| MvReg::default().write(MvRegValue::U64(42), cc, id),
                            "bar".into(),
                            cc,
                            id,
                        )
                    },
                    p,
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                // one writer adds a field (baz) to the inner map
                m.apply_to_map(
                    shared_uid,
                    |old, cc, id| {
                        old.apply_to_register(
                            |_old, cc, id| MvReg::default().write(MvRegValue::Bool(true), cc, id),
                            "baz".into(),
                            cc,
                            id,
                        )
                    },
                    p,
                    &cc,
                    id,
                )
            },
            |m, cc, id| {
                // the other writer removes [0]
                m.delete(shared_uid, &cc, id)
            },
            ValueCountingValidator::default(),
            |CausalDotStore { store: m, .. }, sentinel| {
                // _not_ bottom because this is still an active array
                assert!(!m.is_bottom());
                // the semantics of observed-remove (remember "*OR*array")
                // is that updates concurrent with removes leave the updates intact,
                // so we'll end up with [{baz: true}]
                // as counter-intuitive as that may seem
                let values = m.values();
                let AllValues::Map(m) = values.get(0).unwrap() else {
                    panic!("[0] isn't a map even though we only wrote map");
                };
                assert_eq!(values.len(), 1);
                let AllValues::Register(r) = m
                    .get(&String::from("baz"))
                    .expect("baz key isn't preserved")
                else {
                    panic!("baz isn't a register though we only wrote a register ")
                };
                assert_eq!(m.len(), 1);
                assert_eq!(r, [MvRegValue::Bool(true)]);
                // where's the update? from the perspective of the sentinel
                // we're joining a delta that is attempting to remove a value
                // that hasn't been observed... so nothing happens
                assert!(sentinel.added.is_empty());
                assert!(sentinel.removed.is_empty());
            },
        );
    }

    // TODO: test relocating an element

    #[quickcheck]
    fn order_invariant(ops: Ops<OrArray>, seed: u64) -> quickcheck::TestResult {
        ops.check_order_invariance(seed)
    }
}
