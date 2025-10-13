// (c) Copyright 2025 Helsing GmbH. All rights reserved.
use crate::{
    CausalContext, CausalDotStore, ExtensionType, Identifier, MvReg, OrArray, OrMap,
    crdts::{
        TypeVariantValue, Value,
        orarray::{Position, Uid},
        snapshot::{self, ToValue},
    },
};
use std::{convert::Infallible, fmt};

/*
/// insert(𝑖𝑑𝑥, 𝑜𝛿 𝑖 ) – given an index 𝑖𝑑𝑥 and a method 𝑜𝛿
/// 𝑖 from
/// the API of some CRDT of type 𝑉 , The method assigns a
/// unique id 𝑢𝑖𝑑, assigns a stable position identifier 𝑝 such that
/// the new element in the sorted array appears at index 𝑖𝑑𝑥,
/// and invokes apply(𝑢𝑖𝑑, 𝑜𝛿
/// 𝑖 , 𝑝).
update(𝑖𝑑𝑥, 𝑜𝛿
𝑖 ) – given an index 𝑖𝑑𝑥 and a method 𝑜𝛿
𝑖 of
some CRDT type 𝑉 , The method finds the 𝑢𝑖𝑑 corresponding
to the element at index 𝑖𝑑𝑥, finds the position 𝑝, and invokes
apply(𝑢𝑖𝑑, 𝑜𝛿
𝑖 , 𝑝).
move(𝑜𝑙𝑑_𝑖𝑑𝑥, 𝑛𝑒𝑤_𝑖𝑑𝑥) – given two indexes, finds the ele-
ment 𝑢𝑖𝑑 corresponding to the element at index 𝑜𝑙𝑑_𝑖𝑑𝑥,
calculates the stable position identifier 𝑝 such that the el-
ement in the sorted array will be at index 𝑛𝑒𝑤_𝑖𝑑𝑥, and
invokes move(𝑢𝑖𝑑, 𝑝).
delete(𝑖𝑑𝑥) – given an index 𝑖𝑑𝑥, finds the element 𝑢𝑖𝑑 corre-
sponding to the element at index 𝑖𝑑𝑥, and invokes delete(𝑢𝑖𝑑).
get(𝑖𝑑𝑥) – given an index 𝑖𝑑𝑥, finds the element 𝑢𝑖𝑑 corre-
sponding to the element at index 𝑖𝑑𝑥, and invokes get(𝑢𝑖𝑑).
*/

/// Returns the values of this array without collapsing conflicts.
pub fn values<C>(m: &OrArray<C>) -> snapshot::OrArray<snapshot::AllValues<'_, C::ValueRef<'_>>>
where
    C: ExtensionType,
{
    m.values()
}

/// Returns the values of this array assuming (and asserting) no conflicts on element values.
// NOTE: A type alias won't help much here :melt:.
#[allow(clippy::type_complexity)]
pub fn value<C>(
    m: &OrArray<C>,
) -> Result<
    snapshot::OrArray<snapshot::CollapsedValue<'_, C::ValueRef<'_>>>,
    Box<snapshot::SingleValueError<<&OrArray<C> as ToValue>::LeafValue>>,
>
where
    C: ExtensionType,
{
    m.value()
}

/// Creates a new array.
pub fn create<C>() -> impl Fn(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>
where
    C: ExtensionType + fmt::Debug + PartialEq,
{
    move |m, cc, id| m.create(cc, id)
}

/// Inserts a new element at the given index.
pub fn insert<O, C>(
    o: O,
    idx: usize,
) -> impl FnOnce(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>
where
    O: FnOnce(&CausalContext, Identifier) -> CausalDotStore<Value<C>>,
    C: ExtensionType + fmt::Debug + PartialEq,
{
    move |m, cc, id| {
        let uid = cc.next_dot_for(id).into();
        let p = create_position_for_index(m, idx);
        m.insert(uid, o, p, cc, id)
    }
}

/// Inserts a new map at the given index.
pub fn insert_map<O, C>(
    o: O,
    idx: usize,
) -> impl FnOnce(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>
where
    O: FnOnce(&CausalContext, Identifier) -> CausalDotStore<OrMap<String, C>>,
    C: ExtensionType + fmt::Debug + PartialEq,
{
    insert(move |cc, id| (o)(cc, id).map_store(Value::Map), idx)
}

/// Inserts a new array at the given index.
pub fn insert_array<O, C>(
    o: O,
    idx: usize,
) -> impl FnOnce(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>
where
    O: FnOnce(&CausalContext, Identifier) -> CausalDotStore<OrArray<C>>,
    C: ExtensionType + fmt::Debug + PartialEq,
{
    insert(move |cc, id| (o)(cc, id).map_store(Value::Array), idx)
}

/// Inserts a new register at the given index.
pub fn insert_register<O, C>(
    o: O,
    idx: usize,
) -> impl FnOnce(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>
where
    O: FnOnce(&CausalContext, Identifier) -> CausalDotStore<MvReg>,
    C: ExtensionType + fmt::Debug + PartialEq,
{
    insert(move |cc, id| (o)(cc, id).map_store(Value::Register), idx)
}

/// Applies a function to the element at the given index.
pub fn apply<O, C>(
    o: O,
    idx: usize,
) -> impl FnOnce(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>
where
    O: FnOnce(&TypeVariantValue<C>, &CausalContext, Identifier) -> CausalDotStore<Value<C>>,
    C: ExtensionType + fmt::Debug + PartialEq,
{
    move |m, cc, id| {
        let uid = uid_from_index(m, idx);
        assert_ne!(idx, m.len(), "index out of bounds");
        let p = create_position_for_index(m, idx);
        m.apply(uid, o, p, cc, id)
    }
}

/// Applies a function to the map at the given index.
pub fn apply_to_map<O, C>(
    o: O,
    idx: usize,
) -> impl FnOnce(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>
where
    O: FnOnce(&OrMap<String, C>, &CausalContext, Identifier) -> CausalDotStore<OrMap<String, C>>,
    C: ExtensionType + fmt::Debug + PartialEq,
{
    apply(
        move |m, cc, id| (o)(&m.map, cc, id).map_store(Value::Map),
        idx,
    )
}

/// Applies a function to the array at the given index.
pub fn apply_to_array<O, C>(
    o: O,
    idx: usize,
) -> impl FnOnce(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>
where
    O: FnOnce(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>,
    C: ExtensionType + fmt::Debug + PartialEq,
{
    apply(
        move |m, cc, id| (o)(&m.array, cc, id).map_store(Value::Array),
        idx,
    )
}

/// Applies a function to the register at the given index.
pub fn apply_to_register<O, C>(
    o: O,
    idx: usize,
) -> impl FnOnce(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>
where
    O: FnOnce(&MvReg, &CausalContext, Identifier) -> CausalDotStore<MvReg>,
    C: ExtensionType + fmt::Debug + PartialEq,
{
    apply(
        move |m, cc, id| (o)(&m.reg, cc, id).map_store(Value::Register),
        idx,
    )
}

/// Moves an element from one index to another.
pub fn mv<C>(
    from: usize,
    to: usize,
) -> impl Fn(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>
where
    C: ExtensionType + fmt::Debug + PartialEq,
{
    move |m, cc, id| {
        let uid = uid_from_index(m, from);
        let p = create_position_for_index(m, to);
        m.mv(uid, p, cc, id)
    }
}

/// Deletes an element at the given index.
pub fn delete<'s, C>(
    idx: usize,
) -> impl Fn(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>> + 's
where
    C: ExtensionType + fmt::Debug + PartialEq,
{
    move |m, cc, id| {
        let uid = uid_from_index(m, idx);
        m.delete(uid, cc, id)
    }
}

/// Clears the array.
pub fn clear<C>() -> impl Fn(&OrArray<C>, &CausalContext, Identifier) -> CausalDotStore<OrArray<C>>
where
    C: ExtensionType + fmt::Debug + PartialEq,
{
    move |m, cc, id| m.clear(cc, id)
}

fn ids<C>(m: &OrArray<C>) -> Vec<((), Uid, Position)> {
    // TODO(https://github.com/rust-lang/rust/issues/61695): use into_ok
    m.with_list(|_, _, _| Ok::<_, Infallible>(Some(())))
        .unwrap()
}

/// Computes the [`Position`] a new element should have to end up at `[idx]`.
///
/// Inserting a new element with the given [`Position`] will end up shifting all later elements to
/// the rigth by one. For example, inserting an element with position `create_position_for_index(_,
/// 0)` will make the current `[0]` be at `[1]`, the current `[1]` at `[2]`, and so on.
fn create_position_for_index<C>(m: &OrArray<C>, idx: usize) -> Position {
    // NOTE: the original code passes cc.id() to the Position::between calls here, but that
    // argument is ignored, so it's removed in our implementation;

    // we don't have to sort all the items to resolve the first/last position.
    // not doing the sort saves us from the `.collect` in `with_list`, which would result in a
    // `Vec` that gets pretty much immediately thrown away afterwards.
    // TODO: cache min/max Position inside OrArray maybe?
    if idx == 0 {
        let min_p = m.iter_as_is().map(|(_, _, p)| p).min();
        return Position::between(None, min_p);
    }
    if idx == m.len() {
        let max_p = m.iter_as_is().map(|(_, _, p)| p).max();
        return Position::between(max_p, None);
    }

    assert!(
        idx < m.len(),
        "index out of bounds ({idx} when length is {})",
        m.len()
    );
    // NOTE: we know here that !m.is_empty(), otherwise we'd either hit idx == 0 or the asset.

    let ids = ids(m);
    let pos_at_index = ids.get(idx).map(|(_, _, p)| *p);
    let pos_at_previous_index = if idx == 0 {
        None
    } else {
        Some(
            ids.get(idx - 1)
                .expect("we check for out-of-bounds above")
                .2,
        )
    };
    Position::between(pos_at_previous_index, pos_at_index)
}

fn uid_from_index<C>(m: &OrArray<C>, idx: usize) -> Uid {
    ids(m)[idx].1
}
