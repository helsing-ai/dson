use super::ConflictedValue;
use crate::crdts::TypeVariantValue;
use crate::{ExtensionType, MvReg, OrArray, OrMap};
use std::{fmt, hash::Hash};

/// Result of reading a value from a transaction.
///
/// DSON preserves type conflicts, so reads must handle multiple possibilities.
/// This enum forces explicit handling of:
/// - Map
/// - Array
/// - Register
/// - Concurrent type conflicts
/// - Missing key
///
/// # Example
///
/// ```no_run
/// # use dson::transaction::{MapTransaction, CrdtValue};
/// # let tx: MapTransaction<String> = todo!();
/// match tx.get(&"user".to_string()) {
///     Some(CrdtValue::Map(map)) => { /* work with map */ }
///     Some(CrdtValue::Conflicted(conflicts)) => { /* resolve conflict */ }
///     None => { /* key doesn't exist */ }
///     _ => { /* other types */ }
/// }
/// ```
#[derive(Debug)]
pub enum CrdtValue<'tx, K, C = crate::crdts::NoExtensionTypes>
where
    K: Hash + Eq + fmt::Debug + Clone,
    C: ExtensionType,
{
    /// The value is a map (no type conflict).
    Map(&'tx OrMap<String, C>),

    /// The value is an array (no type conflict).
    Array(&'tx OrArray<C>),

    /// The value is a register (no type conflict).
    Register(&'tx MvReg),

    /// The value has concurrent type conflicts.
    Conflicted(ConflictedValue<'tx, K, C>),

    /// The key exists but all types are empty (bottom).
    Empty,
}

impl<'tx, K, C> CrdtValue<'tx, K, C>
where
    K: Hash + Eq + fmt::Debug + Clone,
    C: ExtensionType,
{
    /// Creates a CrdtValue by classifying a TypeVariantValue.
    ///
    /// Inspects which CRDT types are non-empty (non-bottom) and returns
    /// the appropriate variant:
    /// - If multiple types are present: `Conflicted`
    /// - If only one type is present: the specific variant (Map/Array/Register)
    /// - If all types are empty: `Empty`
    pub fn from_type_variant(value: &'tx TypeVariantValue<C>) -> Self {
        use crate::dotstores::DotStore;

        // Check if there's a type conflict (multiple types are non-bottom)
        let has_multiple_types = {
            let mut count = 0;
            if !value.map.is_bottom() {
                count += 1;
            }
            if !value.array.is_bottom() {
                count += 1;
            }
            if !value.reg.is_bottom() {
                count += 1;
            }
            count > 1
        };

        if has_multiple_types {
            CrdtValue::Conflicted(ConflictedValue::new(value))
        } else if !value.reg.is_bottom() {
            CrdtValue::Register(&value.reg)
        } else if !value.map.is_bottom() {
            CrdtValue::Map(&value.map)
        } else if !value.array.is_bottom() {
            CrdtValue::Array(&value.array)
        } else {
            CrdtValue::Empty
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crdts::mvreg::MvRegValue;
    use crate::crdts::{NoExtensionTypes, TypeVariantValue};
    use crate::dotstores::DotStore;
    use crate::sentinel::DummySentinel;
    use crate::{CausalDotStore, Identifier, OrMap};

    #[test]
    fn from_type_variant_register_only() {
        // Create a TypeVariantValue with only register populated
        let store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        let delta = store.store.apply_to_register(
            |reg, ctx, id| reg.write(MvRegValue::U64(42), ctx, id),
            "key".to_string(),
            &store.context,
            id,
        );

        let type_variant = delta.store.get(&"key".to_string()).unwrap();

        // Test from_type_variant
        let value: CrdtValue<'_, String> = CrdtValue::from_type_variant(type_variant);

        match value {
            CrdtValue::Register(reg) => {
                use crate::crdts::snapshot::ToValue;
                assert_eq!(reg.value().unwrap(), &MvRegValue::U64(42));
            }
            _ => panic!("Expected Register variant"),
        }
    }

    #[test]
    fn from_type_variant_empty() {
        // Empty TypeVariantValue (all fields are bottom)
        let type_variant = TypeVariantValue::<NoExtensionTypes>::default();
        let value: CrdtValue<'_, String> = CrdtValue::from_type_variant(&type_variant);

        match value {
            CrdtValue::Empty => { /* expected */ }
            _ => panic!("Expected Empty variant"),
        }
    }

    #[test]
    fn from_type_variant_map_only() {
        // Create a TypeVariantValue with only map populated
        let store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        // Create nested map
        let delta = store.store.apply_to_map(
            |map, ctx, id| {
                map.apply_to_register(
                    |reg, ctx, id| reg.write(MvRegValue::String("test".to_string()), ctx, id),
                    "field".to_string(),
                    ctx,
                    id,
                )
            },
            "key".to_string(),
            &store.context,
            id,
        );

        let type_variant = delta.store.get(&"key".to_string()).unwrap();
        let value: CrdtValue<'_, String> = CrdtValue::from_type_variant(type_variant);

        match value {
            CrdtValue::Map(map) => {
                assert!(!map.is_bottom());
            }
            _ => panic!("Expected Map variant"),
        }
    }

    #[test]
    fn from_type_variant_array_only() {
        // Create a TypeVariantValue with only array populated
        let store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        // Create array with one element
        let delta = store.store.apply_to_array(
            |array, ctx, id| array.insert_idx_register(0, MvRegValue::U64(1), ctx, id),
            "key".to_string(),
            &store.context,
            id,
        );

        let type_variant = delta.store.get(&"key".to_string()).unwrap();
        let value: CrdtValue<'_, String> = CrdtValue::from_type_variant(type_variant);

        match value {
            CrdtValue::Array(array) => {
                assert_eq!(array.len(), 1);
            }
            _ => panic!("Expected Array variant"),
        }
    }

    #[test]
    fn from_type_variant_conflicted() {
        // Create a TypeVariantValue with multiple types (type conflict)
        let store = CausalDotStore::<OrMap<String>>::default();
        let id1 = Identifier::new(0, 0);
        let id2 = Identifier::new(1, 0);

        // Replica 1 writes register
        let delta1 = store.store.apply_to_register(
            |reg, ctx, id| reg.write(MvRegValue::U64(42), ctx, id),
            "key".to_string(),
            &store.context,
            id1,
        );

        // Replica 2 writes array (concurrent with delta1)
        let delta2 = store.store.apply_to_array(
            |array, ctx, id| {
                array.insert_idx_register(0, MvRegValue::String("conflict".to_string()), ctx, id)
            },
            "key".to_string(),
            &store.context,
            id2,
        );

        // Join both deltas to create conflict
        let combined = delta1.join(delta2, &mut DummySentinel).unwrap();
        let type_variant = combined.store.get(&"key".to_string()).unwrap();

        let value: CrdtValue<'_, String> = CrdtValue::from_type_variant(type_variant);

        match value {
            CrdtValue::Conflicted(conflicts) => {
                assert!(conflicts.has_register());
                assert!(conflicts.has_array());
                assert_eq!(conflicts.conflict_count(), 2);
            }
            _ => panic!("Expected Conflicted variant, got {value:?}"),
        }
    }
}
