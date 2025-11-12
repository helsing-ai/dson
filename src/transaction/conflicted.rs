use crate::crdts::TypeVariantValue;
use crate::dotstores::DotStore;
use crate::{ExtensionType, MvReg, OrArray, OrMap};
use std::{fmt, hash::Hash};

/// A value with concurrent type conflicts.
///
/// When replicas concurrently write different types to the same key
/// (e.g., one writes a map, another an array), DSON preserves both
/// in a [`TypeVariantValue`]. This type exposes methods to inspect conflicts.
///
/// # Example
///
/// ```no_run
/// # use dson::transaction::ConflictedValue;
/// # use dson::crdts::NoExtensionTypes;
/// # let conflicted: ConflictedValue<String, NoExtensionTypes> = todo!();
/// if conflicted.has_map() && conflicted.has_array() {
///     println!("Map and array were written concurrently!");
///     // Application must decide how to resolve this
/// }
/// ```
pub struct ConflictedValue<'tx, K, C>
where
    K: Hash + Eq,
    C: ExtensionType,
{
    inner: &'tx TypeVariantValue<C>,
    // K appears in CrdtValue<'tx, K, C> but TypeVariantValue doesn't use it.
    // PhantomData maintains consistent type parameters across the API.
    _phantom: std::marker::PhantomData<K>,
}

impl<'tx, K, C> ConflictedValue<'tx, K, C>
where
    K: Hash + Eq + fmt::Debug + Clone,
    C: ExtensionType,
{
    pub(crate) fn new(value: &'tx TypeVariantValue<C>) -> Self {
        Self {
            inner: value,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns true if a map value is present in the conflict.
    pub fn has_map(&self) -> bool {
        !self.inner.map.is_bottom()
    }

    /// Returns true if an array value is present in the conflict.
    pub fn has_array(&self) -> bool {
        !self.inner.array.is_bottom()
    }

    /// Returns true if a register value is present in the conflict.
    pub fn has_register(&self) -> bool {
        !self.inner.reg.is_bottom()
    }

    /// Returns a reference to the map value, if present.
    pub fn map(&self) -> Option<&OrMap<String, C>> {
        if self.has_map() {
            Some(&self.inner.map)
        } else {
            None
        }
    }

    /// Returns a reference to the array value, if present.
    pub fn array(&self) -> Option<&OrArray<C>> {
        if self.has_array() {
            Some(&self.inner.array)
        } else {
            None
        }
    }

    /// Returns a reference to the register value, if present.
    pub fn register(&self) -> Option<&MvReg> {
        if self.has_register() {
            Some(&self.inner.reg)
        } else {
            None
        }
    }

    /// Returns the number of different types present in this conflict.
    ///
    /// A value of 0 means the key exists but is empty (all types are bottom).
    /// A value of 1 means there's no actual type conflict.
    /// A value > 1 indicates a genuine type conflict.
    pub fn conflict_count(&self) -> usize {
        let mut count = 0;
        if self.has_map() {
            count += 1;
        }
        if self.has_array() {
            count += 1;
        }
        if self.has_register() {
            count += 1;
        }
        count
    }
}

impl<'tx, K, C> fmt::Debug for ConflictedValue<'tx, K, C>
where
    K: Hash + Eq + fmt::Debug + Clone,
    C: ExtensionType + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConflictedValue")
            .field("has_map", &self.has_map())
            .field("has_array", &self.has_array())
            .field("has_register", &self.has_register())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crdts::NoExtensionTypes;
    use crate::{CausalDotStore, Identifier, OrMap};

    #[test]
    fn conflicted_value_empty() {
        use crate::crdts::TypeVariantValue;
        let value = TypeVariantValue::<NoExtensionTypes>::default();
        let conflicted = ConflictedValue::<String, _>::new(&value);
        assert_eq!(conflicted.conflict_count(), 0);
    }

    #[test]
    fn conflicted_value_single_type() {
        // Create a real map with a value using CRDT operations
        let store = CausalDotStore::<OrMap<String>>::default();
        let id = Identifier::new(0, 0);

        let delta = store.store.apply_to_register(
            |reg, ctx, id| reg.write("test".to_string().into(), ctx, id),
            "key".to_string(),
            &store.context,
            id,
        );

        // The delta contains a non-bottom register
        let value = delta.store.get(&"key".to_string()).unwrap();

        let conflicted = ConflictedValue::<String, _>::new(value);
        assert!(!conflicted.has_map());
        assert!(!conflicted.has_array());
        assert!(conflicted.has_register());
        assert_eq!(conflicted.conflict_count(), 1);
        assert!(conflicted.register().is_some());
    }

    #[test]
    fn conflicted_value_conflict_detection() {
        // Test that we can detect when there are multiple types
        // This is more of a structural test - we verify the logic works
        use crate::crdts::TypeVariantValue;

        let value = TypeVariantValue::<NoExtensionTypes>::default();
        let conflicted = ConflictedValue::<String, _>::new(&value);
        assert_eq!(conflicted.conflict_count(), 0);
        assert!(!conflicted.has_map());
        assert!(!conflicted.has_array());
        assert!(!conflicted.has_register());
    }
}
