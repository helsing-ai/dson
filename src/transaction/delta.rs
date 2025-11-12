/// Changes to a CRDT, not full state.
///
/// Prevents accidental misuse through type safety. Access the inner
/// `CausalDotStore` via the public field.
///
/// # Example
/// ```
/// use dson::{Delta, CausalDotStore, OrMap};
///
/// # fn example(delta: Delta<CausalDotStore<OrMap<String>>>) {
/// // Access inner value
/// let store = delta.0;
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
#[must_use = "deltas should be sent to other replicas or applied to stores"]
pub struct Delta<T>(pub T);

impl<T> Delta<T> {
    /// Creates a new Delta wrapping the given value.
    pub fn new(value: T) -> Self {
        Self(value)
    }

    /// Unwraps the Delta, returning the inner value.
    pub fn into_inner(self) -> T {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CausalDotStore, OrMap};

    #[test]
    fn delta_new_and_into_inner() {
        let store = CausalDotStore::<OrMap<String>>::default();
        let delta = Delta::new(store.clone());
        assert_eq!(delta.into_inner(), store);
    }

    #[test]
    fn delta_access_inner_via_field() {
        let store = CausalDotStore::<OrMap<String>>::default();
        let delta = Delta::new(store.clone());
        assert_eq!(delta.0, store);
    }
}
