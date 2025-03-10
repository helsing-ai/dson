// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! Observe and validate changes to a CRDT.
//!
//! Sentinels are types that can be used to inspect the changes being applied to a CRDT. They are
//! useful for validating that the changes conform to a schema, or simply to observe the changes
//! for any other purpose (for example, logging, metrics, etc).
//!
//! The main entry point for this module is the [`Sentinel`] trait, which is composed of more
//! specialized traits that can be implemented to observe different kinds of changes.
//!
//! For a testing-oriented example, see the `recording_sentinel` module.

use crate::crdts::ValueType;
use std::convert::Infallible;

/// Observes and optionally stops a change being applied to a CRDT.
///
/// This is the base trait that all Sentinels should implement. Different Sentinels may observe
/// different changes, and different data structures may produce different changes, so we've split
/// the actual behaviour into several specialized traits.
///
/// This trait should normally be paired with [`Visit`] so that the Sentinel can keep track of
/// which node is currently being modified in the document tree.
///
/// If Error = Infallible, the Sentinel is referred to as an Observer. If it can produce an error, it
/// may be referred to as a Validator.
pub trait Sentinel {
    type Error;
}

/// Observe when a key is added or removed from the document tree.
///
/// This is how Sentinels can track when container nodes are created. Register values changes can
/// be tracked via the [`ValueSentinel`] trait.
pub trait KeySentinel: Sentinel {
    /// Observe and validate the creation of a new entry under the current path.
    ///
    /// This method may be called _after_ [`ValueSentinel::set`] for a given entry.
    fn create_key(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Observe and validate the deletion of the entry under the current path.
    fn delete_key(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Observes when a value's type changes.
///
/// This is useful for tracking changes involving container values - particularly, when
/// transitioning a value from a container to register or vice-versa, or when creating empty
/// containers. The first case is because updates that switch to/from register values only produce
/// one [`ValueSentinel`] event, as there is no set/unset counterpart for the container value.
/// This leads to incorrectly interpreting the change as an addition or removal. The second case
/// is because no [`ValueSentinel`] events are produced at all, which leaves the container type
/// ambiguous. In either case these type change events are the only way to get the complete picture.
#[expect(unused_variables)]
pub trait TypeSentinel<Custom>: Sentinel {
    /// Observe and validate setting a type at the current path.
    fn set_type(&mut self, value_type: ValueType<Custom>) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Observe and validate unsetting a type at the current path.
    fn unset_type(&mut self, value_type: ValueType<Custom>) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Observe when values are set or unset at the current path.
///
/// Updates are represented as a value unset and another one set. There are no ordering
/// guarantees between the calls.
#[expect(unused_variables)]
pub trait ValueSentinel<V>: Sentinel {
    /// Observe and validate setting a new value under the current path.
    fn set(&mut self, value: &V) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Observe and validate unsetting the value under the current path.
    fn unset(&mut self, value: V) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// Enables a Sentinel to keep track of document traversal.
///
/// During a document mutation (typically via [`DotStoreJoin::join`](crate::DotStoreJoin)), the
/// document tree is traversed in a depth-first manner and each map field or array element visited
/// is reported via this interface, so that the Sentinel can update its internal pointer.
///
/// Typically, you want to implement this for [`String`] (to visit [`OrMap`](crate::OrMap) values)
/// and [`Uid`](crate::crdts::orarray::Uid) (to visit [`OrArray`](crate::OrArray)), for example.
///
/// NOTE: any nodes in the document tree may be visited, regardless of whether they contain a change.
/// Additionally, nodes that are visited may not exist in the final tree.
#[expect(unused_variables)]
pub trait Visit<K>: Sentinel {
    /// Descend into a map field or array element.
    fn enter(&mut self, key: &K) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Backtrack to the parent container.
    ///
    /// NOTE: may not be called if the Sentinel produces an Err.
    fn exit(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// A Sentinel that does nothing.
///
/// This is useful when the join doesn't need any introspection. Using it helps the compiler
/// optimise some code away.
pub struct DummySentinel;

impl Sentinel for DummySentinel {
    type Error = Infallible;
}

impl KeySentinel for DummySentinel {}

impl<C> TypeSentinel<C> for DummySentinel {}

impl<K> Visit<K> for DummySentinel {}

impl<V> ValueSentinel<V> for DummySentinel {}

#[cfg(test)]
pub(crate) mod test {
    use super::*;
    use std::{collections::BTreeMap, fmt};

    /// A Sentinel that always rejects changes.
    pub struct NoChangeValidator;

    impl Sentinel for NoChangeValidator {
        type Error = ();
    }

    impl<K> Visit<K> for NoChangeValidator {}

    impl KeySentinel for NoChangeValidator {
        fn create_key(&mut self) -> Result<(), Self::Error> {
            Err(())
        }

        fn delete_key(&mut self) -> Result<(), Self::Error> {
            Err(())
        }
    }

    impl<C> TypeSentinel<C> for NoChangeValidator {
        fn set_type(&mut self, _value_type: ValueType<C>) -> Result<(), Self::Error> {
            Err(())
        }

        fn unset_type(&mut self, _value_type: ValueType<C>) -> Result<(), Self::Error> {
            Err(())
        }
    }

    impl<V> ValueSentinel<V> for NoChangeValidator {
        fn set(&mut self, _value: &V) -> Result<(), Self::Error> {
            Err(())
        }

        fn unset(&mut self, _value: V) -> Result<(), Self::Error> {
            Err(())
        }
    }

    /// A Sentinel that counts keys added or removed and rejects other changes.
    #[derive(Debug, Default)]
    pub struct KeyCountingValidator {
        pub added: usize,
        pub removed: usize,
    }
    impl Sentinel for KeyCountingValidator {
        type Error = ();
    }
    impl<K> Visit<K> for KeyCountingValidator {}
    impl KeySentinel for KeyCountingValidator {
        fn create_key(&mut self) -> Result<(), Self::Error> {
            self.added += 1;
            Ok(())
        }

        fn delete_key(&mut self) -> Result<(), Self::Error> {
            self.removed += 1;
            Ok(())
        }
    }
    impl<C> TypeSentinel<C> for KeyCountingValidator {
        fn set_type(&mut self, _value_type: crate::crdts::ValueType<C>) -> Result<(), Self::Error> {
            Err(())
        }

        fn unset_type(
            &mut self,
            _value_type: crate::crdts::ValueType<C>,
        ) -> Result<(), Self::Error> {
            Err(())
        }
    }
    impl<V> ValueSentinel<V> for KeyCountingValidator {}

    /// A Sentinel that counts changes to values and rejects other changes.
    ///
    /// Setting `permissive` to true disables erroring on key and type changes.
    #[derive(Debug)]
    pub struct ValueCountingValidator<V> {
        pub added: BTreeMap<V, usize>,
        pub removed: BTreeMap<V, usize>,
        path: Vec<String>,
        permissive: bool,
    }

    impl<V> Default for ValueCountingValidator<V> {
        fn default() -> Self {
            Self {
                added: Default::default(),
                removed: Default::default(),
                path: Default::default(),
                permissive: false,
            }
        }
    }

    impl<V> ValueCountingValidator<V> {
        pub fn new(permissive: bool) -> Self {
            Self {
                permissive,
                ..Default::default()
            }
        }
    }

    impl<V> Sentinel for ValueCountingValidator<V> {
        type Error = String;
    }

    impl<K, V> Visit<K> for ValueCountingValidator<V>
    where
        K: std::fmt::Debug,
    {
        fn enter(&mut self, key: &K) -> Result<(), Self::Error> {
            self.path.push(format!("{key:?}"));
            Ok(())
        }

        fn exit(&mut self) -> Result<(), Self::Error> {
            self.path.pop();
            Ok(())
        }
    }

    impl<V> KeySentinel for ValueCountingValidator<V> {
        fn create_key(&mut self) -> Result<(), Self::Error> {
            self.permissive
                .then_some(())
                .ok_or(format!("create_key at {}", self.path.join("/")))
        }

        fn delete_key(&mut self) -> Result<(), Self::Error> {
            self.permissive
                .then_some(())
                .ok_or(format!("delete_key at {}", self.path.join("/")))
        }
    }

    impl<C, V> TypeSentinel<C> for ValueCountingValidator<V>
    where
        C: fmt::Debug,
    {
        fn set_type(&mut self, value_type: crate::crdts::ValueType<C>) -> Result<(), Self::Error> {
            self.permissive.then_some(()).ok_or(format!(
                "set_type: {value_type:?} at {}",
                self.path.join("/")
            ))
        }

        fn unset_type(
            &mut self,
            value_type: crate::crdts::ValueType<C>,
        ) -> Result<(), Self::Error> {
            self.permissive.then_some(()).ok_or(format!(
                "unset_type: {value_type:?} at {}",
                self.path.join("/")
            ))
        }
    }

    impl<V> ValueSentinel<V> for ValueCountingValidator<V>
    where
        V: std::fmt::Debug + Ord + Clone,
    {
        fn set(&mut self, value: &V) -> Result<(), Self::Error> {
            *self.added.entry(value.clone()).or_default() += 1;
            Ok(())
        }

        fn unset(&mut self, value: V) -> Result<(), Self::Error> {
            *self.removed.entry(value).or_default() += 1;
            Ok(())
        }
    }
}
