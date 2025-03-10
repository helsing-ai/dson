// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! This module contains an implementation of Sentinel that simply records
//! all calls in a human readable form. This is mostly useful for tests.

use crate::{
    crdts::ValueType,
    sentinel::{KeySentinel, Sentinel, TypeSentinel, ValueSentinel, Visit},
};
use std::{convert::Infallible, fmt::Debug};

/// A sentinel that records all calls.
#[derive(Default)]
pub struct RecordingSentinel {
    path: Vec<String>,
    /// A string-representation of each call that the sentinel has received.
    /// This is mostly useful for tests.
    pub changes_seen: Vec<String>,
}
impl RecordingSentinel {
    /// Create a new PeekingSentinel
    pub fn new() -> RecordingSentinel {
        RecordingSentinel {
            path: vec![],
            changes_seen: vec![],
        }
    }
}
impl Sentinel for RecordingSentinel {
    type Error = Infallible;
}
impl<K: Debug> Visit<K> for RecordingSentinel {
    fn enter(&mut self, key: &K) -> Result<(), Self::Error> {
        self.path.push(format!("{key:?}"));
        Ok(())
    }
    fn exit(&mut self) -> Result<(), Self::Error> {
        self.path.pop();
        Ok(())
    }
}
impl KeySentinel for RecordingSentinel {
    fn create_key(&mut self) -> Result<(), Self::Error> {
        self.changes_seen
            .push(format!("create_key at {}", self.path.join("/")));
        Ok(())
    }

    fn delete_key(&mut self) -> Result<(), Self::Error> {
        self.changes_seen
            .push(format!("delete_key at {}", self.path.join("/")));
        Ok(())
    }
}
impl<V: Debug> ValueSentinel<V> for RecordingSentinel {
    fn set(&mut self, value: &V) -> Result<(), Self::Error> {
        self.changes_seen.push(format!("set {value:?}"));
        Ok(())
    }
    fn unset(&mut self, value: V) -> Result<(), Self::Error> {
        self.changes_seen.push(format!("unset {:?}", &value));
        Ok(())
    }
}
impl<V: Debug> TypeSentinel<V> for RecordingSentinel {
    fn set_type(&mut self, value_type: ValueType<V>) -> Result<(), Self::Error> {
        self.changes_seen.push(format!("set_type {value_type:?}"));
        Ok(())
    }
    fn unset_type(&mut self, value_type: ValueType<V>) -> Result<(), Self::Error> {
        self.changes_seen.push(format!("unset_type {value_type:?}"));
        Ok(())
    }
}
