// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! This module provides a `Timestamp` type for efficient encoding of UTC datetimes.
//!
//! The `Timestamp` is represented as a 64-bit integer of milliseconds since the
//! UNIX epoch, but is constrained to a range of years from 0 to 9999.
//! This allows for compact and performant representation of datetimes.
use std::fmt;

#[cfg(feature = "chrono")]
use crate::datetime;
#[cfg(feature = "chrono")]
use chrono::{DateTime, Datelike, Utc};
#[cfg(feature = "chrono")]
use std::str::FromStr;

/// Error returned when creating or parsing a `Timestamp`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimestampError {
    /// The year is outside the supported range of `0` to `9999`.
    InvalidYear(i32),
    /// The string could not be parsed as a valid RFC 3339 datetime.
    Parse(String),
}

impl fmt::Display for TimestampError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimestampError::InvalidYear(year) => write!(
                f,
                "invalid year {year}, supported years are between 0 and 9999 included"
            ),
            TimestampError::Parse(s) => {
                write!(f, "failed to parse date {s} in rfc3339 format")
            }
        }
    }
}

impl std::error::Error for TimestampError {}

/// Represents a UTC datetime with millisecond precision.
///
/// `Timestamp` is stored as an `i64` representing the number of milliseconds since the
/// UNIX epoch.
///
/// The valid range for a `Timestamp` is from `0000-01-01T00:00:00.000Z` to
/// `9999-12-31T23:59:59.999Z`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(::serde::Deserialize, ::serde::Serialize))]
pub struct Timestamp(i64);

impl Timestamp {
    /// Creates a new `Timestamp` from a `chrono::DateTime<Utc>`.
    ///
    /// The datetime is truncated to millisecond precision.
    ///
    /// # Errors
    ///
    /// Returns an error if the year is outside the supported range of `0` to `9999`.
    #[cfg(feature = "chrono")]
    pub fn new(datetime: DateTime<Utc>) -> Result<Timestamp, TimestampError> {
        let year = datetime.year();
        // NOTE: This is arguably more clear.
        #[expect(clippy::manual_range_contains)]
        if year < 0 || year > 9999 {
            return Err(TimestampError::InvalidYear(year));
        }
        let truncated_timestamp = datetime.timestamp_millis();
        Ok(Timestamp(truncated_timestamp))
    }

    #[cfg(not(feature = "chrono"))]
    /// Creates a new `Timestamp` from an i64. This operation always succeeds.
    pub fn new(val: i64) -> Result<Timestamp, TimestampError> {
        Ok(Self(val))
    }

    /// Creates a `Timestamp` from a number of milliseconds since the UNIX epoch.
    ///
    /// Returns `None` if the number of milliseconds corresponds to a datetime outside
    /// the supported range.
    #[cfg(feature = "chrono")]
    pub fn from_millis(milliseconds: i64) -> Option<Self> {
        (Self::MIN.as_millis()..=Self::MAX.as_millis())
            .contains(&milliseconds)
            .then_some(Self(milliseconds))
    }

    #[cfg(not(feature = "chrono"))]
    /// Creates a `Timestamp` from a number of milliseconds since the UNIX epoch.
    /// This operation always succeeds.
    pub fn from_millis(milliseconds: i64) -> Option<Self> {
        Some(Self(milliseconds))
    }

    /// Returns the number of milliseconds since the UNIX epoch as an `i64`.
    pub fn as_millis(&self) -> i64 {
        self.0
    }

    /// Converts the `Timestamp` to a `chrono::DateTime<Utc>`.
    #[cfg(feature = "chrono")]
    pub(crate) fn as_datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.0)
            .expect("roundtrips with `DateTime::timestamp_millis`")
    }

    /// The minimum supported `Timestamp`: `0000-01-01T00:00:00.000Z`.
    #[cfg(feature = "chrono")]
    pub const MIN: Timestamp = Timestamp(datetime!(0000-01-01 00:00:00 Z).timestamp_millis());
    /// The maximum supported `Timestamp`: `9999-12-31T23:59:59.999Z`.
    #[cfg(feature = "chrono")]
    pub const MAX: Timestamp = Timestamp(datetime!(10000-01-01 00:00:00 Z).timestamp_millis() - 1);
}

#[cfg(feature = "chrono")]
impl fmt::Display for Timestamp {
    // Formats the `Timestamp` as an RFC 3339 string.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_datetime().fmt(f)
    }
}

#[cfg(not(feature = "chrono"))]
impl fmt::Display for Timestamp {
    // Formats the `Timestamp` as an RFC 3339 string.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Debug for Timestamp {
    // Formats the `Timestamp` as an RFC 3339 string for debugging.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(all(feature = "serde", feature = "chrono"))]
impl From<Timestamp> for serde_json::Value {
    // The string is formatted according to RFC 3339 with millisecond precision.
    fn from(value: Timestamp) -> Self {
        serde_json::Value::String(
            value
                .as_datetime()
                .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
                .to_string(),
        )
    }
}

#[cfg(feature = "chrono")]
impl FromStr for Timestamp {
    type Err = TimestampError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let datetime =
            DateTime::parse_from_rfc3339(s).map_err(|_| TimestampError::Parse(s.to_string()))?;
        Timestamp::new(datetime.to_utc())
    }
}

#[cfg(all(test, feature = "chrono"))]
mod tests {
    use super::*;
    use chrono::{DateTime, Utc};

    #[test]
    fn new_timestamp_truncates_at_millisecond_precision() {
        assert_eq!(
            "1996-12-19T16:39:57.123555Z".parse::<Timestamp>().unwrap(),
            "1996-12-19T16:39:57.123Z".parse::<Timestamp>().unwrap()
        )
    }

    #[test]
    fn constants_are_correctly_computed() {
        assert_eq!(
            "0000-01-01T00:00:00Z".parse::<Timestamp>().unwrap(),
            Timestamp::MIN
        );

        assert_eq!(
            "9999-12-31T23:59:59.999Z".parse::<Timestamp>().unwrap(),
            Timestamp::MAX
        );
    }

    #[test]
    fn timestamp_constructors() {
        let unparsable_timestamp: Result<Timestamp, _> = "0000-01-01T00:00:00ZTR".parse();
        assert!(unparsable_timestamp.is_err());

        let out_of_range_year = DateTime::<Utc>::UNIX_EPOCH.with_year(10_000).unwrap();
        assert!(Timestamp::new(out_of_range_year).is_err());

        let parseable_timestamp: Result<Timestamp, _> = "0000-01-01T00:00:00Z".parse();
        assert!(parseable_timestamp.is_ok())
    }

    #[test]
    fn parse_accepts_any_timezone() {
        assert_eq!(
            "0000-01-01T00:00:00Z".parse::<Timestamp>().unwrap(),
            "0000-01-01T01:00:00+01:00".parse::<Timestamp>().unwrap()
        );
    }
}
