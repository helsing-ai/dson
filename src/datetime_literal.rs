// (c) Copyright 2025 Helsing GmbH. All rights reserved.
/// Declarative macro to create a [`chrono::DateTime<chrono::Utc>`] suitable
/// for const evaluation, as this is otherwise cumbersome.
///
/// Usage:
/// ```rust
///    # use chrono::{DateTime, Utc};
///    # use dson::datetime;
///    let datetime: DateTime<Utc> = datetime!( 2024-12-24 15:00:00 Z);
///    # let _ = datetime;
/// ```
#[macro_export]
macro_rules! datetime {
    ( $year:literal-$month:literal-$day:literal $(T)? $hour:literal:$min:literal:$second:literal Z)  => {
        const {
            #[allow(clippy::zero_prefixed_literal)]
            $crate::chrono::DateTime::<$crate::chrono::Utc>::from_naive_utc_and_offset(
            datetime!($year - $month - $day $hour:$min:$second),
            $crate::chrono::Utc
        ) }
    };
    ( $year:literal-$month:literal-$day:literal $(T)? $hour:literal:$min:literal:$second:literal)  => {
        const {
            #[allow(clippy::zero_prefixed_literal)]
            $crate::chrono::NaiveDateTime::new(
                match $crate::chrono::NaiveDate::from_ymd_opt($year, $month, $day) {
                    Some(date) => date,
                    None => ::std::panic!("year-month-day outside expected range.")
                },
                match $crate::chrono::NaiveTime::from_hms_opt($hour, $min, $second) {
                    Some(time) => time,
                    None => ::std::panic!("hour:min:second outside expected range.")
                }
        ) }
    };
}
