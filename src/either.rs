// (c) Copyright 2025 Helsing GmbH. All rights reserved.

//! The enum Either with variants Left and Right is a general purpose sum type
//! with two cases.
#[derive(Debug, Clone, Ord, PartialOrd, PartialEq, Eq)]
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<A, B> Either<Either<A, B>, B> {
    /// Converts from `Either<Either<A, B, B>>` to `Either<A, B>`.
    pub fn flatten(self) -> Either<A, B> {
        match self {
            Either::Left(nested) => nested,
            Either::Right(b) => Either::Right(b),
        }
    }
}

impl<A, B> Either<A, Either<A, B>> {
    /// Converts from `Either<A, Either<A, B>>` to `Either<A, B>`.
    pub fn flatten(self) -> Either<A, B> {
        match self {
            Either::Left(a) => Either::Left(a),
            Either::Right(nested) => nested,
        }
    }
}
