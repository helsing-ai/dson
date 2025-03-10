// (c) Copyright 2025 Helsing GmbH. All rights reserved.
//! Implementation of the quickcheck::Arbitrary trait for all of CRDT types.

use crate::{
    CausalContext, CausalDotStore, Dot, DotFun, DotFunMap, DotStore, DotStoreJoin, ExtensionType,
    Identifier, MvReg, OrArray, OrMap,
    api::timestamp::Timestamp,
    crdts::{
        NoExtensionTypes, TypeVariantValue, Value,
        mvreg::MvRegValue,
        orarray::{PairMap, Position, Uid},
    },
    dotstores::{DotMapValue, recording_sentinel::RecordingSentinel},
};
use quickcheck::{Arbitrary, Gen};
use std::{collections::HashMap, fmt, hash::Hash, num::NonZeroU64};

impl Arbitrary for NoExtensionTypes {
    fn arbitrary(_: &mut Gen) -> Self {
        Self
    }
}

impl quickcheck::Arbitrary for Identifier {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        // Skew the distribution to increase the likelihood of triggering bugs.
        // Most interesting behavior occurs when the same Identifier occurs multiple times
        // in the same test.
        let node_choices = [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            u8::arbitrary(g).saturating_add(1),
        ];
        let app_choices = [
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            u16::arbitrary(g).saturating_add(1) % 4096,
        ];
        let node = g.choose(&node_choices).unwrap();
        let app = g.choose(&app_choices).unwrap();
        Self::new(*node, *app)
    }
}

impl quickcheck::Arbitrary for Dot {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        // Skew the distribution to increase the likelihood of triggering bugs.
        // Large distinct values of u64:s often cause the same code paths.
        // Interesting code paths often happen when several values are close to each other.
        let seq_choices = [1, 1, 1, 2, 2, 3, 4, 5, u64::arbitrary(g).saturating_add(1)];
        let seq = g.choose(&seq_choices).unwrap();
        (Identifier::arbitrary(g), NonZeroU64::new(*seq).unwrap()).into()
    }
}

impl<C> Arbitrary for Value<C>
where
    C: Arbitrary + ExtensionType + DotStoreJoin<RecordingSentinel> + fmt::Debug + Clone + PartialEq,
    C::Value: Arbitrary,
{
    fn arbitrary(g: &mut Gen) -> Self {
        match *g.choose(&["map", "array", "reg"]).unwrap() {
            "map" => {
                // penalize nesting
                let mut g = Gen::new(g.size() / 2);
                Self::Map(OrMap::arbitrary(&mut g))
            }
            "array" => {
                // penalize nesting
                let mut g = Gen::new(g.size() / 2);
                Self::Array(OrArray::arbitrary(&mut g))
            }
            "reg" => Self::Register(MvReg::arbitrary(g)),
            _ => unreachable!(),
        }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        match self {
            Self::Map(m) => Box::new(m.shrink().map(Self::Map)),
            Self::Array(a) => Box::new(a.shrink().map(Self::Array)),
            Self::Register(r) => Box::new(r.shrink().map(Self::Register)),
            Self::Custom(c) => Box::new(c.shrink().map(Self::Custom)),
        }
    }
}

impl<C> Arbitrary for TypeVariantValue<C>
where
    C: Arbitrary + ExtensionType + DotStoreJoin<RecordingSentinel> + fmt::Debug + Clone + PartialEq,
    C::Value: Arbitrary,
{
    fn arbitrary(g: &mut Gen) -> Self {
        Self {
            map: if bool::arbitrary(g) {
                // penalize nesting
                let mut g = Gen::new(g.size() / 2);
                <_>::arbitrary(&mut g)
            } else {
                Default::default()
            },
            array: if bool::arbitrary(g) {
                // penalize nesting
                let mut g = Gen::new(g.size() / 2);
                <_>::arbitrary(&mut g)
            } else {
                Default::default()
            },
            reg: <_>::arbitrary(g),
            custom: <_>::arbitrary(g),
        }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        let mut vars = Vec::new();
        if !self.map.is_bottom() && (!self.array.is_bottom() || !self.reg.is_bottom()) {
            vars.push({
                let mut v = self.clone();
                v.map = Default::default();
                v
            })
        }
        if !self.array.is_bottom() && (!self.map.is_bottom() || !self.reg.is_bottom()) {
            vars.push({
                let mut v = self.clone();
                v.array = Default::default();
                v
            })
        }
        if !self.reg.is_bottom() && (!self.array.is_bottom() || !self.map.is_bottom()) {
            vars.push({
                let mut v = self.clone();
                v.reg = Default::default();
                v
            })
        }
        Box::new(vars.into_iter())
    }
}

impl Arbitrary for MvReg {
    fn arbitrary(g: &mut Gen) -> Self {
        if g.size() == 0 || bool::arbitrary(g) {
            MvReg::default()
        } else {
            let reg = MvReg::default();
            let src = MvRegValue::arbitrary(g);
            let id = Identifier::arbitrary(g);
            let cc = CausalContext::new();
            reg.write(src, &cc, id).store
        }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        if self.is_bottom() {
            quickcheck::empty_shrinker()
        } else {
            assert_eq!(
                self.0.len(),
                1,
                "our arbitrary never generates multi-dot MvRegs (atm)"
            );
            let dot = self.0.keys().next().unwrap();
            let v = self.0.values().next().unwrap().clone();
            Box::new(Arbitrary::shrink(&(dot, v)).map(|(dot, v)| {
                let mut dot_fun = DotFun::default();
                dot_fun.set(dot, v);
                Self(dot_fun)
            }))
        }
    }
}

// This cannot be implemented from chrono's arbitrary trait.
// 1. The implementation is for arbitrary crate, not quickcheck
// 2. This would produce values outside of dson::timestamp::Timestamp's supported range
#[cfg(feature = "chrono")]
impl Arbitrary for Timestamp {
    // A random date between 0 and 9999
    fn arbitrary(g: &mut Gen) -> Timestamp {
        use chrono::DateTime;
        let range =
            Timestamp::MAX.as_datetime().timestamp() - Timestamp::MIN.as_datetime().timestamp();
        let random_number: i64 = Arbitrary::arbitrary(g);
        let random_secs =
            Timestamp::MIN.as_datetime().timestamp() + (random_number.rem_euclid(range));

        let random_number: u32 = Arbitrary::arbitrary(g);
        let random_nanoseconds = random_number % 1_000_000_000;
        let random_datetime = DateTime::from_timestamp(random_secs, random_nanoseconds)
            .expect("random timestamp is within accepted range");

        Timestamp::new(random_datetime).expect("random timestamp is within accepted range")
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        use chrono::{DateTime, Datelike, NaiveTime, Utc};
        let datetime = self.as_datetime();
        let midnight = datetime.date_naive().and_time(NaiveTime::MIN);
        let first_day_of_the_month = datetime
            .date_naive()
            .with_day(1)
            .expect("1 is a valid day")
            .and_time(NaiveTime::MIN);
        let first_month_of_the_year = datetime
            .date_naive()
            .with_month(1)
            .expect("1 is a valid month")
            .and_time(NaiveTime::MIN);
        let epoch = DateTime::<Utc>::UNIX_EPOCH;
        let this = *self;
        let shrunk_datetimes = [midnight, first_day_of_the_month, first_month_of_the_year]
            .into_iter()
            .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
            .chain(std::iter::once(epoch))
            .map(|dt| Timestamp::new(dt).expect("Static datetime is a valid Timestamp"))
            // repeated calls to shrink must eventually end up with an empty result
            // so we make sure the shrunk stamps are always strictly smaller than 'self'
            .filter(move |x| x < &this);
        Box::new(shrunk_datetimes)
    }
}

#[cfg(not(feature = "chrono"))]
impl Arbitrary for Timestamp {
    fn arbitrary(g: &mut Gen) -> Timestamp {
        Timestamp::new(i64::arbitrary(g)).unwrap()
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        Box::new(
            self.as_millis()
                .shrink()
                .map(|v| Timestamp::new(v).unwrap()),
        )
    }
}

impl Arbitrary for MvRegValue {
    fn arbitrary(g: &mut Gen) -> Self {
        let mut choices = vec![
            "bytes",
            "string",
            "double",
            "u64",
            "i64",
            "bool",
            "timestamp",
        ];
        if cfg!(feature = "ulid") {
            choices.push("ulid");
        }
        match *g.choose(&choices).unwrap() {
            "bytes" => Self::Bytes(<_>::arbitrary(g)),
            "string" => Self::String(<_>::arbitrary(g)),
            "double" => Self::Double(<_>::arbitrary(g)),
            "u64" => Self::U64(<_>::arbitrary(g)),
            "i64" => Self::I64(<_>::arbitrary(g)),
            "bool" => Self::Bool(<_>::arbitrary(g)),
            "timestamp" => Self::Timestamp(<_>::arbitrary(g)),
            #[cfg(feature = "ulid")]
            "ulid" => Self::Ulid(ulid::Ulid(<_>::arbitrary(g))),
            _ => unreachable!(),
        }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        match self {
            MvRegValue::Bytes(v) => Box::new(
                v.shrink()
                    .map(MvRegValue::Bytes)
                    .chain(std::iter::once(MvRegValue::Bool(true))),
            ),
            MvRegValue::String(v) => Box::new(
                v.shrink()
                    .map(MvRegValue::String)
                    .chain(std::iter::once(MvRegValue::Bool(true))),
            ),
            MvRegValue::Double(v) => Box::new(
                v.shrink()
                    .map(MvRegValue::Double)
                    .chain(std::iter::once(MvRegValue::Bool(true))),
            ),
            MvRegValue::Float(v) => Box::new(
                v.shrink()
                    .map(MvRegValue::Float)
                    .chain(std::iter::once(MvRegValue::Bool(true))),
            ),
            MvRegValue::U64(v) => Box::new(
                v.shrink()
                    .map(MvRegValue::U64)
                    .chain(std::iter::once(MvRegValue::Bool(true))),
            ),
            MvRegValue::I64(v) => Box::new(
                v.shrink()
                    .map(MvRegValue::I64)
                    .chain(std::iter::once(MvRegValue::Bool(true))),
            ),
            MvRegValue::Bool(v) => Box::new(v.shrink().map(MvRegValue::Bool)),
            MvRegValue::Timestamp(v) => Box::new(
                v.shrink()
                    .map(MvRegValue::Timestamp)
                    .chain(std::iter::once(MvRegValue::Bool(true))),
            ),
            #[cfg(feature = "ulid")]
            MvRegValue::Ulid(v) => Box::new(
                v.0.shrink()
                    .map(|u| MvRegValue::Ulid(ulid::Ulid(u)))
                    .chain(std::iter::once(MvRegValue::Bool(true))),
            ),
        }
    }
}

impl<V> Arbitrary for DotMapValue<V>
where
    V: Arbitrary,
{
    fn arbitrary(g: &mut Gen) -> Self {
        Self {
            dots: None,
            value: V::arbitrary(g),
        }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        Box::new(self.value.shrink().map(|value| Self { dots: None, value }))
    }
}

impl<K, C> Arbitrary for OrMap<K, C>
where
    K: Hash + Eq + fmt::Debug + Clone + Arbitrary,
    C: Arbitrary + ExtensionType + DotStoreJoin<RecordingSentinel> + Clone + PartialEq + fmt::Debug,
    C::Value: Arbitrary,
{
    fn arbitrary(g: &mut Gen) -> Self {
        if g.size() == 0 || bool::arbitrary(g) {
            OrMap::default()
        } else {
            let map = OrMap::default();
            let src = HashMap::<K, super::Value<C>>::arbitrary(g);
            let id = Identifier::arbitrary(g);
            let cc = CausalContext::new();
            src.into_iter()
                .fold(map.create(&cc, id), |mut map, (k, v)| {
                    let map_update = map.store.apply(
                        |_old, cc, _id| CausalDotStore {
                            store: v,
                            context: cc.clone(),
                        },
                        k,
                        &map.context,
                        id,
                    );
                    map.test_consume(map_update);
                    map
                })
                .store
        }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        if self.is_bottom() {
            return quickcheck::empty_shrinker();
        }

        let mut vars = Vec::new();

        vars.push(OrMap::default());

        for map in self.0.shrink() {
            vars.push(Self(map));
        }
        Box::new(vars.into_iter())
    }
}

impl<C> Arbitrary for OrArray<C>
where
    C: Arbitrary + ExtensionType + DotStoreJoin<RecordingSentinel> + Clone + PartialEq + fmt::Debug,
    C::Value: Arbitrary,
{
    fn arbitrary(g: &mut Gen) -> Self {
        if g.size() == 0 || bool::arbitrary(g) {
            OrArray::default()
        } else {
            let list = OrArray::default();
            let src = Vec::<super::Value<C>>::arbitrary(g);

            let cc = CausalContext::new();

            // Find an unused causal track, which we can use as 'our' id, so
            // we're sure to have a compact track (otherwise we will trigger asserts).
            let id = src
                .iter()
                .map(|x| x.dots())
                .fold(CausalContext::default(), |mut a, b| {
                    a.union(&b);
                    a
                })
                .unused_identifier()
                .expect("test case is small enough that some Identifier is always unused");

            src.into_iter()
                .fold(list.create(&cc, id), |mut list, v| {
                    let uid = list.context.next_dot_for(id).into();
                    let list_update = list.store.insert(
                        uid,
                        |cc, _id| {
                            let mut cc = cc.clone();
                            cc.union(&v.dots());
                            CausalDotStore {
                                store: v,
                                context: cc.clone(),
                            }
                        },
                        Position::arbitrary(g),
                        &list.context,
                        id,
                    );
                    list.test_consume(list_update);
                    list
                })
                .store
        }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        if self.is_bottom() {
            return quickcheck::empty_shrinker();
        }

        let mut vars = Vec::new();

        vars.push(OrArray::default());

        for map in self.0.shrink() {
            vars.push(Self(map));
        }
        Box::new(vars.into_iter())
    }
}

impl<C> Arbitrary for PairMap<C>
where
    C: Arbitrary + ExtensionType + DotStoreJoin<RecordingSentinel> + fmt::Debug + Clone + PartialEq,
    C::Value: Arbitrary,
{
    fn arbitrary(g: &mut Gen) -> Self {
        let first = if g.size() == 0 {
            // bottom value
            Default::default()
        } else {
            Arbitrary::arbitrary(g)
        };

        let mut dot_fun_map = DotFunMap::default();
        for (d, p) in Vec::<(Dot, Position)>::arbitrary(g) {
            let mut dot_fun = DotFun::default();
            dot_fun.set(d, p);
            dot_fun_map.set(d, dot_fun);
        }

        Self {
            value: first,
            positions: dot_fun_map,
        }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        if self.value.is_bottom() {
            return quickcheck::empty_shrinker();
        }

        let mut vars = Vec::new();
        vars.push(Self {
            value: Default::default(),
            positions: self.positions.clone(),
        });

        // NOTE: based on the impl Arbitrary for tuples,
        // the recommendation is that shrinking is one-at-a-time
        // rather than carthesian product.

        for shrunk in self.value.shrink() {
            vars.push(Self {
                value: shrunk,
                positions: self.positions.clone(),
            });
        }

        let dps: Vec<(Dot, Position)> = self
            .positions
            .values()
            .flat_map(|v| v.iter())
            .map(|(dot, &p)| (dot, p))
            .collect();

        for shrunk in dps.shrink() {
            let mut dot_fun_map = DotFunMap::default();
            for (d, p) in shrunk {
                let mut dot_fun = DotFun::default();
                dot_fun.set(d, p);
                dot_fun_map.set(d, dot_fun);
            }

            vars.push(Self {
                value: self.value.clone(),
                positions: dot_fun_map.clone(),
            });
        }

        Box::new(vars.into_iter())
    }
}

impl Arbitrary for Position {
    fn arbitrary(g: &mut Gen) -> Self {
        let val = u64::arbitrary(g) % (Position::UPPER as u64);
        Self(
            val as f64
                / g.choose(&[1.0, 2.0, 5.0, 10.0])
                    .expect("choose non empty slice"),
        )
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        Box::new(self.0.shrink().map(Self))
    }
}

impl Arbitrary for Uid {
    fn arbitrary(g: &mut Gen) -> Self {
        Self::from(Dot::arbitrary(g))
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        Box::new(self.dot().shrink().map(Self::from))
    }
}
