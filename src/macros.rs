// (c) Copyright 2025 Helsing GmbH. All rights reserved.
/// Convenience macro for creating dot values.
///
/// NOTE! This is mostly useful for tests, since it does not provide control
/// over the app or priority fields of a dot.
#[macro_export]
macro_rules! dot {
    ($seq:expr) => {
        const {
            $crate::causal_context::Dot::mint(
                $crate::causal_context::Identifier::new(1, 1),
                $seq,
            )
        }
    };
    ($node:expr, $seq:expr) => {
        const {
            $crate::causal_context::Dot::mint(
                $crate::causal_context::Identifier::new($node, 1),
                $seq,
            )
        }
    };
    ($node:expr, $app:expr, $seq:expr) => {
        const {
            $crate::causal_context::Dot::mint(
                $crate::causal_context::Identifier::new($node, $app),
                $seq,
            )
        }
    };
}

/// Convenience macro for creating a OrMap instance.
///
/// Use the [`crdt_map_store`](crate::crdt_map_store) literal to also create a matching
/// CausalContext.
///
/// ```rust
/// # use dson::{crdt_map_literal, dot};
/// let map = crdt_map_literal! {
///     "field_x" => ("Hello", dot!(1,2)),
///     "field_y" => ("World", dot!(1,3)),
///     "field_z" => {
///         "field_x" => ("Nested", dot!(1,4)),
///         "field_y" => ("Nested", dot!(1,5))
///     }
/// };
/// ```
///
#[macro_export]
macro_rules! crdt_map_literal {
   ($($k:literal => $v:tt),*) => {
        $crate::crdt_literal!( { $( $k => $v ),* } ).map
    };
}

/// Convenience macro for creating a TypeVariantValue, of either map, array or register type.
///
///
/// Register literal:
/// ```rust
/// # use dson::{crdt_literal, dot};
/// let reg = crdt_literal!( ("hello", dot!(1)));
/// ```
///
/// Conflicted register literal:
/// ```rust
/// # use dson::{crdt_literal, dot};
/// let reg = crdt_literal!( ("Hello", dot!(1); "Bonjour", dot!(2); ));
/// ```
///
/// Map literal (note the '{' and '}'):
/// ```rust
/// # use dson::{crdt_literal, dot};
/// let reg = crdt_literal!( {
///     "Greeting" => ("Hello", dot!(1))
/// } );
/// ```
///
/// Array literal (note the '[' and ']'):
/// ```rust
/// # use dson::{crdt_literal, dot};
/// let reg = crdt_literal!( [
///     (("Banana", dot!(3)), dot!(4), dot!(5), dot!(6), 42.0),
///     (("Cantaloupe", dot!(7)), dot!(8), dot!(9), dot!(10), 43.0)
/// ] );
/// ```
/// The first tuple is the actual value in the array, with its dot.
/// The remaining 4 parameters are: Uid, 2 array position dots (for dotfunmap +
/// dotfun), and the f64 value that decides the sorting order of the array.
///
/// See section 5, about the OrArray algorithm, in the DSON paper for more information.
///
/// Note that this macro does not generate a CausalContext.
#[macro_export]
macro_rules! crdt_literal {
    // Map
    ({$($k:literal => $v:tt),*}) => {
        {
            let mut map = $crate::OrMap::<String, $crate::crdts::NoExtensionTypes>::default();
            $( { $crate::crdt_literal!(map_insert, map, $k, $v); } )*
            $crate::crdts::TypeVariantValue {
                map,
                ..$crate::crdts::TypeVariantValue::<$crate::crdts::NoExtensionTypes>::default()
            }
        }
    };

    // Array
    ([$($v:tt),*]) => {
        {
            let mut array = $crate::OrArray::<$crate::crdts::NoExtensionTypes>::default();
            $( $crate::crdt_literal!(array_element, array, $v); )*
            $crate::crdts::TypeVariantValue {
                array,
                ..$crate::crdts::TypeVariantValue::<$crate::crdts::NoExtensionTypes>::default()
            }
        }
    };

    // Mvreg
    ( ($($v:expr, $dot:expr $(;)? )* ) ) => {
        {
            let mut reg = $crate::crdts::mvreg::MvReg::default();
            $( reg.push($dot, $v); )*
            $crate::crdts::TypeVariantValue {
                reg,
                ..$crate::crdts::TypeVariantValue::<$crate::crdts::NoExtensionTypes>::default()
            }
        }
    };

    // Helper for creating map elements
    (map_insert, $temp:ident, $k:literal , $v: tt) => {
        $temp.insert($k.into(), $crate::crdt_literal!($v));
    };

    // Helper for creating array elements
    (array_element, $temp:ident, ($v:tt, $uid: expr, $dot1:expr, $dot2:expr, $pos_f64:expr)) => {
        let val = $crate::crdt_literal!($v);
        $temp.insert_raw($crate::crdts::orarray::Uid::from($uid), std::iter::once(($dot1,$dot2,$pos_f64)), val);
    };

}

#[macro_export]
macro_rules! crdt_map_store {
    ($($k:literal => $v:tt),*) => {
        {
            use $crate::{DotStore, CausalDotStore};
            let ormap = $crate::crdt_map_literal!($($k => $v),*);
            let dots = ormap.dots();
            CausalDotStore {
                store: ormap,
                context: dots
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::enable_determinism;
    use insta::assert_debug_snapshot;

    #[test]
    fn crdt_map_literal_macro() {
        enable_determinism();
        let map = crdt_map_literal! {
            "field_x" => ("Hello", dot!(1,2)),
            "field_y" => ("World", dot!(1,3)),
            "field_z" => {
                "field_x" => ("Nested", dot!(1,4)),
                "field_y" => ("Nested", dot!(1,5))
            }
        };
        assert_debug_snapshot!(map);
    }
    #[test]
    fn crdt_map_store_macro() {
        enable_determinism();
        let map = crdt_map_store! {
            "field_x" => ("Hello", dot!(1,2)),
            "field_y" => ("World", dot!(1,3)),
            "field_z" => {
                "field_x" => ("Nested", dot!(1,4)),
                "field_y" => ("Nested", dot!(1,5))
            }
        };
        assert_debug_snapshot!(map);
    }
    #[test]
    fn crdt_map_literal_macro_array() {
        enable_determinism();
        let map = crdt_map_literal! {
            "field_x" => ("Hello", dot!(1)),
            "field_y" => ("World", dot!(2)),
            "field_z" => [
                (("Banana", dot!(3)), dot!(4), dot!(5), dot!(6), 42.0),
                (("Cantaloupe", dot!(7)), dot!(8), dot!(9), dot!(10), 43.0)
            ]
        };
        assert_debug_snapshot!(map);
    }
}
