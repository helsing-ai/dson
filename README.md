# DSON: A Delta-State CRDT for JSON-like Data Structures

[![Crates.io](https://img.shields.io/crates/v/dson.svg)](https://crates.io/crates/dson)
[![Docs.rs](https://docs.rs/dson/badge.svg)](https://docs.rs/dson)

This crate provides a Rust implementation of **DSON**, a space-efficient, delta-state Conflict-Free Replicated Datatype (CRDT) for JSON-like data structures. It is based on the research paper ["DSON: JSON CRDT Using Delta-Mutations For Document Stores"][dson-paper] and started as a port of the original author's [JavaScript implementation][js-impl].

[dson-paper]: https://dl.acm.org/doi/10.14778/3510397.3510403
[js-impl]: https://github.com/crdt-ibm-research/json-delta-crdt


The primary goal of this library is to enable robust, and efficient
multi-writer collaboration in extremely constrained environments (high
latency and low bandwidth; opportunistic networking).

## Core Concepts

DSON provides three fundamental, composable CRDTs:

- `OrMap`: An **Observed-Remove Map**, mapping string keys to other CRDT values.
- `OrArray`: An **Observed-Remove Array**, providing a list-like structure.
- `MvReg`: A **Multi-Value Register**, for storing primitive values. When
           concurrent writes occur, the register holds all conflicting values.

These primitives can be nested to create arbitrarily complex data structures.
All modifications produce a **delta**, which is a small set of changes that can
be transmitted to other replicas.

## Observed-Remove Semantics

DSON uses **Observed-Remove (OR)** semantics. This means an element can only be
removed if its addition has been observed. If an element is updated concurrently
with its removal, the update "wins" and the element is preserved.

## Causal CRDTs and Tombstone-Free Removals

DSON is a **causal** CRDT, using causal history to resolve conflicts. A key
advantage of this model is the elimination of **tombstones**, which prevents
unbounded metadata growth in long-lived documents.

## Scope of this Crate

This crate provides the core data structures and algorithms for DSON. It is a
low-level library, and you will likely want to build a typed abstraction layer
on top of it.

**It does not include any networking protocols.** You are responsible for
*implementing the transport layer to broadcast deltas to other replicas.

## Attribution

The initial version of this crate was based on the
[JavaScript implementation][js-impl] by the [DSON paper][dson-paper]
authors.

The following people have contributed to this implementation:

- [@jonhoo](https://github.com/jonhoo)
- [@wngr](https://github.com/wngr)
- [@asmello](https://github.com/asmello)
- [@avl](https://github.com/avl)
- [@oktal](https://github.com/oktal)


## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Further Resources

- [Talk on DSON and CRDTs](https://www.youtube.com/watch?v=4QkLD7JhD_I) - A presentation covering CRDTs in general and DSON in particular.

## Development

This repository provides a [Nix](https://nixos.org) development shell.
If you have Nix installed, you can enter the shell by running:

```sh
nix develop
```

This will provide you with a consistent development environment, including the
correct Rust toolchain and other helpful dependencies.

In general, `dson` has very little dependencies, so you should expect to run `cargo build`/`cargo test`
just fine.

## Documentation

For a complete guide, including detailed explanations of the core concepts,
advanced topics, and API usage, please refer to the 
[**generated docs**](https://docs.rs/dson).
