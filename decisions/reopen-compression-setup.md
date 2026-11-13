# Reopen a compression setup

## Status

Accepted.

## Problem

A prover must reuse completed compression setup files after its process exits.
The files must not determine the relation or supply a trusted key.

## SuperNeo

SuperNeo v1.2 fixes the folding relation and its Ajtai commitments.
It does not select the file lifecycle of Nightstream's compression setup.

## Decision

`Prover::open_compression_setup(dir)` derives both layer-1 keys from the current
prover's relations. It opens the saved light and refresh setups against those
keys. It keeps the norm bound and security allocation from `compression_keys`.
The Nebula prover forwards the same operation.

[SPEC.md Algorithms 7 and 8](../SPEC.md#algorithm-7-finish) keep the trusted
relations as authority. The implementation is in
[compression.rs](../crates/nightstream/src/lifecycle/bounded_depth/compression.rs)
and [Setup::open](../crates/neo-spartan/src/lib.rs).
