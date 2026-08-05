# Rust conformance contract

Status: **selected boundary; G3 evidence open**.

Rust conformance means that the pinned Rust verifier refines the normative
verifier for every supported input. A passing fixture, source path, or valid
JSON file is not universal refinement.

## Pinned build identity

Each evidence set must bind the repository revision and dirty state,
`Cargo.lock`, toolchain, target, release profile, feature set, source-tree hash,
producer-binary hash, contract hash, profile hash, command, run ID, trace hash,
and run attestation.

## Verifier boundary

After verifier-derived challenges are fixed, the entry point must be
deterministic and return one result:

```text
Accept(canonical output)
Reject(first normative rule, stable detail code)
```

A panic, debug-only check, ignored unknown field, implicit default, or fallback
to a legacy proof variant is not normative rejection.

## Required trace

The Rust process must emit the ordered event IDs from `src/protocol/events.jsonl`.
Each event must bind its rule IDs, exact Rust symbol, canonical input hash,
canonical output hash, observed branch, and first rejection when applicable.
The trace ends at the first rejection.

The independent semantic checker receives the full canonical input and build
identity. It computes its own expected result. The artifact must not supply a
trusted expected acceptance value.

## Minimum suites

The selected profile needs:

- positive boundary and randomized executions;
- one-rule negative mutations where rejection is meaningful;
- canonical decoder truncation, extension, ordering, count, and alias tests;
- sparse Structure order, duplicate, index, coefficient, zero-padding, M0,
  source-order, and terminal-order mutations;
- PiCCS round degree, recurrence, challenge-order, and terminal mutations;
- norm terminal substitution from a different witness;
- non-constant ring-action PiRLC vectors;
- sampler threshold, retry, ordering, and exhaustion cases;
- PiDEC sign, digit, child count, and recomposition mutations;
- transcript tag, frame, padding, ratchet, and continuation mutations;
- verifier-key seed-lane, Structure-stream, count, and digest mutations;
- native/circuit differential records;
- terminal public-image and unsupported-manifest mutations.

Tests for beta coins or a column carrier are not part of v1. Such fields must
be rejection tests because the selected protocol forbids them.

## Closure rule

Finite vectors show tested agreement. G3 needs a pinned build, a complete
Rust-origin suite, and either a universal refinement proof, verified small
kernels with verified composition, or an explicit implementation-trust term in
the final theorem. G3 remains open until that evidence exists for the selected
contract and profile hashes.
