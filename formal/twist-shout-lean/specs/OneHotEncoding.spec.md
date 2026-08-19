# OneHotEncoding

## Purpose

Specify plain and `d`-dimensional one-hot encodings for memory addresses.

## Target Formulas

- The one-hot encoding of an address `z` as the indicator vector `e_z`.
- The `d`-dimensional one-hot factorization `e_z(k) = ∏_i v_i(k_i)`.

## Paper Anchors

- `docs/twist-and-shout-paper/3_technical_preliminaries.md`
- `docs/twist-and-shout-ai-summary.md`

## Module Mapping

- Spec: `specs/OneHotEncoding.spec.md`
- Interface: `TwistShout/OneHotEncoding.lean`
- Implementation: `TwistShout/OneHotEncoding.lean`

## Contract Surface

- Definitions of plain and `d`-dimensional one-hot address encodings.
- Booleanity, support, and Hamming-weight-one properties for valid encodings.
- Factorization lemmas consumed by Shout and Twist protocol identities.

## Boundary Assumptions

- The address domain has a fixed finite size.
- The factorized representation uses the same decomposition convention as the paper.

## Dependency and Consumer Map

- Depends on: finite indexing and basic field-parametric algebra.
- Consumed by: `ShoutCore`, `ShoutOneHot`, `TwistCore`, `FastShoutSmallMemory`, `FastShoutStructuredMemory`.

## Out of Scope

- Read-only or read-write memory semantics themselves.
- Commitment-side encoding checks beyond the mathematical one-hot relation.
