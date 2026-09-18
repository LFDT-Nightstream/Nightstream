# ShoutOneHot

## Purpose

Specify the one-hot checking layer that makes Shout sound against malformed address columns.

## Target Formulas

- Booleanity checks for each indicator entry.
- Hamming-weight-one checks for each cycle.
- The address-value oracle identity used to recover the integer address from one-hot columns.

## Paper Anchors

- `docs/twist-and-shout-paper/4_the_shout_piop.md`
- `docs/twist-and-shout-ai-summary.md`

## Module Mapping

- Spec: `specs/ShoutOneHot.spec.md`
- Interface: `TwistShout/ShoutOneHot.lean`
- Implementation: `TwistShout/ShoutOneHot.lean`

## Contract Surface

- Definitions of the Booleanity and Hamming-weight-one relations for address columns.
- Sum-check statements used by the one-hot checking PIOP.
- Theorems that valid one-hot columns support the Shout read-checking theorem boundary.

## Boundary Assumptions

- Address columns use the `d`-dimensional one-hot decomposition fixed by `OneHotEncoding`.
- The ambient field and sum-check layer satisfy the prerequisites of the paper's soundness argument.

## Dependency and Consumer Map

- Depends on: `OneHotEncoding`, `EqPoly`, `MLE`, `SumCheck`.
- Consumed by: `ShoutCore`, `FastShoutSmallMemory`, `FastShoutStructuredMemory`, `ShoutLinearVariant`.

## Out of Scope

- The read-only memory relation itself.
- Read-write memory constraints.
