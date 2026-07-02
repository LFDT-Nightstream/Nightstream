# ShoutCore

## Purpose

Specify the read-only memory-checking argument at the heart of Shout.

## Target Formulas

- Equation (4): `rv~(r_cycle) = Σ_k ra~(k, r_cycle) * Val~(k)` in the `d = 1` presentation.
- Equation (66): the `d`-dimensional read-checking identity used by the Shout sum-check.

## Paper Anchors

- `docs/twist-and-shout-paper/4_the_shout_piop.md`
- `docs/twist-and-shout-ai-summary.md`

## Module Mapping

- Spec: `specs/ShoutCore.spec.md`
- Interface: `TwistShout/ShoutCore.lean`
- Implementation: `TwistShout/ShoutCore.lean`

## Contract Surface

- Definitions of the read-only memory relation and its multilinear form.
- The Shout read-checking sum-check statement and final-round verifier obligations.
- Theorems connecting the read relation to the paper's random-point identity.

## Boundary Assumptions

- The table `Val` is public, structured, or otherwise available at the verifier boundary exactly as required by the paper.
- Address columns satisfy the encoding prerequisites expected by `OneHotEncoding`.

## Dependency and Consumer Map

- Depends on: `EqPoly`, `MLE`, `SumCheck`, `OneHotEncoding`.
- Consumed by: `ShoutOneHot`, `FastShoutSmallMemory`, `FastShoutStructuredMemory`, `SpeedySpartan`, `SpartanPP`.

## Out of Scope

- Soundness of malformed address columns without the one-hot checking layer.
- Read-write memory semantics.
