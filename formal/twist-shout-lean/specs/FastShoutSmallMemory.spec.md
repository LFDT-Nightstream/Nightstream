# FastShoutSmallMemory

## Purpose

Specify the Section 6 prover specialization for Shout on small memories.

## Target Formulas

- The prover-side identities and work reductions from Section 6 for small-memory tables.
- Equivalence between the optimized prover path and the Shout theorem surface it instantiates.

## Paper Anchors

- `docs/twist-and-shout-paper/6_fast_shout_prover_implementation_small_memories.md`

## Module Mapping

- Spec: `specs/FastShoutSmallMemory.spec.md`
- Interface: `TwistShout/FastShoutSmallMemory.lean`
- Implementation: `TwistShout/FastShoutSmallMemory.lean`

## Contract Surface

- Definitions for the small-memory prover data flow.
- Theorem statements showing the specialized prover realizes the same Shout claim as the core protocol.
- Complexity-facing identities needed by downstream applications.

## Boundary Assumptions

- The ambient field and address encoding satisfy the prerequisites of `ShoutCore` and `ShoutOneHot`.
- The small-memory regime matches the parameter conditions stated in Section 6.

## Dependency and Consumer Map

- Depends on: `ShoutCore`, `ShoutOneHot`, `MLE`, `SumCheck`.
- Consumed by: `SpeedySpartan` and any later bridge to executable prover implementations.

## Out of Scope

- Structured-memory specializations from Section 7.
- Twist-specific prover logic.
