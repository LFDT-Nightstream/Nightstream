# TwistCore

## Purpose

Specify the core read-write memory-checking argument of Twist.

## Target Formulas

- Equation (8): the random-point read-check identity for read-write memory.
- Equation (9): the increment relation `Inc(k, j) = wa(k, j) * (wv(j) - Val(k, j))`.

## Paper Anchors

- `docs/twist-and-shout-paper/5_the_twist_piop.md`
- `docs/twist-and-shout-ai-summary.md`

## Module Mapping

- Spec: `specs/TwistCore.spec.md`
- Interface: `TwistShout/TwistCore.lean`
- Implementation: `TwistShout/TwistCore.lean`

## Contract Surface

- Definitions of the read-write memory relation and the increment representation.
- The Twist read-checking and write-checking sum-check statements.
- Theorems connecting increment soundness to the paper's virtual-memory semantics.

## Boundary Assumptions

- Read and write address columns satisfy the one-hot encoding prerequisites used by the paper.
- The initial memory convention matches the paper's stated initialization assumptions.

## Dependency and Consumer Map

- Depends on: `EqPoly`, `MLE`, `SumCheck`, `OneHotEncoding`, `LessThanPoly`.
- Consumed by: `TwistValueEval`, `FastTwistProver`, `SpartanPP`.

## Out of Scope

- Shout's read-only memory protocol.
- Application-level SNARK reductions.
