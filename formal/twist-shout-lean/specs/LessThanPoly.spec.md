# LessThanPoly

## Purpose

Specify the multilinear less-than polynomial used to express Twist's time-prefix sums.

## Target Formulas

- The multilinear extension of the Boolean predicate `LT(j', j)`.
- The prefix-sum identity that rewrites time-indexed accumulation using `LT~`.

## Paper Anchors

- `docs/twist-and-shout-paper/3_technical_preliminaries.md`
- `docs/twist-and-shout-ai-summary.md`

## Module Mapping

- Spec: `specs/LessThanPoly.spec.md`
- Interface: `TwistShout/LessThanPoly.lean`
- Implementation: `TwistShout/LessThanPoly.lean`

## Contract Surface

- A field-parametric definition of the multilinear less-than polynomial.
- Theorems connecting Boolean less-than on indices to multilinear evaluation.
- Prefix-sum lemmas used by Twist's virtual-memory reconstruction.

## Boundary Assumptions

- The ambient scalar carrier is a field.
- Time indices are represented as finite bit-tuples.

## Dependency and Consumer Map

- Depends on: field-parametric algebra, Boolean-cube indexing, and `EqPoly`-style multilinear reasoning.
- Consumed by: `TwistCore`, `TwistValueEval`, `FastTwistProver`.

## Out of Scope

- Address one-hot constraints.
- Shout's read-only memory arguments.
