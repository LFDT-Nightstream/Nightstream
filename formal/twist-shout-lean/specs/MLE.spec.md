# MLE

## Purpose

Specify multilinear extensions and the folding identities that let the paper reduce Boolean-cube sums to random-point evaluations.

## Target Formulas

- The multilinear-extension formula `f~(x) = Σ_b f(b) * eq~(x, b)`.
- The multilinearity identity `g(c, x') = (1 - c) g(0, x') + c g(1, x')`.

## Paper Anchors

- `docs/twist-and-shout-paper/3_technical_preliminaries.md`
- `docs/twist-and-shout-ai-summary.md`

## Module Mapping

- Spec: `specs/MLE.spec.md`
- Interface: `TwistShout/MLE.lean`
- Implementation: `TwistShout/MLE.lean`

## Contract Surface

- Definitions of Boolean-cube tables and their multilinear extensions.
- Theorems for folding, interpolation on the hypercube, and random-point evaluation.
- Bridge lemmas needed by sum-check and memory-checking identities.

## Boundary Assumptions

- The ambient scalar carrier is a field.
- The Boolean cube is represented by finite bit-tuples.

## Dependency and Consumer Map

- Depends on: `EqPoly` and field-parametric finite-sum machinery.
- Consumed by: `SumCheck`, `ShoutCore`, `ShoutOneHot`, `TwistCore`, `TwistValueEval`.

## Out of Scope

- Commitment schemes or transcript machinery.
- Protocol-specific soundness accounting.
