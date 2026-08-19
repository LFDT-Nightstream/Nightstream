# SumCheck

## Purpose

Specify the paper-level sum-check protocol used throughout Shout and Twist.

## Target Formulas

- Equation (16): `H = Σ_{b in {0,1}^ℓ} g(b)`.
- Round consistency equations and the final random-point reduction described in Section 3.

## Paper Anchors

- `docs/twist-and-shout-paper/3_technical_preliminaries.md`
- `docs/twist-and-shout-ai-summary.md`

## Module Mapping

- Spec: `specs/SumCheck.spec.md`
- Interface: `TwistShout/SumCheck.lean`
- Implementation: `TwistShout/SumCheck.lean`

## Contract Surface

- Definitions for paper-level sum-check instances, transcripts, and verifier checks.
- Theorems that express round consistency and the final reduction to a random evaluation.
- Honest-prover and verifier-facing surfaces that later protocol modules can consume.

## Boundary Assumptions

- The ambient scalar carrier is a field.
- The summed polynomial satisfies the degree bounds supplied to the protocol.

## Dependency and Consumer Map

- Depends on: `EqPoly`, `MLE`, and finite-sum polynomial reasoning.
- Consumed by: `ShoutCore`, `ShoutOneHot`, `TwistCore`, `TwistValueEval`, and the fast-prover modules.

## Out of Scope

- Protocol-specific batching or commitment schemes.
- SuperNeo proof-system wrappers.
