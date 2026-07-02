# TwistValueEval

## Purpose

Specify the reconstruction of `Val` from committed increments inside Twist.

## Target Formulas

- The prefix-sum reconstruction of `Val` from `Inc`.
- The sum-check identities that let the verifier validate `Val~(r_address, r_cycle)` without a full `Val` commitment.

## Paper Anchors

- `docs/twist-and-shout-paper/5_the_twist_piop.md`
- `docs/twist-and-shout-paper/B_details_of_the_widetilde_text_val_evaluation_sum_check_prover.md`
- `docs/twist-and-shout-ai-summary.md`

## Module Mapping

- Spec: `specs/TwistValueEval.spec.md`
- Interface: `TwistShout/TwistValueEval.lean`
- Implementation: `TwistShout/TwistValueEval.lean`

## Contract Surface

- Definitions for the virtual-memory reconstruction used by Twist.
- Theorems expressing `Val` as a prefix sum of increments.
- Sum-check theorem surfaces for the verifier-facing evaluation of reconstructed memory values.

## Boundary Assumptions

- The increment relation from `TwistCore` holds.
- The less-than polynomial and multilinear extension primitives satisfy the prerequisites used by the paper.

## Dependency and Consumer Map

- Depends on: `TwistCore`, `LessThanPoly`, `EqPoly`, `MLE`, `SumCheck`.
- Consumed by: `FastTwistProver`, `SpartanPP`, and later integration bridges.

## Out of Scope

- Standalone soundness of the write-checking relation without the Twist core assumptions.
- Read-only memory specializations.
