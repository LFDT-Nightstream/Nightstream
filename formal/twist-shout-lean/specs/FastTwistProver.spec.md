# FastTwistProver

## Purpose

Specify the Section 8 prover specialization for Twist.

## Target Formulas

- The prover-side identities and work reductions in Section 8.
- Equivalence between the optimized prover path and the Twist theorem surface it instantiates.

## Paper Anchors

- `docs/twist-and-shout-paper/8_fast_twist_prover_implementation.md`
- `docs/twist-and-shout-paper/B_details_of_the_widetilde_text_val_evaluation_sum_check_prover.md`

## Module Mapping

- Spec: `specs/FastTwistProver.spec.md`
- Interface: `TwistShout/FastTwistProver.lean`
- Implementation: `TwistShout/FastTwistProver.lean`

## Contract Surface

- Definitions for the optimized prover-side state used by Twist.
- Theorems that the specialized prover realizes the same read/write claim as `TwistCore` and `TwistValueEval`.
- Boundary lemmas for the `Val`-reconstruction prover used inside the protocol.

## Boundary Assumptions

- The ambient field, less-than polynomial, and one-hot encodings satisfy the prerequisites of the core Twist modules.
- The parameter regime matches the hypotheses of Section 8.

## Dependency and Consumer Map

- Depends on: `TwistCore`, `TwistValueEval`, `LessThanPoly`, `MLE`, `SumCheck`.
- Consumed by: later executable prover bridges and application-level protocol instantiations.

## Out of Scope

- Shout-only prover optimizations.
- Integration-specific commitment or transcript machinery.
