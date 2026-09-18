# ShoutLinearVariant

## Purpose

Specify the Appendix C Shout variation with linear prover dependence on `d`.

## Target Formulas

- The Appendix C reformulation of the Shout prover work.
- Equivalence between the variant protocol and the same read-only memory claim enforced by the core Shout layer.

## Paper Anchors

- `docs/twist-and-shout-paper/C_a_shout_variation_with_a_linear_prover_dependence_on_d.md`

## Module Mapping

- Spec: `specs/ShoutLinearVariant.spec.md`
- Interface: `TwistShout/ShoutLinearVariant.lean`
- Implementation: `TwistShout/ShoutLinearVariant.lean`

## Contract Surface

- Definitions specific to the linear-in-`d` Shout variant.
- Theorems that the variant realizes the same mathematical read-check relation as `ShoutCore`.
- Boundary lemmas for any auxiliary batching or re-indexing used by the appendix construction.

## Boundary Assumptions

- The same read-only memory and one-hot validity assumptions as `ShoutCore`.
- Any extra appendix-specific parameter constraints stated in Appendix C.

## Dependency and Consumer Map

- Depends on: `ShoutCore`, `ShoutOneHot`, `MLE`, `SumCheck`.
- Consumed by: later comparisons of Shout prover variants and downstream application modules when this appendix path is chosen.

## Out of Scope

- The base Shout soundness theorem itself.
- Twist-specific stateful-memory arguments.
