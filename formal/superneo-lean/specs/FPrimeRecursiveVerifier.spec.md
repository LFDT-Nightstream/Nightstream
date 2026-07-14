# FPrimeRecursiveVerifier

## Purpose

`FPrimeRecursiveVerifier` is the theorem-facing boundary for minimizing the
R1CS that verifies one recursive augmented-function step. It keeps the target
relation fixed while allowing independently compiled check blocks to be added
or removed.

The module answers two different questions with different certificates:

1. Does the selected set of checks accept exactly the intended `F'` relation?
2. Does each emitted R1CS block implement its named check exactly enough for
   circuit soundness, and can the honest compiler produce a satisfying witness?

A row count is never accepted as evidence for either property.

## Paper Anchors

- HyperNova, Construction 2: recursive augmented function and IVC transition.
- SuperNeo, Section 7: composition through `Pi_CCS`, `Pi_RLC`, and `Pi_DEC`.
- `SuperNeo/FiatShamirReroute.lean`: the proved parent-authority reroute used at
  the post-`Pi_RLC` boundary.

## Fixed Semantic Target

`PaperRecursiveStep predicates step` is the conjunction of these independent
obligations:

1. verifier context validity;
2. canonical encoding;
3. application transition;
4. recursive public link;
5. `Pi_CCS` verification;
6. `Pi_RLC` verification;
7. DEC-child recomposition/validation;
8. parent-authoritative transcript derivation;
9. incoming accumulator binding;
10. outgoing accumulator binding;
11. recursive state transition;
12. public output binding.

The following implementation sidecars are not part of the target relation:

- DEC-children transcript hash;
- duplicate accumulator hash;
- serialization sidecar consistency.

They may appear in a legacy candidate only when `DerivedCheckLaws` proves each
sidecar follows from a valid target step.

## Authority Boundary

`AuthorityCoreAccepts` requires all of the following:

- the parent object is valid;
- children validate against that parent;
- the parent digest is recomputed from the checked parent;
- the challenge is derived from that recomputed parent digest;
- the continuation accepts the derived challenge.

`AuthorityLegacyAccepts` additionally carries a digest of the child payload.
`canonical_legacy_accepts_iff_core` proves that this digest is a removable
serialization sidecar. The theorem does not remove child validation and does
not treat a digest as authority.

At the concrete SuperNeo post-`Pi_RLC` boundary,
`minimalPostRlc_accepts_iff_target` uses the existing `piDEC_of_weak` theorem to
show that the theorem-level DEC statement is derived from the valid RLC parent.
`postRlcCheckedMinimal_iff_target` keeps concrete child recomposition as a
separate mandatory predicate.

## Modular Check Plans

- `Accepts semantics checks input`: every selected check holds.
- `Sound`: selected-check acceptance implies the fixed target.
- `Complete`: every target input passes every selected check.
- `Exact`: selected-check acceptance is equivalent to the target.
- `Redundant`: the other selected checks imply the candidate check.
- `NecessaryForSoundness`: a concrete invalid input passes after one check is
  removed.
- `InclusionMinimalSound`: the plan is sound and every retained check has a
  removal witness.
- `CertifiedPlan`: packages soundness and completeness.

Adding checks preserves soundness. Removing checks preserves completeness.
Removing a check preserves exactness only through `eraseRedundant`, which
requires a proof that the remaining checks imply it.

`booleanEssentialPlan_inclusionMinimalSound` is an executable independence
regression for the generic check vocabulary. A concrete implementation must
provide its own `EssentialNecessityWitnesses` before claiming that its emitted
blocks are inclusion-minimal.

## R1CS Refinement Contract

The R1CS layer defines sparse linear combinations and standard rows
`A(z) * B(z) = C(z)`. A modular encoding supplies one local block and one local
assignment projection per semantic check.

A trusted candidate requires all three certificates:

| Certificate | Guarantee |
|---|---|
| `CertifiedPlan` | selected semantic checks equal the fixed target language |
| `BlockRefinement` | every block is well formed and any satisfying assignment implies its named semantic check |
| `PlanWitnessComplete` | every semantically accepted input has one compiler witness satisfying all selected blocks |

`CertifiedR1csPlan.exact` proves equality of the existential R1CS language and
the target. `CertifiedR1csPlan.eraseRedundant` removes a proved-redundant block,
restricts the existing witness compiler, and preserves exactness.

`R1csBlock.cost` computes rows and sparse nonzeros structurally. `compiledCost`
sums independently concatenated block costs. Cost is reporting data and never
participates in a soundness proof.

## Module Mapping

| Lean file | Ownership |
|---|---|
| `SuperNeo/FPrimeRecursiveVerifier/Plan.lean` | check selection, exactness, redundancy, minimality |
| `SuperNeo/FPrimeRecursiveVerifier/Cost.lean` | per-block and selected-plan costs |
| `SuperNeo/FPrimeRecursiveVerifier/Semantics.lean` | complete `F'` obligation vocabulary and legacy pruning |
| `SuperNeo/FPrimeRecursiveVerifier/Authority.lean` | parent authority and child-sidecar erasure |
| `SuperNeo/FPrimeRecursiveVerifier/SuperNeoBridge.lean` | concrete post-`Pi_RLC`/`Pi_DEC` theorem bridge |
| `SuperNeo/FPrimeRecursiveVerifier/R1csRefinement.lean` | sparse R1CS semantics and lowering certificates |
| `SuperNeo/FPrimeRecursiveVerifier/NecessityModel.lean` | executable independence witnesses |
| `SuperNeo/FPrimeRecursiveVerifier.lean` | certified construction entrypoints |
| `SuperNeo/FPrimeRecursiveVerifierInterface.lean` | curated theorem-facing surface |

## Assumption Ledger

- The module adds no cryptographic, random-oracle, collision-resistance, or
  R1CS-soundness axiom.
- Kernel audit: the generic `r1csExact_of_certifiedPlan` theorem depends only
  on Lean/mathlib's `propext` and `Quot.sound`. The concrete post-`Pi_RLC`
  bridge inherits `Lean.ofReduceBool` and `Lean.trustCompiler` from the existing
  `ProtocolTargetContext`/`piDEC_of_weak` theorem chain; this component does not
  introduce that dependency.
- `FPrimePredicates` is an instantiation boundary. A concrete implementation
  must refine each field to authoritative protocol data; naming a predicate is
  not a proof that Rust enforces it.
- `BlockRefinement` and `PlanWitnessComplete` are proof obligations supplied by
  a concrete lowering. The generic framework does not manufacture them.
- `piDEC_of_weak` is imported from the existing SuperNeo formalization.
- Child payload validation remains mandatory even when the theorem-level DEC
  statement is derived from the RLC parent.
- The framework proves inclusion-minimality relative to a declared check
  vocabulary, not a global lower bound over all possible arithmetic circuits.

## Out of Scope

- Claiming that the current Rust full-history audit circuit satisfies a
  `BlockRefinement`; that requires an explicit Lean-Rust conformance artifact.
- Proving a universal lower bound on R1CS rows.
- Choosing a concrete field encoding, Poseidon2 gadget, or witness layout.
- Replacing R1CS verification with Lean at runtime.

## Acceptance Criteria

1. `lake build SuperNeo.FPrimeRecursiveVerifier` succeeds.
2. `lake build SuperNeo.FPrimeRecursiveVerifierInterface` succeeds.
3. `lake build tests.FPrimeRecursiveVerifierTests` succeeds.
4. `lake build` and `lake exe check` succeed within the repository test cap.
5. No `sorry`, `admit`, new axiom, or vacuous `Prop := True` appears in the
   component.
6. Any removed block is justified by a redundancy theorem and the resulting
   `CertifiedR1csPlan.exact` theorem.
