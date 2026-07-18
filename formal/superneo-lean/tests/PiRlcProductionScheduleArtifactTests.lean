import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.Refinement.ProductionScheduleArtifact

namespace tests.PiRlcProductionScheduleArtifact

open SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifact

/-!
External regression and axiom-surface checks for the generated fixed PiRLC
transcript schedule artifact.

| Check | Mathematical property | Production scope |
|---|---|---|
| schedule | 15 samples by four rounds by four lane decompositions | exact stage order only |
| tree | transcript and challenge immediate children sum componentwise | source plus estimator plus trace |
| digest formula | 15 first rounds plus 45 later rounds explain the dominant cost | exact generated arithmetic |
| nonlinear | 78 permutations own 6,708 S-boxes | diagnostic census only |
| dimensions | source and estimated-low-norm dimensions remain distinct | no materialized low-norm claim |
| axioms | generated theorems introduce no `sorry` or project axiom | Lean kernel report |
-/

example : FixedScheduleOrder
    SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifactData.samples :=
  generated_schedule_order_exact

example :
    (SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifactData.challenge).estimatedLowNormColumns =
      370383 :=
  generated_challenge_dimensions_exact.2.2.2

example :
    SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifactData.digestRounds.estimatedLowNormColumns =
      15 * 7052 + 45 * 3526 :=
  generated_digest_round_cost_formula.2.2.2.1

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifact.generated_schedule_order_exact' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler] -/
#guard_msgs in
#print axioms generated_schedule_order_exact

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifact.generated_tree_reconciles' depends on axioms: [Lean.ofReduceBool,
 Lean.trustCompiler] -/
#guard_msgs in
#print axioms generated_tree_reconciles

/-- info: 'SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifact.generated_nonlinear_census_exact' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler] -/
#guard_msgs in
#print axioms generated_nonlinear_census_exact

end tests.PiRlcProductionScheduleArtifact
