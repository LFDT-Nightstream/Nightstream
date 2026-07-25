import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong

/-!
Exact boundary of the finite PiCCS strong game.

Owns: definitional exposure of the extractor-runtime field used by
`finitePaperStrong`. It records that the field is exactly a uniform family of
finite truncation bounds, not an unbounded sampler termination or expected
runtime statement.

Does not own: a probability law on infinite tapes, a stopping time,
almost-sure termination, asymptotic complexity, or a replacement strong game.

Emits constraints: no.

| Declaration | Owns | Excluded boundary |
|---|---|---|
| `extractorRuntime_iff_uniformTruncatedWorkBound` | exact unfolding of the finite game runtime field | no unbounded sampler |
| `extractorRuntime_iff_all_finite_cutoffs` | the literal quantifier over finite Cartesian cutoffs | no stopping time, limit law, or asymptotic family |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrongBoundary

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uExtension uCommitment uPublicInput uProverSeed uTargetSeed uProverTape

/-- The finite game's runtime field contains exactly the named finite-cutoff
contract and nothing stronger. -/
theorem extractorRuntime_iff_uniformTruncatedWorkBound
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (alphabet : Support Extension)
    (adversaryExpectedPolynomialTime :
      Adversary context ProverSeed TargetSeed ProverTape -> Prop)
    (successFloor : Rat)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape) :
    (finiteStrongGame context alphabet adversaryExpectedPolynomialTime
        successFloor).extractorExpectedPolynomialTime adversary
          .firstSuccessFreshSecond <->
      UniformTruncatedWorkBound context alphabet adversary successFloor := by
  simp [finiteStrongGame]

/-- Exposing the definition further shows the quantifier ranges over one
finite Cartesian tape experiment for each cutoff. -/
theorem extractorRuntime_iff_all_finite_cutoffs
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (alphabet : Support Extension)
    (adversaryExpectedPolynomialTime :
      Adversary context ProverSeed TargetSeed ProverTape -> Prop)
    (successFloor : Rat)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape) :
    (finiteStrongGame context alphabet adversaryExpectedPolynomialTime
        successFloor).extractorExpectedPolynomialTime adversary
          .firstSuccessFreshSecond <->
      forall attemptLimit,
        ((OperationalExperiment.experiment context alphabet adversary
            ).truncatedFirstSuccess
              (OperationalExperiment.success context) attemptLimit
          ).expectedCost
            ((OperationalExperiment.experiment context alphabet adversary
              ).truncatedQueryCost
                (OperationalExperiment.success context) attemptLimit) + 1 <=
          1 / successFloor + 1 := by
  simp [finiteStrongGame, UniformTruncatedWorkBound]

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrongBoundary
