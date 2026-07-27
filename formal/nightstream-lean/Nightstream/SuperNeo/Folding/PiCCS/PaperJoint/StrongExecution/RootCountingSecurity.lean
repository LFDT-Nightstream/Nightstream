import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness

/-!
Concrete finite root-counting security for the causal paper `Pi_CCS`
experiment.

Assurance tier: model-level.

Owns: construction of both named algebraic probability contracts from the
explicit finite verifier alphabet, and their insertion into the unchanged
Appendix-D.4 loss order.

Does not own: first-success runtime, Fiat--Shamir, Poseidon2, the production
Split-NC challenge carrier, Rust, R1CS, artifacts, or costs.

Emits constraints: no.

| Input boundary | Ownership | Output |
|---|---|---|
| alpha/gamma event | `MixingSoundness` root count | concrete mixing contract |
| SumCheck event | `SumCheckSoundness` root count | concrete SumCheck contract |
| first-success loss | existing operational theorem | unchanged ordered bound |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck.Finite

universe uExtension uCommitment uPublicInput uProverSeed uTargetSeed uProverTape

/-- Both exact fresh-run algebraic events are bounded by root counting. Neither
`MixingRootProbabilityContract` nor `SumCheckSoundnessContract` is a premise. -/
theorem fixedFirstBadBound_of_rootCounting
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (noZeroDivisors :
      FiniteRootCounting.NoZeroDivisors context.extensionOps)
    (alphabet : Support Extension)
    (challengeSetSize_eq :
      context.challengeSetSize = alphabet.cardinality)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape) :
    FixedFirstBadBound context alphabet adversary
      (ratio
          (shape.cubeVariables + (shape.jointCoefficientCount - 1))
          alphabet.cardinality +
        ratio (shape.cubeVariables * context.sumcheckWidth)
          alphabet.cardinality) := by
  exact fixedFirstBadBound_of_securityContracts context alphabet adversary
    (ratio
      (shape.cubeVariables + (shape.jointCoefficientCount - 1))
      alphabet.cardinality)
    (ratio (shape.cubeVariables * context.sumcheckWidth)
      alphabet.cardinality)
    (MixingSoundness.mixingRootProbabilityContract_of_rootCounting
      context noZeroDivisors alphabet adversary)
    (SumCheckSoundness.sumCheckSoundnessContract_of_rootCounting
      context exact noZeroDivisors alphabet challengeSetSize_eq adversary)

/-- Appendix D.4 extraction with both algebraic contracts constructed from
finite root counting. The syntactic order remains
`(mixing + SumCheck) + rawMismatch / successFloor`. -/
theorem extraction_after_first_success_of_rootCounting
    {Extension : Type uExtension}
    [DecidableEq Extension]
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (exact : PaperDegreeWidthExact context)
    (noZeroDivisors :
      FiniteRootCounting.NoZeroDivisors context.extensionOps)
    (alphabet : Support Extension)
    (challengeSetSize_eq :
      context.challengeSetSize = alphabet.cardinality)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape)
    (successFloor rawMismatchBudget : Rat)
    (floorPos : 0 < successFloor)
    (floorBound : successFloor <=
      (experiment context alphabet adversary).probabilityBool
        (success context))
    (rawMismatchBound :
      (experiment context alphabet adversary).iidPair.probabilityBool
          (witnessDisagreement context) <= rawMismatchBudget) :
    let mixingBudget :=
      ratio
        (shape.cubeVariables + (shape.jointCoefficientCount - 1))
        alphabet.cardinality
    let sumCheckBudget :=
      ratio (shape.cubeVariables * context.sumcheckWidth)
        alphabet.cardinality
    let base := experiment context alphabet adversary
    let nonempty :
        base.support.values.filter
          (fun seed => success context (base.outcome seed)) ≠ [] :=
      successfulSupport_nonempty_of_floor context alphabet adversary
        successFloor floorPos floorBound
    base.probabilityBool (success context) -
          ((mixingBudget + sumCheckBudget) +
            rawMismatchBudget / successFloor) <=
      (base.firstConditionedFreshSecond
        (success context) nonempty).probabilityBool
          (sourceExtracted context) := by
  exact extraction_after_first_success context alphabet adversary
    successFloor rawMismatchBudget
    (ratio
        (shape.cubeVariables + (shape.jointCoefficientCount - 1))
        alphabet.cardinality +
      ratio (shape.cubeVariables * context.sumcheckWidth)
        alphabet.cardinality)
    floorPos floorBound rawMismatchBound
    (fixedFirstBadBound_of_rootCounting context exact noZeroDivisors
      alphabet challengeSetSize_eq adversary)

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity
