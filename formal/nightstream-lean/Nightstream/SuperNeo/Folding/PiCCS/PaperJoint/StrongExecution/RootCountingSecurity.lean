import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.MixingSoundness
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SumCheckSoundness

/-!
Concrete finite root-counting security for the causal paper `Pi_CCS`
experiment.

Assurance tier: model-level.

Owns: construction of both named algebraic probability contracts from the
explicit finite verifier alphabet, and their insertion into the corrected
Appendix-D.4 success-gated loss order.

Does not own: first-success runtime, Fiat--Shamir, Poseidon2, the production
challenge carrier, Rust, R1CS, artifacts, or costs.

Emits constraints: no.

| Input boundary | Ownership | Output |
|---|---|---|
| alpha/gamma event | `MixingSoundness` root count | concrete mixing contract |
| SumCheck event | `SumCheckSoundness` root count | concrete SumCheck contract |
| success-gated loss | existing operational theorem | root-envelope bound |
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
          (MixingSoundness.paperMixingNumerator shape)
          alphabet.cardinality +
        ratio (shape.cubeVariables * context.sumcheckWidth)
          alphabet.cardinality) := by
  exact fixedFirstBadBound_of_securityContracts context alphabet adversary
    (ratio
      (MixingSoundness.paperMixingNumerator shape)
      alphabet.cardinality)
    (ratio (shape.cubeVariables * context.sumcheckWidth)
      alphabet.cardinality)
    (MixingSoundness.mixingRootProbabilityContract_of_rootCounting
      context noZeroDivisors alphabet adversary)
    (SumCheckSoundness.sumCheckSoundnessContract_of_rootCounting
      context exact noZeroDivisors alphabet challengeSetSize_eq adversary)

/-- Legacy floor-based extraction with both algebraic contracts constructed
from finite root counting. This is retained only as a comparison lemma; it is
not the corrected paper-facing extractor. -/
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
        (MixingSoundness.paperMixingNumerator shape)
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
        (MixingSoundness.paperMixingNumerator shape)
        alphabet.cardinality +
      ratio (shape.cubeVariables * context.sumcheckWidth)
        alphabet.cardinality)
    floorPos floorBound rawMismatchBound
    (fixedFirstBadBound_of_rootCounting context exact noZeroDivisors
      alphabet challengeSetSize_eq adversary)

/-- Corrected Appendix D.4 success-gated extraction with both algebraic
contracts constructed from finite root counting. The raw two-run disagreement
budget is charged through a nonnegative root envelope. No pointwise success
floor is required. -/
theorem extraction_after_success_gate_of_rootCounting
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
    (rawMismatchBudget rootMismatchBudget : Rat)
    (rootNonnegative : 0 <= rootMismatchBudget)
    (rawBudget_le_rootSquare :
      rawMismatchBudget <= rootMismatchBudget * rootMismatchBudget)
    (rawMismatchBound :
      (experiment context alphabet adversary).iidPair.probabilityBool
          (witnessDisagreement context) <= rawMismatchBudget)
    (nonempty :
      (experiment context alphabet adversary).support.values.filter
        (fun seed => success context
          ((experiment context alphabet adversary).outcome seed)) ≠ []) :
    let mixingBudget :=
      ratio
        (MixingSoundness.paperMixingNumerator shape)
        alphabet.cardinality
    let sumCheckBudget :=
      ratio (shape.cubeVariables * context.sumcheckWidth)
        alphabet.cardinality
    let base := experiment context alphabet adversary
    base.probabilityBool (success context) -
          ((mixingBudget + sumCheckBudget) + rootMismatchBudget) <=
      (base.firstConditionedFreshSecond
        (success context) nonempty).probabilityBool
          (successGatedSourceExtracted context) := by
  exact extraction_after_success_gate_of_securityContracts
    context alphabet adversary rawMismatchBudget rootMismatchBudget
    (ratio
      (MixingSoundness.paperMixingNumerator shape)
      alphabet.cardinality)
    (ratio (shape.cubeVariables * context.sumcheckWidth)
      alphabet.cardinality)
    rootNonnegative rawBudget_le_rootSquare rawMismatchBound
    (MixingSoundness.mixingRootProbabilityContract_of_rootCounting
      context noZeroDivisors alphabet adversary)
    (SumCheckSoundness.sumCheckSoundnessContract_of_rootCounting
      context exact noZeroDivisors alphabet challengeSetSize_eq adversary)
    nonempty

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.RootCountingSecurity
