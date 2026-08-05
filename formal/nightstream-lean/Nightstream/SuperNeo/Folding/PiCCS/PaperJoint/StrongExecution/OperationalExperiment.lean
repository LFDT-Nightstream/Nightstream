import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.HonestCompleteness
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.VerifierCoins
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessExtraction

/-!
Exact finite experiment for the causal paper Pi_CCS strong reduction.

Owns: independent finite prover, target, and verifier-coin supports; execution
of the causal protocol before target attachment; successful-support
nonemptiness derived from a positive success floor; and the finite
Appendix-D.4 extraction inequality on the literal operational events.

Does not own: an infinite rejection sampler, expected-polynomial-time
semantics, Schwartz--Zippel or SumCheck probability bounds, Fiat--Shamir,
Rust, R1CS, artifacts, or costs.

Emits constraints: no.

The target receives only a completed PrefixExecution. The run support is the
explicit product prover × (target × verifier coins), so a fresh second run
re-samples all three components independently.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uExtension uCommitment uPublicInput uProverSeed uTargetSeed uProverTape

/-- A probabilistic causal prover and a separate post-prefix target algorithm.
The two finite random tapes have distinct supports and explicit dataflow. -/
structure Adversary
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (ProverSeed : Type uProverSeed)
    (TargetSeed : Type uTargetSeed)
    (ProverTape : Type uProverTape) where
  proverSupport : Support ProverSeed
  targetSupport : Support TargetSeed
  strategy : Strategy Extension shape ProverTape
  proverTape : ProverSeed -> ProverTape
  target : TargetSeed -> PrefixExecution Extension shape ->
    Option (OutputWitness shape columns)

/-- One exact base-experiment seed. -/
abbrev RunSeed
    (Extension : Type uExtension)
    (shape : Shape)
    (ProverSeed : Type uProverSeed)
    (TargetSeed : Type uTargetSeed) :=
  ProverSeed ×
    (TargetSeed × VerifierCoins.Seed Extension shape.cubeVariables)

/-- Exact factorized support for one execution. -/
def runSupport
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (alphabet : Support Extension)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape) :
    Support (RunSeed Extension shape ProverSeed TargetSeed) :=
  adversary.proverSupport.product
    (adversary.targetSupport.product
      (VerifierCoins.support alphabet shape.cubeVariables))

/-- Membership exposes the exact provenance of every random component. -/
theorem mem_runSupport_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (alphabet : Support Extension)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape)
    (seed : RunSeed Extension shape ProverSeed TargetSeed) :
    seed ∈ (runSupport context alphabet adversary).values ↔
      seed.1 ∈ adversary.proverSupport.values ∧
      seed.2.1 ∈ adversary.targetSupport.values ∧
      seed.2.2 ∈
        (VerifierCoins.support alphabet shape.cubeVariables).values := by
  unfold runSupport
  rw [Support.mem_product_iff, Support.mem_product_iff]

/-- Execute the causal prefix first, then invoke the target on that prefix. -/
def run
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape)
    (seed : RunSeed Extension shape ProverSeed TargetSeed) :
    Execution Extension shape columns :=
  let causalRun := execute adversary.strategy
    (adversary.proverTape seed.1) (VerifierCoins.toPublicCoins seed.2.2)
  attachWitness causalRun (adversary.target seed.2.1 causalRun)

@[simp] theorem run_causalRun
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape)
    (seed : RunSeed Extension shape ProverSeed TargetSeed) :
    (run context adversary seed).causalRun =
      execute adversary.strategy (adversary.proverTape seed.1)
        (VerifierCoins.toPublicCoins seed.2.2) := by
  rfl

@[simp] theorem run_target
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape)
    (seed : RunSeed Extension shape ProverSeed TargetSeed) :
    (run context adversary seed).target =
      adversary.target seed.2.1
        (execute adversary.strategy (adversary.proverTape seed.1)
          (VerifierCoins.toPublicCoins seed.2.2)) := by
  rfl

/-- The exact finite-uniform base experiment. -/
def experiment
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    {ProverSeed : Type uProverSeed}
    {TargetSeed : Type uTargetSeed}
    {ProverTape : Type uProverTape}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (alphabet : Support Extension)
    (adversary : Adversary context ProverSeed TargetSeed ProverTape) :
    Experiment (Execution Extension shape columns) where
  Seed := RunSeed Extension shape ProverSeed TargetSeed
  support := runSupport context alphabet adversary
  outcome := run context adversary

/-- Literal ambient success for the base experiment. -/
def success
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount) :
    Execution Extension shape columns -> Bool :=
  ambientCheck context

/-- A positive lower bound on actual success implies a nonempty successful
seed filter. -/
theorem successfulSupport_nonempty_of_floor
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
    (adversary : Adversary context ProverSeed TargetSeed ProverTape)
    (successFloor : Rat)
    (floorPos : 0 < successFloor)
    (floorBound : successFloor <=
      (experiment context alphabet adversary).probabilityBool
        (success context)) :
    (experiment context alphabet adversary).support.values.filter
      (fun seed =>
        success context
          ((experiment context alphabet adversary).outcome seed)) ≠ [] := by
  intro filterEmpty
  have countZero :
      (experiment context alphabet adversary).countBool
          (success context) = 0 := by
    unfold Experiment.countBool
    rw [List.countP_eq_length_filter, filterEmpty]
    rfl
  have probabilityZero :
      (experiment context alphabet adversary).probabilityBool
          (success context) = 0 := by
    unfold Experiment.probabilityBool
    rw [countZero]
    simp [Rat.div_def]
  rw [probabilityZero] at floorBound
  exact (Rat.not_lt.mpr floorBound) floorPos

/-- For every successful first seed, the fresh second execution hits the
literal fixed-first bad event with at most the supplied budget. Later generic
root and SumCheck theorems discharge this actual-experiment contract. -/
def FixedFirstBadBound
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
    (adversary : Adversary context ProverSeed TargetSeed ProverTape)
    (budget : Rat) : Prop :=
  let base := experiment context alphabet adversary
  forall firstSeed,
    firstSeed ∈ base.support.values.filter
      (fun seed => success context (base.outcome seed)) ->
    base.probabilityBool (fun second =>
      fixedFirstBad context (base.outcome firstSeed, second)) <= budget

/-- Exact finite Appendix-D.4 extraction bound for the operational experiment.
No acceptance, witness equality, bad-event truth, or extraction conclusion is
a premise. -/
theorem extraction_after_first_success
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
    (adversary : Adversary context ProverSeed TargetSeed ProverTape)
    (successFloor rawMismatchBudget badBudget : Rat)
    (floorPos : 0 < successFloor)
    (floorBound : successFloor <=
      (experiment context alphabet adversary).probabilityBool
        (success context))
    (rawMismatchBound :
      (experiment context alphabet adversary).iidPair.probabilityBool
          (witnessDisagreement context) <= rawMismatchBudget)
    (fixedBadBound :
      FixedFirstBadBound context alphabet adversary badBudget) :
    let base := experiment context alphabet adversary
    let nonempty :
        base.support.values.filter
          (fun seed => success context (base.outcome seed)) ≠ [] :=
      successfulSupport_nonempty_of_floor context alphabet adversary
        successFloor floorPos floorBound
    base.probabilityBool (success context) -
          (badBudget + rawMismatchBudget / successFloor) <=
      (base.firstConditionedFreshSecond
        (success context) nonempty).probabilityBool
          (sourceExtracted context) := by
  let base := experiment context alphabet adversary
  let nonempty :
      base.support.values.filter
        (fun seed => success context (base.outcome seed)) ≠ [] :=
    successfulSupport_nonempty_of_floor context alphabet adversary
      successFloor floorPos floorBound
  exact Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.extract_after_first_success
      base (success context) nonempty
      (witnessDisagreement context) (fixedFirstBad context)
      (sourceExtracted context)
      rawMismatchBudget badBudget successFloor floorPos floorBound
      (witnessDisagreement_implies_first_success context)
      rawMismatchBound fixedBadBound
      (extraction_or_fixedFirstBad context)

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment
