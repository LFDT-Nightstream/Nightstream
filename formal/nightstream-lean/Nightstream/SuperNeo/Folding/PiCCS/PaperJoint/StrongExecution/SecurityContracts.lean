import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessExtraction

/-!
Named probability contracts for the causal paper `Pi_CCS` experiment.

Owns: literal Boolean events for the fixed-witness joint mixing root and the
finite SumCheck bad challenge; exact contracts bounding those events in the
actual fresh-run experiment; and their finite union-bound composition into
`FixedFirstBadBound`.

Does not own: a Schwartz--Zippel root-count proof, the SumCheck soundness
contract itself, an infinite rejection sampler, Fiat--Shamir, Rust, R1CS,
artifacts, or costs.

Emits constraints: no.

| Owned object | Exact event or bound |
|---|---|
| mixing event | fixed-witness alpha/gamma root event |
| SumCheck event | `sumCheckBadChallengeEvent = true <-> SumCheckFailure` |
| event composition | `FixedFirstBadBound` by the finite union bound |
| success-gated loss | `(mixing + sumcheck) + rootMismatch` |

The contracts quantify over every fixed output witness. They therefore do
not allow either bad event to choose its witness after seeing the fresh
second-run verifier coins.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uExtension uCommitment uPublicInput uProverSeed uTargetSeed uProverTape

private noncomputable def propositionEvent (proposition : Prop) : Bool :=
  @ite Bool proposition (Classical.propDecidable proposition) true false

@[simp] private theorem propositionEvent_eq_true
    (proposition : Prop) :
    propositionEvent proposition = true <-> proposition := by
  simp [propositionEvent]

/-- The exact fixed-witness joint alpha/gamma root event on one fresh causal
execution. -/
noncomputable def mixingRootEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (witness : OutputWitness shape columns) :
    Execution Extension shape columns -> Bool :=
  fun execution =>
    propositionEvent (MixingFailure context execution.causalRun witness)

/-- The exact fixed-witness finite SumCheck collision event on one fresh
causal execution. -/
noncomputable def sumCheckBadChallengeEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (witness : OutputWitness shape columns) :
    Execution Extension shape columns -> Bool :=
  fun execution =>
    propositionEvent (SumCheckFailure context execution.causalRun witness)

/-- Public exact transport for the named Boolean alpha/gamma mixing event. -/
@[simp] theorem mixingRootEvent_eq_true_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (witness : OutputWitness shape columns)
    (execution : Execution Extension shape columns) :
    mixingRootEvent context witness execution = true <->
      MixingFailure context execution.causalRun witness := by
  exact propositionEvent_eq_true _

/-- Public exact transport for the named Boolean SumCheck event. -/
@[simp] theorem sumCheckBadChallengeEvent_eq_true_iff
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (witness : OutputWitness shape columns)
    (execution : Execution Extension shape columns) :
    sumCheckBadChallengeEvent context witness execution = true <->
      SumCheckFailure context execution.causalRun witness := by
  exact propositionEvent_eq_true _

/-- Permitted paper security contract for the joint alpha/gamma polynomial:
every witness fixed before the fresh execution has the stated root bound in
the actual adversary experiment. -/
def MixingRootProbabilityContract
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
  forall witness,
    (experiment context alphabet adversary).probabilityBool
        (mixingRootEvent context witness) <= budget

/-- Permitted finite SumCheck soundness contract on the exact causal
messages-before-challenge execution and every independently fixed witness. -/
def SumCheckSoundnessContract
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
  forall witness,
    (experiment context alphabet adversary).probabilityBool
        (sumCheckBadChallengeEvent context witness) <= budget

/-- For a fixed first target, the operational bad event is exactly the union
of the two named fresh-run events. -/
theorem fixedFirstBad_eq_mixing_or_sumCheck
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (first second : Execution Extension shape columns)
    (witness : OutputWitness shape columns)
    (firstTarget : first.target = some witness) :
    fixedFirstBad context (first, second) =
      (mixingRootEvent context witness second ||
        sumCheckBadChallengeEvent context witness second) := by
  apply Bool.eq_iff_iff.mpr
  simp [fixedFirstBad, firstTarget, mixingRootEvent,
    sumCheckBadChallengeEvent]

/-- The two named security contracts discharge the actual fixed-first bad
event with their exact additive loss. No source truth, acceptance, witness
agreement, or extraction conclusion is a premise. -/
theorem fixedFirstBadBound_of_securityContracts
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
    (mixingBudget sumCheckBudget : Rat)
    (mixingBound : MixingRootProbabilityContract context alphabet adversary
      mixingBudget)
    (sumCheckBound : SumCheckSoundnessContract context alphabet adversary
      sumCheckBudget) :
    FixedFirstBadBound context alphabet adversary
      (mixingBudget + sumCheckBudget) := by
  let base := experiment context alphabet adversary
  change forall firstSeed,
    firstSeed ∈ base.support.values.filter
        (fun seed => success context (base.outcome seed)) ->
      base.probabilityBool (fun second =>
        fixedFirstBad context (base.outcome firstSeed, second)) <=
          mixingBudget + sumCheckBudget
  intro firstSeed firstSuccessful
  have firstSuccess :
      ambientCheck context (base.outcome firstSeed) = true := by
    simpa [base, success] using (List.mem_filter.mp firstSuccessful).2
  have firstSemantic : AmbientSuccess context (base.outcome firstSeed) :=
    (ambientCheck_eq_true_iff context (base.outcome firstSeed)).1 firstSuccess
  cases firstTarget : (base.outcome firstSeed).target with
  | none =>
      exact False.elim (by
        simpa [AmbientSuccess, firstTarget] using firstSemantic)
  | some witness =>
      have eventEquality :
          (fun second => fixedFirstBad context
            (base.outcome firstSeed, second)) =
          (fun second => mixingRootEvent context witness second ||
            sumCheckBadChallengeEvent context witness second) := by
        funext second
        exact fixedFirstBad_eq_mixing_or_sumCheck context
          (base.outcome firstSeed) second witness firstTarget
      rw [eventEquality]
      exact Rat.le_trans
        (base.probabilityBool_or_le
          (mixingRootEvent context witness)
          (sumCheckBadChallengeEvent context witness))
        (Rat.le_trans
          ((Rat.add_le_add_right
            (c := base.probabilityBool
              (sumCheckBadChallengeEvent context witness))).mpr
            (mixingBound witness))
          ((Rat.add_le_add_left (c := mixingBudget)).mpr
            (sumCheckBound witness)))

/-- Operational Appendix-D.4 extraction with the fixed-first premise fully
discharged by the two named paper security contracts. The remaining raw
witness-disagreement bound is exactly Definition 10's external same-phi
premise after conditioning by the positive success floor. -/
theorem extraction_after_first_success_of_securityContracts
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
    (successFloor rawMismatchBudget mixingBudget sumCheckBudget : Rat)
    (floorPos : 0 < successFloor)
    (floorBound : successFloor <=
      (experiment context alphabet adversary).probabilityBool
        (success context))
    (rawMismatchBound :
      (experiment context alphabet adversary).iidPair.probabilityBool
          (witnessDisagreement context) <= rawMismatchBudget)
    (mixingBound : MixingRootProbabilityContract context alphabet adversary
      mixingBudget)
    (sumCheckBound : SumCheckSoundnessContract context alphabet adversary
      sumCheckBudget) :
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
    successFloor rawMismatchBudget (mixingBudget + sumCheckBudget)
    floorPos floorBound rawMismatchBound
      (fixedFirstBadBound_of_securityContracts context alphabet adversary
      mixingBudget sumCheckBudget mixingBound sumCheckBound)

/-- Operational Appendix-D.4 extraction for the corrected success-gated
algorithm. The raw two-run disagreement is charged through a nonnegative root
envelope. No success floor is required. -/
theorem extraction_after_success_gate_of_securityContracts
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
    (rawMismatchBudget rootMismatchBudget mixingBudget sumCheckBudget : Rat)
    (rootNonnegative : 0 <= rootMismatchBudget)
    (rawBudget_le_rootSquare :
      rawMismatchBudget <= rootMismatchBudget * rootMismatchBudget)
    (rawMismatchBound :
      (experiment context alphabet adversary).iidPair.probabilityBool
          (witnessDisagreement context) <= rawMismatchBudget)
    (mixingBound : MixingRootProbabilityContract context alphabet adversary
      mixingBudget)
    (sumCheckBound : SumCheckSoundnessContract context alphabet adversary
      sumCheckBudget)
    (nonempty :
      (experiment context alphabet adversary).support.values.filter
        (fun seed => success context
          ((experiment context alphabet adversary).outcome seed)) ≠ []) :
    let base := experiment context alphabet adversary
    base.probabilityBool (success context) -
          ((mixingBudget + sumCheckBudget) + rootMismatchBudget) <=
      (base.firstConditionedFreshSecond
        (success context) nonempty).probabilityBool
          (successGatedSourceExtracted context) := by
  exact extract_after_success_gate
    (experiment context alphabet adversary) (success context) nonempty
    (witnessDisagreement context) (fixedFirstBad context)
    (successGatedSourceExtracted context) rawMismatchBudget
    rootMismatchBudget (mixingBudget + sumCheckBudget) rootNonnegative
    rawBudget_le_rootSquare
    (witnessDisagreement_implies_first_success context)
    (witnessDisagreement_implies_second_success context)
    rawMismatchBound
    (fixedFirstBadBound_of_securityContracts context alphabet adversary
      mixingBudget sumCheckBudget mixingBound sumCheckBound)
    (successGatedExtraction_or_fixedFirstBad context)

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
