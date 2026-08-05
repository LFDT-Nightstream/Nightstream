import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.TruncatedRejection

/-!
Finite operational `Pi_CCS` strong game.

Owns: the concrete `StrongGame` induced by the causal finite experiment;
literal two-run output, ambient-success, witness-disagreement, and extraction
probabilities; exact finite perfect completeness and public-coin statements;
and the success-gated strong inequality under the two named paper security
contracts.

Does not own: an infinite or Las Vegas rejection sampler, almost-sure
termination, an asymptotic expected-polynomial-time theorem, Fiat--Shamir,
Rust, R1CS, artifacts, or costs.

Emits constraints: no.

The paper-facing game charges one fresh initial execution and enters the retry
loop only after that execution succeeds. Its exact expected execution factor
is at most two, including the zero-success case. The older floor-based objects
remain in this file only as legacy comparison lemmas and are not the headline
paper theorem.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.HonestCompleteness
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uExtension uCommitment uPublicInput uProverSeed uTargetSeed uProverTape

/-- Exact perfect-completeness statement for a single strategy selected before
any verifier coins.  It quantifies over the independently stated source
relation rather than receiving successful execution as a premise. -/
def PerfectComplete
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount) : Prop :=
  forall witness : OutputWitness shape columns,
    SourceHolds context.extensionOps context.lift context.openingMaps
        context.params context.statement witness ->
      exists strategy : Strategy Extension shape PUnit.{1},
        forall coins : PublicCoins Extension shape,
          AmbientSuccess context
            (attachWitness (execute strategy PUnit.unit.{1} coins)
              (some witness))

/-- The corrected ambient bound proves exact finite perfect completeness. -/
theorem perfectComplete
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    [DecidableEq Extension]
    {shape : Shape}
    {columns blockCount : Nat}
    (context : Context Extension Commitment PublicInput shape
      columns blockCount)
    (ambientAdmissible : context.params.b <=
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor
        context.params) :
    PerfectComplete context := by
  intro witness source
  exact exists_uniform_honestStrategy context ambientAdmissible witness source

/-- Public-coin ownership for the causal execution.  The exact verifier coin
record is stored in the probe, and the revealed challenge history is exactly
the verifier-owned round point in order. -/
def PublicCoin
    (Extension : Type uExtension)
    (shape : Shape)
    (ProverTape : Type uProverTape) : Prop :=
  forall
    (strategy : Strategy Extension shape ProverTape)
    (tape : ProverTape)
    (coins : PublicCoins Extension shape),
      (execute strategy tape coins).probe.coins = coins /\
      (execute strategy tape coins).history.challenges =
        coins.roundPoint.coordinates

/-- The finite causal verifier is public coin by construction. -/
theorem publicCoin
    (Extension : Type uExtension)
    (shape : Shape)
    (ProverTape : Type uProverTape) :
    PublicCoin Extension shape ProverTape := by
  intro strategy tape coins
  exact ⟨execute_probe_coins strategy tape coins,
    execute_history_challenges_eq_roundPoint strategy tape coins⟩

/-- The exact positive-success condition used only by the superseded
conditioned-first comparison theorem. -/
def ExtractionEligible
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
    (successFloor : Rat) : Prop :=
  0 < successFloor /\
    successFloor <=
      (experiment context alphabet adversary).probabilityBool (success context)

/-- Every finite rejection cutoff obeys the inverse-floor work bound, with one
additional verifier call charged for the independent fresh second run.  This
is deliberately not named asymptotic expected polynomial time. -/
def UniformTruncatedWorkBound
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
    (successFloor : Rat) : Prop :=
  forall attemptLimit,
    ((experiment context alphabet adversary).truncatedFirstSuccess
        (success context) attemptLimit).expectedCost
          ((experiment context alphabet adversary).truncatedQueryCost
            (success context) attemptLimit) + 1 <=
      1 / successFloor + 1

/-- Positive operational success proves the uniform finite work contract. -/
theorem uniformTruncatedWorkBound_of_eligible
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
    (eligible : ExtractionEligible context alphabet adversary successFloor) :
    UniformTruncatedWorkBound context alphabet adversary successFloor := by
  let base := experiment context alphabet adversary
  have nonempty :
      base.support.values.filter
        (fun seed => success context (base.outcome seed)) ≠ [] :=
    successfulSupport_nonempty_of_floor context alphabet adversary
      successFloor eligible.1 eligible.2
  intro attemptLimit
  exact (Rat.add_le_add_right (c := 1)).mpr
    (base.truncatedFirstSuccess_expectedQueries_le_inverseFloor
      (success context) nonempty attemptLimit successFloor
      eligible.1 eligible.2)

/-- Finite extractor tags. `successGated` is current paper authority;
`firstSuccessFreshSecond` names the legacy conditioned-first comparison. -/
inductive Extractor where
  | firstSuccessFreshSecond
  | successGated
deriving Repr, DecidableEq

/-- Legacy extraction probability of the ideal finite conditioned mixture.
The false branch is unreachable whenever `extractionEligible` is supplied to
`RejectionAdjustedStrong`; keeping it explicit makes the game total. -/
noncomputable def sourceExtractionProbability
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
    (successFloor : Rat) : Rat :=
  @dite Rat (ExtractionEligible context alphabet adversary successFloor)
    (Classical.propDecidable _)
    (fun eligible =>
      let base := experiment context alphabet adversary
      let nonempty :
          base.support.values.filter
            (fun seed => success context (base.outcome seed)) ≠ [] :=
        successfulSupport_nonempty_of_floor context alphabet adversary
          successFloor eligible.1 eligible.2
      (base.firstConditionedFreshSecond
        (success context) nonempty).probabilityBool (sourceExtracted context))
    (fun _ => 0)

theorem sourceExtractionProbability_eq_of_eligible
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
    (eligible : ExtractionEligible context alphabet adversary successFloor) :
    sourceExtractionProbability context alphabet adversary successFloor =
      let base := experiment context alphabet adversary
      let nonempty :
          base.support.values.filter
            (fun seed => success context (base.outcome seed)) ≠ [] :=
        successfulSupport_nonempty_of_floor context alphabet adversary
          successFloor eligible.1 eligible.2
      (base.firstConditionedFreshSecond
        (success context) nonempty).probabilityBool
          (sourceExtracted context) := by
  rw [sourceExtractionProbability, dif_pos eligible]

/-- Exact expected execution factor of the corrected success-gated algorithm:
one fresh initial run, plus a geometric retry entered with the actual success
probability. -/
def SuccessGatedWorkBound
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
    (adversary : Adversary context ProverSeed TargetSeed ProverTape) : Prop :=
  let probability :=
    (experiment context alphabet adversary).probabilityBool (success context)
  1 + probability * (1 / probability) <= 2

/-- Success gating makes the expected execution factor at most two without a
positive pointwise floor. -/
theorem successGatedWorkBound
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
    (adversary : Adversary context ProverSeed TargetSeed ProverTape) :
    SuccessGatedWorkBound context alphabet adversary := by
  let probability :=
    (experiment context alphabet adversary).probabilityBool (success context)
  by_cases probabilityZero : probability = 0
  · unfold SuccessGatedWorkBound
    change 1 + probability * (1 / probability) <= 2
    rw [probabilityZero, Rat.zero_mul, Rat.add_zero]
    decide
  · have cancels : probability * (1 / probability) = 1 := by
      rw [Rat.div_def, Rat.one_mul]
      exact Rat.mul_inv_cancel probability probabilityZero
    unfold SuccessGatedWorkBound
    change 1 + probability * (1 / probability) <= 2
    rw [cancels]
    have twoEq : (1 : Rat) + 1 = 2 :=
      (Rat.natCast_add 1 1).symm
    rw [twoEq]
    exact Rat.le_refl

/-- Total extraction probability for the success-gated finite algorithm.
When no successful retry seed exists, the initial run always fails and the
extractor returns bottom with probability one. -/
noncomputable def successGatedSourceExtractionProbability
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
    (adversary : Adversary context ProverSeed TargetSeed ProverTape) : Rat :=
  let base := experiment context alphabet adversary
  @dite Rat
    (base.support.values.filter
      (fun seed => success context (base.outcome seed)) ≠ [])
    (Classical.propDecidable _)
    (fun nonempty =>
      (base.firstConditionedFreshSecond
        (success context) nonempty).probabilityBool
          (successGatedSourceExtracted context))
    (fun _ => 0)

theorem successGatedSourceExtractionProbability_eq_of_nonempty
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
    (nonempty :
      (experiment context alphabet adversary).support.values.filter
        (fun seed => success context
          ((experiment context alphabet adversary).outcome seed)) ≠ []) :
    successGatedSourceExtractionProbability context alphabet adversary =
      ((experiment context alphabet adversary).firstConditionedFreshSecond
        (success context) nonempty).probabilityBool
          (successGatedSourceExtracted context) := by
  rw [successGatedSourceExtractionProbability, dif_pos nonempty]

private theorem mixture_probability_false
    {Prefix : Type uProverSeed}
    {Outcome : Type uTargetSeed}
    (mixture : Mixture Prefix Outcome) :
    mixture.probability (fun _ => False) = 0 := by
  unfold Mixture.probability
  have mapZero :
      mixture.prefixes.values.map
        (fun outer =>
          (mixture.component outer).probability (fun _ => False)) =
        mixture.prefixes.values.map (fun _ => (0 : Rat)) := by
    apply List.map_congr_left
    intro outer _member
    exact Experiment.probability_false (mixture.component outer)
  rw [mapZero]
  rw [List.map_const']
  have zeroSum :
      (List.replicate mixture.prefixes.values.length (0 : Rat)).sum = 0 := by
    induction mixture.prefixes.values.length with
    | zero => rfl
    | succ length inductionHypothesis =>
        change (0 : Rat) +
          (List.replicate length (0 : Rat)).sum = 0
        rw [inductionHypothesis]
        exact Rat.zero_add 0
  rw [zeroSum]
  simp [Rat.div_def]

/-- The literal repeated-output projection mismatch probability is zero; the
event itself is retained and ruled out pointwise by the verifier's projection
construction. -/
theorem outputPhiMismatchProbability_eq_zero
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
    (adversary : Adversary context ProverSeed TargetSeed ProverTape) :
    (experiment context alphabet adversary).iidPair.probabilityBool
        (outputPhiMismatch context) = 0 := by
  let paired := (experiment context alphabet adversary).iidPair
  calc
    paired.probabilityBool (outputPhiMismatch context) =
        paired.probability
          (fun executions => outputPhiMismatch context executions = true) :=
      (paired.probability_bool_event (outputPhiMismatch context)).symm
    _ = paired.probability (fun _ => False) := by
      congr 1
      funext executions
      apply propext
      simp [outputPhiMismatch_eq_false context executions]
    _ = 0 := mixture_probability_false paired

/-- Actual finite operational `StrongGame`.  The caller supplies the
adversary-side expected-polynomial-time predicate; the extractor-side slot is
the proved, explicitly finite `UniformTruncatedWorkBound`. -/
noncomputable def finiteStrongGame
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
    (successFloor : Rat) :
    StrongGame Rat (Adversary context ProverSeed TargetSeed ProverTape)
      Extractor where
  perfectComplete := PerfectComplete context
  publicCoin := PublicCoin Extension shape ProverTape
  adversaryExpectedPolynomialTime := adversaryExpectedPolynomialTime
  extractorExpectedPolynomialTime := fun adversary extractor =>
    extractor = .firstSuccessFreshSecond /\
      UniformTruncatedWorkBound context alphabet adversary successFloor
  extractionEligible := fun adversary =>
    ExtractionEligible context alphabet adversary successFloor
  repeatedOutputPhiMismatch := fun adversary =>
    (experiment context alphabet adversary).iidPair.probabilityBool
      (outputPhiMismatch context)
  ambientOutputSuccess := fun adversary =>
    (experiment context alphabet adversary).probabilityBool (success context)
  repeatedOutputWitnessDisagreement := fun adversary =>
    (experiment context alphabet adversary).iidPair.probabilityBool
      (witnessDisagreement context)
  sourceWitnessExtracted := fun adversary _ =>
    sourceExtractionProbability context alphabet adversary successFloor

/-- Corrected finite strong game for the paper's success-gated extractor.
Eligibility is total, because the zero-success branch returns bottom after the
initial execution and never enters the retry loop. -/
noncomputable def successGatedFiniteStrongGame
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
      Adversary context ProverSeed TargetSeed ProverTape -> Prop) :
    StrongGame Rat (Adversary context ProverSeed TargetSeed ProverTape)
      Extractor where
  perfectComplete := PerfectComplete context
  publicCoin := PublicCoin Extension shape ProverTape
  adversaryExpectedPolynomialTime := adversaryExpectedPolynomialTime
  extractorExpectedPolynomialTime := fun adversary extractor =>
    extractor = .successGated /\
      SuccessGatedWorkBound context alphabet adversary
  extractionEligible := fun _ => True
  repeatedOutputPhiMismatch := fun adversary =>
    (experiment context alphabet adversary).iidPair.probabilityBool
      (outputPhiMismatch context)
  ambientOutputSuccess := fun adversary =>
    (experiment context alphabet adversary).probabilityBool (success context)
  repeatedOutputWitnessDisagreement := fun adversary =>
    (experiment context alphabet adversary).iidPair.probabilityBool
      (witnessDisagreement context)
  sourceWitnessExtracted := fun adversary _ =>
    successGatedSourceExtractionProbability context alphabet adversary

/-- Security contracts quantified over every operational adversary admitted by
the caller's actual runtime predicate. -/
structure NamedSecurityContracts
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
    (mixingBudget sumCheckBudget : Rat) : Prop where
  mixing : forall adversary,
    adversaryExpectedPolynomialTime adversary ->
      MixingRootProbabilityContract context alphabet adversary mixingBudget
  sumCheck : forall adversary,
    adversaryExpectedPolynomialTime adversary ->
      SumCheckSoundnessContract context alphabet adversary sumCheckBudget

/-- Literal rejection-adjusted strong theorem for the finite operational
`Pi_CCS` game.  The intrinsic error is exactly the sum of the named mixing-root
and SumCheck budgets; the raw witness-disagreement loss is charged exactly
once as `rawMismatchBudget / successFloor`.

No successful run, source witness, mismatch conclusion, or extraction
conclusion is a premise. -/
theorem legacyRejectionAdjustedFinitePaperStrong
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
    (successFloor rawMismatchBudget mixingBudget sumCheckBudget : Rat)
    (ambientAdmissible : context.params.b <=
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor
        context.params)
    (contracts : NamedSecurityContracts context alphabet
      adversaryExpectedPolynomialTime mixingBudget sumCheckBudget) :
    RejectionAdjustedStrong
      Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
      (fun raw floor => raw / floor)
      (finiteStrongGame context alphabet adversaryExpectedPolynomialTime
        successFloor)
      successFloor (mixingBudget + sumCheckBudget) rawMismatchBudget := by
  refine ⟨perfectComplete context ambientAdmissible,
    publicCoin Extension shape ProverTape, ?_, ?_⟩
  · intro adversary _adversaryEpt
    exact outputPhiMismatchProbability_eq_zero context alphabet adversary
  · intro adversary adversaryEpt eligible
    refine ⟨eligible.2, ?_⟩
    intro rawMismatchBound
    refine ⟨.firstSuccessFreshSecond, ?_, ?_⟩
    · exact ⟨rfl,
        uniformTruncatedWorkBound_of_eligible context alphabet adversary
          successFloor eligible⟩
    · change
        (experiment context alphabet adversary).probabilityBool
              (success context) -
            ((mixingBudget + sumCheckBudget) +
              rawMismatchBudget / successFloor) <=
          sourceExtractionProbability context alphabet adversary successFloor
      rw [sourceExtractionProbability_eq_of_eligible context alphabet
        adversary successFloor eligible]
      exact extraction_after_first_success_of_securityContracts
        context alphabet adversary successFloor rawMismatchBudget
        mixingBudget sumCheckBudget eligible.1 eligible.2 rawMismatchBound
        (contracts.mixing adversary adversaryEpt)
        (contracts.sumCheck adversary adversaryEpt)

/-- Exact success-gated strong theorem for the finite operational `Pi_CCS`
game. The raw disagreement budget is charged once through the declared root
envelope. -/
theorem finitePaperStrong
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
    (rawMismatchBudget rootMismatchBudget mixingBudget sumCheckBudget : Rat)
    (rootNonnegative : 0 <= rootMismatchBudget)
    (rawBudget_le_rootSquare :
      rawMismatchBudget <= rootMismatchBudget * rootMismatchBudget)
    (mixingNonnegative : 0 <= mixingBudget)
    (sumCheckNonnegative : 0 <= sumCheckBudget)
    (ambientAdmissible : context.params.b <=
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor
        context.params)
    (contracts : NamedSecurityContracts context alphabet
      adversaryExpectedPolynomialTime mixingBudget sumCheckBudget) :
    SuccessGatedStrong
      Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.scale
      (successGatedFiniteStrongGame context alphabet
        adversaryExpectedPolynomialTime)
      (mixingBudget + sumCheckBudget) rawMismatchBudget
      rootMismatchBudget := by
  refine ⟨perfectComplete context ambientAdmissible,
    publicCoin Extension shape ProverTape, ?_, ?_⟩
  · intro adversary _adversaryEpt
    exact outputPhiMismatchProbability_eq_zero context alphabet adversary
  · intro adversary adversaryEpt _eligible rawMismatchBound
    refine ⟨.successGated, ⟨rfl,
      successGatedWorkBound context alphabet adversary⟩, ?_⟩
    let base := experiment context alphabet adversary
    by_cases nonempty :
        base.support.values.filter
          (fun seed => success context (base.outcome seed)) ≠ []
    · change
        base.probabilityBool (success context) -
              ((mixingBudget + sumCheckBudget) + rootMismatchBudget) <=
          successGatedSourceExtractionProbability context alphabet adversary
      rw [successGatedSourceExtractionProbability_eq_of_nonempty
        context alphabet adversary nonempty]
      exact extraction_after_success_gate_of_securityContracts
        context alphabet adversary rawMismatchBudget rootMismatchBudget
        mixingBudget sumCheckBudget rootNonnegative rawBudget_le_rootSquare
        rawMismatchBound (contracts.mixing adversary adversaryEpt)
        (contracts.sumCheck adversary adversaryEpt) nonempty
    · have filteredEmpty :
          base.support.values.filter
            (fun seed => success context (base.outcome seed)) = [] :=
        Classical.not_not.mp nonempty
      have countZero : base.countBool (success context) = 0 := by
        unfold Experiment.countBool
        rw [List.countP_eq_length_filter, filteredEmpty]
        rfl
      have probabilityZero : base.probabilityBool (success context) = 0 := by
        unfold Experiment.probabilityBool
        rw [countZero]
        simp [Rat.div_def]
      have extractionZero :
          successGatedSourceExtractionProbability context alphabet adversary =
            0 := by
        rw [successGatedSourceExtractionProbability, dif_neg nonempty]
      have totalNonnegative :
          0 <= (mixingBudget + sumCheckBudget) + rootMismatchBudget :=
        Rat.add_nonneg (Rat.add_nonneg mixingNonnegative sumCheckNonnegative)
          rootNonnegative
      change
        base.probabilityBool (success context) -
              ((mixingBudget + sumCheckBudget) + rootMismatchBudget) <=
          successGatedSourceExtractionProbability context alphabet adversary
      rw [probabilityZero, extractionZero, Rat.sub_eq_add_neg, Rat.zero_add]
      simpa using Rat.neg_le_neg totalNonnegative

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong
