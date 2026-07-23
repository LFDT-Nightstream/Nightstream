import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.TruncatedRejection

/-!
Finite operational `Pi_CCS` strong game.

Owns: the concrete `StrongGame` induced by the causal finite experiment;
literal two-run output, ambient-success, witness-disagreement, and extraction
probabilities; exact finite perfect completeness and public-coin statements;
and the rejection-adjusted strong inequality under the two named paper
security contracts.

Does not own: an infinite or Las Vegas rejection sampler, almost-sure
termination, an asymptotic expected-polynomial-time theorem, Fiat--Shamir,
Rust, R1CS, artifacts, or costs.

The `extractorExpectedPolynomialTime` slot of `finiteStrongGame` is therefore
filled by the explicitly named `UniformTruncatedWorkBound`: every finite
cutoff has expected first-success-plus-fresh-run cost at most
`1 / successFloor + 1`.  This is the strongest execution-cost theorem owned by
the current finite probability model.  Consequently `finitePaperStrong` is a
literal `RejectionAdjustedStrong` theorem for this finite game, but it is not
by itself the frozen asymptotic `PiCcsStrong` obligation.  Closing that last
interpretation requires a separate infinite-sampler construction and a
theorem connecting it to the conditioned mixture used here.
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

/-- The exact positive-success condition needed by Appendix D.4's
first-success conditioning. -/
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

/-- The one finite-model extractor: ideal first-success conditioning followed
by one independently fresh second execution. -/
inductive Extractor where
  | firstSuccessFreshSecond
deriving Repr, DecidableEq

/-- Exact extraction probability of the ideal finite conditioned mixture.
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

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong
