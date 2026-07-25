import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime

/-!
Unbounded first-success strong reduction for the operational paper PiCCS
experiment.

Owns: security-parameter-indexed PiCCS contexts and adversaries, their exact
finite one-run experiments, the linked unbounded first-success/fresh-second
extractor game, and the rejection-adjusted strong theorem with exact
SumCheck-then-Schwartz--Zippel loss order.

Does not own: PiRLC, PiDEC, their composition couplings, Fiat--Shamir, Rust,
R1CS, or constraints.

Almost-sure termination, expected polynomial time, conditioned-law equality,
and fresh-second independence are not premises of the headline theorem. They
are inherited from the operational trace/runtime theorems constructed from
the one-run experiment and explicit costs.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalEvents
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.OperationalExperiment
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.SecurityContracts
open Nightstream.SuperNeo.InteractiveReduction.Asymptotic
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessTrace
open Nightstream.SuperNeo.InteractiveReduction.Paper

universe uExtension uCommitment uPublicInput
  uProverSeed uTargetSeed uProverTape

/-- One security-parameter instance of the exact causal PiCCS experiment.
Every type and shape is owned by the point, so the family may vary dimensions
with the security parameter. -/
structure Point where
  Extension : Type uExtension
  Commitment : Type uCommitment
  PublicInput : Type uPublicInput
  extensionDecidableEq : DecidableEq Extension
  shape : Shape
  columns : Nat
  blockCount : Nat
  context :
    Context Extension Commitment PublicInput shape columns blockCount
  degreeWidthExact : PaperDegreeWidthExact context
  alphabet : Support Extension
  ProverSeed : Type uProverSeed
  TargetSeed : Type uTargetSeed
  ProverTape : Type uProverTape
  ambientAdmissible :
    context.params.b <=
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBoundFor
        context.params

/-- Every asymptotic paper point charges a SumCheck degree no larger than
Appendix D.4's conservative expression. -/
theorem Point.sumcheckWidth_le_paperRoundDegreeCeiling
    (point : Point) :
    point.context.sumcheckWidth <=
      (point.context.statement.verifierInput point.context.lift).paperRoundDegreeCeiling
        point.context.params.b :=
  paperDegreeWidthExact_implies_width_le_paperRoundDegreeCeiling
    point.context point.degreeWidthExact

/-- One adversary at a point. -/
abbrev Point.Adversary (point : Point) :=
  OperationalExperiment.Adversary point.context
    point.ProverSeed point.TargetSeed point.ProverTape

/-- One exact execution outcome at a point. -/
abbrev Point.Outcome (point : Point) :=
  Execution point.Extension point.shape point.columns

/-- Exact finite one-run experiment at a point. -/
def Point.experiment
    (point : Point)
    (adversary : point.Adversary) :
    Experiment point.Outcome :=
  OperationalExperiment.experiment point.context point.alphabet adversary

/-- Literal ambient-success predicate at a point. -/
def Point.success
    (point : Point) :
    point.Outcome -> Bool :=
  letI := point.extensionDecidableEq
  OperationalExperiment.success point.context

/-- Literal raw Definition-10 witness disagreement event at a point. -/
noncomputable def Point.witnessDisagreement
    (point : Point) :
    point.Outcome × point.Outcome -> Bool :=
  letI := point.extensionDecidableEq
  OperationalEvents.witnessDisagreement point.context

/-- Literal source-extraction event at a point. -/
noncomputable def Point.sourceExtracted
    (point : Point) :
    point.Outcome × point.Outcome -> Bool :=
  OperationalEvents.sourceExtracted point.context

/-- Literal output-projection disagreement event at a point. -/
noncomputable def Point.outputPhiMismatch
    (point : Point) :
    point.Outcome × point.Outcome -> Bool :=
  OperationalEvents.outputPhiMismatch point.context

/-- Fixed-witness joint alpha/gamma root contract at one point. -/
def Point.SchwartzZippelContract
    (point : Point)
    (adversary : point.Adversary)
    (budget : Rat) : Prop :=
  letI := point.extensionDecidableEq
  MixingRootProbabilityContract point.context point.alphabet adversary budget

/-- Fixed-witness SumCheck contract at one point. -/
def Point.SumCheckContract
    (point : Point)
    (adversary : point.Adversary)
    (budget : Rat) : Prop :=
  letI := point.extensionDecidableEq
  SumCheckSoundnessContract point.context point.alphabet adversary budget

/-- Pointwise perfect-completeness proposition with the point-owned equality
instance installed locally. -/
def Point.PerfectComplete (point : Point) : Prop :=
  letI := point.extensionDecidableEq
  FinitePaperStrong.PerfectComplete point.context

/-- One operational adversary for every security parameter. -/
abbrev AdversaryFamily (points : Nat -> Point) :=
  (securityParameter : Nat) -> (points securityParameter).Adversary

/-- Primitive PiCCS family data. The only runtime assumptions are explicit
one-run costs and a positive inverse-polynomial success floor. -/
structure Family where
  point : Nat -> Point
  successFloor : Weight
  successFloor_pos :
    forall securityParameter, 0 < successFloor securityParameter
  inverseFloorPolynomial :
    PolynomiallyBounded
      (fun securityParameter => 1 / successFloor securityParameter)
  runCost :
    (adversary : AdversaryFamily point) ->
    (securityParameter : Nat) ->
    ((point securityParameter).experiment
      (adversary securityParameter)).Seed -> Nat
  runCostBound :
    AdversaryFamily point -> Nat -> Nat
  runCost_le_bound :
    forall adversary securityParameter seed,
      seed ∈ ((point securityParameter).experiment
        (adversary securityParameter)).support.values ->
      runCost adversary securityParameter seed <=
        runCostBound adversary securityParameter

/-- Adversary-family type owned by a PiCCS family. -/
abbrev Family.Adversary (family : Family) :=
  AdversaryFamily family.point

/-- The generic runtime family is definitionally the exact PiCCS one-run
experiment and cost owner at each security parameter. -/
def Family.runtime (family : Family) :
    Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.Family
      family.Adversary where
  Outcome := fun securityParameter =>
    (family.point securityParameter).Outcome
  experiment := fun adversary securityParameter =>
    (family.point securityParameter).experiment
      (adversary securityParameter)
  success := fun securityParameter =>
    (family.point securityParameter).success
  runCost := family.runCost
  runCostBound := family.runCostBound
  runCost_le_bound := family.runCost_le_bound
  successFloor := family.successFloor
  successFloor_pos := family.successFloor_pos
  inverseFloorPolynomial := family.inverseFloorPolynomial

/-- Exact adversary EPT predicate used by the asymptotic strong game. -/
def Family.AdversaryExpectedPolynomialTime
    (family : Family)
    (adversary : family.Adversary) : Prop :=
  Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.AdversaryExpectedPolynomialTime
    family.runtime adversary

/-- Exact positive-success eligibility predicate used by the strong game. -/
def Family.ExtractionEligible
    (family : Family)
    (adversary : family.Adversary) : Prop :=
  Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.ExtractionEligible
    family.runtime adversary

/-- The unbounded paper extractor. -/
inductive Extractor where
  | firstSuccessFreshSecond
deriving Repr, DecidableEq

/-- The unbounded first-success/fresh-second extraction probability. This is
the trace law itself, not a separately supplied conditioned distribution. -/
noncomputable def sourceExtractionProbability
    (family : Family)
    (adversary : family.Adversary) : Weight :=
  fun securityParameter =>
    let point := family.point securityParameter
    jointProbability
      (point.experiment (adversary securityParameter))
      point.success point.sourceExtracted

/-- Operational security-parameter-indexed PiCCS strong game. -/
noncomputable def strongGame
    (family : Family) :
    StrongGame Weight family.Adversary Extractor where
  perfectComplete :=
    forall securityParameter,
      (family.point securityParameter).PerfectComplete
  publicCoin :=
    forall securityParameter,
      PublicCoin
        (family.point securityParameter).Extension
        (family.point securityParameter).shape
        (family.point securityParameter).ProverTape
  adversaryExpectedPolynomialTime :=
    family.AdversaryExpectedPolynomialTime
  extractorExpectedPolynomialTime := fun adversary extractor =>
    extractor = .firstSuccessFreshSecond /\
      Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.ExtractorExpectedPolynomialTime
        family.runtime adversary
  extractionEligible := family.ExtractionEligible
  repeatedOutputPhiMismatch := fun adversary securityParameter =>
    let point := family.point securityParameter
    (point.experiment (adversary securityParameter)).iidPair.probabilityBool
      point.outputPhiMismatch
  ambientOutputSuccess := fun adversary securityParameter =>
    let point := family.point securityParameter
    (point.experiment (adversary securityParameter)).probabilityBool
      point.success
  repeatedOutputWitnessDisagreement := fun adversary securityParameter =>
    let point := family.point securityParameter
    (point.experiment (adversary securityParameter)).iidPair.probabilityBool
      point.witnessDisagreement
  sourceWitnessExtracted := fun adversary _ =>
    sourceExtractionProbability family adversary

/-- The corrected ambient bound derives perfect completeness pointwise. -/
theorem perfectComplete
    (family : Family) :
    (strongGame family).perfectComplete := by
  intro securityParameter
  let point := family.point securityParameter
  letI := point.extensionDecidableEq
  exact FinitePaperStrong.perfectComplete
    point.context point.ambientAdmissible

/-- Public-coin ownership derives pointwise from the causal execution. -/
theorem publicCoin
    (family : Family) :
    (strongGame family).publicCoin := by
  intro securityParameter
  exact FinitePaperStrong.publicCoin
    (family.point securityParameter).Extension
    (family.point securityParameter).shape
    (family.point securityParameter).ProverTape

/-- Named fixed-witness algebraic contracts, quantified over every
polynomial-time adversary family and every security parameter. -/
structure NamedSecurityContracts
    (family : Family)
    (sumCheckBudget schwartzZippelBudget : Weight) : Prop where
  sumCheck :
    forall adversary,
      family.AdversaryExpectedPolynomialTime adversary ->
      forall securityParameter,
        (family.point securityParameter).SumCheckContract
          (adversary securityParameter)
          (sumCheckBudget securityParameter)
  schwartzZippel :
    forall adversary,
      family.AdversaryExpectedPolynomialTime adversary ->
      forall securityParameter,
        (family.point securityParameter).SchwartzZippelContract
          (adversary securityParameter)
          (schwartzZippelBudget securityParameter)

/-- The exact asymptotic PiCCS strong reduction.

The extractor's termination, EPT, conditioned-first law, and fresh-second
independence are all derived. The raw disagreement premise is the literal
Definition-10 two-run event and is divided by the positive floor exactly
once. The intrinsic loss is stated in frozen order:
`SumCheck + Schwartz--Zippel`. -/
theorem paperStrong
    (family : Family)
    (sumCheckBudget schwartzZippelBudget rawMismatchBudget : Weight)
    (contracts :
      NamedSecurityContracts family
        sumCheckBudget schwartzZippelBudget) :
    RejectionAdjustedStrong
      Nightstream.SuperNeo.InteractiveReduction.Asymptotic.scale
      (fun raw floor securityParameter =>
        raw securityParameter / floor securityParameter)
      (strongGame family)
      family.successFloor
      (Nightstream.SuperNeo.InteractiveReduction.Asymptotic.scale.add
        sumCheckBudget schwartzZippelBudget)
      rawMismatchBudget := by
  refine ⟨perfectComplete family, publicCoin family, ?_, ?_⟩
  · intro adversary _adversaryEpt
    funext securityParameter
    let point := family.point securityParameter
    letI := point.extensionDecidableEq
    exact FinitePaperStrong.outputPhiMismatchProbability_eq_zero
      point.context point.alphabet (adversary securityParameter)
  · intro adversary adversaryEpt eligible
    refine ⟨?_, ?_⟩
    · intro securityParameter
      exact eligible securityParameter
    intro rawMismatchBound
    refine ⟨.firstSuccessFreshSecond, ?_, ?_⟩
    · exact ⟨rfl,
        Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.FirstSuccessRuntime.extractorExpectedPolynomialTime
          family.runtime adversary adversaryEpt eligible⟩
    · intro securityParameter
      let point := family.point securityParameter
      letI := point.extensionDecidableEq
      let base := point.experiment (adversary securityParameter)
      have floorPos :
          0 < family.successFloor securityParameter :=
        family.successFloor_pos securityParameter
      have floorBound :
          family.successFloor securityParameter <=
            base.probabilityBool point.success :=
        eligible securityParameter
      have nonempty :
          base.support.values.filter
            (fun seed => point.success (base.outcome seed)) ≠ [] :=
        OperationalExperiment.successfulSupport_nonempty_of_floor
          point.context point.alphabet (adversary securityParameter)
          (family.successFloor securityParameter)
          floorPos floorBound
      have finiteExtraction :=
        extraction_after_first_success_of_securityContracts
          point.context point.alphabet (adversary securityParameter)
          (family.successFloor securityParameter)
          (rawMismatchBudget securityParameter)
          (schwartzZippelBudget securityParameter)
          (sumCheckBudget securityParameter)
          floorPos floorBound
          (rawMismatchBound securityParameter)
          (contracts.schwartzZippel adversary adversaryEpt securityParameter)
          (contracts.sumCheck adversary adversaryEpt securityParameter)
      change
        base.probabilityBool point.success -
            ((sumCheckBudget securityParameter +
                schwartzZippelBudget securityParameter) +
              rawMismatchBudget securityParameter /
                family.successFloor securityParameter) <=
          jointProbability
            base point.success point.sourceExtracted
      rw [jointProbability_eq_firstConditionedFreshSecond
        base point.success nonempty point.sourceExtracted]
      simpa [Rat.add_comm
        (schwartzZippelBudget securityParameter)
        (sumCheckBudget securityParameter)] using finiteExtraction

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.AsymptoticPaperStrong
