import NightstreamFPrime.Lifecycle.Nifs.BindingReduction
import NightstreamFPrime.Lifecycle.Nifs.InteractiveAgreement
import NightstreamFPrime.Spec.Folding.Nifs.ContextPreparation

/-!
The actual binding reduction samples one of the selected 17 source
coordinates. Its short-kernel success probability bounds the binding event
in the existing NIFS pair experiment. The fixed-key hardness bound remains
the approved public-seed MSIS premise.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.BindingProbability

open scoped BigOperators
attribute [local instance] Classical.propDecidable
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.CoordinateForkLaw PiRLC.PaperForkExtractionWork

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (access : CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))

/-- Uniform finite choice of the trial coordinate in the implemented reduction. -/
noncomputable def coordinateSuccess
    (left right : BindingReduction.Endpoint (logicalWidth := logicalWidth) (publicFits := publicFits)) : ℝ :=
  (∑ coordinate, if BindingReduction.Succeeds ajtai
    (BindingReduction.run program access left right coordinate).value then (1 : ℝ) else 0) /
      (PaperProfile.arity.total : ℝ)

theorem coordinateSuccess_range
    (left right : BindingReduction.Endpoint (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    0 ≤ coordinateSuccess ajtai program access left right ∧
      coordinateSuccess ajtai program access left right ≤ 1 := by
  have positive : (0 : ℝ) < PaperProfile.arity.total := by change (0 : ℝ) < 17; norm_num
  unfold coordinateSuccess
  constructor
  · exact div_nonneg (Finset.sum_nonneg (fun _ _ => by split_ifs <;> norm_num)) positive.le
  · apply (div_le_one positive).mpr
    calc
      _ ≤ ∑ _coordinate : Fin PaperProfile.arity.total, (1 : ℝ) := by
        apply Finset.sum_le_sum
        intro coordinate _
        split_ifs <;> norm_num
      _ = _ := by
        simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one]
        exact_mod_cast Fintype.card_fin PaperProfile.arity.total

theorem success_implies_coordinate_bound
    (left right : BindingReduction.Endpoint (logicalWidth := logicalWidth) (publicFits := publicFits))
    (success : ∃ coordinate, BindingReduction.Succeeds ajtai
      (BindingReduction.run program access left right coordinate).value) :
    1 ≤ coordinateSuccess ajtai program access left right * PaperProfile.arity.total := by
  obtain ⟨coordinate, success⟩ := success
  have nonzero : (PaperProfile.arity.total : ℝ) ≠ 0 := by change (17 : ℝ) ≠ 0; norm_num
  rw [coordinateSuccess, div_mul_cancel₀ _ nonzero]
  have selected := Finset.single_le_sum
    (f := fun trial => if BindingReduction.Succeeds ajtai
      (BindingReduction.run program access left right trial).value then (1 : ℝ) else 0)
    (fun _ _ => by dsimp only; split_ifs <;> norm_num) (Finset.mem_univ coordinate)
  simpa only [if_pos success] using selected

variable {Context State Tape : Type*}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (publicCheck : Context → Probe K productionShape → Bool)
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      WeakExtraction.Continuation Tape relation ajtai (running context) (fresh context) coins output)

/-- An aborted prefix emits no short-kernel output. Present observations keep
the exact endpoint data used by the existing binding event. -/
noncomputable def observationSuccess :
    PaperCompositionAgreement.Observation State (InteractiveComposition.Endpoint relation ajtai) productionShape →
    PaperCompositionAgreement.Observation State (InteractiveComposition.Endpoint relation ajtai) productionShape → ℝ
  | some (_, left), some (_, right) => coordinateSuccess ajtai program access left right
  | _, _ => 0

omit [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)] in
/-- Erasing the clocks of the two actual calls gives exactly the success
observable already used by the binding-event experiment. -/
theorem observationSuccess_eq_runPair
    (left right : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)) :
    observationSuccess ajtai program access relation left.value right.value =
      (∑ coordinate, if BindingReduction.Succeeds ajtai
        (BindingReduction.runPair program access left right coordinate).value then (1 : ℝ) else 0) /
          (PaperProfile.arity.total : ℝ) := by
  rcases left with ⟨left, leftWork⟩
  rcases right with ⟨right, rightWork⟩
  cases left <;> cases right <;>
    simp [observationSuccess, coordinateSuccess, BindingReduction.runPair,
      BindingReduction.Succeeds]
  rfl

omit [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)] in
theorem observationSuccess_range
    (left right : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape) :
    0 ≤ observationSuccess ajtai program access relation left right ∧
      observationSuccess ajtai program access relation left right ≤ 1 := by
  cases left <;> cases right <;> simp only [observationSuccess]
  all_goals first | exact ⟨le_rfl, zero_le_one⟩ | exact coordinateSuccess_range ajtai program access _ _

/-- The same two independent prefix/suffix executions, followed by one
independent uniform source-coordinate choice. -/
noncomputable def localSuccessProbability (context : Context) : ℝ :=
  PaperCompositionAgreement.pairMean
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
    (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
    (observationSuccess ajtai program access relation)

omit [DecidableEq RingF] in
theorem localSuccessProbability_range (context : Context) :
    0 ≤ localSuccessProbability ajtai program access relation running fresh
        originalFirstPhase publicCheck continuation context ∧
      localSuccessProbability ajtai program access relation running fresh
        originalFirstPhase publicCheck continuation context ≤ 1 := by
  apply PaperCompositionAgreement.pairMean_range
  exact observationSuccess_range ajtai program access relation

variable
  (strongSet : StrongSetUnits (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (ProductionKey.key relation ajtai).piRlcAlgebra.challengeValid)
  (correct : Correct (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (PaperExtractionAlgebra.extractionAlgebra ajtai).assignmentModule program)
  (accessCorrect : CostedWitnessProjection.Correct access)

include strongSet correct accessCorrect in
/-- The loss is the selected arity, not the size of the witness space. Every
term uses the actual endpoints and the same charged parent-opening checker. -/
theorem local_binding_le_success (context : Context) :
    InteractiveAgreement.localBindingProbability relation ajtai running fresh
        originalFirstPhase publicCheck continuation program context ≤
      localSuccessProbability ajtai program access relation running fresh
        originalFirstPhase publicCheck continuation context * PaperProfile.arity.total := by
  unfold InteractiveAgreement.localBindingProbability localSuccessProbability
  rw [← PaperCompositionAgreement.pairMean_mul_const]
  apply PaperCompositionAgreement.pairMean_mono
  intro left right leftSupported rightSupported
  dsimp only
  by_cases event : InteractiveAgreement.BindingEvent relation ajtai running fresh program context left right
  · rw [if_pos event]
    cases left with
    | none => cases event
    | some left =>
      rcases left with ⟨leftReceipt, leftEndpoint⟩
      cases right with
      | none => cases event
      | some right =>
        rcases right with ⟨rightReceipt, rightEndpoint⟩
        let leftContinuation := continuation context leftReceipt.1.coins
          leftReceipt.1.response.fullOutput leftReceipt.2
        let rightContinuation := continuation context rightReceipt.1.coins
          rightReceipt.1.response.fullOutput rightReceipt.2
        apply success_implies_coordinate_bound
        apply BindingReduction.bindingEvent_implies_success relation ajtai (running context) (fresh context)
          program correct strongSet access accessCorrect leftReceipt.1 rightReceipt.1
          leftContinuation.chargedOracle rightContinuation.chargedOracle
          leftContinuation.parentChecker rightContinuation.parentChecker
          ?_ ?_ leftEndpoint rightEndpoint leftSupported rightSupported event
        · intro response
          rw [← WeakExtraction.batchForOutput_eq_probe relation ajtai (running context) (fresh context) leftReceipt.1]
          exact leftContinuation.parentChecker_spec response
        · intro response
          rw [← WeakExtraction.batchForOutput_eq_probe relation ajtai (running context) (fresh context) rightReceipt.1]
          exact rightContinuation.parentChecker_spec response
  · rw [if_neg event]
    exact mul_nonneg (observationSuccess_range ajtai program access relation left right).1 (Nat.cast_nonneg _)

/-- Average the actual reduction success over the original context law.
The accessor is the one supplied by that context's checked source program. -/
noncomputable def successProbability
    (accesses : Context → CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (contexts : PMF Context) : ℝ :=
  ∑' context, (contexts context).toReal *
    localSuccessProbability ajtai program (accesses context) relation running fresh
      originalFirstPhase publicCheck continuation context

omit [DecidableEq RingF] in
/-- The actual preparation output selects the same binding experiment.
The probability is taken over the original private tapes, not over an
independently supplied context distribution. -/
theorem prepared_successProbability_eq {SetupTape : Type*}
    (setupTapes : PMF SetupTape) (prepare : SetupTape → Result Context)
    (accesses : Context → CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits)) :
    successProbability ajtai program relation running fresh originalFirstPhase publicCheck
        continuation accesses (ContextPreparation.contexts setupTapes prepare) =
      ∑' tape, (setupTapes tape).toReal * localSuccessProbability ajtai program
        (accesses (prepare tape).value) relation running fresh originalFirstPhase publicCheck
        continuation (prepare tape).value := by
  let value := fun context => localSuccessProbability ajtai program (accesses context) relation
    running fresh originalFirstPhase publicCheck continuation context
  have summable := StrongProbability.clockMean_summable_of_bounded (shape := productionShape)
    setupTapes (fun tape _ _ _ => value (prepare tape).value) 1
    (fun tape _ _ _ => localSuccessProbability_range ajtai program (accesses (prepare tape).value)
      relation running fresh originalFirstPhase publicCheck continuation (prepare tape).value)
  have actual := ContextPreparation.value_hasSum setupTapes prepare value
    (by simpa only [StrongProbability.verifierMean_const] using summable)
  exact actual.tsum_eq

include strongSet correct in
/-- Global binding failure is bounded by the implemented same-key MSIS
reduction's success probability, with the selected source-count loss. -/
theorem binding_le_success
    (accesses : Context → CostedWitnessProjection.Accessor productionShape (FullShape logicalWidth publicFits))
    (accessesCorrect : ∀ context, CostedWitnessProjection.Correct (accesses context))
    (contexts : PMF Context) :
    InteractiveAgreement.bindingProbability relation ajtai running fresh originalFirstPhase
        publicCheck continuation program contexts ≤
      successProbability ajtai program relation running fresh originalFirstPhase publicCheck
        continuation accesses contexts * PaperProfile.arity.total := by
  let base := fun context (_ : CubePoint K productionShape.cubeVariables) (_ : K)
      (_ : CubePoint K productionShape.cubeVariables) =>
    localSuccessProbability ajtai program (accesses context) relation running fresh
      originalFirstPhase publicCheck continuation context
  let total := fun context (_ : CubePoint K productionShape.cubeVariables) (_ : K)
      (_ : CubePoint K productionShape.cubeVariables) =>
    InteractiveAgreement.localBindingProbability relation ajtai running fresh
      originalFirstPhase publicCheck continuation program context
  have baseSummable := StrongProbability.clockMean_summable_of_bounded (shape := productionShape)
    contexts base 1 (by
      intro context _ _ _
      exact localSuccessProbability_range ajtai program (accesses context) relation running fresh
        originalFirstPhase publicCheck continuation context)
  have averaged := StrongProbability.clockMean_le_mul_add_const (shape := productionShape)
    contexts base total PaperProfile.arity.total 0
    (fun context _ _ _ => (InteractiveAgreement.localBindingProbability_range relation ajtai
      running fresh originalFirstPhase publicCheck continuation program context).1)
    baseSummable (by
      intro context _ _ _
      simpa only [base, total, add_zero] using local_binding_le_success ajtai program
        (accesses context) relation running fresh originalFirstPhase publicCheck continuation
        strongSet correct (accessesCorrect context) context)
  simpa only [StrongProbability.clockMean, base, total,
    StrongProbability.verifierMean_const, add_zero, successProbability,
    InteractiveAgreement.bindingProbability] using averaged.2

end NightstreamFPrime.Lifecycle.Nifs.BindingProbability
