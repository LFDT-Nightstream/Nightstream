import NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingLaw
import NightstreamFPrime.Spec.Folding.Nifs.SequentialObservationLaw
import Mathlib.Analysis.Normed.Group.Tannery

/-!
The actual adaptive MSIS probability on the selected NIFS observation law.
Both acceptance checks remain in the two-call numerator. Each original
context retains its own retry rate, including contexts with zero success.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingProbability

open scoped BigOperators ENNReal
attribute [local instance] Classical.propDecidable
open NightstreamFPrime.Spec NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork PiRLC.CoordinateForkLaw

private theorem event_ne_top {Sample : Type*} (distribution : PMF Sample)
    (event : Set Sample) : distribution.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact distribution.tsum_coe_indicator_ne_top event

private theorem bind_event_toReal {Sample Output : Type*}
    (distribution : PMF Sample) (next : Sample → PMF Output) (event : Set Output) :
    ((distribution.bind next).toOuterMeasure event).toReal =
      ∑' sample, (distribution sample).toReal * ((next sample).toOuterMeasure event).toReal := by
  rw [PMF.toOuterMeasure_bind_apply,
    ENNReal.tsum_toReal_eq (fun sample => ENNReal.mul_ne_top
      (distribution.apply_ne_top sample) (event_ne_top (next sample) event))]
  simp only [ENNReal.toReal_mul]

private theorem event_toReal {Sample : Type*} (distribution : PMF Sample)
    (event : Set Sample) :
    (distribution.toOuterMeasure event).toReal =
      ∑' sample, (distribution sample).toReal * (if sample ∈ event then 1 else 0) := by
  have finite (sample : Sample) : event.indicator distribution sample ≠ ∞ := by
    by_cases member : sample ∈ event
    · simpa only [Set.indicator_of_mem member] using distribution.apply_ne_top sample
    · rw [Set.indicator_of_notMem member]
      exact ENNReal.zero_ne_top
  rw [PMF.toOuterMeasure_apply, ENNReal.tsum_toReal_eq finite]
  apply tsum_congr
  intro sample
  by_cases member : sample ∈ event
  · simp only [Set.indicator_of_mem member, if_pos member, mul_one]
  · simp only [Set.indicator_of_notMem member, if_neg member, ENNReal.toReal_zero, mul_zero]

variable {State : Type*} {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  [DecidableEq RingF]
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (sourceProgram : CheckedWitnessExtraction.Program productionShape (FullShape logicalWidth publicFits))

private theorem runLaw_one
    (values : PMF (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (coordinate : Fin PaperProfile.arity.total) :
    AdaptiveBindingLaw.runLaw relation ajtai program sourceProgram values 1 coordinate =
      values.bind fun left => values.map fun right =>
        if (AdaptiveBinding.check relation ajtai program sourceProgram left).value = true ∧
            (AdaptiveBinding.check relation ajtai program sourceProgram right).value = true then
          (BindingReduction.runPair program sourceProgram.access ⟨left, 0⟩ ⟨right, 0⟩ coordinate).value
        else none := by
  unfold AdaptiveBindingLaw.runLaw
  change ((values.map fun value => (⟨value, 0⟩ : Result _)).bind fun first =>
    (AcceptedRetryLaw.prefixes (values.map fun value => (⟨value, 0⟩ : Result _)) 1).map
      (fun following => (AdaptiveBindingRun.run relation ajtai program sourceProgram
        first following coordinate).value)) = _
  simp only [PMF.bind_map, AcceptedRetryLaw.prefixes, PMF.map_bind,
    PMF.pure_map, Function.comp_def]
  apply congrArg (PMF.bind values)
  funext left
  apply congrArg (PMF.bind values)
  funext right
  apply congrArg PMF.pure
  cases first : (AdaptiveBinding.check relation ajtai program sourceProgram left).value <;>
    cases second : (AdaptiveBinding.check relation ajtai program sourceProgram right).value <;>
    simp only [AdaptiveBindingRun.run, first, second, AcceptedRetry.search,
      Bool.false_eq_true, ↓reduceIte, false_and, and_false, true_and]
  cases left <;> cases right <;> rfl

/-- Every finite driver prefix has an actual MSIS success probability in
the unit interval, without any correctness or hardness premise. -/
theorem finiteSuccessProbability_range
    (values : PMF (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)) (count : Nat) :
    0 ≤ AdaptiveBindingLaw.successProbability relation ajtai program sourceProgram values count ∧
      AdaptiveBindingLaw.successProbability relation ajtai program sourceProgram values count ≤ 1 := by
  have eventRange (coordinate : Fin PaperProfile.arity.total) :
      0 ≤ ((AdaptiveBindingLaw.runLaw relation ajtai program sourceProgram values count coordinate).toOuterMeasure
        {output | BindingReduction.Succeeds ajtai output}).toReal ∧
      ((AdaptiveBindingLaw.runLaw relation ajtai program sourceProgram values count coordinate).toOuterMeasure
        {output | BindingReduction.Succeeds ajtai output}).toReal ≤ 1 := by
    refine ⟨ENNReal.toReal_nonneg, ?_⟩
    apply (ENNReal.toReal_mono ENNReal.one_ne_top ?_).trans_eq ENNReal.toReal_one
    rw [PMF.toOuterMeasure_apply]
    exact (ENNReal.tsum_le_tsum (fun _ => Set.indicator_apply_le fun _ => le_rfl)).trans_eq
      (AdaptiveBindingLaw.runLaw relation ajtai program sourceProgram values count coordinate).tsum_coe
  have positive : (0 : ℝ) < PaperProfile.arity.total := by change (0 : ℝ) < 17; norm_num
  unfold AdaptiveBindingLaw.successProbability
  refine ⟨div_nonneg (Finset.sum_nonneg fun coordinate _ => (eventRange coordinate).1) positive.le, ?_⟩
  apply (div_le_one positive).mpr
  calc
    _ ≤ ∑ _coordinate : Fin PaperProfile.arity.total, (1 : ℝ) :=
      Finset.sum_le_sum fun coordinate _ => (eventRange coordinate).2
    _ = _ := by
      simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one]
      exact_mod_cast Fintype.card_fin PaperProfile.arity.total

variable {Context Tape : Type*}
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (publicCheck : Context → Probe K productionShape → Bool)
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      WeakExtraction.Continuation Tape relation ajtai (running context) (fresh context) coins output)

/-- The executable check's indicator mean is the actual local retry rate.
This identity needs no correctness premise and also supplies work recurrences. -/
theorem checkedMean_eq_rate (context : Context) :
    (∑' observation,
        (SequentialObservationLaw.law
          (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
          (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context) observation).toReal *
        (if (AdaptiveBinding.check relation ajtai program sourceProgram observation).value = true
          then (1 : ℝ) else 0)) =
      AdaptiveBindingLaw.rate relation ajtai program sourceProgram
        (SequentialObservationLaw.law
          (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
          (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)) := by
  let values := SequentialObservationLaw.law
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
    (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
  let accepted := fun observation : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape =>
    (AdaptiveBinding.check relation ajtai program sourceProgram observation).value = true
  have sums (value : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape → ℝ) :
      Summable (fun observation => (values observation).toReal * value observation) :=
    (SequentialObservationLaw.value_hasSum _ _ value).summable
  have reject :
      ((values.map fun observation => if accepted observation then some observation else none) none).toReal =
        ∑' observation, (values observation).toReal * (if accepted observation then (0 : ℝ) else 1) := by
    rw [← PMF.toOuterMeasure_apply_singleton, PMF.toOuterMeasure_map_apply, event_toReal]
    apply tsum_congr
    intro observation
    by_cases checked : accepted observation <;> simp only [checked, ↓reduceIte,
      Set.mem_preimage, Set.mem_singleton_iff, Option.some_ne_none]
  have partition :
      (∑' observation, (values observation).toReal * (if accepted observation then (1 : ℝ) else 0)) +
      (∑' observation, (values observation).toReal * (if accepted observation then (0 : ℝ) else 1)) = 1 := by
    rw [← (sums _).tsum_add (sums _)]
    simp only [← mul_add]
    have each (observation) :
        (if accepted observation then (1 : ℝ) else 0) +
          (if accepted observation then (0 : ℝ) else 1) = 1 := by split_ifs <;> norm_num
    simp only [each, mul_one]
    rw [← ENNReal.tsum_toReal_eq values.apply_ne_top, values.tsum_coe, ENNReal.toReal_one]
  unfold AdaptiveBindingLaw.rate AcceptedRetryLaw.successRate AcceptedRetryLaw.attemptLaw
  change _ = 1 - (((values.map fun observation => (⟨observation, 0⟩ : Result _)).map
    (fun packet => if accepted packet.value then some packet.value else none)) none).toReal
  rw [PMF.map_comp]
  change _ = 1 - ((values.map fun observation => if accepted observation then some observation else none) none).toReal
  rw [reject]
  linarith only [partition]

/-- The first call uses the exact relaxed-success rate at this context. -/
theorem rate_eq (context : Context)
    (sourceCorrect : CheckedWitnessExtraction.Correct (width := 9) sourceProgram
      (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
      ((ProductionKey.key relation ajtai).statement (running context) (fresh context))) :
    AdaptiveBindingLaw.rate relation ajtai program sourceProgram
      (SequentialObservationLaw.law
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
        (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)) =
      ∑' observation,
        (SequentialObservationLaw.law
          (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
          (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context) observation).toReal *
        (if StrongProbability.RelaxedSuccess (width := 9) (PaperAlgebra.openingMaps ajtai)
          productionGlobalParams ((ProductionKey.key relation ajtai).statement (running context) (fresh context))
          (PaperCompositionAgreement.outputOf (InteractiveComposition.consume relation ajtai program context)
            observation) then (1 : ℝ) else 0) := by
  rw [← checkedMean_eq_rate relation ajtai program sourceProgram running fresh
    originalFirstPhase publicCheck continuation context]
  apply tsum_congr
  intro observation
  have checked : (AdaptiveBinding.check relation ajtai program sourceProgram observation).value = true ↔
      StrongProbability.RelaxedSuccess (width := 9) (PaperAlgebra.openingMaps ajtai)
        productionGlobalParams ((ProductionKey.key relation ajtai).statement (running context) (fresh context))
        (PaperCompositionAgreement.outputOf (InteractiveComposition.consume relation ajtai program context)
          observation) :=
    AdaptiveBinding.check_correct relation ajtai program sourceProgram
      (running context) (fresh context) sourceCorrect observation
  simp only [checked]

omit [DecidableEq RingF] in
private theorem pairMean_sum {SampleState SampleEndpoint : Type*} [Fintype SampleEndpoint]
    {shape : Shape} {width : Nat}
    (firstPhase : InteractivePrefix.Prover SampleState shape width)
    (suffixLaw : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      SampleState → PMF SampleEndpoint)
    (value : Fin PaperProfile.arity.total →
      PaperCompositionAgreement.Observation SampleState SampleEndpoint shape →
      PaperCompositionAgreement.Observation SampleState SampleEndpoint shape → ℝ) :
    PaperCompositionAgreement.pairMean firstPhase suffixLaw
      (fun left right => ∑ coordinate, value coordinate left right) =
      ∑ coordinate, PaperCompositionAgreement.pairMean firstPhase suffixLaw (value coordinate) := by
  have endpointSum (terms : Fin PaperProfile.arity.total →
      PaperCompositionAgreement.Observation SampleState SampleEndpoint shape → ℝ) :
      PaperCompositionAgreement.endpointMean firstPhase suffixLaw (fun observation => ∑ coordinate, terms coordinate observation) =
        fun alpha gamma point => ∑ coordinate,
          PaperCompositionAgreement.endpointMean firstPhase suffixLaw (terms coordinate) alpha gamma point := by
    funext alpha gamma point
    cases returned : InteractivePrefix.run firstPhase alpha gamma point <;>
      simp only [PaperCompositionAgreement.endpointMean, returned]
    simp only [Finset.mul_sum]
    exact Finset.sum_comm
  simp only [PaperCompositionAgreement.pairMean]
  simp_rw [endpointSum, StrongProbability.verifierMean_sum]
  rw [endpointSum, StrongProbability.verifierMean_sum]

/-- One retry attempt uses two fresh actual observations and both executable
acceptance checks before computing the uniform-coordinate integer output. -/
theorem onePairSuccess_eq (context : Context) :
    AdaptiveBindingLaw.successProbability relation ajtai program sourceProgram
      (SequentialObservationLaw.law
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
        (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)) 1 =
      PaperCompositionAgreement.pairMean
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
        (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
        (fun left right =>
          if (AdaptiveBinding.check relation ajtai program sourceProgram left).value = true ∧
              (AdaptiveBinding.check relation ajtai program sourceProgram right).value = true then
            BindingProbability.observationSuccess ajtai program sourceProgram.access relation left right
          else 0) := by
  let firstPhase := InteractiveComposition.firstPhase originalFirstPhase publicCheck context
  let suffixLaw := InteractiveComposition.suffixLaw relation ajtai running fresh continuation context
  let values := SequentialObservationLaw.law firstPhase suffixLaw
  let trial : Fin PaperProfile.arity.total →
      PaperCompositionAgreement.Observation State (InteractiveComposition.Endpoint relation ajtai) productionShape →
      PaperCompositionAgreement.Observation State (InteractiveComposition.Endpoint relation ajtai) productionShape → ℝ :=
    fun coordinate left right =>
      if (AdaptiveBinding.check relation ajtai program sourceProgram left).value = true ∧
          (AdaptiveBinding.check relation ajtai program sourceProgram right).value = true then
        if BindingReduction.Succeeds ajtai
          (BindingReduction.runPair program sourceProgram.access ⟨left, 0⟩ ⟨right, 0⟩ coordinate).value then 1 else 0
      else 0
  have each (coordinate : Fin PaperProfile.arity.total) :
      ((AdaptiveBindingLaw.runLaw relation ajtai program sourceProgram values 1 coordinate).toOuterMeasure
        {output | BindingReduction.Succeeds ajtai output}).toReal =
      PaperCompositionAgreement.pairMean firstPhase suffixLaw (trial coordinate) := by
    rw [runLaw_one, bind_event_toReal, ← SequentialObservationLaw.pairMean_eq]
    apply tsum_congr
    intro left
    apply congrArg (fun value : ℝ => (values left).toReal * value)
    rw [PMF.toOuterMeasure_map_apply, event_toReal]
    apply tsum_congr
    intro right
    apply congrArg (fun value : ℝ => (values right).toReal * value)
    by_cases accepted : (AdaptiveBinding.check relation ajtai program sourceProgram left).value = true ∧
        (AdaptiveBinding.check relation ajtai program sourceProgram right).value = true
    · simp only [trial, if_pos accepted, Set.mem_preimage, Set.mem_setOf_eq]
    · simp [trial, accepted, BindingReduction.Succeeds]
  change (∑ coordinate,
    ((AdaptiveBindingLaw.runLaw relation ajtai program sourceProgram values 1 coordinate).toOuterMeasure
      {output | BindingReduction.Succeeds ajtai output}).toReal) / (PaperProfile.arity.total : ℝ) = _
  simp_rw [each]
  rw [div_eq_mul_inv, ← pairMean_sum, ← PaperCompositionAgreement.pairMean_mul_const]
  apply congrArg (PaperCompositionAgreement.pairMean firstPhase suffixLaw)
  funext left right
  by_cases accepted : (AdaptiveBinding.check relation ajtai program sourceProgram left).value = true ∧
      (AdaptiveBinding.check relation ajtai program sourceProgram right).value = true
  · simp only [trial, if_pos accepted]
    rw [BindingProbability.observationSuccess_eq_runPair ajtai program sourceProgram.access relation
      ⟨left, 0⟩ ⟨right, 0⟩, div_eq_mul_inv]
    rfl
  · simp only [trial, if_neg accepted, Finset.sum_const_zero, zero_mul]

/-- Actual stopped MSIS success at one original context. A zero-rate
context contributes zero, as proved by the finite driver limit. -/
noncomputable def localSuccessProbability (context : Context) : ℝ :=
  let values := SequentialObservationLaw.law
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
    (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
  AdaptiveBindingLaw.successProbability relation ajtai program sourceProgram values 1 /
    AdaptiveBindingLaw.rate relation ajtai program sourceProgram values

theorem localSuccessProbability_range (context : Context) :
    0 ≤ localSuccessProbability relation ajtai program sourceProgram running fresh
      originalFirstPhase publicCheck continuation context ∧
    localSuccessProbability relation ajtai program sourceProgram running fresh
      originalFirstPhase publicCheck continuation context ≤ 1 := by
  let values := SequentialObservationLaw.law
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
    (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
  have limit := AdaptiveBindingLaw.successProbability_tendsto relation ajtai program sourceProgram values
  exact ⟨ge_of_tendsto' limit (fun count =>
    (finiteSuccessProbability_range relation ajtai program sourceProgram values count).1),
    le_of_tendsto' limit (fun count =>
      (finiteSuccessProbability_range relation ajtai program sourceProgram values count).2)⟩

variable (sourcePrograms : Context → CheckedWitnessExtraction.Program productionShape
  (FullShape logicalWidth publicFits))

/-- Average actual stopped success under the original context PMF. No
context is removed or reweighted by its local acceptance rate. -/
noncomputable def successProbability (contexts : PMF Context) : ℝ :=
  ∑' context, (contexts context).toReal * localSuccessProbability relation ajtai program
    (sourcePrograms context) running fresh originalFirstPhase publicCheck continuation context

/-- The original-context average is the limit of the actual finite driver
probabilities. The PMF weights dominate all terms, including zero-rate contexts. -/
theorem successProbability_tendsto (contexts : PMF Context) :
    Filter.Tendsto (fun count : Nat =>
      ∑' context, (contexts context).toReal * AdaptiveBindingLaw.successProbability
        relation ajtai program (sourcePrograms context)
        (SequentialObservationLaw.law
          (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
          (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)) count)
      Filter.atTop (nhds (successProbability relation ajtai program running fresh
        originalFirstPhase publicCheck continuation sourcePrograms contexts)) := by
  apply tendsto_tsum_of_dominated_convergence
    (ENNReal.summable_toReal contexts.tsum_coe_ne_top)
  · intro context
    exact (AdaptiveBindingLaw.successProbability_tendsto relation ajtai program
      (sourcePrograms context) _).const_mul (contexts context).toReal
  · apply Filter.Eventually.of_forall
    intro count context
    have range := finiteSuccessProbability_range relation ajtai program (sourcePrograms context)
      (SequentialObservationLaw.law
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
        (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)) count
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg ENNReal.toReal_nonneg, abs_of_nonneg range.1]
    exact (mul_le_mul_of_nonneg_left range.2 ENNReal.toReal_nonneg).trans_eq (mul_one _)

variable
  (strongSet : StrongSetUnits (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (ProductionKey.key relation ajtai).piRlcAlgebra.challengeValid)
  (correct : Correct (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (PaperExtractionAlgebra.extractionAlgebra ajtai).assignmentModule program)

include strongSet correct in
/-- Successful disagreement in two actual calls gives a computed MSIS
success after both acceptance gates. Division uses this same context's rate. -/
theorem local_retryDisagreement_le_success (context : Context)
    (accessCorrect : CostedWitnessProjection.Correct sourceProgram.access)
    (sourceCorrect : CheckedWitnessExtraction.Correct (width := 9) sourceProgram
      (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
      ((ProductionKey.key relation ajtai).statement (running context) (fresh context))) :
    StrongProbability.retryDisagreementProbability
      (InteractiveDistribution.tapes
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
        (none : InteractiveComposition.Endpoint relation ajtai)
        (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context))
      (InteractiveDistribution.coupled
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
        (InteractiveComposition.consume relation ajtai program context))
      (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement (running context) (fresh context)) ≤
      localSuccessProbability relation ajtai program sourceProgram running fresh
        originalFirstPhase publicCheck continuation context * PaperProfile.arity.total := by
  let firstPhase := InteractiveComposition.firstPhase originalFirstPhase publicCheck context
  let suffixLaw := InteractiveComposition.suffixLaw relation ajtai running fresh continuation context
  let values := SequentialObservationLaw.law firstPhase suffixLaw
  let oracle := fun coins output state => (continuation context coins output state).chargedOracle
  let checker := fun coins output state => (continuation context coins output state).parentChecker
  have checkSpec : ∀ (probe : Probe K productionShape) (state : State) response,
      (checker probe.coins probe.response.fullOutput state response).accepted = true ↔
        response.Success (ProductionKey.key relation ajtai).piRlcSemantics
          (ProductionKey.key relation ajtai).params (ProductionKey.key relation ajtai).piRlcAlgebra
          (PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai)
            (running context) (fresh context) probe) := by
    intro probe state response
    rw [← WeakExtraction.batchForOutput_eq_probe relation ajtai (running context) (fresh context) probe]
    exact (continuation context probe.coins probe.response.fullOutput state).parentChecker_spec response
  have numerator :
      PaperCompositionAgreement.pairMean firstPhase suffixLaw (fun left right =>
        if StrongProbability.SuccessfulDisagreement (width := 9) (PaperAlgebra.openingMaps ajtai)
          productionGlobalParams ((ProductionKey.key relation ajtai).statement (running context) (fresh context))
          (PaperCompositionAgreement.outputOf (InteractiveComposition.consume relation ajtai program context) left)
          (PaperCompositionAgreement.outputOf (InteractiveComposition.consume relation ajtai program context) right)
          then (1 : ℝ) else 0) ≤
        AdaptiveBindingLaw.successProbability relation ajtai program sourceProgram values 1 * PaperProfile.arity.total := by
    rw [onePairSuccess_eq relation ajtai program sourceProgram running fresh
      originalFirstPhase publicCheck continuation context, ← PaperCompositionAgreement.pairMean_mul_const]
    apply PaperCompositionAgreement.pairMean_mono
    intro left right leftSupported rightSupported
    by_cases disagreement : StrongProbability.SuccessfulDisagreement (width := 9)
        (PaperAlgebra.openingMaps ajtai) productionGlobalParams
        ((ProductionKey.key relation ajtai).statement (running context) (fresh context))
        (PaperCompositionAgreement.outputOf (InteractiveComposition.consume relation ajtai program context) left)
        (PaperCompositionAgreement.outputOf (InteractiveComposition.consume relation ajtai program context) right)
    · have firstAccepted := (AdaptiveBinding.check_correct relation ajtai program sourceProgram
        (running context) (fresh context) sourceCorrect left).mpr disagreement.1
      have secondAccepted := (AdaptiveBinding.check_correct relation ajtai program sourceProgram
        (running context) (fresh context) sourceCorrect right).mpr disagreement.2.1
      have binding : InteractiveAgreement.BindingEvent relation ajtai running fresh program context left right :=
        PaperCompositionAgreement.successful_disagreement_implies_bindingEvent
          (ProductionKey.key relation ajtai) (running context) (fresh context) oracle checker program
          (Phi81Relation.PiRLCAlgebra.Binding.relaxedOps
            (shape := FullShape logicalWidth publicFits) (rows := productionProfile.commitmentWidth))
          (PaperExtractionAlgebra.extractionAlgebra ajtai) strongSet correct
          (BindingBridge.compatible relation ajtai) checkSpec
          (PaperAlgebra.openingMaps ajtai) productionGlobalParams
          ((ProductionKey.key relation ajtai).statement (running context) (fresh context))
          left right leftSupported rightSupported disagreement
      rw [if_pos disagreement, if_pos ⟨firstAccepted, secondAccepted⟩]
      simpa only [if_pos binding] using BindingProbability.supported_binding_le_success ajtai program
        sourceProgram.access relation running fresh continuation strongSet correct accessCorrect
        context left right leftSupported rightSupported
    · rw [if_neg disagreement]
      apply mul_nonneg _ (Nat.cast_nonneg _)
      split_ifs
      · exact (BindingProbability.observationSuccess_range ajtai program sourceProgram.access relation left right).1
      · exact le_rfl
  rw [SequentialObservationLaw.retryDisagreement_eq, SequentialObservationLaw.pairMean_eq]
  have rateNonnegative : 0 ≤ AdaptiveBindingLaw.rate relation ajtai program sourceProgram values := by
    unfold AdaptiveBindingLaw.rate
    exact (AcceptedRetryLaw.successRate_range _ _).1
  have divided := div_le_div_of_nonneg_right numerator rateNonnegative
  convert divided using 1
  · congr 1
    exact (rate_eq relation ajtai program sourceProgram running fresh originalFirstPhase
      publicCheck continuation context sourceCorrect).symm
  · change
      (AdaptiveBindingLaw.successProbability relation ajtai program sourceProgram values 1 /
        AdaptiveBindingLaw.rate relation ajtai program sourceProgram values) * PaperProfile.arity.total =
      (AdaptiveBindingLaw.successProbability relation ajtai program sourceProgram values 1 * PaperProfile.arity.total) /
        AdaptiveBindingLaw.rate relation ajtai program sourceProgram values
    ring

include strongSet correct in
/-- The exact retry term in the selected linear source bound is at most
the arity times actual adaptive MSIS success under the original context law. -/
theorem retryDisagreement_le_success (contexts : PMF Context)
    (accessesCorrect : ∀ context, CostedWitnessProjection.Correct (sourcePrograms context).access)
    (sourcesCorrect : ∀ context, CheckedWitnessExtraction.Correct (width := 9) (sourcePrograms context)
      (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
      ((ProductionKey.key relation ajtai).statement (running context) (fresh context))) :
    PaperCompositionProbability.retryDisagreementProbability contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck)
      (none : InteractiveComposition.Endpoint relation ajtai)
      (InteractiveComposition.suffixLaw relation ajtai running fresh continuation)
      (InteractiveComposition.consume relation ajtai program)
      (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
      (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context)) ≤
      successProbability relation ajtai program running fresh originalFirstPhase publicCheck
        continuation sourcePrograms contexts * PaperProfile.arity.total := by
  let base := fun context (_ : CubePoint K productionShape.cubeVariables) (_ : K)
      (_ : CubePoint K productionShape.cubeVariables) =>
    localSuccessProbability relation ajtai program (sourcePrograms context) running fresh
      originalFirstPhase publicCheck continuation context
  let total := fun context (_ : CubePoint K productionShape.cubeVariables) (_ : K)
      (_ : CubePoint K productionShape.cubeVariables) =>
    StrongProbability.retryDisagreementProbability
      (InteractiveDistribution.tapes
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
        (none : InteractiveComposition.Endpoint relation ajtai)
        (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context))
      (InteractiveDistribution.coupled
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
        (InteractiveComposition.consume relation ajtai program context))
      (PaperAlgebra.openingMaps ajtai) productionGlobalParams
      ((ProductionKey.key relation ajtai).statement (running context) (fresh context))
  have baseSummable := StrongProbability.clockMean_summable_of_bounded (shape := productionShape)
    contexts base 1 (fun context _ _ _ => localSuccessProbability_range relation ajtai program
      (sourcePrograms context) running fresh originalFirstPhase publicCheck continuation context)
  have averaged := StrongProbability.clockMean_le_mul_add_const (shape := productionShape)
    contexts base total PaperProfile.arity.total 0
    (fun context _ _ _ => (StrongProbability.retryDisagreementProbability_range _ _ _ _ _).1)
    baseSummable (by
      intro context _ _ _
      simpa only [base, total, add_zero] using local_retryDisagreement_le_success relation ajtai program
        (sourcePrograms context) running fresh originalFirstPhase publicCheck continuation strongSet correct
        context (accessesCorrect context) (sourcesCorrect context))
  simpa only [StrongProbability.clockMean, StrongProbability.verifierMean_const,
    base, total, add_zero, successProbability, PaperCompositionProbability.retryDisagreementProbability,
    StrongProbability.globalRetryDisagreementProbability] using averaged.2

end NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingProbability
