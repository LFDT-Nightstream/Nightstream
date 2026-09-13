import NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingRun
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.AcceptedRetryLaw

/-!
Probability semantics of the adaptive binding driver. Values determine all
branches, so recorded call clocks can be erased when deriving its output
law. The executable driver keeps those clocks, and its work proof counts
them separately under the existing NIFS mean-cost model.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingLaw

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork PiRLC.CoordinateForkLaw

private def eraseWork {Value : Type*} (packet : Result Value) : Result Value :=
  ⟨packet.value, 0⟩

private theorem search_value_erase {Value : Type*} (check : Value → Result Bool)
    (packets : List (Result Value)) :
    (AcceptedRetry.search check (packets.map eraseWork)).value =
      (AcceptedRetry.search check packets).value := by
  induction packets with
  | nil => rfl
  | cons packet following induction =>
      cases accepted : (check packet.value).value <;>
        simp only [List.map_cons, eraseWork, AcceptedRetry.search, accepted,
          Bool.false_eq_true, ↓reduceIte, induction]

variable {State : Type*} {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  [DecidableEq RingF]
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (sourceProgram : CheckedWitnessExtraction.Program productionShape (FullShape logicalWidth publicFits))

/-- Erasing recorded call work preserves the actual emitted value on every
finite execution prefix. This is used only for probability semantics; the
driver and its cost bound retain all original work fields. -/
theorem run_value_erase
    (first : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (following : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (coordinate : Fin PaperProfile.arity.total) :
    (AdaptiveBindingRun.run relation ajtai program sourceProgram
      (eraseWork first) (following.map eraseWork) coordinate).value =
      (AdaptiveBindingRun.run relation ajtai program sourceProgram first following coordinate).value := by
  rcases first with ⟨first, firstWork⟩
  dsimp only [AdaptiveBindingRun.run, eraseWork]
  cases accepted : (AdaptiveBinding.check relation ajtai program sourceProgram first).value with
  | false => simp only [Bool.false_eq_true, ↓reduceIte]
  | true =>
      simp only [↓reduceIte]
      rw [search_value_erase]
      cases selected : (AcceptedRetry.search
        (AdaptiveBinding.check relation ajtai program sourceProgram) following).value.1 with
      | none => rfl
      | some second => cases first <;> cases second <;> rfl

private def selected
    (first : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (following : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))) :=
  if (AdaptiveBinding.check relation ajtai program sourceProgram first.value).value then
    (AcceptedRetry.search (AdaptiveBinding.check relation ajtai program sourceProgram) following).value.1.map
      fun second => (first.value, second)
  else none

/-- The actual emitted value is obtained by applying the existing integer
reduction to the driver's selected pair. This view preserves failure and
does not add any extra witness or commitment checks. -/
theorem run_value_eq_selected
    (first : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (following : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (coordinate : Fin PaperProfile.arity.total) :
    (AdaptiveBindingRun.run relation ajtai program sourceProgram first following coordinate).value =
      (match selected relation ajtai program sourceProgram first following with
      | none => none
      | some (left, right) =>
          (BindingReduction.runPair program sourceProgram.access ⟨left, 0⟩ ⟨right, 0⟩ coordinate).value) := by
  rcases first with ⟨first, firstWork⟩
  dsimp only [AdaptiveBindingRun.run, selected]
  cases accepted : (AdaptiveBinding.check relation ajtai program sourceProgram first).value with
  | false => simp only [Bool.false_eq_true, ↓reduceIte]
  | true =>
      simp only [↓reduceIte]
      cases next : (AcceptedRetry.search
        (AdaptiveBinding.check relation ajtai program sourceProgram) following).value.1 with
      | none => rfl
      | some second => cases first <;> cases second <;> rfl

variable (values : PMF (PaperCompositionAgreement.Observation State
  (InteractiveComposition.Endpoint relation ajtai) productionShape))

private noncomputable def calls := values.map fun value => (⟨value, 0⟩ : Result _)

/-- The exact selected-pair law on independent latent prefixes. Work is
erased only for this value calculation, as justified by `run_value_erase`.
The executable driver consumes only the visited part of the prefix. -/
noncomputable def pairLaw (count : Nat) :=
  (calls relation ajtai values).bind fun first =>
    (AcceptedRetryLaw.prefixes (calls relation ajtai values) count).map
      (selected relation ajtai program sourceProgram first)

/-- Only first-call acceptance enters the retry law. Its returned endpoint
is paired with that first observation; all failures remain `none`. -/
theorem pairLaw_eq (count : Nat) :
    pairLaw relation ajtai program sourceProgram values count =
      values.bind fun first =>
        if (AdaptiveBinding.check relation ajtai program sourceProgram first).value then
          (AcceptedRetryLaw.valueLaw (calls relation ajtai values)
            (AdaptiveBinding.check relation ajtai program sourceProgram) count).map
              (fun second => second.map fun last => (first, last))
        else PMF.pure none := by
  simp only [pairLaw, calls, PMF.bind_map, AcceptedRetryLaw.valueLaw,
    AcceptedRetryLaw.runLaw, PMF.map_comp, Function.comp_def]
  apply congrArg (PMF.bind values)
  funext first
  have view : selected relation ajtai program sourceProgram ⟨first, 0⟩ =
      (fun following => if (AdaptiveBinding.check relation ajtai program sourceProgram first).value then
        (AcceptedRetry.search (AdaptiveBinding.check relation ajtai program sourceProgram) following).value.1.map
          (fun last => (first, last)) else none) := rfl
  rw [view]
  cases accepted : (AdaptiveBinding.check relation ajtai program sourceProgram first).value with
  | false =>
      simp only [Bool.false_eq_true, ↓reduceIte]
      exact PMF.map_const _ _
  | true => rfl

private theorem pairLaw_eq_attempt (count : Nat) :
    pairLaw relation ajtai program sourceProgram values count =
      (AcceptedRetryLaw.attemptLaw (calls relation ajtai values)
        (AdaptiveBinding.check relation ajtai program sourceProgram)).bind fun first =>
          match first with
          | none => PMF.pure none
          | some left =>
              (AcceptedRetryLaw.valueLaw (calls relation ajtai values)
                (AdaptiveBinding.check relation ajtai program sourceProgram) count).map
                  (fun second => second.map fun right => (left, right)) := by
  rw [pairLaw_eq]
  simp only [AcceptedRetryLaw.attemptLaw, calls, PMF.bind_map, Function.comp_def]
  congr 1
  funext first
  cases (AdaptiveBinding.check relation ajtai program sourceProgram first).value <;>
    simp only [Bool.false_eq_true, ↓reduceIte]

/-- Every nonabort pair event has the finite geometric first-hit factor
times its actual two-call mass. The event may depend on both observations;
in particular it can be binding failure or computed MSIS success. -/
theorem pairLaw_event
    (event : Set (Option (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape ×
        PaperCompositionAgreement.Observation State
          (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (excludesAbort : none ∉ event) (count : Nat) :
    (pairLaw relation ajtai program sourceProgram values count).toOuterMeasure event =
      (∑ index ∈ Finset.range count,
        (AcceptedRetryLaw.attemptLaw (calls relation ajtai values)
          (AdaptiveBinding.check relation ajtai program sourceProgram) none) ^ index) *
      (pairLaw relation ajtai program sourceProgram values 1).toOuterMeasure event := by
  classical
  rw [pairLaw_eq, pairLaw_eq, PMF.toOuterMeasure_bind_apply,
    PMF.toOuterMeasure_bind_apply, ← ENNReal.tsum_mul_left]
  apply tsum_congr
  intro first
  cases accepted : (AdaptiveBinding.check relation ajtai program sourceProgram first).value with
  | false =>
      simp only [Bool.false_eq_true, ↓reduceIte, PMF.toOuterMeasure_pure_apply,
        if_neg excludesAbort, mul_zero]
  | true =>
      have noAbort : none ∉ (fun second : Option (PaperCompositionAgreement.Observation State
          (InteractiveComposition.Endpoint relation ajtai) productionShape) =>
          second.map fun last => (first, last)) ⁻¹' event := by
        simpa only [Set.mem_preimage, Option.map_none] using excludesAbort
      simp only [↓reduceIte, PMF.toOuterMeasure_map_apply,
        AcceptedRetryLaw.valueLaw_event _ _ _ noAbort, Finset.sum_range_one, pow_zero, one_mul]
      ring

/-- Success rate of the actual checker on one observation at the fixed
original context. Work was erased only in its probability marginal. -/
noncomputable def rate : ℝ :=
  AcceptedRetryLaw.successRate (calls relation ajtai values)
    (AdaptiveBinding.check relation ajtai program sourceProgram)

/-- A zero-success context aborts before entering retries. This case is
retained in the original context distribution. -/
theorem pairLaw_zero (count : Nat)
    (zero : rate relation ajtai program sourceProgram values = 0) :
    pairLaw relation ajtai program sourceProgram values count = PMF.pure none := by
  classical
  let attempted := AcceptedRetryLaw.attemptLaw (calls relation ajtai values)
    (AdaptiveBinding.check relation ajtai program sourceProgram)
  have failure : attempted none = 1 := by
    apply (ENNReal.toReal_eq_one_iff _).mp
    change 1 - (attempted none).toReal = 0 at zero
    linarith only [zero]
  have onlyAbort : attempted = PMF.pure none := by
    have support := (PMF.apply_eq_one_iff attempted none).mp failure
    ext outcome
    cases outcome with
    | none => simpa only [PMF.pure_apply_self] using failure
    | some value =>
        have absent : attempted (some value) = 0 := by
          by_contra positive
          have member := (PMF.mem_support_iff attempted (some value)).mpr positive
          rw [support, Set.mem_singleton_iff] at member
          cases member
        simpa only [PMF.pure_apply, Option.some_ne_none, ↓reduceIte] using absent
  dsimp only [attempted] at onlyAbort
  rw [pairLaw_eq_attempt, onlyAbort, PMF.pure_bind]

/-- At positive local rate, each stopped nonabort pair event has its actual
two-call mass divided by that same rate. No ratio of global means appears. -/
theorem pairLaw_event_tendsto
    (event : Set (Option (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape ×
        PaperCompositionAgreement.Observation State
          (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (excludesAbort : none ∉ event)
    (positive : 0 < rate relation ajtai program sourceProgram values) :
    Filter.Tendsto (fun count : Nat =>
      ((pairLaw relation ajtai program sourceProgram values count).toOuterMeasure event).toReal)
      Filter.atTop (nhds
        (((pairLaw relation ajtai program sourceProgram values 1).toOuterMeasure event).toReal /
          rate relation ajtai program sourceProgram values)) := by
  have firstHits := AcceptedRetry.firstHit_hasSum (rate relation ajtai program sourceProgram values) positive
    (AcceptedRetryLaw.successRate_range (calls relation ajtai values)
      (AdaptiveBinding.check relation ajtai program sourceProgram)).2
    (((pairLaw relation ajtai program sourceProgram values 1).toOuterMeasure event).toReal)
  have actual (count : Nat) :
      ((pairLaw relation ajtai program sourceProgram values count).toOuterMeasure event).toReal =
        ∑ index ∈ Finset.range count,
          (1 - rate relation ajtai program sourceProgram values) ^ index *
            ((pairLaw relation ajtai program sourceProgram values 1).toOuterMeasure event).toReal := by
    rw [pairLaw_event relation ajtai program sourceProgram values event excludesAbort,
      ENNReal.toReal_mul, ENNReal.toReal_sum (fun index _ => ENNReal.pow_ne_top
        ((AcceptedRetryLaw.attemptLaw (calls relation ajtai values)
          (AdaptiveBinding.check relation ajtai program sourceProgram)).apply_ne_top none))]
    simp only [rate, AcceptedRetryLaw.successRate, sub_sub_cancel, ENNReal.toReal_pow, Finset.sum_mul]
  have sequences :
      (fun count : Nat =>
        ((pairLaw relation ajtai program sourceProgram values count).toOuterMeasure event).toReal) =
      (fun count : Nat => ∑ index ∈ Finset.range count,
        (1 - rate relation ajtai program sourceProgram values) ^ index *
          ((pairLaw relation ajtai program sourceProgram values 1).toOuterMeasure event).toReal) :=
    funext actual
  rw [sequences]
  exact firstHits.tendsto_sum_nat

/-- Distribution of the actual emitted integer vector on the same latent
prefixes as `pairLaw`. Erasing input clocks preserves this value by
`run_value_erase`; no successful-vector oracle replaces the computation. -/
noncomputable def runLaw (count : Nat) (coordinate : Fin PaperProfile.arity.total) :
    PMF (Option (List Int)) :=
  (calls relation ajtai values).bind fun first =>
    (AcceptedRetryLaw.prefixes (calls relation ajtai values) count).map fun following =>
      (AdaptiveBindingRun.run relation ajtai program sourceProgram first following coordinate).value

/-- The selected-pair distribution feeds the existing executable integer
reduction and gives exactly the actual driver's output distribution. -/
theorem runLaw_eq (count : Nat) (coordinate : Fin PaperProfile.arity.total) :
    runLaw relation ajtai program sourceProgram values count coordinate =
      (pairLaw relation ajtai program sourceProgram values count).map
        (fun pair => match pair with
          | none => none
          | some (left, right) =>
              (BindingReduction.runPair program sourceProgram.access
                ⟨left, 0⟩ ⟨right, 0⟩ coordinate).value) := by
  unfold runLaw pairLaw
  simp only [PMF.map_bind, PMF.map_comp, Function.comp_def]
  congr 1
  funext first
  congr 1
  funext following
  exact run_value_eq_selected relation ajtai program sourceProgram first following coordinate

private theorem runLaw_event_tendsto_pos (coordinate : Fin PaperProfile.arity.total)
    (event : Set (Option (List Int))) (excludesAbort : none ∉ event)
    (positive : 0 < rate relation ajtai program sourceProgram values) :
    Filter.Tendsto (fun count : Nat =>
      ((runLaw relation ajtai program sourceProgram values count coordinate).toOuterMeasure event).toReal)
      Filter.atTop (nhds
        (((runLaw relation ajtai program sourceProgram values 1 coordinate).toOuterMeasure event).toReal /
          rate relation ajtai program sourceProgram values)) := by
  let reduced : Option (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape ×
        PaperCompositionAgreement.Observation State
          (InteractiveComposition.Endpoint relation ajtai) productionShape) → Option (List Int) :=
    fun pair => match pair with
    | none => none
    | some (left, right) =>
        (BindingReduction.runPair program sourceProgram.access
          ⟨left, 0⟩ ⟨right, 0⟩ coordinate).value
  have masses (count : Nat) :
      ((runLaw relation ajtai program sourceProgram values count coordinate).toOuterMeasure event).toReal =
      ((pairLaw relation ajtai program sourceProgram values count).toOuterMeasure
        (reduced ⁻¹' event)).toReal := by
    rw [runLaw_eq, PMF.toOuterMeasure_map_apply]
  have sequences := funext masses
  rw [sequences, masses 1]
  exact pairLaw_event_tendsto relation ajtai program sourceProgram values
    (reduced ⁻¹' event) excludesAbort positive

/-- A context with no accepted call cannot emit an MSIS vector. -/
theorem runLaw_zero (count : Nat) (coordinate : Fin PaperProfile.arity.total)
    (zero : rate relation ajtai program sourceProgram values = 0) :
    runLaw relation ajtai program sourceProgram values count coordinate = PMF.pure none := by
  rw [runLaw_eq, pairLaw_zero relation ajtai program sourceProgram values count zero,
    PMF.pure_map]

/-- Every nonabort event of the actual emitted vector has the stopped
two-call mass divided by the local acceptance rate. Zero-success contexts
remain in the experiment and contribute zero. -/
theorem runLaw_event_tendsto (coordinate : Fin PaperProfile.arity.total)
    (event : Set (Option (List Int))) (excludesAbort : none ∉ event) :
    Filter.Tendsto (fun count : Nat =>
      ((runLaw relation ajtai program sourceProgram values count coordinate).toOuterMeasure event).toReal)
      Filter.atTop (nhds
        (((runLaw relation ajtai program sourceProgram values 1 coordinate).toOuterMeasure event).toReal /
          rate relation ajtai program sourceProgram values)) := by
  by_cases zero : rate relation ajtai program sourceProgram values = 0
  · have noEvent (count : Nat) :
        ((runLaw relation ajtai program sourceProgram values count coordinate).toOuterMeasure event).toReal =
          0 := by
      rw [runLaw_zero relation ajtai program sourceProgram values count coordinate zero,
        PMF.toOuterMeasure_pure_apply, if_neg excludesAbort, ENNReal.toReal_zero]
    simp only [noEvent, zero_div]
    exact tendsto_const_nhds
  · exact runLaw_event_tendsto_pos relation ajtai program sourceProgram values coordinate event excludesAbort
      (lt_of_le_of_ne (AcceptedRetryLaw.successRate_range (calls relation ajtai values)
        (AdaptiveBinding.check relation ajtai program sourceProgram)).1 (Ne.symm zero))

/-- Success on a finite prefix, with the same independent uniform source
coordinate as the existing binding reduction. The event concerns the integer
vector actually emitted for the selected key and strict norm bound. -/
noncomputable def successProbability (count : Nat) : ℝ :=
  (∑ coordinate : Fin PaperProfile.arity.total,
    ((runLaw relation ajtai program sourceProgram values count coordinate).toOuterMeasure
      {output | BindingReduction.Succeeds ajtai output}).toReal) / (PaperProfile.arity.total : ℝ)

/-- The actual adaptive MSIS success mass is the limit of the finite
execution-prefix masses. The formula includes zero-success contexts; it
does not replace the emitted vector with an existential witness. -/
theorem successProbability_tendsto :
    Filter.Tendsto (successProbability relation ajtai program sourceProgram values)
      Filter.atTop (nhds
        (successProbability relation ajtai program sourceProgram values 1 /
          rate relation ajtai program sourceProgram values)) := by
  have each (coordinate : Fin PaperProfile.arity.total) :=
    runLaw_event_tendsto relation ajtai program sourceProgram values coordinate
      {output | BindingReduction.Succeeds ajtai output}
      (by rintro ⟨witness, impossible⟩; cases impossible)
  have summed := (tendsto_finsetSum Finset.univ (fun coordinate _ => each coordinate)).div_const
    (PaperProfile.arity.total : ℝ)
  simpa only [successProbability, ← Finset.sum_div, div_right_comm] using summed

end NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingLaw
