import NightstreamFPrime.Lifecycle.Nifs.AdaptiveBinding
import NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingProbability
import NightstreamFPrime.Lifecycle.Nifs.BindingWork
import Mathlib.Topology.Order.MonotoneConvergence

/-!
Expected work of the adaptive NIFS driver under its actual finite control
recursion. The prefix and charged query keep their existing clocks. The
endpoint law selects both the actual check and the continuation work in the
same sum. The finite means converge to the complete stopped mean.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingWork

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork PiRLC.CoordinateForkLaw

variable {Context State Tape : Type*} {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      WeakExtraction.Continuation Tape relation ajtai (running context) (fresh context) coins output)
  (call : Context → CubePoint K productionShape.cubeVariables → K →
    CubePoint K productionShape.cubeVariables → Result (Option (Probe K productionShape × State)))
  (sourceProgram : Context → CheckedWitnessExtraction.Program productionShape (FullShape logicalWidth publicFits))

/-- Execute the same phases as `AdaptiveBinding.callClock`, then charge the
work selected by the observation. Query work precedes the endpoint branch;
the check and continuation are averaged together over that branch. -/
noncomputable def callThenClock (context : Context)
    (nextWork : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape → ℝ)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) : ℝ :=
  let issued := call context alpha gamma point
  (issued.work : ℝ) + (match issued.value with
  | none =>
      ((AdaptiveBinding.check relation ajtai program (sourceProgram context)
        (none : PaperCompositionAgreement.Observation State
          (InteractiveComposition.Endpoint relation ajtai) productionShape)).work : ℝ) + nextWork none
  | some receipt =>
      let algebra := (ProductionKey.key relation ajtai).piRlcAlgebra
      let law := InteractiveWork.law relation ajtai running fresh continuation context receipt
      let parentChecker := InteractiveWork.parentChecker relation ajtai running fresh continuation context receipt
      let typed := PiRLC.CoordinateExtraction.typedChecker algebra parentChecker
      let charged := PiRLC.CoordinateCheckedCalls.withChecker law typed
      PiRLC.CoordinateRetryWork.expectedQueryWork charged (PiRLC.CoordinateCheckedCalls.check typed) +
        ∑ endpoint, (PaperCompositionWork.endpointLaw algebra law parentChecker endpoint).toReal *
          (((AdaptiveBinding.check relation ajtai program (sourceProgram context)
            (some (receipt, endpoint))).work : ℝ) + nextWork (some (receipt, endpoint)))) + 1

variable
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (publicCheck : Context → Probe K productionShape → Bool)
  (callCorrect : ∀ context alpha gamma point,
    (call context alpha gamma point).value =
      InteractivePrefix.run (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
        alpha gamma point)

include callCorrect in
/-- The complete mean adds the continuation under the actual retained
observation law. Finite requests and endpoints suffice for every real-valued
continuation; no independence between response and query work is required. -/
theorem callThenClock_mean (context : Context)
    (nextWork : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape → ℝ) :
    StrongProbability.verifierMean
      (callThenClock relation ajtai program running fresh continuation call sourceProgram context nextWork) =
      StrongProbability.verifierMean
        (AdaptiveBinding.callClock relation ajtai program running fresh continuation call sourceProgram context) +
      ∑' observation,
        (SequentialObservationLaw.law
          (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
          (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
          observation).toReal * nextWork observation := by
  have split :
      callThenClock relation ajtai program running fresh continuation call sourceProgram context nextWork =
        fun alpha gamma point =>
          AdaptiveBinding.callClock relation ajtai program running fresh continuation call sourceProgram
            context alpha gamma point +
          PaperCompositionAgreement.endpointMean
            (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
            (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
            nextWork alpha gamma point := by
    funext alpha gamma point
    dsimp only [callThenClock, AdaptiveBinding.callClock, PaperCompositionAgreement.endpointMean]
    rw [← callCorrect context alpha gamma point]
    cases returned : (call context alpha gamma point).value with
    | none => ring
    | some receipt =>
        simp only [mul_add, Finset.sum_add_distrib]
        rw [BindingWork.suffixLaw_eq_workLaw ajtai relation running fresh continuation context receipt]
        ring_nf
        rfl
  rw [split, StrongProbability.verifierMean_add,
    ← (SequentialObservationLaw.value_hasSum
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
      (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context) nextWork).tsum_eq]

/-- Mean of the finite retry work. This follows `AdaptiveBindingRun.retryWork_cons`:
the actual call and checker, three loop transitions, then either the actual
integer-vector work or the remaining retry prefix. The empty prefix charges
the exhaustion transition in `AdaptiveBindingRun.retryWork_nil`. -/
noncomputable def retryMean (context : Context)
    (first : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)
    (coordinate : Fin PaperProfile.arity.total) : Nat → ℝ
  | 0 => 1
  | count + 1 =>
      StrongProbability.verifierMean
        (callThenClock relation ajtai program running fresh continuation call sourceProgram context
          fun observation => 3 +
            if (AdaptiveBinding.check relation ajtai program (sourceProgram context) observation).value then
              ((BindingReduction.runPair program (sourceProgram context).access
                ⟨first, 0⟩ ⟨observation, 0⟩ coordinate).work : ℝ)
            else retryMean context first coordinate count)

/-- Mean of the actual finite driver clock from `AdaptiveBindingRun.run_work_eq`.
The first checked call enters retries only on acceptance. Its branch uses the
same four return transitions, while rejection uses two. -/
noncomputable def driverMean (context : Context)
    (coordinate : Fin PaperProfile.arity.total) (count : Nat) : ℝ :=
  StrongProbability.verifierMean
    (callThenClock relation ajtai program running fresh continuation call sourceProgram context
      fun first =>
        if (AdaptiveBinding.check relation ajtai program (sourceProgram context) first).value then
          retryMean relation ajtai program running fresh continuation call sourceProgram
            context first coordinate count + 4
        else 2)

/-- The complete mean is the supremum of the actual finite-prefix means.
The bounds and convergence below justify this supremum as the stopped work. -/
noncomputable def expectedWork (context : Context)
    (coordinate : Fin PaperProfile.arity.total) : ℝ :=
  ⨆ count, driverMean relation ajtai program running fresh continuation call sourceProgram
    context coordinate count

variable (context : Context)

local notation "callLaw" => SequentialObservationLaw.law
  (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
  (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
local notation "callMean" => StrongProbability.verifierMean
  (AdaptiveBinding.callClock relation ajtai program running fresh continuation call sourceProgram context)
local notation "callRate" => AdaptiveBindingLaw.rate relation ajtai program (sourceProgram context) callLaw
local notation "checked" => AdaptiveBinding.check relation ajtai program (sourceProgram context)

include callCorrect in
private theorem callThenClock_mean_mono
    (left right : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape → ℝ)
    (ordered : ∀ observation, left observation ≤ right observation) :
    StrongProbability.verifierMean
        (callThenClock relation ajtai program running fresh continuation call sourceProgram context left) ≤
      StrongProbability.verifierMean
        (callThenClock relation ajtai program running fresh continuation call sourceProgram context right) := by
  rw [callThenClock_mean relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect,
    callThenClock_mean relation ajtai program running fresh continuation call sourceProgram
      originalFirstPhase publicCheck callCorrect]
  apply _root_.add_le_add le_rfl
  exact Summable.tsum_le_tsum
    (fun observation => mul_le_mul_of_nonneg_left (ordered observation) ENNReal.toReal_nonneg)
    (SequentialObservationLaw.value_hasSum _ _ left).summable
    (SequentialObservationLaw.value_hasSum _ _ right).summable

include callCorrect in
private theorem callThenClock_branch_mean (accepted rejected : ℝ) :
    StrongProbability.verifierMean
      (callThenClock relation ajtai program running fresh continuation call sourceProgram context
        fun observation => if (checked observation).value then accepted else rejected) =
      callMean + callRate * accepted + (1 - callRate) * rejected := by
  rw [callThenClock_mean relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect]
  have weights : HasSum (fun observation => (callLaw observation).toReal) 1 := by
    have total : (∑' observation, (callLaw observation).toReal) = 1 := by
      rw [← ENNReal.tsum_toReal_eq (callLaw).apply_ne_top, (callLaw).tsum_coe, ENNReal.toReal_one]
    exact total ▸ (ENNReal.summable_toReal (callLaw).tsum_coe_ne_top).hasSum
  have indicator : HasSum (fun observation => (callLaw observation).toReal *
      (if (checked observation).value then (1 : ℝ) else 0)) callRate := by
    have summed := (SequentialObservationLaw.value_hasSum
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
      (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
      (fun observation => if (checked observation).value then (1 : ℝ) else 0)).summable.hasSum
    rw [AdaptiveBindingProbability.checkedMean_eq_rate] at summed
    exact summed
  have combined := (weights.mul_right rejected).add (indicator.mul_right (accepted - rejected))
  have same : (fun observation => (callLaw observation).toReal *
        (if (checked observation).value then accepted else rejected)) =
      (fun observation => (callLaw observation).toReal * rejected +
        ((callLaw observation).toReal * (if (checked observation).value then (1 : ℝ) else 0)) *
          (accepted - rejected)) := by
    funext observation
    cases (checked observation).value <;> simp only [Bool.false_eq_true, ↓reduceIte] <;> ring
  rw [same, combined.tsum_eq]
  ring

private theorem callMean_nonnegative : 0 ≤ callMean := by
  have nonnegative := StrongProbability.verifierMean_mono
    (fun _ _ _ => (0 : ℝ))
    (AdaptiveBinding.callClock relation ajtai program running fresh continuation call sourceProgram context)
    (fun alpha gamma point => by
      rw [AdaptiveBinding.callClock_eq]
      exact add_nonneg
        (PaperCompositionWork.totalClock_nonnegative (ProductionKey.key relation ajtai).piRlcAlgebra
          call (InteractiveWork.law relation ajtai running fresh continuation)
          (InteractiveWork.parentChecker relation ajtai running fresh continuation)
          program sourceProgram (PaperWeakOutput.decode (ProductionKey.key relation ajtai))
          context alpha gamma point) zero_le_one)
  simpa only [StrongProbability.verifierMean_const] using nonnegative

private theorem callRate_range : 0 ≤ callRate ∧ callRate ≤ 1 :=
  AcceptedRetryLaw.successRate_range _ _

include callCorrect in
private theorem retryMean_one_le
    (first : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)
    (coordinate : Fin PaperProfile.arity.total) (count : Nat) :
    1 ≤ retryMean relation ajtai program running fresh continuation call sourceProgram
      context first coordinate count := by
  induction count with
  | zero => exact le_rfl
  | succ count induction =>
      have comparison := callThenClock_mean_mono relation ajtai program running fresh continuation
        call sourceProgram originalFirstPhase publicCheck callCorrect context
        (fun observation => if (checked observation).value then (3 : ℝ) else 3)
        (fun observation => 3 + if (checked observation).value then
          ((BindingReduction.runPair program (sourceProgram context).access
            ⟨first, 0⟩ ⟨observation, 0⟩ coordinate).work : ℝ)
          else retryMean relation ajtai program running fresh continuation call sourceProgram
            context first coordinate count)
        (by
          intro observation
          cases accepted : (checked observation).value <;>
            simp only [accepted, Bool.false_eq_true, ↓reduceIte]
          · linarith only [induction]
          · have workNonnegative : 0 ≤ ((BindingReduction.runPair program
                (sourceProgram context).access ⟨first, 0⟩ ⟨observation, 0⟩ coordinate).work : ℝ) :=
              Nat.cast_nonneg _
            linarith only [workNonnegative])
      rw [callThenClock_branch_mean relation ajtai program running fresh continuation call sourceProgram
        originalFirstPhase publicCheck callCorrect context] at comparison
      change 1 ≤ StrongProbability.verifierMean _
      have nonnegative := callMean_nonnegative relation ajtai program running fresh continuation
        call sourceProgram context
      linarith only [comparison, nonnegative]

include callCorrect in
private theorem retryMean_mono
    (first : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)
    (coordinate : Fin PaperProfile.arity.total) :
    Monotone (retryMean relation ajtai program running fresh continuation call sourceProgram
      context first coordinate) := by
  apply monotone_nat_of_le_succ
  intro count
  induction count with
  | zero =>
      exact retryMean_one_le relation ajtai program running fresh continuation call sourceProgram
        originalFirstPhase publicCheck callCorrect context first coordinate 1
  | succ count induction =>
      dsimp only [retryMean]
      apply callThenClock_mean_mono relation ajtai program running fresh continuation call sourceProgram
        originalFirstPhase publicCheck callCorrect context
      intro observation
      cases accepted : (checked observation).value <;>
        simp only [Bool.false_eq_true, ↓reduceIte]
      · exact add_le_add le_rfl induction
      · exact le_rfl

include callCorrect in
private theorem driverMean_mono (coordinate : Fin PaperProfile.arity.total) :
    Monotone (driverMean relation ajtai program running fresh continuation call sourceProgram
      context coordinate) := by
  intro left right ordered
  unfold driverMean
  apply callThenClock_mean_mono relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect context
  intro first
  cases accepted : (checked first).value <;>
    simp only [Bool.false_eq_true, ↓reduceIte]
  · exact le_rfl
  · exact add_le_add
      (retryMean_mono relation ajtai program running fresh continuation call sourceProgram
        originalFirstPhase publicCheck callCorrect context first coordinate ordered) le_rfl

include callCorrect in
private theorem driverMean_nonnegative (coordinate : Fin PaperProfile.arity.total) (count : Nat) :
    0 ≤ driverMean relation ajtai program running fresh continuation call sourceProgram
      context coordinate count := by
  unfold driverMean
  rw [callThenClock_mean relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect]
  apply add_nonneg
    (callMean_nonnegative relation ajtai program running fresh continuation call sourceProgram context)
  apply tsum_nonneg
  intro first
  apply mul_nonneg ENNReal.toReal_nonneg
  cases accepted : (checked first).value <;>
    simp only [Bool.false_eq_true, ↓reduceIte]
  · norm_num
  · have one := retryMean_one_le relation ajtai program running fresh continuation call sourceProgram
      originalFirstPhase publicCheck callCorrect context first coordinate count
    linarith only [one]

include callCorrect in
private theorem retryMean_le
    (postBound : ℝ) (postNonnegative : 0 ≤ postBound)
    (postBounded : ∀ first second coordinate,
      ((BindingReduction.runPair (State := State) program (sourceProgram context).access
        ⟨first, 0⟩ ⟨second, 0⟩ coordinate).work : ℝ) ≤ postBound)
    (first : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)
    (coordinate : Fin PaperProfile.arity.total) (count : Nat) :
    retryMean relation ajtai program running fresh continuation call sourceProgram
        context first coordinate count ≤
      (callMean + 3) * (∑ index ∈ Finset.range count, (1 - callRate) ^ index) +
        (1 - callRate) ^ count + postBound := by
  induction count with
  | zero =>
      simp only [retryMean, Finset.sum_range_zero, mul_zero, pow_zero, zero_add]
      linarith only [postNonnegative]
  | succ count induction =>
      let previous := (callMean + 3) * (∑ index ∈ Finset.range count, (1 - callRate) ^ index) +
        (1 - callRate) ^ count
      have comparison := callThenClock_mean_mono relation ajtai program running fresh continuation
        call sourceProgram originalFirstPhase publicCheck callCorrect context
        (fun observation => 3 + if (checked observation).value then
          ((BindingReduction.runPair program (sourceProgram context).access
            ⟨first, 0⟩ ⟨observation, 0⟩ coordinate).work : ℝ)
          else retryMean relation ajtai program running fresh continuation call sourceProgram
            context first coordinate count)
        (fun observation => if (checked observation).value then 3 + postBound
          else 3 + (previous + postBound))
        (by
          intro observation
          cases accepted : (checked observation).value <;>
            simp only [accepted, Bool.false_eq_true, ↓reduceIte]
          · exact add_le_add le_rfl induction
          · exact add_le_add le_rfl (postBounded first observation coordinate))
      rw [callThenClock_branch_mean relation ajtai program running fresh continuation call sourceProgram
        originalFirstPhase publicCheck callCorrect context] at comparison
      calc
        _ ≤ callMean + callRate * (3 + postBound) +
            (1 - callRate) * (3 + (previous + postBound)) := comparison
        _ = _ := by
          dsimp only [previous]
          simp only [Finset.sum_range_succ', pow_zero, pow_succ, ← Finset.sum_mul]
          ring

include callCorrect in
private theorem driverMean_le_postBound
    (postBound : ℝ) (postNonnegative : 0 ≤ postBound)
    (postBounded : ∀ first second coordinate,
      ((BindingReduction.runPair (State := State) program (sourceProgram context).access
        ⟨first, 0⟩ ⟨second, 0⟩ coordinate).work : ℝ) ≤ postBound)
    (coordinate : Fin PaperProfile.arity.total) (count : Nat) :
    driverMean relation ajtai program running fresh continuation call sourceProgram
        context coordinate count ≤
      callMean + callRate * (callMean + 3) *
        (∑ index ∈ Finset.range count, (1 - callRate) ^ index) +
        callRate * (1 - callRate) ^ count + postBound + 4 := by
  let cap := (callMean + 3) * (∑ index ∈ Finset.range count, (1 - callRate) ^ index) +
    (1 - callRate) ^ count
  have comparison := callThenClock_mean_mono relation ajtai program running fresh continuation
    call sourceProgram originalFirstPhase publicCheck callCorrect context
    (fun first => if (checked first).value then
      retryMean relation ajtai program running fresh continuation call sourceProgram
        context first coordinate count + 4 else 2)
    (fun first => if (checked first).value then cap + postBound + 4 else 2)
    (by
      intro first
      cases accepted : (checked first).value <;>
        simp only [accepted, Bool.false_eq_true, ↓reduceIte]
      · exact le_rfl
      · exact add_le_add
          (retryMean_le relation ajtai program running fresh continuation call sourceProgram
            originalFirstPhase publicCheck callCorrect context postBound postNonnegative postBounded
            first coordinate count) le_rfl)
  rw [callThenClock_branch_mean relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect context] at comparison
  have range := callRate_range relation ajtai program running fresh continuation sourceProgram
    originalFirstPhase publicCheck context
  have remaining := mul_nonneg (sub_nonneg.mpr range.2)
    (by linarith only [postNonnegative] : 0 ≤ postBound + 2)
  dsimp only [cap] at comparison
  unfold driverMean
  linarith only [comparison, remaining]

variable (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds)
  (accessBound : Nat)
  (accessBounded : CostedWitnessProjection.Bounded (sourceProgram context).access accessBound)

local notation "postBound" => ((BindingOutput.crossWork bounds +
  Phi81Relation.Shape.carrierWidth (FullShape logicalWidth publicFits) * (accessBound + 6) + 12 : Nat) : ℝ)
local notation "overhead" => ((BindingOutput.crossWork bounds +
  Phi81Relation.Shape.carrierWidth (FullShape logicalWidth publicFits) * (accessBound + 6) + 16 : Nat) : ℝ)

include callCorrect bounded accessBounded in
/-- Finite work of the actual driver, with its entered rejected-call tail
and exhaustion transition. The overhead is the existing integer-vector
bound plus the four actual outer transitions from `run_work_le`. -/
theorem driverMean_le (coordinate : Fin PaperProfile.arity.total) (count : Nat) :
    driverMean relation ajtai program running fresh continuation call sourceProgram
        context coordinate count ≤
      callMean + callRate * (callMean + 3) *
        (∑ index ∈ Finset.range count, (1 - callRate) ^ index) +
        callRate * (1 - callRate) ^ count + overhead := by
  have postBounded : ∀ first second coordinate,
      ((BindingReduction.runPair (State := State) program (sourceProgram context).access
        ⟨first, 0⟩ ⟨second, 0⟩ coordinate).work : ℝ) ≤ postBound := by
    intro first second coordinate
    have actual := BindingReduction.runPair_work_le ajtai program bounds bounded
      (sourceProgram context).access accessBound accessBounded
      ⟨first, 0⟩ ⟨second, 0⟩ coordinate
    dsimp only at actual
    simp only [Nat.zero_add] at actual
    exact_mod_cast actual
  have actual := driverMean_le_postBound relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect context postBound (Nat.cast_nonneg _) postBounded coordinate count
  simp only [Nat.cast_add, Nat.cast_ofNat] at actual ⊢
  linarith only [actual]

include callCorrect bounded accessBounded in
/-- The same bound holds for every finite prefix. Multiplying by the actual
entry rate cancels the retry factor, including a zero-rate context. -/
theorem driverMean_uniform_bound (coordinate : Fin PaperProfile.arity.total) (count : Nat) :
    driverMean relation ajtai program running fresh continuation call sourceProgram
      context coordinate count ≤ 2 * callMean + 3 + overhead := by
  have range := callRate_range relation ajtai program running fresh continuation sourceProgram
    originalFirstPhase publicCheck context
  have meanNonnegative := callMean_nonnegative relation ajtai program running fresh continuation
    call sourceProgram context
  have mass : callRate * (∑ index ∈ Finset.range count, (1 - callRate) ^ index) =
      1 - (1 - callRate) ^ count := by
    simpa only [sub_sub_cancel] using mul_neg_geom_sum (1 - callRate) count
  have tailNonnegative := mul_nonneg
    (pow_nonneg (sub_nonneg.mpr range.2) count)
    (by linarith only [meanNonnegative, range.2] : 0 ≤ callMean + 3 - callRate)
  have entered : callRate * (callMean + 3) *
        (∑ index ∈ Finset.range count, (1 - callRate) ^ index) +
        callRate * (1 - callRate) ^ count ≤ callMean + 3 := by
    calc
      _ = (callMean + 3) *
          (callRate * (∑ index ∈ Finset.range count, (1 - callRate) ^ index)) +
          callRate * (1 - callRate) ^ count := by ring
      _ = (callMean + 3) * (1 - (1 - callRate) ^ count) +
          callRate * (1 - callRate) ^ count := by rw [mass]
      _ ≤ callMean + 3 := by nlinarith only [tailNonnegative]
  have actual := driverMean_le relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect context bounds bounded accessBound accessBounded coordinate count
  linarith only [actual, entered]

include callCorrect bounded accessBounded in
/-- The supremum is finite and bounds the complete work of this same
stopped driver. Source-query costs and every failed call remain in callMean. -/
theorem expectedWork_range (coordinate : Fin PaperProfile.arity.total) :
    0 ≤ expectedWork relation ajtai program running fresh continuation call sourceProgram context coordinate ∧
      expectedWork relation ajtai program running fresh continuation call sourceProgram context coordinate ≤
        2 * callMean + 3 + overhead := by
  have upper := driverMean_uniform_bound relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect context bounds bounded accessBound accessBounded coordinate
  have boundedAbove : BddAbove (Set.range
      (driverMean relation ajtai program running fresh continuation call sourceProgram context coordinate)) := by
    refine ⟨2 * callMean + 3 + overhead, ?_⟩
    rintro _ ⟨count, rfl⟩
    exact upper count
  unfold expectedWork
  exact ⟨(driverMean_nonnegative relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect context coordinate 0).trans (le_ciSup boundedAbove 0),
    ciSup_le upper⟩

include callCorrect bounded accessBounded in
/-- Increasing the actual oracle prefix converges to the complete mean.
Monotonicity follows from the executed retry branches; boundedness comes
from the entered-work calculation above. -/
theorem driverMean_tendsto (coordinate : Fin PaperProfile.arity.total) :
    Filter.Tendsto
      (driverMean relation ajtai program running fresh continuation call sourceProgram context coordinate)
      Filter.atTop
      (nhds (expectedWork relation ajtai program running fresh continuation call sourceProgram
        context coordinate)) := by
  apply tendsto_atTop_ciSup
    (driverMean_mono relation ajtai program running fresh continuation call sourceProgram
      originalFirstPhase publicCheck callCorrect context coordinate)
  refine ⟨2 * callMean + 3 + overhead, ?_⟩
  rintro _ ⟨count, rfl⟩
  exact driverMean_uniform_bound relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect context bounds bounded accessBound accessBounded coordinate count

/-- The actual rate weights entry into the rejected-prefix tail. Its mass
vanishes for every context; a zero-rate context never enters retries. -/
theorem entered_exhaustion_tendsto :
    Filter.Tendsto (fun count : Nat => callRate * (1 - callRate) ^ count)
      Filter.atTop (nhds 0) := by
  have range := callRate_range relation ajtai program running fresh continuation sourceProgram
    originalFirstPhase publicCheck context
  exact AcceptedRetry.exhaustion_tendsTo_zero callRate range.1 range.2

/-- The mean recursion starts at the actual empty-prefix retry clock. -/
theorem retryMean_zero_eq_runWork
    (first : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)
    (coordinate : Fin PaperProfile.arity.total) :
    retryMean relation ajtai program running fresh continuation call sourceProgram
        context first coordinate 0 =
      (AdaptiveBindingRun.retryWork relation ajtai program (sourceProgram context) first [] coordinate : ℝ) := by
  simp only [retryMean, AdaptiveBindingRun.retryWork_nil, Nat.cast_one]

/-- The continuation is the actual one-packet retry frame. The checker is
already charged by `callThenClock`; on rejection, the next mean replaces
the frame's empty-prefix exhaustion transition. -/
theorem retryMean_succ_eq_runWork
    (first : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)
    (coordinate : Fin PaperProfile.arity.total) (count : Nat) :
    retryMean relation ajtai program running fresh continuation call sourceProgram
        context first coordinate (count + 1) =
      StrongProbability.verifierMean
        (callThenClock relation ajtai program running fresh continuation call sourceProgram context
          fun observation =>
            (AdaptiveBindingRun.retryWork relation ajtai program (sourceProgram context)
              first [⟨observation, 0⟩] coordinate : ℝ) - ((checked observation).work : ℝ) +
            if (checked observation).value then 0 else
              retryMean relation ajtai program running fresh continuation call sourceProgram
                context first coordinate count - 1) := by
  change StrongProbability.verifierMean
    (callThenClock relation ajtai program running fresh continuation call sourceProgram context _) = _
  apply congrArg (fun nextWork => StrongProbability.verifierMean
    (callThenClock relation ajtai program running fresh continuation call sourceProgram context nextWork))
  funext observation
  rw [AdaptiveBindingRun.retryWork_cons, AdaptiveBindingRun.retryWork_nil]
  cases (checked observation).value <;>
    simp only [Bool.false_eq_true, ↓reduceIte, Nat.cast_add, Nat.cast_zero, Nat.cast_ofNat] <;> ring

/-- The first-call continuation is the actual driver's empty-prefix frame.
Its checker is already charged, and an entered retry mean replaces exactly
the empty retry transition. This preserves both outer return branches. -/
theorem driverMean_eq_runWork (coordinate : Fin PaperProfile.arity.total) (count : Nat) :
    driverMean relation ajtai program running fresh continuation call sourceProgram context coordinate count =
      StrongProbability.verifierMean
        (callThenClock relation ajtai program running fresh continuation call sourceProgram context
          fun first =>
            ((AdaptiveBindingRun.run relation ajtai program (sourceProgram context)
              ⟨first, 0⟩ [] coordinate).work : ℝ) - ((checked first).work : ℝ) +
            if (checked first).value then
              retryMean relation ajtai program running fresh continuation call sourceProgram
                context first coordinate count - 1 else 0) := by
  unfold driverMean
  apply congrArg (fun nextWork => StrongProbability.verifierMean
    (callThenClock relation ajtai program running fresh continuation call sourceProgram context nextWork))
  funext first
  rw [AdaptiveBindingRun.run_work_eq, AdaptiveBindingRun.retryWork_nil]
  cases (checked first).value <;>
    simp only [Bool.false_eq_true, ↓reduceIte, Nat.cast_add, Nat.cast_zero, Nat.cast_ofNat] <;> ring

variable (allAccessBounded : ∀ context, CostedWitnessProjection.Bounded
  (sourceProgram context).access accessBound)

include callCorrect bounded allAccessBounded in
/-- The original context law inherits the source's finite first moment.
Individual contexts need no common source-work bound. The two presence
tests add two to the local bound, giving the displayed five plus overhead. -/
theorem expected_work_bound (contexts : PMF Context) (coordinate : Fin PaperProfile.arity.total)
    (sourceSummable : Summable fun context => (contexts context).toReal *
      StrongProbability.verifierMean
        (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram context)) :
    Summable (fun context => (contexts context).toReal *
      expectedWork relation ajtai program running fresh continuation call sourceProgram context coordinate) ∧
    (∑' context, (contexts context).toReal *
      expectedWork relation ajtai program running fresh continuation call sourceProgram context coordinate) ≤
      2 * StrongProbability.clockMean contexts
        (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram) +
        5 + overhead := by
  let source := fun context (_ : CubePoint K productionShape.cubeVariables) (_ : K)
      (_ : CubePoint K productionShape.cubeVariables) =>
    StrongProbability.verifierMean
      (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram context)
  let total := fun context (_ : CubePoint K productionShape.cubeVariables) (_ : K)
      (_ : CubePoint K productionShape.cubeVariables) =>
    expectedWork relation ajtai program running fresh continuation call sourceProgram context coordinate
  have lifted := StrongProbability.clockMean_le_mul_add_const contexts source total 2 (5 + overhead)
    (fun context _ _ _ =>
      (expectedWork_range relation ajtai program running fresh continuation call sourceProgram
        originalFirstPhase publicCheck callCorrect context bounds bounded accessBound
        (allAccessBounded context) coordinate).1)
    (by simpa only [source, StrongProbability.verifierMean_const] using sourceSummable)
    (by
      intro context _ _ _
      have actual := (expectedWork_range relation ajtai program running fresh continuation call sourceProgram
        originalFirstPhase publicCheck callCorrect context bounds bounded accessBound
        (allAccessBounded context) coordinate).2
      have clocks :
          AdaptiveBinding.callClock relation ajtai program running fresh continuation call sourceProgram context =
            fun alpha gamma point =>
              InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram
                context alpha gamma point + 1 := by
        funext alpha gamma point
        exact AdaptiveBinding.callClock_eq relation ajtai program running fresh continuation call sourceProgram
          context alpha gamma point
      rw [clocks, StrongProbability.verifierMean_add, StrongProbability.verifierMean_const] at actual
      dsimp only [source, total]
      linarith only [actual])
  simpa only [StrongProbability.clockMean, source, total, StrongProbability.verifierMean_const,
    mul_comm _ (2 : ℝ), _root_.add_assoc] using lifted

include callCorrect bounded allAccessBounded in
/-- Preparation returns the context used by this same stopped reduction.
Its observed work and its dispatch transition are charged once. Both input
moments remain explicit premises on the actual source and preparation. -/
theorem prepared_expected_work_bound {SetupTape : Type*}
    (setupTapes : PMF SetupTape) (prepare : SetupTape → Result Context)
    (coordinate : Fin PaperProfile.arity.total)
    (preparationSummable : Summable fun tape => (setupTapes tape).toReal * (prepare tape).work)
    (sourceSummable : Summable fun context => (ContextPreparation.contexts setupTapes prepare context).toReal *
      StrongProbability.verifierMean
        (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram context)) :
    let continuationMean := fun context =>
      expectedWork relation ajtai program running fresh continuation call sourceProgram context coordinate
    Summable (fun tape => (setupTapes tape).toReal * ContextPreparation.clock prepare continuationMean tape) ∧
    (∑' tape, (setupTapes tape).toReal * ContextPreparation.clock prepare continuationMean tape) ≤
      (∑' tape, (setupTapes tape).toReal * (prepare tape).work) +
      2 * StrongProbability.clockMean (ContextPreparation.contexts setupTapes prepare)
        (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram) +
        6 + overhead := by
  dsimp only
  have reduced := expected_work_bound relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect bounds bounded accessBound allAccessBounded
    (ContextPreparation.contexts setupTapes prepare) coordinate sourceSummable
  have full := ContextPreparation.expected_work_eq setupTapes prepare
    (fun context => expectedWork relation ajtai program running fresh continuation call sourceProgram context coordinate)
    (fun context =>
      (expectedWork_range relation ajtai program running fresh continuation call sourceProgram
        originalFirstPhase publicCheck callCorrect context bounds bounded accessBound
        (allAccessBounded context) coordinate).1)
    preparationSummable reduced.1
  refine ⟨full.1, ?_⟩
  rw [full.2]
  linarith only [reduced.2]

include callCorrect bounded allAccessBounded in
/-- Supplied source, primitive, access and preparation polynomials bound the
same prepared reduction. The constant is five plus the existing overhead's
sixteen plus one preparation dispatch; crossWork contributes its proved factor three. -/
theorem prepared_expected_work_polynomial_bound {SetupTape : Type*}
    (setupTapes : PMF SetupTape) (prepare : SetupTape → Result Context)
    (coordinate : Fin PaperProfile.arity.total)
    (preparationSummable : Summable fun tape => (setupTapes tape).toReal * (prepare tape).work)
    (sourceSummable : Summable fun context => (ContextPreparation.contexts setupTapes prepare context).toReal *
      StrongProbability.verifierMean
        (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram context))
    (securityParameter : Nat)
    (preparationPolynomial sourcePolynomial primitivePolynomial accessPolynomial : Polynomial ℝ)
    (preparationPPT : (∑' tape, (setupTapes tape).toReal * (prepare tape).work) ≤
      preparationPolynomial.eval (securityParameter : ℝ))
    (sourcePPT : StrongProbability.clockMean (ContextPreparation.contexts setupTapes prepare)
      (InteractiveWork.totalClock relation ajtai running fresh continuation call program sourceProgram) ≤
        sourcePolynomial.eval (securityParameter : ℝ))
    (primitivePPT : (bounds.coordinateWork : ℝ) ≤ primitivePolynomial.eval (securityParameter : ℝ))
    (accessPPT : (accessBound : ℝ) ≤ accessPolynomial.eval (securityParameter : ℝ)) :
    let continuationMean := fun context =>
      expectedWork relation ajtai program running fresh continuation call sourceProgram context coordinate
    Summable (fun tape => (setupTapes tape).toReal * ContextPreparation.clock prepare continuationMean tape) ∧
    (∑' tape, (setupTapes tape).toReal * ContextPreparation.clock prepare continuationMean tape) ≤
      (preparationPolynomial + Polynomial.C 2 * sourcePolynomial + Polynomial.C 3 * primitivePolynomial +
        Polynomial.C ((FullShape logicalWidth publicFits).carrierWidth : ℝ) *
          (accessPolynomial + Polynomial.C 6) + Polynomial.C 22).eval (securityParameter : ℝ) := by
  dsimp only
  have actual := prepared_expected_work_bound relation ajtai program running fresh continuation call sourceProgram
    originalFirstPhase publicCheck callCorrect bounds bounded accessBound allAccessBounded
    setupTapes prepare coordinate preparationSummable sourceSummable
  refine ⟨actual.1, ?_⟩
  have cross : (BindingOutput.crossWork bounds : ℝ) ≤ 3 * bounds.coordinateWork := by
    exact_mod_cast BindingOutput.crossWork_le_coordinateWork bounds
  have access := mul_le_mul_of_nonneg_left accessPPT
    (Nat.cast_nonneg (FullShape logicalWidth publicFits).carrierWidth : (0 : ℝ) ≤
      (FullShape logicalWidth publicFits).carrierWidth)
  have total := actual.2
  push_cast at total
  simp only [Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_C]
  nlinarith only [total, preparationPPT, sourcePPT, primitivePPT, access, cross]

end NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingWork
