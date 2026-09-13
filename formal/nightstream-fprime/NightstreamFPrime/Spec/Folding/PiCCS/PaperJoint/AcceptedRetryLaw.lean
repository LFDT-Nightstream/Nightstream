import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.AcceptedRetry
import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-!
The finite-prefix probability law of `AcceptedRetry.search`. Fresh calls draw
independently from the supplied law of actual responses and their clocks.
The driver, rather than the length of the sampled prefix, determines which
responses execute and which clocks are charged. Prefix sampling is a
mathematical coupling, not a runtime table generator.

This module supplies the first-hit law needed by the v1.2 uniqueness retry.
The selected NIFS consumer must identify its actual checked call law with the
law supplied here; no Poseidon2 distribution or oracle translation is assumed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.AcceptedRetryLaw

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)
open AcceptedRetry

variable {Value : Type*}

/-- Independent latent call prefixes used to prove the driver's law. The
executed search determines which calls and clocks are consumed; this PMF
does not prescribe runtime construction of its unused suffix. -/
noncomputable def prefixes (calls : PMF (Result Value)) :
    Nat → PMF (List (Result Value))
  | 0 => PMF.pure []
  | count + 1 => calls.bind fun packet => (prefixes calls count).map (List.cons packet)

/-- One checked call retains its actual accepted value or records rejection. -/
noncomputable def attemptLaw (calls : PMF (Result Value)) (check : Value → Result Bool) :
    PMF (Option Value) :=
  calls.map fun packet => if (check packet.value).value then some packet.value else none

/-- Exact distribution of the finite driver, including its consumed-call
count and observed oracle, checker and control work. -/
noncomputable def runLaw (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) : PMF (Result (Option Value × Nat)) :=
  (prefixes calls count).map (search check)

/-- The full distribution keeps oracle and checker work correlated with the
response. Only a rejected response executes the next fresh prefix. -/
theorem runLaw_succ (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) :
    runLaw calls check (count + 1) = calls.bind fun packet =>
      let checked := check packet.value
      if checked.value then
        PMF.pure ⟨(some packet.value, 1), packet.work + checked.work + 3⟩
      else
        (runLaw calls check count).map fun next =>
          ⟨(next.value.1, next.value.2 + 1), packet.work + checked.work + next.work + 3⟩ := by
  simp only [runLaw, prefixes, PMF.map_bind, PMF.map_comp]
  apply congrArg (PMF.bind calls)
  funext packet
  cases accepted : (check packet.value).value <;>
    simp [Function.comp_def, search, accepted]
  exact PMF.map_const _ _

/-- Erase the clock and call count only when asking about the returned value. -/
noncomputable def valueLaw (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) : PMF (Option Value) :=
  (runLaw calls check count).map fun result => result.value.1

/-- One finite step keeps the first accepted value; only rejection advances
to the remaining fresh prefix. This is derived from the actual driver. -/
theorem valueLaw_succ (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) :
    valueLaw calls check (count + 1) =
      (attemptLaw calls check).bind fun outcome =>
        match outcome with
        | none => valueLaw calls check count
        | some value => PMF.pure (some value) := by
  simp only [valueLaw, runLaw, prefixes, PMF.map_bind, PMF.map_comp,
    attemptLaw, PMF.bind_map]
  apply congrArg (PMF.bind calls)
  funext packet
  cases accepted : (check packet.value).value <;>
    simp [Function.comp_def, search, accepted]
  exact PMF.map_const _ _

/-- The actual finite driver returns no value exactly when every checked
call rejects. The independent-prefix law gives the corresponding power. -/
theorem valueLaw_none (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) :
    (valueLaw calls check count) none = ((attemptLaw calls check) none) ^ count := by
  classical
  induction count with
  | zero => simp [valueLaw, runLaw, prefixes, search, PMF.pure_map]
  | succ count induction =>
      rw [valueLaw_succ, PMF.bind_apply, tsum_eq_single none]
      · rw [induction, pow_succ]
        exact mul_comm _ _
      · intro outcome different
        cases outcome with
        | none => exact (different rfl).elim
        | some value => simp

/-- An accepted value comes either from the first call, or from the remaining
prefix after a rejected first call. The two terms use the same actual law. -/
theorem valueLaw_some_succ (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) (value : Value) :
    (valueLaw calls check (count + 1)) (some value) =
      (attemptLaw calls check) none * (valueLaw calls check count) (some value) +
        (attemptLaw calls check) (some value) := by
  classical
  rw [valueLaw_succ, PMF.bind_apply, ENNReal.tsum_eq_add_tsum_ite none,
    tsum_eq_single (some value)]
  · simp
  · intro outcome different
    cases outcome with
    | none => simp
    | some other => simp [PMF.pure_apply, Ne.symm different]

/-- Exact accepted-value mass after a finite fresh prefix. Each term counts
one rejected prefix followed by this actual accepted value. -/
theorem valueLaw_some (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) (value : Value) :
    (valueLaw calls check count) (some value) =
      (∑ index ∈ Finset.range count, ((attemptLaw calls check) none) ^ index) *
        (attemptLaw calls check) (some value) := by
  induction count with
  | zero => simp [valueLaw, runLaw, prefixes, search, PMF.pure_map]
  | succ count induction =>
      rw [valueLaw_some_succ, induction, Finset.sum_range_succ']
      simp only [pow_succ, pow_zero, ← Finset.sum_mul]
      ring

/-- One minus the rejection atom is the success rate of the actual checked
call, with no conditioning of the surrounding input context. -/
noncomputable def successRate (calls : PMF (Result Value)) (check : Value → Result Bool) : ℝ :=
  1 - ((attemptLaw calls check) none).toReal

/-- The success rate is a probability, including the always-rejecting law. -/
theorem successRate_range (calls : PMF (Result Value)) (check : Value → Result Bool) :
    0 ≤ successRate calls check ∧ successRate calls check ≤ 1 := by
  have lower := ENNReal.toReal_nonneg (a := (attemptLaw calls check) none)
  have upper := ENNReal.toReal_mono ENNReal.one_ne_top
    ((attemptLaw calls check).coe_le_one none)
  simp only [ENNReal.toReal_one] at upper
  dsimp only [successRate]
  constructor <;> linarith

/-- The actual exhausted-output mass, weighted by base acceptance, vanishes.
This includes an always-rejecting law: its retry loop is never entered. -/
theorem entered_exhaustion_tendsTo_zero
    (calls : PMF (Result Value)) (check : Value → Result Bool) :
    Filter.Tendsto (fun count : Nat =>
      successRate calls check * ((valueLaw calls check count) none).toReal)
      Filter.atTop (nhds 0) := by
  have range := successRate_range calls check
  simpa only [valueLaw_none, ENNReal.toReal_pow, successRate, sub_sub_cancel] using
    AcceptedRetry.exhaustion_tendsTo_zero (successRate calls check) range.1 range.2

private noncomputable def mean {Sample : Type*} (distribution : PMF Sample)
    (value : Sample → ℝ≥0∞) : ℝ≥0∞ :=
  ∑' sample, distribution sample * value sample

private theorem mean_pure {Sample : Type*} (sample : Sample) (value : Sample → ℝ≥0∞) :
    mean (PMF.pure sample) value = value sample := by
  unfold mean
  rw [tsum_eq_single sample]
  · rw [PMF.pure_apply_self, one_mul]
  · intro other different
    rw [PMF.pure_apply_of_ne sample other different, zero_mul]

private theorem mean_bind {Sample Output : Type*} (distribution : PMF Sample)
    (next : Sample → PMF Output) (value : Output → ℝ≥0∞) :
    mean (distribution.bind next) value = mean distribution (fun sample => mean (next sample) value) := by
  unfold mean
  simp only [PMF.bind_apply, ← ENNReal.tsum_mul_right]
  rw [ENNReal.tsum_comm]
  apply tsum_congr
  intro sample
  simp only [mul_assoc, ENNReal.tsum_mul_left]

private theorem mean_map {Sample Output : Type*} (distribution : PMF Sample)
    (map : Sample → Output) (value : Output → ℝ≥0∞) :
    mean (distribution.map map) value = mean distribution (fun sample => value (map sample)) := by
  rw [PMF.map, mean_bind]
  simp only [Function.comp_apply, mean_pure]

private theorem mean_add {Sample : Type*} (distribution : PMF Sample)
    (left right : Sample → ℝ≥0∞) :
    mean distribution (fun sample => left sample + right sample) =
      mean distribution left + mean distribution right := by
  simp only [mean, mul_add, ENNReal.tsum_add]

private theorem mean_const {Sample : Type*} (distribution : PMF Sample) (value : ℝ≥0∞) :
    mean distribution (fun _ => value) = value := by
  simp only [mean, ENNReal.tsum_mul_right, distribution.tsum_coe, one_mul]

private theorem mean_rejected (calls : PMF (Result Value)) (check : Value → Result Bool)
    (value : ℝ≥0∞) :
    mean calls (fun packet => if (check packet.value).value then 0 else value) =
      (attemptLaw calls check) none * value := by
  classical
  rw [mean, attemptLaw, PMF.map_apply, ← ENNReal.tsum_mul_right]
  apply tsum_congr
  intro packet
  cases (check packet.value).value <;> simp

/-- Mean of one actual call, its checker, and the declared local transitions.
It can be infinite; a finite-work claim must supply a finite bound. -/
noncomputable def callWork (calls : PMF (Result Value)) (check : Value → Result Bool) : ℝ≥0∞ :=
  mean calls fun packet => ((packet.work + (check packet.value).work + 3 : Nat) : ℝ≥0∞)

/-- Expected work of the full finite driver, with the actual response/work
joint law retained. No independence within a call is required. -/
noncomputable def expectedWork (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) : ℝ≥0∞ :=
  mean (runLaw calls check count) fun result => (result.work : ℝ≥0∞)

/-- Freshness is used only between calls. Every call's own correlated oracle
and checker cost is charged before deciding whether another call executes. -/
theorem expectedWork_succ (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) :
    expectedWork calls check (count + 1) = callWork calls check +
      (attemptLaw calls check) none * expectedWork calls check count := by
  unfold expectedWork
  rw [runLaw_succ, mean_bind]
  calc
    _ = mean calls (fun packet =>
        ((packet.work + (check packet.value).work + 3 : Nat) : ℝ≥0∞) +
          if (check packet.value).value then 0 else expectedWork calls check count) := by
      apply congrArg (mean calls)
      funext packet
      cases accepted : (check packet.value).value <;>
        simp [accepted, mean_pure, mean_map, Nat.cast_add, mean_add, mean_const,
          expectedWork, add_assoc, add_comm, add_left_comm]
    _ = _ := by rw [mean_add, mean_rejected]; rfl

/-- Exact finite expected work. The final term is the exhaustion transition;
all earlier terms are complete calls, including rejected responses. -/
theorem expectedWork_eq (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) :
    expectedWork calls check count =
      callWork calls check *
        (∑ index ∈ Finset.range count, ((attemptLaw calls check) none) ^ index) +
          ((attemptLaw calls check) none) ^ count := by
  induction count with
  | zero => simp [expectedWork, runLaw, prefixes, search, PMF.pure_map, mean_pure]
  | succ count induction =>
      rw [expectedWork_succ, induction, Finset.sum_range_succ']
      simp only [pow_succ, pow_zero, ← Finset.sum_mul]
      ring

/-- Weighting the actual retry work by base acceptance cancels the reciprocal
success rate. The exhaustion charge vanishes. Only finite one-call mean work
is required; response and runtime may be correlated. -/
theorem entered_expectedWork_tendsto
    (calls : PMF (Result Value)) (check : Value → Result Bool)
    (finite : callWork calls check ≠ ∞) :
    Filter.Tendsto (fun count : Nat =>
      successRate calls check * (expectedWork calls check count).toReal)
      Filter.atTop
      (nhds (if successRate calls check = 0 then 0 else (callWork calls check).toReal)) := by
  have powers (index : Nat) : ((attemptLaw calls check) none) ^ index ≠ ∞ :=
    ENNReal.pow_ne_top ((attemptLaw calls check).apply_ne_top none)
  have actual (count : Nat) :
      (expectedWork calls check count).toReal =
        (callWork calls check).toReal *
          (∑ index ∈ Finset.range count, (((attemptLaw calls check) none).toReal) ^ index) +
            (((attemptLaw calls check) none).toReal) ^ count := by
    rw [expectedWork_eq, ENNReal.toReal_add
      (ENNReal.mul_ne_top finite (ENNReal.sum_ne_top.mpr (fun index _ => powers index)))
      (powers count), ENNReal.toReal_mul,
      ENNReal.toReal_sum (fun index _ => powers index)]
    simp only [ENNReal.toReal_pow]
  have range := successRate_range calls check
  have summed := (AcceptedRetry.entered_work_hasSum (successRate calls check)
    (callWork calls check).toReal range.1 range.2).tendsto_sum_nat
  have tail := AcceptedRetry.exhaustion_tendsTo_zero (successRate calls check) range.1 range.2
  simpa only [actual, successRate, sub_sub_cancel, mul_add, Finset.mul_sum,
    Finset.sum_mul, mul_assoc, mul_comm, mul_left_comm, add_zero] using summed.add tail

/-- Mean number of actual oracle responses consumed by the finite driver. -/
noncomputable def expectedCalls (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) : ℝ≥0∞ :=
  mean (runLaw calls check count) fun result => (result.value.2 : ℝ≥0∞)

/-- Invocation counting uses the returned call count, including rejected
calls, and does not infer a query count from a machine-time estimate. -/
theorem expectedCalls_eq (calls : PMF (Result Value)) (check : Value → Result Bool)
    (count : Nat) :
    expectedCalls calls check count =
      ∑ index ∈ Finset.range count, ((attemptLaw calls check) none) ^ index := by
  induction count with
  | zero => simp [expectedCalls, runLaw, prefixes, search, PMF.pure_map, mean_pure]
  | succ count induction =>
      change mean (runLaw calls check (count + 1)) _ = _
      rw [runLaw_succ, mean_bind]
      calc
        _ = mean calls (fun packet => (1 : ℝ≥0∞) +
            if (check packet.value).value then 0 else expectedCalls calls check count) := by
          apply congrArg (mean calls)
          funext packet
          cases accepted : (check packet.value).value <;>
            simp [accepted, mean_pure, mean_map, Nat.cast_add, mean_add, mean_const,
              expectedCalls, add_comm]
        _ = 1 + (attemptLaw calls check) none * expectedCalls calls check count := by
          rw [mean_add, mean_const, mean_rejected]
        _ = _ := by
          rw [induction, Finset.sum_range_succ']
          simp only [pow_succ, pow_zero, ← Finset.sum_mul]
          ring

/-- The entered retry contributes at most one oracle call in expectation.
Adding the initial checked call therefore gives at most two calls. This is
an invocation bound; permutation-query accounting belongs to the caller. -/
theorem entered_expectedCalls_tendsto
    (calls : PMF (Result Value)) (check : Value → Result Bool) :
    Filter.Tendsto (fun count : Nat =>
      successRate calls check * (expectedCalls calls check count).toReal)
      Filter.atTop (nhds (if successRate calls check = 0 then 0 else 1)) := by
  have actual (count : Nat) :
      (expectedCalls calls check count).toReal =
        ∑ index ∈ Finset.range count, (((attemptLaw calls check) none).toReal) ^ index := by
    rw [expectedCalls_eq, ENNReal.toReal_sum (fun index _ =>
      ENNReal.pow_ne_top ((attemptLaw calls check).apply_ne_top none))]
    simp only [ENNReal.toReal_pow]
  have range := successRate_range calls check
  have summed := (AcceptedRetry.entered_work_hasSum (successRate calls check)
    1 range.1 range.2).tendsto_sum_nat
  simpa only [actual, successRate, sub_sub_cancel, mul_one, Finset.mul_sum] using summed

/-- The same finite first-hit formula holds for any accepted-output event,
including a witness-disagreement event fixed by an earlier response. -/
theorem valueLaw_event (calls : PMF (Result Value)) (check : Value → Result Bool)
    (event : Set (Option Value)) (excludesAbort : none ∉ event) (count : Nat) :
    (valueLaw calls check count).toOuterMeasure event =
      (∑ index ∈ Finset.range count, ((attemptLaw calls check) none) ^ index) *
        (attemptLaw calls check).toOuterMeasure event := by
  classical
  rw [PMF.toOuterMeasure_apply, PMF.toOuterMeasure_apply, ← ENNReal.tsum_mul_left]
  apply tsum_congr
  intro outcome
  by_cases member : outcome ∈ event
  · cases outcome with
    | none => exact (excludesAbort member).elim
    | some value => simp only [Set.indicator_of_mem member, valueLaw_some]
  · simp only [Set.indicator_of_notMem member, mul_zero]

/-- The actual stopped law normalizes every accepted-output event by the
same success rate. This covers the binding reduction's event probabilities. -/
theorem valueLaw_event_tendsto
    (calls : PMF (Result Value)) (check : Value → Result Bool)
    (event : Set (Option Value)) (excludesAbort : none ∉ event)
    (positive : 0 < successRate calls check) :
    Filter.Tendsto (fun count : Nat => ((valueLaw calls check count).toOuterMeasure event).toReal)
      Filter.atTop
      (nhds (((attemptLaw calls check).toOuterMeasure event).toReal / successRate calls check)) := by
  have firstHits := AcceptedRetry.firstHit_hasSum (successRate calls check) positive
    (successRate_range calls check).2 (((attemptLaw calls check).toOuterMeasure event).toReal)
  have actual (count : Nat) :
      ((valueLaw calls check count).toOuterMeasure event).toReal =
        ∑ index ∈ Finset.range count,
          (((attemptLaw calls check) none).toReal) ^ index *
            ((attemptLaw calls check).toOuterMeasure event).toReal := by
    rw [valueLaw_event calls check event excludesAbort, ENNReal.toReal_mul,
      ENNReal.toReal_sum (fun index _ => ENNReal.pow_ne_top
        ((attemptLaw calls check).apply_ne_top none))]
    simp only [ENNReal.toReal_pow, Finset.sum_mul]
  simpa only [actual, successRate, sub_sub_cancel] using firstHits.tendsto_sum_nat

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.AcceptedRetryLaw
