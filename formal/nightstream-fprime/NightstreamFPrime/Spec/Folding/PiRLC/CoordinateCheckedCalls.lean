import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateRetryWork

/-!
The actual CE checker and its work belong to one invocation. This provider
adds that returned work to each oracle clock and proves that clock erasure
preserves the response distribution used by coordinate extraction.

The oracle clock is unbounded with a summable first moment. The deterministic
CE checker returns its Boolean result and its work together; the closing
theorem must identify that Boolean projection with the CE response relation.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateCheckedCalls

open scoped BigOperators
open CoordinateOracle CoordinateChargedOracle

/-- Result and observed work of the same CE checker invocation. -/
structure CheckResult where
  accepted : Bool
  work : Nat

abbrev Checker (Index Challenge Assignment : Type*) :=
  (Index → Challenge) → Assignment → CheckResult

variable {Index Challenge Assignment : Type*}

def check (checker : Checker Index Challenge Assignment)
    (vector : Index → Challenge) (assignment : Assignment) : Bool :=
  (checker vector assignment).accepted

/-- An oracle abort has no assignment on which to run the CE checker. -/
def checkerWork (checker : Checker Index Challenge Assignment)
    (vector : Index → Challenge) : Option Assignment → Nat
  | none => 0
  | some assignment => (checker vector assignment).work

/-- Execute the check on the returned assignment, retaining the oracle clock.
The driver's stopping transition is charged even on an abort. -/
def runQuery (checker : Checker Index Challenge Assignment)
    (vector : Index → Challenge) (result : Option Assignment) (oracleSteps : Nat) : Bool × Nat :=
  match result with
  | none => (false, oracleSteps + 1)
  | some assignment =>
      let checked := checker vector assignment
      (checked.accepted, oracleSteps + checked.work + 1)

/-- The shifted clock is exactly the work of the actual checker invocation;
an unrelated Boolean function cannot supply this equality. -/
theorem runQuery_eq_queryStep (checker : Checker Index Challenge Assignment)
    (vector : Index → Challenge) (result : Option Assignment) (oracleSteps : Nat) :
    runQuery checker vector result oracleSteps =
      queryStep (check checker) vector result
        (oracleSteps + checkerWork checker vector result) := by
  cases result <;> simp [runQuery, queryStep, checkerWork, check, CoordinateOracle.accepted]

private theorem clock_injective (extra : Nat) :
    Function.Injective (fun steps : Nat => steps + extra) :=
  fun _ _ equal => Nat.add_right_cancel equal

private noncomputable def shift (extra : Nat) (mass : Nat → ℝ) : Nat → ℝ :=
  Function.extend (fun steps => steps + extra) mass 0

private theorem shift_hasSum (extra : Nat) {mass : Nat → ℝ} {total : ℝ}
    (summed : HasSum mass total) : HasSum (shift extra mass) total :=
  (hasSum_extend_zero (clock_injective extra)).mpr summed

private theorem shift_nonnegative (extra : Nat) (mass : Nat → ℝ)
    (nonnegative : ∀ steps, 0 ≤ mass steps) (clock : Nat) :
    0 ≤ shift extra mass clock := by
  by_cases reached : ∃ steps, steps + extra = clock
  · rcases reached with ⟨steps, rfl⟩
    simpa only [shift, (clock_injective extra).extend_apply] using nonnegative steps
  · simp only [shift, Function.extend_apply' _ _ _ reached, Pi.zero_apply, le_refl]

private theorem weighted_shift (extra : Nat) (mass : Nat → ℝ) (clock : Nat) :
    shift extra mass clock * ((clock : ℝ) + 1) =
      shift extra (fun steps => mass steps * ((steps : ℝ) + 1) +
        mass steps * (extra : ℝ)) clock := by
  by_cases reached : ∃ steps, steps + extra = clock
  · rcases reached with ⟨steps, rfl⟩
    simp only [shift, (clock_injective extra).extend_apply, Nat.cast_add]
    ring
  · simp only [shift, Function.extend_apply' _ _ _ reached, Pi.zero_apply, zero_mul]

variable [Fintype Assignment]

private theorem checked_firstMoment (law : Law Index Challenge Assignment)
    (checker : Checker Index Challenge Assignment)
    (vector : Index → Challenge) (result : Option Assignment) :
    HasSum (fun clock =>
      shift (checkerWork checker vector result) (law.mass vector result) clock *
        ((clock : ℝ) + 1))
      ((∑' steps, law.mass vector result steps * ((steps : ℝ) + 1)) +
        law.oracle.mass vector result * (checkerWork checker vector result : ℝ)) := by
  simp_rw [weighted_shift]
  exact shift_hasSum _ ((law.workSummable vector result).hasSum.add
    ((law.response_hasSum vector result).mul_right (checkerWork checker vector result : ℝ)))

/-- Construct the clock law by running this checker on each actual response.
Only the clock changes; no accepting response or abort is removed. -/
noncomputable def withChecker (law : Law Index Challenge Assignment)
    (checker : Checker Index Challenge Assignment) : Law Index Challenge Assignment where
  mass vector result := shift (checkerWork checker vector result) (law.mass vector result)
  nonnegative vector result := shift_nonnegative _ _ (law.nonnegative vector result)
  summable vector result :=
    (shift_hasSum (checkerWork checker vector result) (law.response_hasSum vector result)).summable
  normalized vector := by
    calc
      (∑ result, ∑' clock,
          shift (checkerWork checker vector result) (law.mass vector result) clock) =
          ∑ result, law.oracle.mass vector result := by
        apply Finset.sum_congr rfl
        intro result _
        exact (shift_hasSum _ (law.response_hasSum vector result)).tsum_eq
      _ = 1 := law.oracle.normalized vector
  workSummable vector result := (checked_firstMoment law checker vector result).summable

/-- Erasing the combined clock gives the original oracle response law. -/
theorem withChecker_response (law : Law Index Challenge Assignment)
    (checker : Checker Index Challenge Assignment)
    (vector : Index → Challenge) (result : Option Assignment) :
    (withChecker law checker).oracle.mass vector result = law.oracle.mass vector result :=
  (shift_hasSum _ (law.response_hasSum vector result)).tsum_eq

/-- The computed query mean contains the exact expected checker work. -/
theorem withChecker_meanWork (law : Law Index Challenge Assignment)
    (checker : Checker Index Challenge Assignment) (vector : Index → Challenge) :
    (withChecker law checker).meanWork vector = law.meanWork vector +
      ∑ result, law.oracle.mass vector result * (checkerWork checker vector result : ℝ) := by
  change (∑ result, ∑' clock,
    shift (checkerWork checker vector result) (law.mass vector result) clock *
      ((clock : ℝ) + 1)) = _
  simp_rw [(checked_firstMoment law checker vector _).tsum_eq]
  rw [Finset.sum_add_distrib]
  rfl

/-- First moment of the program that actually calls the costed checker. This
identifies the provider's mean with execution, not merely a renamed clock. -/
theorem runQuery_work_hasSum (law : Law Index Challenge Assignment)
    (checker : Checker Index Challenge Assignment) (vector : Index → Challenge) :
    HasSum (fun steps => ∑ result, law.mass vector result steps *
      ((runQuery checker vector result steps).2 : ℝ))
      ((withChecker law checker).meanWork vector) := by
  have each : ∀ result,
      HasSum (fun steps => law.mass vector result steps *
        ((runQuery checker vector result steps).2 : ℝ))
      ((∑' steps, law.mass vector result steps * ((steps : ℝ) + 1)) +
        law.oracle.mass vector result * (checkerWork checker vector result : ℝ)) := by
    intro result
    have summed := (law.workSummable vector result).hasSum.add
      ((law.response_hasSum vector result).mul_right (checkerWork checker vector result : ℝ))
    apply summed.congr_fun
    intro steps
    rw [runQuery_eq_queryStep]
    simp only [queryStep, Nat.cast_add, Nat.cast_one]
    ring
  have summed := hasSum_sum (s := Finset.univ) (fun result _ => each result)
  rw [withChecker_meanWork]
  simpa only [Finset.sum_add_distrib, Law.meanWork] using summed

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateCheckedCalls
