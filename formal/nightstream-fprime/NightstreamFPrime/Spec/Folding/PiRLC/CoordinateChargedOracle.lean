import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateOracleCost

/-!
Clocked calls for the interactive coordinate extractor. The joint law retains
the returned assignment, aborts, and the number of steps used by the oracle and
CE check. An EPT call has a summable first moment; no bound on an individual
call or independence of time from its response is required.

The driver charges one query transition in addition to the observed call work.
This transition performs the challenge request, response dispatch, and branch
of the oracle algorithm. Its implementation must use the efficient uniform
challenge primitive of the interactive experiment. It is not a Poseidon law.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateChargedOracle

open scoped BigOperators
open CoordinateOracle

/-- A completed oracle/verifier call and its observed clock. The mass at clock
`steps` includes both accepting and aborting calls. Summability is the finite
expected-time premise on the actual clocked call, not a free cost estimate. -/
structure Law (Index Challenge Assignment : Type*) [Fintype Assignment] where
  mass : (Index → Challenge) → Option Assignment → Nat → ℝ
  nonnegative : ∀ vector result steps, 0 ≤ mass vector result steps
  summable : ∀ vector result, Summable (mass vector result)
  normalized : ∀ vector, ∑ result, ∑' steps, mass vector result steps = 1
  workSummable : ∀ vector result,
    Summable (fun steps => mass vector result steps * ((steps : ℝ) + 1))

variable {Index Challenge Assignment : Type*} [Fintype Assignment]

/-- Execute the query's stopping check. The clock includes the call work and
the driver transition, even when the oracle returns `none`. -/
def queryStep (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (result : Option Assignment) (steps : Nat) : Bool × Nat :=
  (accepted check vector result, steps + 1)

namespace Law

/-- Forget only the clock. The response distribution used in the existing
fork theorems is derived from this joint law. -/
noncomputable def oracle (law : Law Index Challenge Assignment) :
    Oracle Index Challenge Assignment where
  mass vector result := ∑' steps, law.mass vector result steps
  nonnegative vector result := tsum_nonneg (law.nonnegative vector result)
  normalized := law.normalized

theorem response_hasSum (law : Law Index Challenge Assignment)
    (vector : Index → Challenge) (result : Option Assignment) :
    HasSum (law.mass vector result) (law.oracle.mass vector result) :=
  (law.summable vector result).hasSum

/-- Mean of the actual query-step clock, with all responses retained. -/
noncomputable def meanWork (law : Law Index Challenge Assignment)
    (vector : Index → Challenge) : ℝ :=
  ∑ result, ∑' steps, law.mass vector result steps * ((steps : ℝ) + 1)

theorem meanWork_nonnegative (law : Law Index Challenge Assignment)
    (vector : Index → Challenge) : 0 ≤ law.meanWork vector := by
  apply Finset.sum_nonneg
  intro result _
  exact tsum_nonneg fun steps =>
    mul_nonneg (law.nonnegative vector result steps) (by positivity)

/-- The mean work is the first moment of the executed query transition, not
the number of successful responses or a supplied time constant. -/
theorem queryStep_work_hasSum (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) (vector : Index → Challenge) :
    HasSum (fun steps => ∑ result,
      law.mass vector result steps * ((queryStep check vector result steps).2 : ℝ))
      (law.meanWork vector) := by
  have summed := hasSum_sum (s := Finset.univ)
    (fun result _ => (law.workSummable vector result).hasSum)
  simpa only [queryStep, Nat.cast_add, Nat.cast_one] using summed

variable [DecidableEq Index] [Fintype Challenge]

/-- Clocked form of exactly one uniform coordinate query. -/
noncomputable def callMass (law : Law Index Challenge Assignment)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment))
    (steps : Nat) : ℝ :=
  law.mass (callVector coordinate rest result.1) result.2 steps / Fintype.card Challenge

theorem callMass_nonnegative (law : Law Index Challenge Assignment)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) (steps : Nat) :
    0 ≤ law.callMass coordinate rest result steps :=
  div_nonneg (law.nonnegative _ _ _) (Nat.cast_nonneg _)

/-- Erasing the clock recovers the existing abort-inclusive coordinate law. -/
theorem callMass_hasSum (law : Law Index Challenge Assignment)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) :
    HasSum (law.callMass coordinate rest result)
      (CoordinateOracle.callMass law.oracle coordinate rest result) :=
  (response_hasSum law (callVector coordinate rest result.1) result.2).div_const _

/-- Mean work of one coordinate query, before any acceptance conditioning. -/
noncomputable def lineWork (law : Law Index Challenge Assignment)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge) : ℝ :=
  𝔼 challenge, law.meanWork (callVector coordinate rest challenge)

theorem lineWork_nonnegative (law : Law Index Challenge Assignment)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge) :
    0 ≤ law.lineWork coordinate rest := by
  exact Finset.expect_nonneg fun challenge _ => law.meanWork_nonnegative _

/-- Sum over response values and clocks of this exact uniform query. This
retains correlation between the response, its challenge, and its running time. -/
theorem call_work_hasSum (law : Law Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge) :
    HasSum (fun steps => ∑ result,
      law.callMass coordinate rest result steps *
        ((queryStep check (callVector coordinate rest result.1) result.2 steps).2 : ℝ))
      (law.lineWork coordinate rest) := by
  have summed := hasSum_sum (s := Finset.univ) (fun challenge _ =>
    queryStep_work_hasSum law check (callVector coordinate rest challenge))
  have divided := summed.div_const (Fintype.card Challenge : ℝ)
  simp only [lineWork, Fintype.expect_eq_sum_div_card] at divided ⊢
  convert divided using 1
  funext steps
  rw [Fintype.sum_prod_type, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro challenge _
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro result _
  unfold callMass
  ring

end Law

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateChargedOracle
