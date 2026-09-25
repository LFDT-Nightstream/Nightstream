import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateCheckedCalls

/-!
Terminal distribution of the same coordinate searches, before repeated
challenges are rejected. It retains failure endpoints so terminal validation
work is charged on failures as well as on successful forks. Initial rejection
has a separate one-step failure exit.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalLaw

open scoped BigOperators
open CoordinateRetry CoordinateOracle CoordinateOracleStar

variable {Index Challenge Assignment : Type*} [Fintype Index] [DecidableEq Index]
  [Fintype Challenge] [Nonempty Challenge] [Fintype Assignment]

omit [Fintype Index] [DecidableEq Index] [Fintype Challenge] [Nonempty Challenge] in
private theorem responseMass_nonnegative (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (initial : Option Assignment) :
    0 ≤ responseAcceptedMass oracle check vector initial := by
  unfold responseAcceptedMass
  split
  · exact oracle.nonnegative _ _
  · exact le_rfl

/-- All first-accepted coordinate endpoints, including repeated challenges. -/
noncomputable def endpointMass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (initial : Option Assignment)
    (outputs : Index → Outcome (Challenge := Challenge) (Assignment := Assignment)) : ℝ :=
  responseAcceptedMass oracle check vector initial *
    ∏ coordinate, stoppedMass oracle check vector coordinate (outputs coordinate)

theorem endpointMass_nonnegative (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (initial : Option Assignment)
    (outputs : Index → Outcome (Challenge := Challenge) (Assignment := Assignment)) :
    0 ≤ endpointMass oracle check vector initial outputs :=
  mul_nonneg (responseMass_nonnegative oracle check vector initial)
    (Finset.prod_nonneg fun coordinate _ => stoppedMass_nonnegative oracle check vector coordinate _)

/-- The accepted-base mass reaches a terminal endpoint with total mass one
conditionally. Zero-rate searches are never entered with positive base mass. -/
theorem endpointMass_total (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) (vector : Index → Challenge) :
    (∑ initial, ∑ outputs, endpointMass oracle check vector initial outputs) =
      (line oracle check).acceptance vector := by
  unfold endpointMass
  simp only [← Finset.mul_sum, ← Fintype.prod_sum, ← Finset.sum_mul]
  change (line oracle check).acceptance vector *
    (∏ coordinate, ∑ result, stoppedMass oracle check vector coordinate result) = _
  by_cases zero : (line oracle check).acceptance vector = 0
  · simp [zero]
  · have positive : 0 < (line oracle check).acceptance vector :=
      lt_of_le_of_ne ((line oracle check).nonnegative vector) (Ne.symm zero)
    have total : (∏ coordinate, ∑ result, stoppedMass oracle check vector coordinate result) = 1 := by
      simp_rw [stoppedMass_total oracle check vector _
        (sourceLine_positive oracle check vector _ positive)]
      simp
    rw [total, mul_one]

/-- Expected terminal clock, including the initial failure exit. The clock
argument is the observed work returned by the terminal program. -/
noncomputable def expectedTerminalWork (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (clock : (Index → Challenge) → Option Assignment →
      (Index → Outcome (Challenge := Challenge) (Assignment := Assignment)) → Nat) : ℝ :=
  (1 - (line oracle check).rate) +
    𝔼 vector, ∑ initial, ∑ outputs,
      endpointMass oracle check vector initial outputs * (clock vector initial outputs : ℝ)

/-- A bound on the actual terminal program's clock bounds its mean, including
all failure paths. The explicit one-step rejected-base exit is also covered. -/
theorem expectedTerminalWork_le (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (clock : (Index → Challenge) → Option Assignment →
      (Index → Outcome (Challenge := Challenge) (Assignment := Assignment)) → Nat)
    (bound : Nat) (exitBound : 1 ≤ bound)
    (bounded : ∀ vector initial outputs, clock vector initial outputs ≤ bound) :
    expectedTerminalWork oracle check clock ≤ (bound : ℝ) := by
  have each : ∀ vector,
      (∑ initial, ∑ outputs, endpointMass oracle check vector initial outputs *
        (clock vector initial outputs : ℝ)) ≤
        (line oracle check).acceptance vector * (bound : ℝ) := by
    intro vector
    calc
      _ ≤ ∑ initial, ∑ outputs,
          endpointMass oracle check vector initial outputs * (bound : ℝ) := by
        apply Finset.sum_le_sum
        intro initial _
        apply Finset.sum_le_sum
        intro outputs _
        apply mul_le_mul_of_nonneg_left _ (endpointMass_nonnegative oracle check vector initial outputs)
        exact_mod_cast bounded vector initial outputs
      _ = _ := by
        simp_rw [← Finset.sum_mul]
        rw [endpointMass_total]
  have averaged := Finset.expect_le_expect (s := Finset.univ) (fun vector _ => each vector)
  rw [← Finset.expect_mul] at averaged
  have rate : (𝔼 vector, (line oracle check).acceptance vector) = (line oracle check).rate := by
    simp only [Fintype.expect_eq_sum_div_card, Line.rate, Line.weight, ← Finset.sum_div]
  rw [rate] at averaged
  have boundReal : (1 : ℝ) ≤ (bound : ℝ) := by exact_mod_cast exitBound
  have remaining : 0 ≤ (1 - (line oracle check).rate) * ((bound : ℝ) - 1) :=
    mul_nonneg (sub_nonneg.mpr (Line.rate_le_one _)) (sub_nonneg.mpr boundReal)
  unfold expectedTerminalWork
  nlinarith

/-- Probability that the actual terminal program returns a successful value. -/
noncomputable def returningProbability (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (returns : (Index → Challenge) → Option Assignment →
      (Index → Outcome (Challenge := Challenge) (Assignment := Assignment)) → Bool) : ℝ :=
  𝔼 vector, ∑ initial, ∑ outputs,
    if returns vector initial outputs then endpointMass oracle check vector initial outputs else 0

variable [DecidableEq Challenge]

omit [Nonempty Challenge] in
/-- Distinct terminal coordinates are exactly the successful-fork branch of
the existing stopped-output law. -/
theorem endpointMass_eq_forkMass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (initial : Option Assignment)
    (outputs : Index → Outcome (Challenge := Challenge) (Assignment := Assignment))
    (different : ∀ coordinate, (outputs coordinate).1 ≠ vector coordinate) :
    endpointMass oracle check vector initial outputs =
      outcomeMass oracle check vector initial outputs := by
  unfold endpointMass outcomeMass
  apply congrArg (fun mass : ℝ => responseAcceptedMass oracle check vector initial * mass)
  apply Finset.prod_congr rfl
  intro coordinate _
  unfold returningMass
  rw [if_pos (different coordinate)]

private theorem forkMass_nonnegative (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (initial : Option Assignment)
    (outputs : Index → Outcome (Challenge := Challenge) (Assignment := Assignment)) :
    0 ≤ outcomeMass oracle check vector initial outputs :=
  mul_nonneg (responseMass_nonnegative oracle check vector initial)
    (Finset.prod_nonneg fun coordinate _ => returningMass_nonnegative oracle check vector coordinate _)

private theorem forkMass_le_endpointMass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (initial : Option Assignment)
    (outputs : Index → Outcome (Challenge := Challenge) (Assignment := Assignment)) :
    outcomeMass oracle check vector initial outputs ≤ endpointMass oracle check vector initial outputs := by
  apply mul_le_mul_of_nonneg_left _ (responseMass_nonnegative oracle check vector initial)
  apply Finset.prod_le_prod
  · intro coordinate _
    exact returningMass_nonnegative oracle check vector coordinate _
  · intro coordinate _
    unfold returningMass
    split
    · exact le_rfl
    · exact stoppedMass_nonnegative oracle check vector coordinate _

/-- If every positive successful-fork endpoint makes the terminal program
return, the existing fork probability lower bound applies to that program. -/
theorem returningProbability_lower_bound (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (returns : (Index → Challenge) → Option Assignment →
      (Index → Outcome (Challenge := Challenge) (Assignment := Assignment)) → Bool)
    (returnsOnFork : ∀ vector initial outputs,
      0 < outcomeMass oracle check vector initial outputs → returns vector initial outputs = true) :
    (line oracle check).rate - (Fintype.card Index : ℝ) / Fintype.card Challenge ≤
      returningProbability oracle check returns := by
  apply (successMass_lower_bound oracle check).trans
  apply Finset.expect_le_expect
  intro vector _
  apply Finset.sum_le_sum
  intro initial _
  apply Finset.sum_le_sum
  intro outputs _
  by_cases positive : 0 < outcomeMass oracle check vector initial outputs
  · rw [returnsOnFork vector initial outputs positive]
    exact forkMass_le_endpointMass oracle check vector initial outputs
  · have zero : outcomeMass oracle check vector initial outputs = 0 :=
      le_antisymm (le_of_not_gt positive) (forkMass_nonnegative oracle check vector initial outputs)
    rw [zero]
    split
    · exact endpointMass_nonnegative oracle check vector initial outputs
    · exact le_rfl

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalLaw
