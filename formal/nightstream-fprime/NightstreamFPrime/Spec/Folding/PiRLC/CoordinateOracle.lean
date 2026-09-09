import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkProbability
import Mathlib.Data.Fintype.Option

/-!
Distribution of actual oracle-response traces for one interactive coordinate
search. `none` is an oracle failure and remains a counted call. Assignment
probabilities may be arbitrary; only the challenge coordinate is uniform.

The oracle's random coins are marginalized into its response distribution.
No uniform-assignment or nonzero-success premise is used. These definitions
are proof-only and do not construct concrete field enumerations in executables.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateOracle

open scoped BigOperators
open CoordinateRetry CoordinateForkProbability

/-- The response law of an oracle on each challenge vector. Failure is a
response outcome, so normalization does not exclude aborting calls. -/
structure Oracle (Index Challenge Assignment : Type*) [Fintype Assignment] where
  mass : (Index → Challenge) → Option Assignment → ℝ
  nonnegative : ∀ vector result, 0 ≤ mass vector result
  normalized : ∀ vector, ∑ result, mass vector result = 1

variable {Index Challenge Assignment : Type*} [DecidableEq Index]
  [Fintype Challenge] [Nonempty Challenge] [Fintype Assignment]

/-- The verifier checks returned assignments; an oracle failure is rejected. -/
def accepted (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (result : Option Assignment) : Bool :=
  (result.map (check vector)).getD false

/-- Accepted response mass at one challenge vector. -/
noncomputable def responseAcceptedMass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (result : Option Assignment) : ℝ :=
  if accepted check vector result then oracle.mass vector result else 0

/-- The actual oracle acceptance probabilities supplied to the numeric model. -/
noncomputable def line (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) : Line (Index → Challenge) where
  acceptance vector := ∑ result, responseAcceptedMass oracle check vector result
  nonnegative vector := by
    apply Finset.sum_nonneg
    intro result _
    unfold responseAcceptedMass
    split
    · exact oracle.nonnegative vector result
    · exact le_rfl
  atMostOne vector := by
    rw [← oracle.normalized vector]
    apply Finset.sum_le_sum
    intro result _
    unfold responseAcceptedMass
    split
    · exact le_rfl
    · exact oracle.nonnegative vector result

/-- Keep the other coordinates fixed during every call in the search. -/
def callVector (coordinate : Index)
    (rest : {index // index ≠ coordinate} → Challenge) (challenge : Challenge) :
    Index → Challenge :=
  (Equiv.funSplitAt coordinate Challenge).symm (challenge, rest)

/-- One uniform coordinate and its actual response, including oracle failure. -/
abbrev Outcome := Challenge × Option Assignment

noncomputable def callMass (oracle : Oracle Index Challenge Assignment)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) : ℝ :=
  oracle.mass (callVector coordinate rest result.1) result.2 / Fintype.card Challenge

def callAccepted (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) : Bool :=
  accepted check (callVector coordinate rest result.1) result.2

noncomputable def acceptedCallMass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) : ℝ :=
  if callAccepted check coordinate rest result then callMass oracle coordinate rest result else 0

noncomputable def rejectedCallMass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) : ℝ :=
  if callAccepted check coordinate rest result then 0 else callMass oracle coordinate rest result

omit [Nonempty Challenge] in
theorem callMass_nonnegative (oracle : Oracle Index Challenge Assignment)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) :
    0 ≤ callMass oracle coordinate rest result := by
  exact div_nonneg (oracle.nonnegative _ _) (Nat.cast_nonneg _)

theorem callMass_normalized (oracle : Oracle Index Challenge Assignment)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge) :
    (∑ result, callMass oracle coordinate rest result) = 1 := by
  rw [Fintype.sum_prod_type]
  simp only [callMass, ← Finset.sum_div, oracle.normalized]
  have positive : 0 < (Fintype.card Challenge : ℝ) := by
    exact_mod_cast Fintype.card_pos
  simp [positive.ne']

omit [Nonempty Challenge] in
theorem acceptedCallMass_coordinate (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (challenge : Challenge) :
    (∑ result, acceptedCallMass oracle check coordinate rest (challenge, result)) =
      (coordinateLine (line oracle check) coordinate rest).weight challenge := by
  change _ = (∑ result, responseAcceptedMass oracle check
    (callVector coordinate rest challenge) result) / Fintype.card Challenge
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro result _
  unfold acceptedCallMass responseAcceptedMass callAccepted callMass
  split <;> simp_all

omit [Nonempty Challenge] in
theorem acceptedCallMass_total (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge) :
    (∑ result, acceptedCallMass oracle check coordinate rest result) =
      (coordinateLine (line oracle check) coordinate rest).rate := by
  rw [Fintype.sum_prod_type]
  simp_rw [acceptedCallMass_coordinate]
  rfl

theorem rejectedCallMass_total (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge) :
    (∑ result, rejectedCallMass oracle check coordinate rest result) =
      1 - (coordinateLine (line oracle check) coordinate rest).rate := by
  have partition :
      (∑ result, rejectedCallMass oracle check coordinate rest result) +
        (∑ result, acceptedCallMass oracle check coordinate rest result) = 1 := by
    rw [← Finset.sum_add_distrib, ← callMass_normalized oracle coordinate rest]
    apply Finset.sum_congr rfl
    intro result _
    unfold rejectedCallMass acceptedCallMass
    split <;> simp
  rw [acceptedCallMass_total] at partition
  linarith

/-- Density of a concrete first-hit trace: every earlier call is rejected and
the last call is accepted. All calls, including `none`, are retained. -/
noncomputable def traceMass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    {rejections : Nat}
    (before : Fin rejections → Outcome (Challenge := Challenge) (Assignment := Assignment))
    (last : Outcome (Challenge := Challenge) (Assignment := Assignment)) : ℝ :=
  (∏ position, rejectedCallMass oracle check coordinate rest (before position)) *
    acceptedCallMass oracle check coordinate rest last

/-- Sum of all concrete rejected prefixes of this length. This derives the
geometric factor from the actual independent call distribution. -/
theorem traceMass_sum_prefixes (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (rejections : Nat)
    (last : Outcome (Challenge := Challenge) (Assignment := Assignment)) :
    (∑ before : Fin rejections → Outcome (Challenge := Challenge) (Assignment := Assignment),
      traceMass oracle check coordinate rest before last) =
      (1 - (coordinateLine (line oracle check) coordinate rest).rate) ^ rejections *
        acceptedCallMass oracle check coordinate rest last := by
  unfold traceMass
  rw [← Finset.sum_mul, ← Fintype.sum_pow, rejectedCallMass_total]

/-- The previous numeric first-hit law is exactly the sum over actual response
traces with this accepted challenge; assignment probabilities are unchanged. -/
theorem traceMass_sum_assignments (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    (rejections : Nat) (challenge : Challenge) :
    (∑ result, ∑ before : Fin rejections →
        Outcome (Challenge := Challenge) (Assignment := Assignment),
      traceMass oracle check coordinate rest before (challenge, result)) =
      (coordinateLine (line oracle check) coordinate rest).firstHitTerm challenge rejections := by
  simp_rw [traceMass_sum_prefixes]
  rw [← Finset.mul_sum, acceptedCallMass_coordinate]
  rfl

omit [Nonempty Challenge] in
/-- A positive trace density implies the exact first-accepted stopping checks. -/
theorem traceMass_positive_implies_checks (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (coordinate : Index) (rest : {index // index ≠ coordinate} → Challenge)
    {rejections : Nat}
    (before : Fin rejections → Outcome (Challenge := Challenge) (Assignment := Assignment))
    (last : Outcome (Challenge := Challenge) (Assignment := Assignment))
    (positive : 0 < traceMass oracle check coordinate rest before last) :
    (∀ position, callAccepted check coordinate rest (before position) = false) ∧
      callAccepted check coordinate rest last = true := by
  constructor
  · intro position
    by_cases hit : callAccepted check coordinate rest (before position) = true
    · have zero :
          (∏ index, rejectedCallMass oracle check coordinate rest (before index)) = 0 :=
        Finset.prod_eq_zero (Finset.mem_univ position) (by simp [rejectedCallMass, hit])
      simp [traceMass, zero] at positive
    · exact Bool.eq_false_iff.mpr hit
  · by_cases hit : callAccepted check coordinate rest last = true
    · exact hit
    · simp [traceMass, acceptedCallMass, hit] at positive

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateOracle
