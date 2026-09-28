import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateOracle

/-!
Joint stopped-output law for interactive coordinate extraction. Every query
restarts the same fixed oracle context and uses fresh coins. Only verifier
challenges are uniform; response probabilities and aborts are unchanged.

The first-hit output masses are derived from complete rejected-prefix traces.
Their product describes independent coordinate searches after an accepted base.
Zero-rate lines receive no accepted-base mass and require no success premise.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateOracleStar

open scoped BigOperators
open CoordinateRetry CoordinateForkProbability CoordinateOracle

variable {Index Challenge Assignment : Type*} [DecidableEq Index]
  [Fintype Challenge] [Nonempty Challenge] [Fintype Assignment]

noncomputable def sourceLine (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) : Line Challenge :=
  coordinateLine (line oracle check) coordinate
    (Equiv.funSplitAt coordinate Challenge vector).2

/-- Distribution of the final accepted response in one coordinate search. -/
noncomputable def stoppedMass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) : ℝ :=
  acceptedCallMass oracle check coordinate (Equiv.funSplitAt coordinate Challenge vector).2 result /
    (sourceLine oracle check vector coordinate).rate

theorem stoppedMass_nonnegative (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) :
    0 ≤ stoppedMass oracle check vector coordinate result := by
  apply div_nonneg _ (Line.rate_nonnegative _)
  unfold acceptedCallMass
  split
  · exact callMass_nonnegative oracle coordinate _ result
  · exact le_rfl

/-- The output law is the sum over all finite stopping traces, not a supplied
fork-success probability. Positivity is required only for this entered line. -/
theorem stoppedMass_hasSum (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index)
    (positive : 0 < (sourceLine oracle check vector coordinate).rate)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) :
    HasSum (fun rejections : Nat =>
      ∑ before : Fin rejections → Outcome (Challenge := Challenge) (Assignment := Assignment),
        traceMass oracle check coordinate (Equiv.funSplitAt coordinate Challenge vector).2
          before result) (stoppedMass oracle check vector coordinate result) := by
  simp_rw [traceMass_sum_prefixes]
  have geometric := (sourceLine oracle check vector coordinate).conditional_calls_hasSum positive
  simpa [stoppedMass, sourceLine, div_eq_mul_inv, mul_comm] using
    geometric.mul_right (acceptedCallMass oracle check coordinate
      (Equiv.funSplitAt coordinate Challenge vector).2 result)

omit [Nonempty Challenge] in
theorem stoppedMass_total (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index)
    (positive : 0 < (sourceLine oracle check vector coordinate).rate) :
    (∑ result, stoppedMass oracle check vector coordinate result) = 1 := by
  unfold stoppedMass
  rw [← Finset.sum_div, acceptedCallMass_total]
  exact div_self positive.ne'

omit [Nonempty Challenge] in
theorem stoppedMass_coordinate (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) (challenge : Challenge) :
    (∑ result, stoppedMass oracle check vector coordinate (challenge, result)) =
      (sourceLine oracle check vector coordinate).weight challenge /
        (sourceLine oracle check vector coordinate).rate := by
  unfold stoppedMass
  rw [← Finset.sum_div, acceptedCallMass_coordinate]
  rfl

/-- A positive chance of accepting this base forces every coordinate search
that it enters to have a positive rate. This is derived, not assumed. -/
theorem sourceLine_positive (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index)
    (positive : 0 < (line oracle check).acceptance vector) :
    0 < (sourceLine oracle check vector coordinate).rate := by
  have current : (sourceLine oracle check vector coordinate).weight (vector coordinate) =
      (line oracle check).acceptance vector / Fintype.card Challenge := by
    change (line oracle check).acceptance
      ((Equiv.funSplitAt coordinate Challenge).symm
        (Equiv.funSplitAt coordinate Challenge vector)) / _ = _
    rw [Equiv.symm_apply_apply]
  have cardinality : 0 < (Fintype.card Challenge : ℝ) := by
    exact_mod_cast Fintype.card_pos
  have positiveWeight : 0 < (sourceLine oracle check vector coordinate).weight
      (vector coordinate) := by
    rw [current]
    exact div_pos positive cardinality
  exact lt_of_lt_of_le positiveWeight (Line.weight_le_rate _ _)

variable [DecidableEq Challenge]

/-- Accepted first-hit responses at a different challenge form a fork. -/
noncomputable def returningMass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) : ℝ :=
  if result.1 ≠ vector coordinate then stoppedMass oracle check vector coordinate result else 0

theorem returningMass_nonnegative (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index)
    (result : Outcome (Challenge := Challenge) (Assignment := Assignment)) :
    0 ≤ returningMass oracle check vector coordinate result := by
  unfold returningMass
  split
  · exact stoppedMass_nonnegative oracle check vector coordinate result
  · exact le_rfl

omit [Nonempty Challenge] in
theorem returningMass_total (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index)
    (positive : 0 < (sourceLine oracle check vector coordinate).rate) :
    (∑ result, returningMass oracle check vector coordinate result) =
      1 - repeatChance (line oracle check) vector coordinate := by
  have partition :
      (∑ result, returningMass oracle check vector coordinate result) +
        (∑ assignment, stoppedMass oracle check vector coordinate
          (vector coordinate, assignment)) =
        ∑ result, stoppedMass oracle check vector coordinate result := by
    rw [Fintype.sum_prod_type, Fintype.sum_prod_type]
    have same :
        (∑ assignment, stoppedMass oracle check vector coordinate (vector coordinate, assignment)) =
          ∑ challenge, (if challenge = vector coordinate then
            ∑ assignment, stoppedMass oracle check vector coordinate (challenge, assignment)
            else 0) := by simp
    rw [same, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro challenge _
    by_cases equal : challenge = vector coordinate
    · simp [returningMass, equal]
    · simp [returningMass, equal]
  rw [stoppedMass_total oracle check vector coordinate positive,
    stoppedMass_coordinate] at partition
  change _ + repeatChance (line oracle check) vector coordinate = 1 at partition
  linarith

section Joint

variable [Fintype Index]

/-- Joint density conditioned on the base challenge vector. The initial oracle
outcome and every stopped coordinate outcome remain explicit. -/
noncomputable def outcomeMass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (initial : Option Assignment)
    (outputs : Index → Outcome (Challenge := Challenge) (Assignment := Assignment)) : ℝ :=
  responseAcceptedMass oracle check vector initial *
    ∏ coordinate, returningMass oracle check vector coordinate (outputs coordinate)

noncomputable def successMass (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) : ℝ :=
  𝔼 vector, ∑ initial, ∑ outputs, outcomeMass oracle check vector initial outputs

/-- Sum over actual response assignments gives the previously checked numeric
kernel. The zero-success branch vanishes before division by any line rate. -/
theorem outcomeMass_total (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) (vector : Index → Challenge) :
    (∑ initial, ∑ outputs, outcomeMass oracle check vector initial outputs) =
      (line oracle check).acceptance vector *
        ∏ coordinate, (1 - repeatChance (line oracle check) vector coordinate) := by
  unfold outcomeMass
  simp only [← Finset.mul_sum, ← Fintype.prod_sum, ← Finset.sum_mul]
  change (line oracle check).acceptance vector *
    (∏ coordinate, ∑ result, returningMass oracle check vector coordinate result) = _
  by_cases zero : (line oracle check).acceptance vector = 0
  · simp [zero]
  · have positive : 0 < (line oracle check).acceptance vector :=
      lt_of_le_of_ne ((line oracle check).nonnegative vector) (Ne.symm zero)
    congr 1
    apply Finset.prod_congr rfl
    intro coordinate _
    exact returningMass_total oracle check vector coordinate
      (sourceLine_positive oracle check vector coordinate positive)

theorem successMass_eq_probability (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) :
    successMass oracle check = successProbability (line oracle check) := by
  unfold successMass successProbability
  apply Finset.expect_congr rfl
  intro vector _
  exact outcomeMass_total oracle check vector

/-- The loss bound now applies to the actual stopped-response distribution. -/
theorem successMass_lower_bound (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) :
    (line oracle check).rate - (Fintype.card Index : ℝ) / Fintype.card Challenge ≤
      successMass oracle check := by
  rw [successMass_eq_probability]
  exact successProbability_lower_bound _

end Joint

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateOracleStar
