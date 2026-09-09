import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateOracleStar

/-!
Oracle-call expectation of the same abort-inclusive coordinate search traces.
A retry is made exactly when the base was accepted and every earlier retry
was rejected. Summing those probabilities counts all invocations, not just
the accepted responses retained by the extractor.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateOracleCost

open scoped BigOperators
open CoordinateRetry CoordinateForkProbability CoordinateOracle CoordinateOracleStar

variable {Index Challenge Assignment : Type*} [Fintype Index] [DecidableEq Index]
  [Fintype Challenge] [Nonempty Challenge] [Fintype Assignment]

/-- Joint probability that this coordinate's next call is made after this many
rejections. Every possible earlier oracle response is included in the sum. -/
noncomputable def queryTailTerm (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) (priorCalls : Nat) : ℝ :=
  (line oracle check).acceptance vector *
    ∑ before : Fin priorCalls → Outcome (Challenge := Challenge) (Assignment := Assignment),
      ∏ position, rejectedCallMass oracle check coordinate
        (Equiv.funSplitAt coordinate Challenge vector).2 (before position)

omit [Fintype Index] in
theorem queryTailTerm_eq (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) (priorCalls : Nat) :
    queryTailTerm oracle check vector coordinate priorCalls =
      (line oracle check).acceptance vector *
        (1 - (sourceLine oracle check vector coordinate).rate) ^ priorCalls := by
  unfold queryTailTerm
  rw [← Fintype.sum_pow, rejectedCallMass_total]
  rfl

/-- The reciprocal-rate expression is the sum of the actual invocation tails.
The zero-rate case implies zero base-acceptance mass and is proved separately. -/
theorem queryTailTerm_hasSum (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool)
    (vector : Index → Challenge) (coordinate : Index) :
    HasSum (queryTailTerm oracle check vector coordinate)
      ((line oracle check).acceptance vector *
        conditionalCalls (line oracle check) vector coordinate) := by
  change HasSum (fun priorCalls : Nat => queryTailTerm oracle check vector coordinate priorCalls) _
  simp_rw [queryTailTerm_eq]
  by_cases zero : (sourceLine oracle check vector coordinate).rate = 0
  · have noBase : (line oracle check).acceptance vector = 0 := by
      by_contra nonzero
      have positive := sourceLine_positive oracle check vector coordinate
        (lt_of_le_of_ne ((line oracle check).nonnegative vector) (Ne.symm nonzero))
      rw [zero] at positive
      exact (lt_irrefl 0) positive
    simpa only [noBase, zero_mul] using
      (hasSum_zero : HasSum (fun _ : Nat => (0 : ℝ)) 0)
  · have positive : 0 < (sourceLine oracle check vector coordinate).rate :=
      lt_of_le_of_ne (Line.rate_nonnegative _) (Ne.symm zero)
    simpa only [sourceLine, conditionalCalls] using
      ((sourceLine oracle check vector coordinate).conditional_calls_hasSum positive).mul_left
        ((line oracle check).acceptance vector)

theorem coordinateTail_hasSum (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) (coordinate : Index) :
    HasSum (fun priorCalls => 𝔼 vector, queryTailTerm oracle check vector coordinate priorCalls)
      (𝔼 vector, (line oracle check).acceptance vector *
        conditionalCalls (line oracle check) vector coordinate) := by
  have summed := hasSum_sum (s := Finset.univ)
    (fun vector _ => queryTailTerm_hasSum oracle check vector coordinate)
  simpa only [Fintype.expect_eq_sum_div_card] using
    summed.div_const (Fintype.card (Index → Challenge) : ℝ)

/-- Sum of all coordinate invocation tails after the one initial oracle call. -/
noncomputable def invocationTail (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) (priorCalls : Nat) : ℝ :=
  ∑ coordinate, 𝔼 vector, queryTailTerm oracle check vector coordinate priorCalls

theorem invocationTail_hasSum (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) :
    HasSum (invocationTail oracle check)
      (expectedOracleCalls (line oracle check) - 1) := by
  have summed := hasSum_sum (s := Finset.univ)
    (fun coordinate _ => coordinateTail_hasSum oracle check coordinate)
  have total :
      (∑ coordinate, 𝔼 vector, (line oracle check).acceptance vector *
        conditionalCalls (line oracle check) vector coordinate) =
        expectedOracleCalls (line oracle check) - 1 := by
    unfold expectedOracleCalls
    ring
  rw [← total]
  exact summed

/-- The actual trace-law expectation, including the first oracle call and every
failed retry, satisfies the paper's unconditional `ℓ + 1` bound. -/
theorem oracleCalls_bound (oracle : Oracle Index Challenge Assignment)
    (check : (Index → Challenge) → Assignment → Bool) :
    1 + ∑' priorCalls : Nat, invocationTail oracle check priorCalls ≤
      (Fintype.card Index : ℝ) + 1 := by
  rw [(invocationTail_hasSum oracle check).tsum_eq]
  have bound := expectedOracleCalls_le (line oracle check)
  linarith

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateOracleCost
