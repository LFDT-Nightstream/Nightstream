import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
One interactive coordinate retry in SuperNeo v1.1 Appendix B.3.

Fix all other coordinates. A call samples this coordinate uniformly and uses
fresh oracle coins. After an accepted base call, retry until the first accepted
response. A repeated base challenge is extraction failure, not a valid fork.
The first-hit law below counts every retry, including rejected responses.

The cost is unconditional: the retry loop is entered only after base acceptance.
Its conditional expected call count is the reciprocal acceptance probability.
This module proves the per-coordinate loss and cost used by the coordinate-fork
extractor. It does not identify the bounded Poseidon sampler with uniform coins.
The cited primary result is Fenzi--Moghaddas--Nguyen, Lemma 7.1,
https://doi.org/10.1007/s00145-024-09511-8 (specialized to two transcripts).
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateRetry

open scoped BigOperators
open Filter

variable {Challenge : Type*} [Fintype Challenge] [Nonempty Challenge]

/-- Acceptance probability of each challenge, with all other coordinates fixed.
Randomness of the oracle response and its verifier is included in this value. -/
structure Line (Challenge : Type*) where
  acceptance : Challenge → ℝ
  nonnegative : ∀ challenge, 0 ≤ acceptance challenge
  atMostOne : ∀ challenge, acceptance challenge ≤ 1

namespace Line

/-- Joint mass of one uniform challenge and an accepted response. -/
noncomputable def weight (line : Line Challenge) (challenge : Challenge) : ℝ :=
  line.acceptance challenge / Fintype.card Challenge

/-- Probability that one uniform oracle call is accepted. -/
noncomputable def rate (line : Line Challenge) : ℝ :=
  ∑ challenge, line.weight challenge

private theorem card_pos : 0 < (Fintype.card Challenge : ℝ) := by
  exact_mod_cast Fintype.card_pos

theorem weight_nonnegative (line : Line Challenge) (challenge : Challenge) :
    0 ≤ line.weight challenge :=
  div_nonneg (line.nonnegative challenge) card_pos.le

theorem weight_le_inverse (line : Line Challenge) (challenge : Challenge) :
    line.weight challenge ≤ 1 / Fintype.card Challenge :=
  div_le_div_of_nonneg_right (line.atMostOne challenge) card_pos.le

theorem rate_nonnegative (line : Line Challenge) : 0 ≤ line.rate := by
  exact Finset.sum_nonneg fun challenge _ => line.weight_nonnegative challenge

theorem rate_le_one (line : Line Challenge) : line.rate ≤ 1 := by
  calc
    line.rate ≤ ∑ _ : Challenge, (1 : ℝ) / Fintype.card Challenge :=
      Finset.sum_le_sum fun challenge _ => line.weight_le_inverse challenge
    _ = 1 := by simp [card_pos.ne']

theorem weight_le_rate (line : Line Challenge) (challenge : Challenge) :
    line.weight challenge ≤ line.rate := by
  exact Finset.single_le_sum (fun value _ => line.weight_nonnegative value)
    (Finset.mem_univ challenge)

/-- Mass of first acceptance at this challenge after exactly `rejections`
failed independent calls. -/
noncomputable def firstHitTerm (line : Line Challenge)
    (challenge : Challenge) (rejections : Nat) : ℝ :=
  (1 - line.rate) ^ rejections * line.weight challenge

/-- Mass that a retry call is made: the base was accepted and all prior retry
calls failed. Summing these tails counts oracle calls, not accepted outputs. -/
noncomputable def queryTail (line : Line Challenge) (priorCalls : Nat) : ℝ :=
  line.rate * (1 - line.rate) ^ priorCalls

/-- Accepted base calls whose retry loop has not terminated after `calls`.
This is explicit exhaustion of a finite observation prefix. -/
noncomputable def exhaustionMass (line : Line Challenge) (calls : Nat) : ℝ :=
  line.rate * (1 - line.rate) ^ calls

theorem firstHitTerm_nonnegative (line : Line Challenge)
    (challenge : Challenge) (rejections : Nat) :
    0 ≤ line.firstHitTerm challenge rejections := by
  exact mul_nonneg (pow_nonneg (sub_nonneg.mpr line.rate_le_one) _)
    (line.weight_nonnegative challenge)

theorem firstHitTerm_hasSum (line : Line Challenge) (positive : 0 < line.rate)
    (challenge : Challenge) :
    HasSum (line.firstHitTerm challenge)
      (line.weight challenge / line.rate) := by
  have geometric := hasSum_geometric_of_lt_one
    (sub_nonneg.mpr line.rate_le_one) (sub_lt_self 1 positive)
  simpa [firstHitTerm, sub_sub_cancel, div_eq_mul_inv, mul_comm] using
    geometric.mul_right (line.weight challenge)

/-- The retry loop's conditional expected call count is `1 / rate`. -/
theorem conditional_calls_hasSum (line : Line Challenge)
    (positive : 0 < line.rate) :
    HasSum (fun priorCalls : Nat => (1 - line.rate) ^ priorCalls)
      (1 / line.rate) := by
  simpa only [sub_sub_cancel, one_div] using
    hasSum_geometric_of_lt_one (sub_nonneg.mpr line.rate_le_one)
      (sub_lt_self 1 positive)

/-- Base acceptance cancels the reciprocal retry cost. This includes every
rejected oracle response before the first accepted response. -/
theorem entered_calls_hasSum (line : Line Challenge)
    (positive : 0 < line.rate) : HasSum line.queryTail 1 := by
  simpa [queryTail, div_eq_mul_inv, positive.ne'] using
    (line.conditional_calls_hasSum positive).mul_left line.rate

omit [Nonempty Challenge] in
theorem zero_rate_hasSum (line : Line Challenge) (zero : line.rate = 0) :
    HasSum line.queryTail 0 := by
  change HasSum (fun priorCalls : Nat => line.rate * (1 - line.rate) ^ priorCalls) 0
  simpa only [zero, zero_mul] using
    (hasSum_zero : HasSum (fun _ : Nat => (0 : ℝ)) 0)

/-- The unconditional expected coordinate retry count is at most one, also
when no challenge can produce an accepted response. -/
theorem expected_calls_le_one (line : Line Challenge) :
    ∃ expected : ℝ, HasSum line.queryTail expected ∧ expected ≤ 1 := by
  by_cases zero : line.rate = 0
  · exact ⟨0, line.zero_rate_hasSum zero, by norm_num⟩
  · exact ⟨1, line.entered_calls_hasSum
      (lt_of_le_of_ne line.rate_nonnegative (Ne.symm zero)), le_rfl⟩

/-- If each fresh retry has the same expected work `work`, including oracle
and verification work, base-weighted expected retry work equals `work`. -/
theorem entered_work_hasSum (line : Line Challenge)
    (positive : 0 < line.rate) (work : ℝ) :
    HasSum (fun priorCalls => line.queryTail priorCalls * work) work := by
  simpa using (line.entered_calls_hasSum positive).mul_right work

theorem exhaustion_tendsTo_zero (line : Line Challenge) :
    Tendsto line.exhaustionMass atTop (nhds 0) := by
  by_cases zero : line.rate = 0
  · change Tendsto (fun calls : Nat => line.rate * (1 - line.rate) ^ calls)
      atTop (nhds 0)
    simpa only [zero, zero_mul] using
      (tendsto_const_nhds : Tendsto (fun _ : Nat => (0 : ℝ)) atTop (nhds 0))
  · have positive : 0 < line.rate :=
      lt_of_le_of_ne line.rate_nonnegative (Ne.symm zero)
    simpa [exhaustionMass] using
      (tendsto_pow_atTop_nhds_zero_of_lt_one
        (sub_nonneg.mpr line.rate_le_one) (sub_lt_self 1 positive)).const_mul line.rate

/-- Joint mass of accepted base and a first accepted retry at that same
challenge. Fresh oracle coins may give different answers at this challenge. -/
noncomputable def repeatedBaseMass (line : Line Challenge) : ℝ :=
  (∑ challenge, line.weight challenge * line.weight challenge) / line.rate

/-- Joint mass of accepted base and a first accepted retry at another challenge.
This is the successful coordinate-fork event. -/
noncomputable def distinctMass (line : Line Challenge) : ℝ :=
  (∑ challenge, line.weight challenge * (line.rate - line.weight challenge)) /
    line.rate

theorem repeatedBaseMass_le_inverse (line : Line Challenge)
    (positive : 0 < line.rate) :
    line.repeatedBaseMass ≤ 1 / Fintype.card Challenge := by
  have squares :
      (∑ challenge, line.weight challenge * line.weight challenge) ≤
        line.rate * (1 / Fintype.card Challenge) := by
    calc
      (∑ challenge, line.weight challenge * line.weight challenge) ≤
          ∑ challenge, line.weight challenge * (1 / Fintype.card Challenge) :=
        Finset.sum_le_sum fun challenge _ => mul_le_mul_of_nonneg_left
          (line.weight_le_inverse challenge) (line.weight_nonnegative challenge)
      _ = line.rate * (1 / Fintype.card Challenge) := by
        rw [← Finset.sum_mul]
        rfl
  calc
    line.repeatedBaseMass ≤
        (line.rate * (1 / Fintype.card Challenge)) / line.rate :=
      div_le_div_of_nonneg_right squares positive.le
    _ = 1 / Fintype.card Challenge := by field_simp

theorem distinctMass_nonnegative (line : Line Challenge) :
    0 ≤ line.distinctMass := by
  apply div_nonneg _ line.rate_nonnegative
  exact Finset.sum_nonneg fun challenge _ => mul_nonneg
    (line.weight_nonnegative challenge)
    (sub_nonneg.mpr (line.weight_le_rate challenge))

omit [Nonempty Challenge] in
/-- The only loss after an accepted base is repetition of its challenge;
the infinite retry loop has no positive exhaustion mass. -/
theorem distinct_add_repeated (line : Line Challenge)
    (positive : 0 < line.rate) :
    line.distinctMass + line.repeatedBaseMass = line.rate := by
  unfold distinctMass repeatedBaseMass
  rw [← add_div, ← Finset.sum_add_distrib]
  have pointwise :
      (∑ challenge, (line.weight challenge * (line.rate - line.weight challenge) +
        line.weight challenge * line.weight challenge)) = line.rate * line.rate := by
    calc
      _ = ∑ challenge, line.weight challenge * line.rate := by
        apply Finset.sum_congr rfl
        intro challenge _
        ring
      _ = line.rate * line.rate := by rw [← Finset.sum_mul]; rfl
  rw [pointwise]
  exact mul_div_cancel_right₀ line.rate positive.ne'

/-- Per-coordinate loss `1 / |C|` for the uniform interactive experiment.
The acceptance rate is that of the actual probabilistic response oracle. -/
theorem distinctMass_lower_bound (line : Line Challenge)
    (positive : 0 < line.rate) :
    line.rate - 1 / Fintype.card Challenge ≤ line.distinctMass := by
  have partition := line.distinct_add_repeated positive
  have loss := line.repeatedBaseMass_le_inverse positive
  linarith

end Line

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateRetry
