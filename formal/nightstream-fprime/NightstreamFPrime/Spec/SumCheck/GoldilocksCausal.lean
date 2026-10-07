import NightstreamFPrime.Spec.SumCheck.GoldilocksRoots
import NightstreamFPrime.Spec.SumCheck.FixedPhase.Sequential
import Mathlib.Algebra.Order.BigOperators.Expect
import Mathlib.Data.Real.Basic

/-!
Owns the bad challenge of one SumCheck round over the actual `K` field.
`Strategy` chooses each message from the earlier challenges only; `none` is a
prover abort. A round `Hit`s when its message differs from the semantic round
polynomial but agrees with it at the challenge. `hit_probability_le` bounds
that chance by `degree / |samples|` for any finite sample set.
-/

namespace NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal

open scoped BigOperators
open GoldilocksRoots (ops)
attribute [local instance] Classical.propDecidable

/-- Each message is chosen from past challenges only. None is prover abort. -/
abbrev Strategy (degree : Nat) := List K → Option (FixedPolynomial K degree)

/-- The semantic round is fixed by the prefix and remaining dimension. -/
def expected (q : List K → K) (fixed : List K) (remaining : Nat) (point : K) : K :=
  HypercubeTruth.sumCompletions ops q (fixed ++ [point]) remaining

def Hit {degree : Nat} (q : List K → K) (fixed : List K) (remaining : Nat)
    (message : FixedPolynomial K degree) (challenge : K) : Prop :=
  (∃ point, message.evaluate ops point ≠ expected q fixed remaining point) ∧
    message.evaluate ops challenge = expected q fixed remaining challenge

private theorem hit_count_le {degree : Nat} (samples : Finset K) (q : List K → K)
    (fixed : List K) (remaining : Nat) (message semantic : FixedPolynomial K degree)
    (represents : FixedPhase.Represents ops semantic (expected q fixed remaining)) :
    (samples.filter (Hit q fixed remaining message)).card ≤ degree := by
  classical
  by_cases different : ∃ point, message.evaluate ops point ≠ expected q fixed remaining point
  · have distinct : ∃ point, message.evaluate ops point ≠ semantic.evaluate ops point := by
      obtain ⟨point, unequal⟩ := different
      exact ⟨point, by simpa only [represents point] using unequal⟩
    have subset : samples.filter (Hit q fixed remaining message) ⊆
        samples.filter (fun point => message.evaluate ops point = semantic.evaluate ops point) := by
      intro point member
      obtain ⟨inside, hit⟩ := Finset.mem_filter.mp member
      exact Finset.mem_filter.mpr ⟨inside, hit.2.trans (represents point).symm⟩
    exact (Finset.card_le_card subset).trans
      (GoldilocksRoots.agreement_count_le message semantic samples distinct)
  · have empty : samples.filter (Hit q fixed remaining message) = ∅ := by
      apply Finset.eq_empty_iff_forall_notMem.mpr
      intro point member
      exact different (Finset.mem_filter.mp member).2.1
    simp only [empty, Finset.card_empty]
    exact Nat.zero_le degree

theorem hit_probability_le {degree : Nat} (samples : Finset K) (q : List K → K)
    (fixed : List K) (remaining : Nat) (message semantic : FixedPolynomial K degree)
    (represents : FixedPhase.Represents ops semantic (expected q fixed remaining)) :
    (𝔼 challenge ∈ samples,
      if Hit q fixed remaining message challenge then (1 : ℝ) else 0) ≤
        (degree : ℝ) / samples.card := by
  classical
  rw [Finset.expect_eq_sum_div_card, ← Finset.sum_filter]
  simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
  apply div_le_div_of_nonneg_right _ (Nat.cast_nonneg _)
  exact_mod_cast hit_count_le samples q fixed remaining message semantic represents

end NightstreamFPrime.Spec.SumCheck.Finite.GoldilocksCausal
