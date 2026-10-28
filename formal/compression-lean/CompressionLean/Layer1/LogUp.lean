import CompressionLean.Layer1.Roots
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Polynomial.BigOperators

/-!
Owns the pole argument of logUp (Haböck, ePrint 2022/1530): a fraction sum
`Σ_k a_k / (X - b_k)` that is zero at a challenge `β` off its poles, while
the numerators at some pole do not cancel, makes `β` a root of a nonzero
polynomial of degree below the number of terms (`fraction_sum_bad`).

Both sides of a logUp identity are one sum here: the right side enters with
negated numerators. The numerators grouped at a pole are `residue`.
-/

namespace CompressionLean.Layer1

open Polynomial

variable {K : Type} [Field K] {ι : Type} [DecidableEq K]

/-- The numerators of the terms whose pole is `v`. -/
def residue (terms : Finset ι) (a b : ι → K) (v : K) : K :=
  ∑ k ∈ terms.filter (fun k => b k = v), a k

/-- The distinct poles of the terms. -/
def poles (terms : Finset ι) (b : ι → K) : Finset K := terms.image b

/-- `Σ_v residue(v)·Π_{v' ≠ v} (X - v')` over the distinct poles. -/
noncomputable def residuePoly (terms : Finset ι) (a b : ι → K) : K[X] :=
  ∑ v ∈ poles terms b, C (residue terms a b v) * ∏ w ∈ (poles terms b).erase v, (X - C w)

theorem residuePoly_natDegree (terms : Finset ι) (a b : ι → K) :
    (residuePoly terms a b).natDegree ≤ terms.card := by
  unfold residuePoly
  apply Polynomial.natDegree_sum_le_of_forall_le
  intro v member
  refine (natDegree_C_mul_le _ _).trans ?_
  refine (Polynomial.natDegree_prod_le _ _).trans ?_
  calc ∑ w ∈ (poles terms b).erase v, (X - C w).natDegree
      ≤ ∑ _w ∈ (poles terms b).erase v, 1 := by
        apply Finset.sum_le_sum
        intro w _
        exact natDegree_X_sub_C_le w
    _ = ((poles terms b).erase v).card := by simp
    _ ≤ (poles terms b).card := Finset.card_erase_le
    _ ≤ terms.card := Finset.card_image_le

/-- At a pole, only that pole's term survives. -/
theorem residuePoly_eval_pole (terms : Finset ι) (a b : ι → K) {v : K} (member : v ∈ poles terms b) :
    (residuePoly terms a b).eval v =
      residue terms a b v * ∏ w ∈ (poles terms b).erase v, (v - w) := by
  unfold residuePoly
  rw [eval_finsetSum, Finset.sum_eq_single v]
  · simp [eval_prod]
  · intro u _ different
    rw [eval_mul, eval_prod]
    apply mul_eq_zero_of_right
    exact Finset.prod_eq_zero (Finset.mem_erase.mpr ⟨fun h => different h.symm, member⟩) (by simp)
  · exact fun outside => absurd member outside

/-- A pole whose numerators do not cancel makes the polynomial nonzero. -/
theorem residuePoly_ne_zero (terms : Finset ι) (a b : ι → K) {v : K}
    (live : residue terms a b v ≠ 0) : residuePoly terms a b ≠ 0 := by
  have member : v ∈ poles terms b := by
    by_contra outside
    apply live
    unfold residue
    apply Finset.sum_eq_zero
    intro k inside
    exact absurd (Finset.mem_image.mpr ⟨k, (Finset.mem_filter.mp inside).1,
      (Finset.mem_filter.mp inside).2⟩) outside
  intro zero
  have value := residuePoly_eval_pole terms a b member
  rw [zero, eval_zero] at value
  refine (mul_ne_zero live ?_) value.symm
  exact Finset.prod_ne_zero_iff.mpr fun w inside =>
    sub_ne_zero.mpr fun equal => (Finset.mem_erase.mp inside).1 equal.symm

/-- Off the poles, the fraction sum times the pole product is the polynomial. -/
theorem fraction_sum_mul (terms : Finset ι) (a b : ι → K) (β : K)
    (no_pole : ∀ k ∈ terms, β ≠ b k) :
    (∑ k ∈ terms, a k / (β - b k)) * ∏ w ∈ poles terms b, (β - w) =
      (residuePoly terms a b).eval β := by
  have grouped : ∑ k ∈ terms, a k / (β - b k) =
      ∑ v ∈ poles terms b, residue terms a b v / (β - v) := by
    unfold residue poles
    rw [Finset.sum_image' (fun k => a k / (β - b k))]
    intro v member
    rw [Finset.sum_div]
    apply Finset.sum_congr rfl
    intro k inside
    rw [(Finset.mem_filter.mp inside).2]
  rw [grouped, Finset.sum_mul]
  unfold residuePoly
  rw [eval_finsetSum]
  apply Finset.sum_congr rfl
  intro v member
  have nonzero : β - v ≠ 0 := by
    obtain ⟨k, inside, rfl⟩ := Finset.mem_image.mp member
    exact sub_ne_zero.mpr (no_pole k inside)
  rw [← Finset.mul_prod_erase _ _ member, eval_mul, eval_C, eval_prod]
  simp only [eval_sub, eval_X, eval_C]
  rw [div_mul_eq_mul_div, mul_div_assoc, mul_comm (β - v), mul_div_assoc, div_self nonzero, mul_one]

/-- The pole argument: a zero fraction sum at `β` off its poles, with a live
pole, puts `β` among at most `terms.card` bad challenges. -/
theorem fraction_sum_bad [Fintype K] (terms : Finset ι) (a b : ι → K) {v : K}
    (live : residue terms a b v ≠ 0) (β : K) (no_pole : ∀ k ∈ terms, β ≠ b k)
    (zero : ∑ k ∈ terms, a k / (β - b k) = 0) : β ∈ zeros (residuePoly terms a b) := by
  unfold zeros
  rw [if_neg (residuePoly_ne_zero terms a b live)]
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  rw [← fraction_sum_mul terms a b β no_pole, zero, zero_mul]

end CompressionLean.Layer1
