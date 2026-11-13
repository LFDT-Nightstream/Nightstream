import NightstreamFPrime.Spec.Compression.Lib.Density
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Polynomial.BigOperators

/-! The lookup of Haböck, ePrint 2022/1530, as Rust `lookup` runs it: the cells `w x` for `x < N`
are values `value t` for `t < T` exactly when, for a random `τ`,
`Σ_x 1/(τ − w x) = Σ_t m t/(τ − value t)`. The difference of the two sides is a sum of fractions
(`residue`, `residuePoly`). If a cell is not a value, its pole has a nonzero residue, so a
challenge that balances the sums is a root of a nonzero polynomial of degree at most `N + T`
(`lookup_root`). `L` independent challenges of the base field all land on roots with density at
most `((N + T) / q)^L` (`density_lookupBad_le`). -/

namespace NightstreamFPrime.Spec.Compression.Lib

open Polynomial

variable {K : Type} [Field K] {ι : Type} [DecidableEq K]

def residue (terms : Finset ι) (a b : ι → K) (v : K) : K :=
  ∑ k ∈ terms.filter (fun k => b k = v), a k

def poles (terms : Finset ι) (b : ι → K) : Finset K := terms.image b

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

theorem fraction_sum_bad [Fintype K] (terms : Finset ι) (a b : ι → K) {v : K}
    (live : residue terms a b v ≠ 0) (β : K) (no_pole : ∀ k ∈ terms, β ≠ b k)
    (zero : ∑ k ∈ terms, a k / (β - b k) = 0) : β ∈ zeros (residuePoly terms a b) := by
  unfold zeros
  rw [ite_eq_right (residuePoly_ne_zero terms a b live)]
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  rw [← fraction_sum_mul terms a b β no_pole, zero, zero_mul]

section Lookup

/-- The terms of a lookup: one for each cell below `N`, then one for each value below `T`. -/
def lookupTerms (N T : ℕ) : Finset (ℕ ⊕ ℕ) := (Finset.range N).disjSum (Finset.range T)

/-- `1` for a cell and `−m t` for value `t`. -/
def lookupNumerator (m : ℕ → K) : ℕ ⊕ ℕ → K
  | .inl _ => 1
  | .inr t => -m t

/-- `w x` for a cell and `value t` for value `t`. -/
def lookupPole (w value : ℕ → K) : ℕ ⊕ ℕ → K
  | .inl x => w x
  | .inr t => value t

/-- The polynomial whose roots hold each bad challenge of the lookup. -/
noncomputable def lookupPoly (N T : ℕ) (w value m : ℕ → K) : K[X] :=
  residuePoly (lookupTerms N T) (lookupNumerator m) (lookupPole w value)

theorem lookupPoly_natDegree (N T : ℕ) (w value m : ℕ → K) :
    (lookupPoly N T w value m).natDegree ≤ N + T := by
  refine (residuePoly_natDegree _ _ _).trans (le_of_eq ?_)
  simp [lookupTerms]

omit [DecidableEq K] in
theorem lookup_fraction_sum (N T : ℕ) (w value m : ℕ → K) (β : K) :
    ∑ k ∈ lookupTerms N T, lookupNumerator m k / (β - lookupPole w value k) =
      ∑ x ∈ Finset.range N, 1 / (β - w x) - ∑ t ∈ Finset.range T, m t / (β - value t) := by
  rw [lookupTerms, Finset.sum_disjSum, sub_eq_add_neg, ← Finset.sum_neg_distrib]
  simp [lookupNumerator, lookupPole, neg_div]

/-- If cell `x` is no value, a challenge that is no pole and balances the two sums is a root of
the lookup polynomial. The counts of equal cells must be nonzero in `K`, as they are when
`N < char K`. -/
theorem lookup_root [Fintype K] (N T : ℕ) (w value m : ℕ → K)
    (count : ∀ n : ℕ, 0 < n → n ≤ N → (n : K) ≠ 0) (x : ℕ) (cell : x < N)
    (outside : ∀ t < T, w x ≠ value t) (β : K)
    (no_pole : (∀ y < N, β ≠ w y) ∧ ∀ t < T, β ≠ value t)
    (balanced : ∑ y ∈ Finset.range N, 1 / (β - w y) = ∑ t ∈ Finset.range T, m t / (β - value t)) :
    β ∈ zeros (lookupPoly N T w value m) := by
  classical
  have residue_eq : residue (lookupTerms N T) (lookupNumerator m) (lookupPole w value) (w x) =
      (((Finset.range N).filter fun y => w y = w x).card : K) := by
    rw [residue, Finset.sum_filter, lookupTerms, Finset.sum_disjSum]
    have values : ∑ t ∈ Finset.range T,
        (if lookupPole w value (.inr t) = w x then lookupNumerator m (.inr t) else 0) = 0 := by
      refine Finset.sum_eq_zero fun t member => ?_
      have : value t ≠ w x := fun same => outside t (Finset.mem_range.mp member) same.symm
      simp [lookupPole, this]
    rw [values, add_zero]
    rw [Finset.natCast_card_filter]
    rfl
  have live : residue (lookupTerms N T) (lookupNumerator m) (lookupPole w value) (w x) ≠ 0 := by
    rw [residue_eq]
    refine count _ (Finset.card_pos.mpr ⟨x, by simp [cell]⟩) ?_
    exact (Finset.card_filter_le _ _).trans (by simp)
  refine fraction_sum_bad _ _ _ live β ?_ ?_
  · rintro (y | t) member
    · exact no_pole.1 y (by simpa [lookupTerms] using member)
    · exact no_pole.2 t (by simpa [lookupTerms] using member)
  · rw [lookup_fraction_sum, balanced, sub_self]

variable [Fintype K]

/-- The bad challenges of `L` lookups of one table: when a cell is no value, each challenge is a
root of the lookup polynomial. -/
noncomputable def lookupBad (L N T : ℕ) (w value m : ℕ → K) : Finset (Fin L → K) := by
  classical
  exact if ∃ x < N, ∀ t < T, w x ≠ value t then
    Finset.univ.filter fun τ => ∀ k, τ k ∈ zeros (lookupPoly N T w value m)
  else ∅

theorem density_lookupBad_le (L N T : ℕ) (w value m : ℕ → K) :
    density (lookupBad L N T w value m) ≤ (((N + T : ℕ) : ℝ) / Fintype.card K) ^ L := by
  classical
  unfold lookupBad
  split
  · rw [density_pi_mem]
    exact pow_le_pow_left₀ (density_nonneg _) (density_zeros_le (lookupPoly_natDegree _ _ _ _ _)) L
  · rw [density_empty]
    positivity

/-- A challenge vector outside the bad set, at which each lookup has no pole and balances, shows
that every cell is a value. -/
theorem lookup_sound (L N T : ℕ) (w value m : ℕ → K)
    (count : ∀ n : ℕ, 0 < n → n ≤ N → (n : K) ≠ 0) (τ : Fin L → K)
    (good : τ ∉ lookupBad L N T w value m)
    (no_pole : ∀ k, (∀ y < N, τ k ≠ w y) ∧ ∀ t < T, τ k ≠ value t)
    (balanced : ∀ k, ∑ y ∈ Finset.range N, 1 / (τ k - w y) =
      ∑ t ∈ Finset.range T, m t / (τ k - value t)) :
    ∀ x < N, ∃ t < T, w x = value t := by
  classical
  by_contra some
  push Not at some
  obtain ⟨x, cell, outside⟩ := some
  apply good
  unfold lookupBad
  rw [ite_eq_left ⟨x, cell, outside⟩]
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  exact fun k => lookup_root N T w value m count x cell outside (τ k) (no_pole k) (balanced k)

end Lookup

end NightstreamFPrime.Spec.Compression.Lib
