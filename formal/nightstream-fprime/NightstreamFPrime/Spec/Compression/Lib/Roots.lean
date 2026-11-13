import Mathlib.Algebra.Polynomial.Roots
import Mathlib.FieldTheory.Finite.Basic

/-! Roots of univariate polynomials: the roots of a nonzero polynomial (`zeros`) and of a line
(`linePoly`), and the roots of `ω^(2^k) = −1`, which give the challenges `ω` without a row point
(Rust `key::row_point`, `rowPoint_degenerate_card_le`). -/

namespace NightstreamFPrime.Spec.Compression.Lib

open Polynomial

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

noncomputable def zeros (p : K[X]) : Finset K :=
  if p = 0 then ∅ else Finset.univ.filter (fun c => p.eval c = 0)

theorem zeros_card_le (p : K[X]) : (zeros p).card ≤ p.natDegree := by
  unfold zeros
  split
  · simp
  · rename_i nonzero
    apply Polynomial.card_le_degree_of_subset_roots
    intro c inside
    simp only [Finset.mem_val, Finset.mem_filter] at inside
    exact (Polynomial.mem_roots nonzero).2 inside.2

theorem eval_ne_of_not_mem_zeros {p q : K[X]} (different : p ≠ q) {c : K}
    (outside : c ∉ zeros (p - q)) : p.eval c ≠ q.eval c := by
  intro equal
  apply outside
  unfold zeros
  rw [ite_eq_right (sub_ne_zero.mpr different)]
  simp [equal]

theorem zeros_card_le_of_natDegree_le {p : K[X]} {d : ℕ} (degree : p.natDegree ≤ d) :
    (zeros p).card ≤ d :=
  (zeros_card_le p).trans degree

omit [Field K] [Fintype K] in
theorem card_biUnion_le_mul {W : Type} (worlds : Finset W) (bad : W → Finset K) (d : ℕ)
    (each : ∀ w ∈ worlds, (bad w).card ≤ d) :
    (worlds.biUnion bad).card ≤ worlds.card * d := by
  calc (worlds.biUnion bad).card ≤ ∑ w ∈ worlds, (bad w).card := Finset.card_biUnion_le
    _ ≤ ∑ _w ∈ worlds, d := Finset.sum_le_sum each
    _ = worlds.card * d := by simp

section Line

omit [Fintype K] [DecidableEq K]

noncomputable def linePoly (a b : K) : K[X] := C a + X * C b

theorem linePoly_eval (a b c : K) : (linePoly a b).eval c = a + c * b := by
  simp only [linePoly, eval_add, eval_mul, eval_C, eval_X]

theorem linePoly_natDegree (a b : K) : (linePoly a b).natDegree ≤ 1 := by
  unfold linePoly
  refine (natDegree_add_le _ _).trans (max_le (by simp) ?_)
  exact (natDegree_mul_le).trans (by simp)

theorem linePoly_ne_zero {a b : K} (nonzero : a ≠ 0 ∨ b ≠ 0) : linePoly a b ≠ 0 := by
  intro zero
  have zeroth := congrArg (fun p => p.coeff 0) zero
  have first := congrArg (fun p => p.coeff 1) zero
  simp [linePoly, coeff_C] at zeroth first
  rcases nonzero with h | h
  · exact h zeroth
  · exact h first

end Line

section TwoPower

/-- `ω^(2^k) = −1` has at most `2^k` roots. -/
theorem twoPow_roots_card_le (k : ℕ) :
    (Finset.univ.filter fun ω : K => ω ^ 2 ^ k = -1).card ≤ 2 ^ k := by
  have degree : (X ^ 2 ^ k + C (1 : K)).natDegree = 2 ^ k :=
    natDegree_X_pow_add_C
  have nonzero : X ^ 2 ^ k + C (1 : K) ≠ 0 := by
    intro zero
    have := congrArg natDegree zero
    rw [degree, natDegree_zero] at this
    exact absurd this (by positivity)
  refine (Finset.card_le_card fun ω member => ?_).trans ((zeros_card_le _).trans (le_of_eq degree))
  unfold zeros
  rw [ite_eq_right nonzero]
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at member ⊢
  simp [member]

omit [DecidableEq K] in
/-- `ω^(2^k) = −1` has no root in a field of odd characteristic unless `2^(k+1)` divides
`|K| − 1`. -/
theorem twoPow_ne_neg_one (k : ℕ) (two : (2 : K) ≠ 0)
    (not_dvd : ¬ 2 ^ (k + 1) ∣ Fintype.card K - 1) (ω : K) : ω ^ 2 ^ k ≠ -1 := by
  intro root
  have one_ne : (1 : K) ≠ -1 := by
    intro same
    apply two
    rw [show (2 : K) = 1 + 1 by norm_num]
    nth_rewrite 2 [same]
    simp
  have nonzero : ω ≠ 0 := by
    rintro rfl
    rw [zero_pow (by positivity)] at root
    exact one_ne (by linear_combination 2 * root)
  have square : ω ^ 2 ^ (k + 1) = 1 := by
    rw [pow_succ, pow_mul, root]
    norm_num
  obtain ⟨j, j_le, order⟩ := (Nat.dvd_prime_pow Nat.prime_two).mp (orderOf_dvd_of_pow_eq_one square)
  rcases Nat.lt_or_ge j (k + 1) with small | large
  · apply one_ne
    rw [← root]
    exact (orderOf_dvd_iff_pow_eq_one.mp
      (order ▸ Nat.pow_dvd_pow 2 (by omega : j ≤ k))).symm
  · have full : orderOf ω = 2 ^ (k + 1) := by rw [order, show j = k + 1 by omega]
    apply not_dvd
    rw [← full]
    exact orderOf_dvd_of_pow_eq_one (FiniteField.pow_card_sub_one_eq_one ω nonzero)

/-- The challenges `ω` without a row point: `1 + ω^(2^k) = 0` for some `k < 5`. When `16` does
not divide `|K| − 1`, only `k ≤ 2` has roots, so there are at most `1 + 2 + 4 = 7`. -/
theorem rowPoint_degenerate_card_le (two : (2 : K) ≠ 0)
    (v2 : ¬ 16 ∣ Fintype.card K - 1) :
    (Finset.univ.filter fun ω : K => ∃ k < 5, 1 + ω ^ 2 ^ k = 0).card ≤ 7 := by
  have cover : (Finset.univ.filter fun ω : K => ∃ k < 5, 1 + ω ^ 2 ^ k = 0) ⊆
      (Finset.range 3).biUnion fun k => Finset.univ.filter fun ω : K => ω ^ 2 ^ k = -1 := by
    intro ω member
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at member
    obtain ⟨k, lt, zero⟩ := member
    have root : ω ^ 2 ^ k = -1 := by linear_combination zero
    have small : k < 3 := by
      by_contra large
      refine twoPow_ne_neg_one k two (fun dvd => v2 ?_) ω root
      exact (Nat.pow_dvd_pow 2 (by omega : 4 ≤ k + 1)).trans dvd
    simp only [Finset.mem_biUnion, Finset.mem_range, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨k, small, root⟩
  refine (Finset.card_le_card cover).trans (Finset.card_biUnion_le.trans ?_)
  calc ∑ k ∈ Finset.range 3, (Finset.univ.filter fun ω : K => ω ^ 2 ^ k = -1).card
      ≤ ∑ k ∈ Finset.range 3, 2 ^ k := Finset.sum_le_sum fun k _ => twoPow_roots_card_le k
    _ = 7 := by decide

end TwoPower

end NightstreamFPrime.Spec.Compression.Lib
