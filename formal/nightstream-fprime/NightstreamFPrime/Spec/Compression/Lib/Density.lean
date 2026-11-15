import NightstreamFPrime.Spec.Compression.Lib.Roots
import NightstreamFPrime.Spec.Compression.Lib.Cube
import Mathlib.Data.Fintype.Pi
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp

/-! Densities of sets of challenges: the probability that a uniform draw lands in a set. One draw
of a transcript can give several challenges, so a bad set is a set of tuples. A set of pairs is
small when its sections are small outside a small set (`density_prod_le`, `density_prod_le'`). The
other lemmas bound the sections that this argument uses: the roots of a nonzero polynomial
(`density_zeros_le`), a nonzero linear form (`density_linear_le`), a nonzero multilinear table
(`density_mle_zero_le`), and `L` independent draws that each land in one set (`density_pi_mem`). -/

namespace NightstreamFPrime.Spec.Compression.Lib

open Finset

/-- The probability that a uniform draw from `C` lands in `S`. -/
noncomputable def density {C : Type} [Fintype C] (S : Finset C) : ℝ :=
  S.card / Fintype.card C

section General

variable {A B C : Type} [Fintype A] [Fintype B] [Fintype C]

theorem density_nonneg (S : Finset C) : 0 ≤ density S := by
  unfold density
  positivity

theorem density_mono {S T : Finset C} (sub : S ⊆ T) : density S ≤ density T :=
  div_le_div_of_nonneg_right (by exact_mod_cast card_le_card sub) (by positivity)

theorem density_union_le [DecidableEq C] (S T : Finset C) :
    density (S ∪ T) ≤ density S + density T := by
  unfold density
  rw [← add_div]
  exact div_le_div_of_nonneg_right (by exact_mod_cast card_union_le S T) (by positivity)

theorem density_empty : density (∅ : Finset C) = 0 := by
  simp [density]

theorem density_le_one (S : Finset C) : density S ≤ 1 := by
  unfold density
  rcases (Fintype.card C).eq_zero_or_pos with zero | positive
  · rw [zero, Nat.cast_zero, div_zero]
    norm_num
  · rw [div_le_one (by exact_mod_cast positive)]
    exact_mod_cast card_le_univ S

/-- Two sets that an equivalence matches have the same density. -/
theorem density_congr (e : A ≃ C) (S : Finset A) (T : Finset C) (same : ∀ a, a ∈ S ↔ e a ∈ T) :
    density S = density T := by
  have image : T = S.map e.toEmbedding := by
    ext c
    rw [mem_map_equiv, same, Equiv.apply_symm_apply]
  unfold density
  rw [image, card_map, Fintype.card_congr e]

theorem density_product [DecidableEq A] [DecidableEq B] (SA : Finset A) (SB : Finset B) :
    density (SA ×ˢ SB) = density SA * density SB := by
  unfold density
  rw [card_product, Fintype.card_prod]
  push_cast
  rw [mul_div_mul_comm]

/-- Pairs whose sections over `A` have density at most `ε` outside the set `SB` of `B`. -/
theorem density_prod_le [Nonempty A] [Nonempty B] [DecidableEq A] [DecidableEq B] (S : Finset (A × B))
    (SB : Finset B) (ε : ℝ) (nonneg : 0 ≤ ε)
    (sections : ∀ b, b ∉ SB → density (univ.filter fun a => (a, b) ∈ S) ≤ ε) :
    density S ≤ ε + density SB := by
  classical
  have positiveA : (0 : ℝ) < Fintype.card A := by exact_mod_cast Fintype.card_pos
  have positiveB : (0 : ℝ) < Fintype.card B := by exact_mod_cast Fintype.card_pos
  have fiber (b : B) :
      (S.filter fun p => p.2 = b).card = (univ.filter fun a => (a, b) ∈ S).card := by
    apply card_bij (fun p _ => p.1)
    · intro p member
      rw [mem_filter] at member
      obtain ⟨inside, rfl⟩ := member
      simpa using inside
    · intro p member p' member' equal
      rw [mem_filter] at member member'
      exact Prod.ext equal (member.2.trans member'.2.symm)
    · intro a member
      exact ⟨(a, b), by simpa using member, rfl⟩
  have each (b : B) : ((univ.filter fun a => (a, b) ∈ S).card : ℝ) ≤
      ε * Fintype.card A + (if b ∈ SB then (Fintype.card A : ℝ) else 0) := by
    by_cases inside : b ∈ SB
    · rw [ite_eq_left inside]
      have : ((univ.filter fun a => (a, b) ∈ S).card : ℝ) ≤ Fintype.card A := by
        exact_mod_cast card_le_univ _
      nlinarith
    · rw [ite_eq_right inside, add_zero]
      have := sections b inside
      unfold density at this
      rwa [div_le_iff₀ positiveA] at this
  have total : (S.card : ℝ) ≤ ε * Fintype.card A * Fintype.card B + Fintype.card A * SB.card := by
    rw [card_eq_sum_card_fiberwise (f := Prod.snd) (t := univ) (by simp)]
    push_cast
    simp only [fiber]
    refine (sum_le_sum fun b _ => each b).trans (le_of_eq ?_)
    rw [sum_add_distrib, sum_const, card_univ, nsmul_eq_mul, sum_ite_mem, univ_inter, sum_const,
      nsmul_eq_mul]
    ring
  unfold density
  rw [Fintype.card_prod, Nat.cast_mul, div_le_iff₀ (by positivity)]
  calc (S.card : ℝ) ≤ ε * Fintype.card A * Fintype.card B + Fintype.card A * SB.card := total
    _ = (ε + SB.card / Fintype.card B) * (Fintype.card A * Fintype.card B) := by
        field_simp

/-- Pairs whose sections over `B` have density at most `ε` outside the set `SA` of `A`. -/
theorem density_prod_le' [Nonempty A] [Nonempty B] [DecidableEq A] [DecidableEq B] (S : Finset (A × B))
    (SA : Finset A) (ε : ℝ) (nonneg : 0 ≤ ε)
    (sections : ∀ a, a ∉ SA → density (univ.filter fun b => (a, b) ∈ S) ≤ ε) :
    density S ≤ ε + density SA := by
  classical
  rw [density_congr (Equiv.prodComm A B) S (univ.filter fun p => p.swap ∈ S) (by simp)]
  exact density_prod_le _ SA ε nonneg fun a outside => by simpa using sections a outside

end General

section Field

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

theorem density_zeros_le {p : Polynomial K} {d : ℕ} (degree : p.natDegree ≤ d) :
    density (zeros p) ≤ d / Fintype.card K :=
  div_le_div_of_nonneg_right (by exact_mod_cast zeros_card_le_of_natDegree_le degree)
    (by positivity)

theorem density_linePoly_le (a b : K) : density (zeros (linePoly a b)) ≤ 1 / Fintype.card K := by
  exact_mod_cast density_zeros_le (linePoly_natDegree a b)

/-- `L` independent draws that each land in `Z`. -/
theorem density_pi_mem {C : Type} [Fintype C] [DecidableEq C] (Z : Finset C) :
    ∀ L : ℕ, density (univ.filter fun τ : Fin L → C => ∀ k, τ k ∈ Z) = density Z ^ L := by
  intro L
  induction L with
  | zero =>
      have : (univ.filter fun τ : Fin 0 → C => ∀ k, τ k ∈ Z) = univ := by
        ext τ
        simp
      rw [this, pow_zero]
      unfold density
      rw [card_univ, div_self (by exact_mod_cast Fintype.card_ne_zero)]
  | succ L ih =>
      rw [← density_congr (Fin.consEquiv fun _ => C)
        (Z ×ˢ (univ.filter fun τ : Fin L → C => ∀ k, τ k ∈ Z)) _ ?_,
        density_product, ih, pow_succ, mul_comm]
      rintro ⟨c, τ⟩
      simp [Fin.forall_fin_succ]

/-- A linear form with a nonzero coefficient takes one value with density at most `1 / |K|`. -/
theorem density_linear_le :
    ∀ (n : ℕ) (a : Fin n → K) (c : K), (∃ i, a i ≠ 0) →
      density (univ.filter fun x : Fin n → K => ∑ i, x i * a i = c) ≤ 1 / Fintype.card K := by
  intro n
  induction n with
  | zero => intro a c nonzero; exact absurd nonzero (by simp)
  | succ n ih =>
      intro a c nonzero
      have one : (0 : ℝ) ≤ 1 / Fintype.card K := by positivity
      rw [density_congr (Fin.consEquiv fun _ => K).symm
        (univ.filter fun x : Fin (n + 1) → K => ∑ i, x i * a i = c)
        (univ.filter fun p : K × (Fin n → K) => p.1 * a 0 + ∑ i, p.2 i * a i.succ = c) (by
          intro x
          simp [Fin.sum_univ_succ, Fin.tail_def])]
      by_cases first : a 0 = 0
      · have rest : ∃ i, a (Fin.succ i) ≠ 0 := by
          obtain ⟨i, value⟩ := nonzero
          cases i using Fin.cases with
          | zero => exact absurd first value
          | succ i => exact ⟨i, value⟩
        refine (density_prod_le' _ ∅ _ one fun x _ => ?_).trans (by rw [density_empty, add_zero])
        simpa [first] using ih (fun i => a i.succ) c rest
      · refine (density_prod_le _ ∅ _ one fun y _ => ?_).trans (by rw [density_empty, add_zero])
        refine (density_mono (T := zeros (linePoly (∑ i, y i * a i.succ - c) (a 0))) ?_).trans
          (density_linePoly_le _ _)
        intro x member
        unfold zeros
        rw [ite_eq_right (linePoly_ne_zero (Or.inr first))]
        simp only [mem_filter, mem_univ, true_and] at member ⊢
        rw [linePoly_eval, ← member]
        ring

/-- A table that is nonzero on the cube of `n` variables has a multilinear extension that is zero
with density at most `n / |K|`. -/
theorem density_mle_zero_le :
    ∀ (n : ℕ) (e : ℕ → K), (∃ i, i < 2 ^ n ∧ e i ≠ 0) →
      density (univ.filter fun β : Fin n → K => mle (List.ofFn β) e = 0) ≤
        n / Fintype.card K := by
  intro n
  induction n with
  | zero =>
      intro e nonzero
      obtain ⟨i, lt, value⟩ := nonzero
      obtain rfl : i = 0 := by simpa using lt
      have : (univ.filter fun β : Fin 0 → K => mle (List.ofFn β) e = 0) = ∅ := by
        ext β
        simp [mle, fold, value]
      rw [this, density_empty]
      simp
  | succ n ih =>
      intro e nonzero
      obtain ⟨i, lt, value⟩ := nonzero
      have pair : e (2 * (i / 2)) ≠ 0 ∨ e (2 * (i / 2) + 1) - e (2 * (i / 2)) ≠ 0 := by
        by_contra both
        push Not at both
        rcases Nat.mod_two_eq_zero_or_one i with even | odd
        · exact value (by rw [show i = 2 * (i / 2) by omega]; exact both.1)
        · exact value (by rw [show i = 2 * (i / 2) + 1 by omega, sub_eq_zero.mp both.2, both.1])
      have line := linePoly_ne_zero pair
      have step : (n : ℝ) / Fintype.card K + 1 / Fintype.card K = ((n + 1 : ℕ) : ℝ) /
          Fintype.card K := by
        push_cast
        ring
      rw [density_congr (Fin.consEquiv fun _ => K).symm
        (univ.filter fun β : Fin (n + 1) → K => mle (List.ofFn β) e = 0)
        (univ.filter fun p : K × (Fin n → K) => mle (List.ofFn p.2) (bindLow p.1 e) = 0) (by
          intro β
          simp [List.ofFn_succ, mle_cons, Fin.tail_def])]
      refine (density_prod_le' _ (zeros (linePoly (e (2 * (i / 2)))
        (e (2 * (i / 2) + 1) - e (2 * (i / 2))))) ((n : ℝ) / Fintype.card K) (by positivity)
        fun c outside => ?_).trans ?_
      swap
      · have := density_linePoly_le (e (2 * (i / 2))) (e (2 * (i / 2) + 1) - e (2 * (i / 2)))
        rw [← step]
        linarith
      have section_eq : (univ.filter fun b => (c, b) ∈ (univ.filter fun p : K × (Fin n → K) =>
          mle (List.ofFn p.2) (bindLow p.1 e) = 0)) =
          univ.filter fun β : Fin n → K => mle (List.ofFn β) (bindLow c e) = 0 := by
        ext
        simp
      rw [section_eq]
      refine ih (bindLow c e) ⟨i / 2, ?_, ?_⟩
      · rw [pow_succ] at lt
        omega
      · intro zero
        apply outside
        unfold zeros
        rw [ite_eq_right line]
        simp only [mem_filter, mem_univ, true_and, linePoly_eval]
        rw [← zero]
        simp only [bindLow]
        ring

end Field

end NightstreamFPrime.Spec.Compression.Lib
