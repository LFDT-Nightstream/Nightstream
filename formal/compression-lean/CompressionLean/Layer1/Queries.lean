import CompressionLean.Layer1.HonestFold
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Real.Basic

/-!
Owns the setup queries of the honest fold (`Relation::query_indices`,
`setup::check`): a P1 table that differs from the honest fold agrees with it
at fewer than half of the leaves' query points, so `t` independent leaves all
agree with probability below `2^-t` (`queryBad_density`).

A leaf `j` has the query point `q_j = (g·ω^j)^8`. The points of distinct
leaves must be distinct (the premise `distinct`); this holds when `ω^8` has
order at least the leaf count.
-/

namespace CompressionLean.Layer1

open Polynomial

variable {K : Type} [Field K]

/-- Distinct tables have forms that agree at fewer than `2^m` distinct points. -/
theorem agreement_card [DecidableEq K] (m : ℕ) (g h : ℕ → K) (different : ∃ i, i < 2 ^ m ∧ g i ≠ h i)
    (q : ℕ → K) (leaves : ℕ) (distinct : ∀ i j, i < leaves → j < leaves → q i = q j → i = j) :
    ((Finset.range leaves).filter fun j => (univ m g).eval (q j) = (univ m h).eval (q j)).card ≤
      2 ^ m - 1 := by
  classical
  have nonzero : univ m (fun i => g i - h i) ≠ 0 := by
    intro zero
    obtain ⟨i, inside, unequal⟩ := different
    exact unequal (sub_eq_zero.mp (univ_eq_zero m _ zero i inside))
  have degree : (univ m (fun i => g i - h i)).natDegree ≤ 2 ^ m - 1 := by
    have := univ_natDegree m (fun i => g i - h i)
    omega
  set S := (Finset.range leaves).filter fun j => (univ m g).eval (q j) = (univ m h).eval (q j)
  have injective : Set.InjOn q S := by
    intro i hi j hj equal
    simp only [S, Finset.coe_filter, Finset.mem_range, Set.mem_setOf_eq] at hi hj
    exact distinct i j hi.1 hj.1 equal
  rw [← Finset.card_image_of_injOn injective]
  refine (Polynomial.card_le_degree_of_subset_roots ?_).trans degree
  intro x member
  obtain ⟨j, inside, rfl⟩ := Finset.mem_image.mp (Finset.mem_val.mp member)
  simp only [S, Finset.mem_filter] at inside
  rw [Polynomial.mem_roots nonzero, IsRoot, univ_sub, eval_sub, inside.2, sub_self]

open Classical in
/-- The query tuples whose every leaf lies in the agreement set. -/
noncomputable def queryBad {L : ℕ} (t : ℕ) (agree : Fin L → Prop) : Finset (Fin t → Fin L) :=
  Fintype.piFinset fun _ => Finset.univ.filter agree

open Classical in
theorem queryBad_density {L : ℕ} (t A : ℕ) (agree : Fin L → Prop)
    (small : (Finset.univ.filter agree).card ≤ A) (positive : 0 < L) :
    ((queryBad t agree).card : ℝ) / Fintype.card (Fin t → Fin L) ≤ ((A : ℝ) / L) ^ t := by
  classical
  unfold queryBad
  rw [Fintype.card_piFinset, Finset.prod_const, Finset.card_univ, Fintype.card_fin, Fintype.card_fun,
    Fintype.card_fin, Fintype.card_fin, div_pow]
  push_cast
  apply div_le_div_of_nonneg_right _ (by positivity)
  exact pow_le_pow_left₀ (Nat.cast_nonneg _) (by exact_mod_cast small) t

end CompressionLean.Layer1
