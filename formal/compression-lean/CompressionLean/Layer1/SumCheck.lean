import CompressionLean.Layer1.Game
import CompressionLean.Layer1.Roots
import CompressionLean.Layer1.Cube
import Mathlib.LinearAlgebra.Lagrange

namespace CompressionLean.Layer1

open Polynomial

variable {K : Type} [Field K]

def node (d : ℕ) (i : Fin (d + 2)) : K := ((i : ℕ) : K)

def roundValues (d : ℕ) (claim : K) (message : Fin (d + 1) → K) (i : Fin (d + 2)) : K :=
  if hi : (i : ℕ) = 0 then message 0
  else if (i : ℕ) = 1 then claim - message 0
  else message ⟨(i : ℕ) - 1, by omega⟩

noncomputable def roundPoly (d : ℕ) (claim : K) (message : Fin (d + 1) → K) : K[X] :=
  Lagrange.interpolate Finset.univ (node d) (roundValues d claim message)

def NodesDistinct (K : Type) [Field K] (d : ℕ) : Prop :=
  Function.Injective (node (K := K) d)

section RoundPoly

variable {d : ℕ}

theorem roundPoly_eval_node (distinct : NodesDistinct K d) (claim : K) (message : Fin (d + 1) → K)
    (i : Fin (d + 2)) : (roundPoly d claim message).eval (node d i) = roundValues d claim message i :=
  Lagrange.eval_interpolate_at_node _ (distinct.injOn) (Finset.mem_univ i)

theorem roundPoly_sum (distinct : NodesDistinct K d) (claim : K) (message : Fin (d + 1) → K) :
    (roundPoly d claim message).eval 0 + (roundPoly d claim message).eval 1 = claim := by
  have zero := roundPoly_eval_node distinct claim message 0
  have one := roundPoly_eval_node distinct claim message 1
  simp only [node, Fin.val_zero, Nat.cast_zero, roundValues, ↓reduceDIte] at zero
  simp only [node, Fin.val_one, Nat.cast_one, roundValues, show ((1 : ℕ) = 0) = False by simp,
    dite_eq_right, ite_eq_left, not_false_eq_true] at one
  rw [zero, one]
  ring

theorem roundPoly_natDegree (distinct : NodesDistinct K d) (claim : K) (message : Fin (d + 1) → K) :
    (roundPoly d claim message).natDegree ≤ d + 1 := by
  have lt : (roundPoly d claim message).degree < (Finset.univ : Finset (Fin (d + 2))).card :=
    Lagrange.degree_interpolate_lt (roundValues d claim message) distinct.injOn
  rw [Finset.card_univ, Fintype.card_fin] at lt
  by_cases zero : roundPoly d claim message = 0
  · rw [zero, natDegree_zero]
    omega
  · rw [degree_eq_natDegree zero] at lt
    exact Nat.lt_succ_iff.mp (by exact_mod_cast lt)

end RoundPoly

variable [Fintype K] [DecidableEq K]

def Path (extra : List K → Finset K) : List K → List K → Prop
  | _, [] => True
  | point, c :: rest => c ∉ extra point ∧ Path extra (point ++ [c]) rest

variable {W : Type}

noncomputable def sumcheck (worlds : Finset W) (d : ℕ) (round : W → List K → K[X])
    (extra : List K → Finset K) : ℕ → List K → K → Game (List K × K)
  | 0, point, claim => .pure (point, claim)
  | rounds + 1, point, claim =>
      .recv (Fin (d + 1) → K) fun message =>
        .draw K (worlds.biUnion (fun w => zeros (roundPoly d claim message - round w point)) ∪
            extra point) fun c =>
          sumcheck worlds d round extra rounds (point ++ [c]) ((roundPoly d claim message).eval c)

structure RoundTruth (d : ℕ) (bound : ℕ) (truth : List K → K) (round : List K → K[X]) : Prop where
  degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ d + 1
  halves : ∀ point : List K, point.length < bound → (round point).eval 0 + (round point).eval 1 = truth point
  binds : ∀ point : List K, ∀ c : K, point.length < bound → (round point).eval c = truth (point ++ [c])

theorem sumcheck_avoids (worlds : Finset W) (d : ℕ) (distinct : NodesDistinct K d)
    (truth : W → List K → K) (round : W → List K → K[X]) (extra : List K → Finset K) (bound : ℕ)
    (true_rounds : ∀ w ∈ worlds, RoundTruth d bound (truth w) (round w)) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      (sumcheck worlds d round extra rounds point claim).Avoids fun result =>
        (∃ challenges, result.1 = point ++ challenges ∧ challenges.length = rounds ∧
          Path extra point challenges) ∧
        ∀ w ∈ worlds, claim ≠ truth w point → result.2 ≠ truth w result.1 := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim _
      exact ⟨⟨[], by simp, rfl, trivial⟩, fun w _ false_claim => false_claim⟩
  | succ rounds ih =>
      intro point claim fits message c outside
      have inside : point.length < bound := by omega
      have rest := ih (point ++ [c]) ((roundPoly d claim message).eval c) (by simp; omega)
      refine rest.mono ?_
      rintro ⟨final, value⟩ ⟨⟨challenges, shape, length, path⟩, transfer⟩
      refine ⟨⟨c :: challenges, by simpa using shape, by simp [length], ?_, path⟩, ?_⟩
      · exact fun hit => outside (Finset.mem_union_right _ hit)
      · intro w member false_claim
        apply transfer w member
        have truths := true_rounds w member
        have differ : roundPoly d claim message ≠ round w point := by
          intro equal
          apply false_claim
          rw [← roundPoly_sum distinct claim message, equal, truths.halves point inside]
        rw [← truths.binds point c inside]
        exact eval_ne_of_not_mem_zeros differ
          (fun hit => outside (Finset.mem_union_left _ (Finset.mem_biUnion.mpr ⟨w, member, hit⟩)))

theorem sumcheck_cost (worlds : Finset W) (d : ℕ) (distinct : NodesDistinct K d)
    (round : W → List K → K[X]) (extra : List K → Finset K) (bound extraBound : ℕ)
    (degrees : ∀ w ∈ worlds, ∀ point : List K, point.length < bound → (round w point).natDegree ≤ d + 1)
    (extra_card : ∀ point : List K, (extra point).card ≤ extraBound) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      (sumcheck worlds d round extra rounds point claim).CostLe
        (rounds * ((worlds.card * (d + 1) + extraBound : ℕ) / (Fintype.card K : ℝ))) := by
  intro rounds
  induction rounds with
  | zero => intro point claim _; simp [sumcheck, Game.CostLe]
  | succ rounds ih =>
      intro point claim fits message
      refine ⟨rounds * ((worlds.card * (d + 1) + extraBound : ℕ) / (Fintype.card K : ℝ)), ?_,
        fun c => ih _ _ (by simp; omega)⟩
      have card : (worlds.biUnion (fun w => zeros (roundPoly d claim message - round w point)) ∪
          extra point).card ≤ worlds.card * (d + 1) + extraBound := by
        refine (Finset.card_union_le _ _).trans (add_le_add ?_ (extra_card point))
        refine card_biUnion_le_mul worlds _ (d + 1) fun w member => zeros_card_le_of_natDegree_le ?_
        exact (natDegree_sub_le _ _).trans
          (max_le (roundPoly_natDegree distinct claim message) (degrees w member point (by omega)))
      have positive : (0 : ℝ) < Fintype.card K := by exact_mod_cast Fintype.card_pos
      have step : ∀ k : ℕ, k ≤ worlds.card * (d + 1) + extraBound →
          (k : ℝ) / Fintype.card K +
              rounds * ((worlds.card * (d + 1) + extraBound : ℕ) / (Fintype.card K : ℝ)) ≤
            ((rounds + 1 : ℕ) : ℝ) *
              ((worlds.card * (d + 1) + extraBound : ℕ) / (Fintype.card K : ℝ)) := by
        intro k le
        have scaled : (k : ℝ) / Fintype.card K ≤
            ((worlds.card * (d + 1) + extraBound : ℕ) : ℝ) / Fintype.card K :=
          div_le_div_of_nonneg_right (by exact_mod_cast le) positive.le
        push_cast at scaled ⊢
        linarith
      exact step _ card

end CompressionLean.Layer1
