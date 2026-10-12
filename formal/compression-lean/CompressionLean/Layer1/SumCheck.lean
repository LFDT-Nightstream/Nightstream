import CompressionLean.Layer1.Game
import CompressionLean.Layer1.Roots
import CompressionLean.Layer1.Cube
import Mathlib.LinearAlgebra.Lagrange

/-!
Owns the sum-check rounds of `crates/neo-spartan` (`sumcheck::replay`) as a
game, and their soundness against several candidate worlds at once.

A round message is `h(0), h(2), ..., h(d + 1)`; the verifier sets
`h(1) = claim - h(0)`, interpolates through the nodes `0, ..., d + 1`, draws
`c`, and continues with `h(c)`. In world `w` the true round polynomial is
`round w point`. While a world's claim is false, its claimed and true round
polynomials differ, so the claim stays false unless `c` is a common root: a
bad set of at most `d + 1` values per world.

`extra` adds bad sets that ride on the same draws (identity tests of other
tables at the sum-check point); `Path` records that the point avoided them.
-/

namespace CompressionLean.Layer1

open Polynomial

variable {K : Type} [Field K]

/-- The interpolation nodes `0, 1, ..., d + 1` as field elements. -/
def node (d : ℕ) (i : Fin (d + 2)) : K := ((i : ℕ) : K)

/-- The node values of a round: `h(0)`, `claim - h(0)`, then `h(2), ..., h(d + 1)`. -/
def roundValues (d : ℕ) (claim : K) (message : Fin (d + 1) → K) (i : Fin (d + 2)) : K :=
  if hi : (i : ℕ) = 0 then message 0
  else if (i : ℕ) = 1 then claim - message 0
  else message ⟨(i : ℕ) - 1, by omega⟩

/-- The round polynomial of degree at most `d + 1` through the node values. -/
noncomputable def roundPoly (d : ℕ) (claim : K) (message : Fin (d + 1) → K) : K[X] :=
  Lagrange.interpolate Finset.univ (node d) (roundValues d claim message)

/-- The nodes are distinct field elements (the characteristic exceeds `d + 1`). -/
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
  simp only [node, Fin.val_zero, Nat.cast_zero, roundValues, dif_pos] at zero
  simp only [node, Fin.val_one, Nat.cast_one, roundValues, show ((1 : ℕ) = 0) = False by simp,
    dif_neg, if_pos, not_false_eq_true] at one
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

/-- The challenges after `point` avoid the riding bad sets. -/
def Path (extra : List K → Finset K) : List K → List K → Prop
  | _, [] => True
  | point, c :: rest => c ∉ extra point ∧ Path extra (point ++ [c]) rest

variable {W : Type}

/-- `rounds` sum-check rounds of degree `d + 1`, continuing `point` and `claim`. -/
noncomputable def sumcheck (worlds : Finset W) (d : ℕ) (round : W → List K → K[X])
    (extra : List K → Finset K) : ℕ → List K → K → Game (List K × K)
  | 0, point, claim => .pure (point, claim)
  | rounds + 1, point, claim =>
      .recv (Fin (d + 1) → K) fun message =>
        .draw K (worlds.biUnion (fun w => zeros (roundPoly d claim message - round w point)) ∪
            extra point) fun c =>
          sumcheck worlds d round extra rounds (point ++ [c]) ((roundPoly d claim message).eval c)

/-- What a world's true round polynomials must satisfy: degree at most `d + 1`,
the two cube halves sum to the partial sum, and evaluation binds the variable.
Only rounds before `bound` coordinates are constrained. -/
structure RoundTruth (d : ℕ) (bound : ℕ) (truth : List K → K) (round : List K → K[X]) : Prop where
  degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ d + 1
  halves : ∀ point : List K, point.length < bound → (round point).eval 0 + (round point).eval 1 = truth point
  binds : ∀ point : List K, ∀ c : K, point.length < bound → (round point).eval c = truth (point ++ [c])

/-- Soundness of the rounds: in every world whose claim is false, the final
claim is false at the final point, unless a challenge hit a bad set; the
point extends the start by `rounds` challenges that avoided `extra`. -/
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

/-- The cost of the rounds: per round, `d + 1` roots per world and the riding sets. -/
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
