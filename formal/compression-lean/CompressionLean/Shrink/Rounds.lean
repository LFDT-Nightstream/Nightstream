import CompressionLean.Ports.Shrink.Rounds
import CompressionLean.Shrink.Ccs
import CompressionLean.Lib.Combination

/-!
Owns the true sums of the shrink sum-checks of `Ports/Shrink/Rounds.lean`.

- Outer: the true sum is the extension of the constraint table at `τ`
  (`outer_truth_nil`); at the end it is `eq(τ, rx)·P(M̃_j·z(rx))`
  (`outer_truth_final`). `outerComb` has degree 8 (`outerComb_degree`).
- Batching: claimed values that differ from the true ones differ in a
  nonzero polynomial of degree at most 7 in `ρ` (`batchPoly_ne_zero`,
  `batch_ne`).
- Inner: the true sum is the batch of the true values (`inner_truth_nil`);
  at the end it is `Σ_j ρ^j·M̃_j(rx, ry)·z̃(ry)` (`inner_truth_final`).
-/

namespace CompressionLean.Shrink

open CompressionLean.Lib Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K]

local notation "ι" => algebraMap F K

section Outer

omit [Field K] in
theorem eval_outerComb [Field K] (f : Option (Fin 8) → K) :
    MvPolynomial.eval f outerComb = f none * constraint fun j => f (some j) := by
  unfold outerComb
  rw [map_mul, map_constraint]
  simp

omit [Field K] in
theorem outerComb_degree [Field K] : (outerComb : MvPolynomial (Option (Fin 8)) K).totalDegree ≤ 7 + 1 := by
  have x (j : Option (Fin 8)) : (MvPolynomial.X j : MvPolynomial (Option (Fin 8)) K).totalDegree = 1 :=
    MvPolynomial.totalDegree_X _
  have pair (i j : Fin 8) :
      (MvPolynomial.X (some i) * MvPolynomial.X (some j) : MvPolynomial (Option (Fin 8)) K).totalDegree ≤ 2 :=
    (MvPolynomial.totalDegree_mul _ _).trans (by rw [x, x])
  have seventh : ((MvPolynomial.X (some 0)) ^ 7 : MvPolynomial (Option (Fin 8)) K).totalDegree ≤ 7 :=
    (MvPolynomial.totalDegree_pow _ _).trans (by rw [x])
  have bound : (constraint fun j => (MvPolynomial.X (some j) : MvPolynomial (Option (Fin 8)) K)).totalDegree ≤ 7 := by
    unfold constraint
    refine (MvPolynomial.totalDegree_sub _ _).trans (max_le ?_ (by rw [x]; norm_num))
    refine (MvPolynomial.totalDegree_add _ _).trans (max_le ?_ ((pair 5 6).trans (by norm_num)))
    refine (MvPolynomial.totalDegree_add _ _).trans (max_le ?_ ((pair 3 4).trans (by norm_num)))
    exact (MvPolynomial.totalDegree_add _ _).trans (max_le seventh ((pair 1 2).trans (by norm_num)))
  unfold outerComb
  refine (MvPolynomial.totalDegree_mul _ _).trans ?_
  rw [x]
  omega

/-- The zero-check's true sum is the constraint table's extension at `τ`. -/
theorem outer_truth_nil (es : List (Entry F)) (z : ℕ → F) (τ : List K) (m : ℕ) (length : τ.length = m) :
    combTruth m (outerTables es z τ) outerComb [] = mle τ (constraintTable es z) := by
  rw [combTruth_nil, mle_eq_cubeSum, length]
  congr 1
  funext i
  rw [eval_outerComb]
  simp only [outerTables, constraintTable, map_constraint]

/-- At the end of the outer rounds the partial sum is `eq(τ, rx)·P(M̃_j·z(rx))`. -/
theorem outer_truth_final (es : List (Entry F)) (z : ℕ → F) (τ rx : List K) (length : rx.length = τ.length) :
    combTruth rx.length (outerTables es z τ) outerComb rx =
      eqEval τ rx * constraint fun j => mle rx fun r => ι (rowValue es z j r) := by
  rw [combTruth_final, eval_outerComb]
  simp only [outerTables]
  rw [mle_eqAt τ rx length]

end Outer

section Batch

theorem batchPoly_eval (v t : Fin 8 → K) (ρ : K) : (batchPoly v t).eval ρ = batch ρ v - batch ρ t := by
  simp only [batchPoly, batch, eval_finsetSum, eval_mul, eval_C, eval_pow, eval_X, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  ring

theorem batchPoly_natDegree (v t : Fin 8 → K) : (batchPoly v t).natDegree ≤ 7 := by
  apply natDegree_sum_le_of_forall_le
  intro j _
  refine (natDegree_C_mul_le _ _).trans ?_
  rw [natDegree_X_pow]
  exact Nat.lt_succ_iff.mp j.isLt

theorem batchPoly_ne_zero {v t : Fin 8 → K} (differ : v ≠ t) : batchPoly v t ≠ 0 := by
  obtain ⟨j, hj⟩ := Function.ne_iff.mp differ
  intro zero
  have coefficient := congrArg (fun p => p.coeff (j : ℕ)) zero
  simp only [batchPoly, finsetSum_coeff, coeff_C_mul_X_pow, coeff_zero, Fin.val_inj] at coefficient
  rw [Finset.sum_ite_eq] at coefficient
  simp only [Finset.mem_univ, if_true] at coefficient
  exact hj (sub_eq_zero.mp coefficient)

/-- Off the roots of `batchPoly`, wrong values give a wrong batch. -/
theorem batch_ne [Fintype K] [DecidableEq K] {v t : Fin 8 → K} {ρ : K} (differ : v ≠ t)
    (outside : ρ ∉ zeros (batchPoly v t)) : batch ρ v ≠ batch ρ t := by
  intro equal
  apply outside
  unfold zeros
  rw [if_neg (batchPoly_ne_zero differ)]
  simp [batchPoly_eval, equal]

end Batch

section Inner

/-- The inner sum-check's true sum is the batch of the true matrix values. -/
theorem inner_truth_nil (es : List (Entry F)) (z : ℕ → F) (rx : List K) (ρ : K) (n : ℕ)
    (rows : ∀ e ∈ es, e.row < 2 ^ rx.length) (cols : ∀ e ∈ es, e.col < 2 ^ n) :
    combTruth n (innerTables es z rx ρ) productComb [] =
      batch ρ fun j => mle rx fun r => ι (rowValue es z j r) := by
  rw [product_truth_nil]
  simp only [innerTables, if_true, show (1 : Fin 2) ≠ 0 by decide, if_false]
  exact cubeSum_weight_rows es z rx ρ n rows cols

/-- At the end of the inner rounds the partial sum is `Σ_j ρ^j·M̃_j(rx, ry)·z̃(ry)`. -/
theorem inner_truth_final (es : List (Entry F)) (z : ℕ → F) (rx ry : List K) (ρ : K)
    (cols : ∀ e ∈ es, e.col < 2 ^ ry.length) :
    combTruth ry.length (innerTables es z rx ρ) productComb ry =
      matrixValue es rx ry ρ * mle ry fun y => ι (z y) := by
  rw [product_truth_final]
  simp only [innerTables, if_true, show (1 : Fin 2) ≠ 0 by decide, if_false]
  rw [mle_weight es rx ry ρ cols]

end Inner

end CompressionLean.Shrink
