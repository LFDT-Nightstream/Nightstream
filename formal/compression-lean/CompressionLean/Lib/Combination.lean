import CompressionLean.Lib.SumCheck
import Mathlib.Algebra.MvPolynomial.Degrees
import Mathlib.Algebra.Polynomial.BigOperators

/-!
Owns the true round polynomials of a sum-check over the cube sum of a
polynomial combination of tables: `Σ_x comb((T_s(x))_s)`, one variable `s` per table.

After the point `p` binds the low variables, every table is a line in the next
variable, so a round polynomial is `comb` of lines: its degree is at most the
total degree of `comb` (`combRound_truth`). After all variables are bound, the
partial sum is `comb` of the table extensions at the point (`combTruth_final`).
-/

namespace CompressionLean.Lib

open Polynomial

variable {K : Type} [Field K] {σ : Type}

/-- The table `t` restricted to the line through indices `2i` and `2i + 1`. -/
noncomputable def lineAt (t : ℕ → K) (i : ℕ) : K[X] :=
  C (t (2 * i)) * (1 - X) + C (t (2 * i + 1)) * X

theorem lineAt_eval (t : ℕ → K) (i : ℕ) (c : K) : (lineAt t i).eval c = bindLow c t i := by
  simp [lineAt, bindLow]
  ring

theorem lineAt_natDegree (t : ℕ → K) (i : ℕ) : (lineAt t i).natDegree ≤ 1 := by
  unfold lineAt
  refine (natDegree_add_le _ _).trans (max_le ?_ ?_)
  · refine (natDegree_C_mul_le _ _).trans ?_
    refine (natDegree_sub_le _ _).trans (max_le ?_ ?_)
    · simp
    · simp
  · exact (natDegree_C_mul_le _ _).trans natDegree_X_le

/-- A combination of lines has degree at most the combination's total degree. -/
theorem natDegree_aeval_le (P : MvPolynomial σ K) (v : σ → K[X])
    (linear : ∀ j, (v j).natDegree ≤ 1) : (MvPolynomial.aeval v P).natDegree ≤ P.totalDegree := by
  rw [MvPolynomial.aeval_def, MvPolynomial.eval₂_eq]
  apply Polynomial.natDegree_sum_le_of_forall_le
  intro m member
  rw [Polynomial.algebraMap_eq]
  refine (natDegree_C_mul_le _ _).trans ((Polynomial.natDegree_prod_le _ _).trans ?_)
  calc ∑ j ∈ m.support, ((v j) ^ (m j)).natDegree ≤ ∑ j ∈ m.support, m j :=
        Finset.sum_le_sum fun j _ =>
          natDegree_pow_le.trans (by simpa using Nat.mul_le_mul_left (m j) (linear j))
    _ ≤ P.totalDegree := by simpa [Finsupp.sum] using MvPolynomial.le_totalDegree member

theorem eval_aeval (P : MvPolynomial σ K) (v : σ → K[X]) (c : K) :
    (MvPolynomial.aeval v P).eval c = MvPolynomial.eval (fun j => (v j).eval c) P := by
  rw [MvPolynomial.aeval_def, ← Polynomial.coe_evalRingHom, MvPolynomial.hom_eval₂]
  have identity : (Polynomial.evalRingHom c).comp (algebraMap K K[X]) = RingHom.id K := by
    ext a
    simp [Polynomial.algebraMap_eq]
  rw [identity]
  rfl

theorem eval_congr (P : MvPolynomial σ K) {x y : σ → K} (equal : ∀ j, x j = y j) :
    MvPolynomial.eval x P = MvPolynomial.eval y P := by
  rw [funext equal]

theorem eval_cubeSum (n : ℕ) (f : ℕ → K[X]) (c : K) :
    (cubeSum n f).eval c = cubeSum n (fun i => (f i).eval c) :=
  map_cubeSum (Polynomial.evalRingHom c).toAddMonoidHom n f

theorem natDegree_cubeSum_le (n : ℕ) (f : ℕ → K[X]) (d : ℕ) (each : ∀ i, (f i).natDegree ≤ d) :
    (cubeSum n f).natDegree ≤ d := by
  induction n generalizing f with
  | zero => exact each 0
  | succ n ih =>
      exact (natDegree_add_le _ _).trans (max_le (ih _ fun i => each _) (ih _ fun i => each _))

/-- The partial sum of `comb` over `n` variables after binding `point`. -/
noncomputable def combTruth (n : ℕ) (tables : σ → ℕ → K) (comb : MvPolynomial σ K)
    (point : List K) : K :=
  cubeSum (n - point.length) fun i => MvPolynomial.eval (fun j => fold point (tables j) i) comb

/-- The true round polynomial after binding `point`. -/
noncomputable def combRound (n : ℕ) (tables : σ → ℕ → K) (comb : MvPolynomial σ K)
    (point : List K) : K[X] :=
  cubeSum (n - point.length - 1) fun i =>
    MvPolynomial.aeval (fun j => lineAt (fold point (tables j)) i) comb

theorem combRound_truth (n d : ℕ) (tables : σ → ℕ → K) (comb : MvPolynomial σ K)
    (degree : comb.totalDegree ≤ d + 1) :
    RoundTruth d n (combTruth n tables comb) (combRound n tables comb) where
  degree point _ :=
    natDegree_cubeSum_le _ _ _ fun i =>
      (natDegree_aeval_le comb _ fun j => lineAt_natDegree _ _).trans degree
  halves point short := by
    unfold combRound combTruth
    rw [eval_cubeSum, eval_cubeSum]
    obtain ⟨m, rest⟩ : ∃ m, n - point.length = m + 1 := ⟨n - point.length - 1, by omega⟩
    rw [rest, show m + 1 - 1 = m by omega]
    simp only [cubeSum, eval_aeval, lineAt_eval, bindLow]
    congr 1 <;> congr 1 <;> funext i <;> exact eval_congr comb fun j => by ring
  binds point c short := by
    unfold combRound combTruth
    rw [eval_cubeSum, List.length_append, List.length_singleton,
      show n - (point.length + 1) = n - point.length - 1 by omega]
    congr 1
    funext i
    rw [eval_aeval]
    exact eval_congr comb fun j => by rw [lineAt_eval, fold_append_single]

/-- After every variable is bound, the partial sum is `comb` of the extensions. -/
theorem combTruth_final (tables : σ → ℕ → K) (comb : MvPolynomial σ K) (point : List K) :
    combTruth point.length tables comb point = MvPolynomial.eval (fun j => mle point (tables j)) comb := by
  simp [combTruth, cubeSum, mle]

/-- Before any variable is bound, the partial sum is the cube sum. -/
theorem combTruth_nil (n : ℕ) (tables : σ → ℕ → K) (comb : MvPolynomial σ K) :
    combTruth n tables comb [] = cubeSum n fun i => MvPolynomial.eval (fun j => tables j i) comb := by
  simp [combTruth, fold]

/-- `X₀·X₁`. -/
noncomputable def productComb : MvPolynomial (Fin 2) K := MvPolynomial.X 0 * MvPolynomial.X 1

theorem product_truth_nil (n : ℕ) (T : Fin 2 → ℕ → K) :
    combTruth n T productComb [] = cubeSum n fun i => T 0 i * T 1 i := by
  rw [combTruth_nil]
  simp [productComb]

theorem product_truth_final (p : List K) (T : Fin 2 → ℕ → K) :
    combTruth p.length T productComb p = mle p (T 0) * mle p (T 1) := by
  rw [combTruth_final]
  simp [productComb]

theorem productComb_degree : (productComb : MvPolynomial (Fin 2) K).totalDegree ≤ 1 + 1 := by
  unfold productComb
  refine (MvPolynomial.totalDegree_mul _ _).trans ?_
  rw [MvPolynomial.totalDegree_X, MvPolynomial.totalDegree_X]

end CompressionLean.Lib
