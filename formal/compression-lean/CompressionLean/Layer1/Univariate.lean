import CompressionLean.Layer1.Cube
import Mathlib.Algebra.Polynomial.Expand
import Mathlib.Algebra.Polynomial.Degree.Lemmas

/-!
Owns the univariate form of a multilinear table (`setup.rs`): with `e`/`o` the
even and odd halves, `Ô_t(Y) = Ô_e(Y²) + Y·(Ô_o - Ô_e)(Y²)`. Equivalently
`Ô_t(Y) = Σ_x c_x·Y^x` for the monomial coefficients `c` of the extension.

- `univ_eval`: `Ô_t(q) = t~(q, q², q⁴, ...)`, the P1 opening at a query point.
- `univ_natDegree`: degree below `2^m`.
- `univ_eq_zero`: only the zero table has the zero form.
-/

namespace CompressionLean.Layer1

open Polynomial

variable {K : Type} [Field K]

/-- The univariate form of a table on the `m`-cube. -/
noncomputable def univ : ℕ → (ℕ → K) → K[X]
  | 0, t => C (t 0)
  | m + 1, t =>
      expand K 2 (univ m fun i => t (2 * i)) +
        X * expand K 2 (univ m (fun i => t (2 * i + 1)) - univ m fun i => t (2 * i))

/-- `(q, q², q⁴, ...)`, the point of a setup query. -/
def powerPoint (q : K) : ℕ → List K
  | 0 => []
  | m + 1 => q :: powerPoint (q ^ 2) m

theorem univ_add (m : ℕ) (t u : ℕ → K) : univ m (fun i => t i + u i) = univ m t + univ m u := by
  induction m generalizing t u with
  | zero => simp [univ]
  | succ m ih =>
      simp only [univ]
      rw [ih (fun i => t (2 * i)), ih (fun i => t (2 * i + 1))]
      simp only [map_add, map_sub]
      ring

theorem univ_mul_left (m : ℕ) (a : K) (t : ℕ → K) : univ m (fun i => a * t i) = C a * univ m t := by
  induction m generalizing t with
  | zero => simp [univ]
  | succ m ih =>
      simp only [univ]
      rw [ih (fun i => t (2 * i)), ih (fun i => t (2 * i + 1))]
      simp only [← mul_sub, map_mul, expand_C]
      ring

theorem univ_sub (m : ℕ) (t u : ℕ → K) : univ m (fun i => t i - u i) = univ m t - univ m u := by
  have := univ_add m t (fun i => -u i)
  have neg := univ_mul_left m (-1) u
  simp only [map_neg, map_one, neg_mul, one_mul] at neg
  simpa [sub_eq_add_neg, neg] using this

theorem univ_eval (m : ℕ) (t : ℕ → K) (q : K) : (univ m t).eval q = mle (powerPoint q m) t := by
  induction m generalizing t q with
  | zero => simp [univ, powerPoint, mle, fold]
  | succ m ih =>
      simp only [univ, eval_add, eval_mul, eval_X, expand_eval, eval_sub, powerPoint, mle_cons]
      rw [ih, ih]
      have rewrite : bindLow q t = fun i => t (2 * i) + q * (t (2 * i + 1) - t (2 * i)) := by
        funext i
        unfold bindLow
        ring
      rw [rewrite, mle_add, mle_mul_left, mle_sub]

theorem univ_natDegree (m : ℕ) (t : ℕ → K) : (univ m t).natDegree < 2 ^ m := by
  induction m generalizing t with
  | zero => simp [univ]
  | succ m ih =>
      simp only [univ]
      have even := ih (fun i => t (2 * i))
      have odd := ih (fun i => t (2 * i + 1))
      refine lt_of_le_of_lt (natDegree_add_le _ _) (max_lt ?_ ?_)
      · rw [natDegree_expand, pow_succ]
        omega
      · refine lt_of_le_of_lt (natDegree_mul_le) ?_
        rw [natDegree_expand]
        have diff := natDegree_sub_le (univ m (fun i => t (2 * i + 1))) (univ m (fun i => t (2 * i)))
        have := natDegree_X_le (R := K)
        rw [pow_succ]
        have : (univ m (fun i => t (2 * i + 1)) - univ m fun i => t (2 * i)).natDegree < 2 ^ m :=
          lt_of_le_of_lt diff (max_lt odd even)
        omega

/-- Even coefficients come from the first part, odd from the second. -/
private theorem split_eq_zero {A B : K[X]} (zero : expand K 2 A + X * expand K 2 B = 0) :
    A = 0 ∧ B = 0 := by
  constructor
  · ext k
    have := congrArg (fun p => p.coeff (2 * k)) zero
    simp only [coeff_add, coeff_zero] at this
    rw [coeff_expand (by norm_num), if_pos (dvd_mul_right 2 k), Nat.mul_div_cancel_left k (by norm_num)] at this
    rcases Nat.eq_zero_or_pos k with rfl | positive
    · simpa [coeff_X_mul_zero] using this
    · rw [show 2 * k = (2 * k - 1) + 1 by omega, coeff_X_mul, coeff_expand (by norm_num),
        if_neg (by omega), add_zero] at this
      simpa using this
  · ext k
    have := congrArg (fun p => p.coeff (2 * k + 1)) zero
    simp only [coeff_add, coeff_zero, coeff_X_mul] at this
    rw [coeff_expand (by norm_num), if_neg (by omega), zero_add, coeff_expand (by norm_num),
      if_pos (dvd_mul_right 2 k), Nat.mul_div_cancel_left k (by norm_num)] at this
    simpa using this

theorem univ_eq_zero (m : ℕ) (t : ℕ → K) (zero : univ m t = 0) : ∀ i, i < 2 ^ m → t i = 0 := by
  induction m generalizing t with
  | zero =>
      intro i lt
      rw [pow_zero, Nat.lt_one_iff] at lt
      subst lt
      simpa [univ] using zero
  | succ m ih =>
      intro i lt
      simp only [univ] at zero
      obtain ⟨even, diff⟩ := split_eq_zero zero
      have odd : univ m (fun i => t (2 * i + 1)) = 0 := by rw [sub_eq_zero.mp diff, even]
      rcases Nat.mod_two_eq_zero_or_one i with parity | parity
      · rw [show i = 2 * (i / 2) by omega]
        exact ih _ even (i / 2) (by rw [pow_succ] at lt; omega)
      · rw [show i = 2 * (i / 2) + 1 by omega]
        exact ih _ odd (i / 2) (by rw [pow_succ] at lt; omega)

end CompressionLean.Layer1
