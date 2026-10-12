import CompressionLean.Layer1.Univariate
import CompressionLean.Layer1.Roots
import Mathlib.Algebra.Field.GeomSum
import Mathlib.Tactic.FieldSimp

/-!
Owns the honest fold of the setup oracles (`setup.rs`, M2 "Honest fold"):

- `univ_split`: `Ô_t(Y) = Σ_{u < 2^f} Y^u·P_u(Y^{2^f})` with `P_u = parts f m t u`;
- `univ_fold`: the table folded at `α` (low variables) has the form
  `Σ_u α^[u]·P_u`, with `α^[u] = Π_i α_i^{bit i of u}` (`monomial`);
- `leaf_formula`: the verifier's weights on the `2^f` coset values
  `Ô_t(y·ω^v)` give the folded form at `y^{2^f}` (an inverse DFT; `ω` of
  order `2^f`).
-/

namespace CompressionLean.Layer1

open Polynomial

variable {K : Type} [Field K]

/-- `α^[u] = Π_i α_i^{bit i of u}`, low bit first. -/
def monomial : List K → ℕ → K
  | [], _ => 1
  | a :: rest, u => (if u % 2 = 1 then a else 1) * monomial rest (u / 2)

/-- The coset parts of the form: `Ô_t(Y) = Σ_u Y^u·P_u(Y^{2^f})`. -/
noncomputable def parts : ℕ → ℕ → (ℕ → K) → ℕ → K[X]
  | 0, m, t, _ => univ m t
  | f + 1, m, t, u =>
      if u % 2 = 0 then parts f m (fun i => t (2 * i)) (u / 2)
      else parts f m (fun i => t (2 * i + 1) - t (2 * i)) (u / 2)

theorem parts_add (f m : ℕ) (t u : ℕ → K) (v : ℕ) :
    parts f m (fun i => t i + u i) v = parts f m t v + parts f m u v := by
  induction f generalizing m t u v with
  | zero => exact univ_add m t u
  | succ f ih =>
      simp only [parts]
      split
      · exact ih _ _ _ _
      · rw [← ih]
        congr 1
        funext i
        ring

theorem parts_mul_left (f m : ℕ) (a : K) (t : ℕ → K) (v : ℕ) :
    parts f m (fun i => a * t i) v = C a * parts f m t v := by
  induction f generalizing m t v with
  | zero => exact univ_mul_left m a t
  | succ f ih =>
      simp only [parts]
      split
      · exact ih _ _ _
      · rw [← ih]
        congr 1
        funext i
        ring

theorem expand_cubeSum (p n : ℕ) (t : ℕ → K[X]) :
    expand K p (cubeSum n t) = cubeSum n fun i => expand K p (t i) :=
  map_cubeSum (expand K p).toAddMonoidHom n t

theorem univ_split (f m : ℕ) (t : ℕ → K) :
    univ (m + f) t = cubeSum f fun u => X ^ u * expand K (2 ^ f) (parts f m t u) := by
  induction f generalizing t with
  | zero => simp [cubeSum, parts]
  | succ f ih =>
      rw [show m + (f + 1) = (m + f) + 1 by ring]
      simp only [univ]
      rw [← univ_sub, ih, ih]
      simp only [cubeSum]
      have even (u : ℕ) : (2 * u) % 2 = 0 := by omega
      have odd (u : ℕ) : (2 * u + 1) % 2 = 1 := by omega
      have half (u : ℕ) : (2 * u) / 2 = u := by omega
      have half' (u : ℕ) : (2 * u + 1) / 2 = u := by omega
      simp only [parts, even, odd, half, half', if_true, show (1 : ℕ) ≠ 0 by omega, if_false]
      rw [expand_cubeSum, expand_cubeSum, ← cubeSum_mul_left]
      have power : 2 * 2 ^ f = 2 ^ (f + 1) := by ring
      congr 1
      · congr 1
        funext u
        rw [map_mul, map_pow, expand_X, expand_expand, power, ← pow_mul]
      · congr 1
        funext u
        rw [map_mul, map_pow, expand_X, expand_expand, power, ← pow_mul, ← mul_assoc, ← pow_succ']

theorem univ_fold (m : ℕ) (t : ℕ → K) (α : List K) :
    univ m (fold α t) = cubeSum α.length fun u => C (monomial α u) * parts α.length m t u := by
  induction α generalizing t with
  | nil => simp [fold, cubeSum, monomial, parts]
  | cons a rest ih =>
      simp only [fold, List.length_cons]
      rw [ih]
      have rewrite : bindLow a t = fun i => t (2 * i) + a * (t (2 * i + 1) - t (2 * i)) := by
        funext i
        unfold bindLow
        ring
      simp only [cubeSum, monomial, parts]
      have even (u : ℕ) : (2 * u) % 2 = 0 := by omega
      have odd (u : ℕ) : (2 * u + 1) % 2 = 1 := by omega
      have half (u : ℕ) : (2 * u) / 2 = u := by omega
      have half' (u : ℕ) : (2 * u + 1) / 2 = u := by omega
      simp only [even, odd, half, half', if_true, show (1 : ℕ) ≠ 0 by omega, if_false,
        show ¬ ((0 : ℕ) = 1) by omega, one_mul]
      rw [← cubeSum_add]
      congr 1
      funext u
      rw [rewrite, parts_add, parts_mul_left, map_mul]
      ring

/-- Characters of the cyclic group of order `2^f` are orthogonal. -/
theorem orthogonal (n : ℕ) (ω : K) (root : ω ^ n = 1) (primitive : ∀ k, 0 < k → k < n → ω ^ k ≠ 1)
    {u u' : ℕ} (hu : u < n) (hu' : u' < n) :
    ∑ v ∈ Finset.range n, ω⁻¹ ^ (u * v) * ω ^ (u' * v) = if u = u' then (n : K) else 0 := by
  have nonzero : ω ≠ 0 := by
    intro zero
    rw [zero, zero_pow (by omega)] at root
    exact zero_ne_one root
  have each (v : ℕ) : ω⁻¹ ^ (u * v) * ω ^ (u' * v) = (ω⁻¹ ^ u * ω ^ u') ^ v := by
    rw [mul_pow, ← pow_mul, ← pow_mul]
  simp only [each]
  split
  · rename_i same
    subst same
    simp [inv_pow, inv_mul_cancel₀ (pow_ne_zero _ nonzero)]
  · rename_i different
    have ratio_ne : ω⁻¹ ^ u * ω ^ u' ≠ 1 := by
      intro one
      rw [inv_pow, inv_mul_eq_one₀ (pow_ne_zero _ nonzero)] at one
      rcases Nat.lt_or_gt_of_ne different with lt | gt
      · apply primitive (u' - u) (by omega) (by omega)
        rw [pow_sub₀ ω nonzero lt.le, ← one, mul_inv_cancel₀ (pow_ne_zero _ nonzero)]
      · apply primitive (u - u') (by omega) (by omega)
        rw [pow_sub₀ ω nonzero gt.le, one, mul_inv_cancel₀ (pow_ne_zero _ nonzero)]
    have power : (ω⁻¹ ^ u * ω ^ u') ^ n = 1 := by
      rw [mul_pow, ← pow_mul, ← pow_mul, mul_comm u n, mul_comm u' n, pow_mul, pow_mul, inv_pow, root,
        inv_one, one_pow, one_pow, mul_one]
    rw [geom_sum_eq ratio_ne, power, sub_self, zero_div]

/-- The verifier's leaf value: weights on the coset values `Ô_t(y·ω^v)` give
the folded form at `y^N`, `N = 2^{α.length}`. -/
theorem leaf_formula (m : ℕ) (t : ℕ → K) (α : List K) (y ω : K) (y_ne : y ≠ 0)
    (root : ω ^ (2 ^ α.length) = 1) (primitive : ∀ k, 0 < k → k < 2 ^ α.length → ω ^ k ≠ 1)
    (size : ((2 ^ α.length : ℕ) : K) ≠ 0) :
    ∑ v ∈ Finset.range (2 ^ α.length),
        (∑ u ∈ Finset.range (2 ^ α.length),
          monomial α u * ((2 ^ α.length : ℕ) : K)⁻¹ * y⁻¹ ^ u * ω⁻¹ ^ (u * v)) *
          (univ (m + α.length) t).eval (y * ω ^ v) =
      (univ m (fold α t)).eval (y ^ 2 ^ α.length) := by
  set N := 2 ^ α.length with N_def
  set P := fun u => (parts α.length m t u).eval (y ^ N)
  have value (v : ℕ) : (univ (m + α.length) t).eval (y * ω ^ v) =
      ∑ u' ∈ Finset.range N, y ^ u' * ω ^ (u' * v) * P u' := by
    rw [univ_split, cubeSum_eq_sum_range, eval_finsetSum]
    apply Finset.sum_congr rfl
    intro u' _
    have coset : (y * ω ^ v) ^ N = y ^ N := by
      rw [mul_pow, ← pow_mul, mul_comm v N, pow_mul, root, one_pow, mul_one]
    rw [eval_mul, eval_pow, eval_X, expand_eval, coset, mul_pow, ← pow_mul, mul_comm v u']
  simp only [value]
  have collapse (u : ℕ) (inside : u < N) :
      ∑ v ∈ Finset.range N, ω⁻¹ ^ (u * v) * ∑ u' ∈ Finset.range N, y ^ u' * ω ^ (u' * v) * P u' =
        (N : K) * (y ^ u * P u) := by
    simp only [Finset.mul_sum]
    rw [Finset.sum_comm]
    have each (u' : ℕ) (inside' : u' ∈ Finset.range N) :
        ∑ v ∈ Finset.range N, ω⁻¹ ^ (u * v) * (y ^ u' * ω ^ (u' * v) * P u') =
          (if u = u' then (N : K) else 0) * (y ^ u' * P u') := by
      rw [← orthogonal N ω root primitive inside (Finset.mem_range.mp inside'), Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro v _
      ring
    rw [Finset.sum_congr rfl each]
    simp only [ite_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_range, if_pos inside]
  calc ∑ v ∈ Finset.range N,
        (∑ u ∈ Finset.range N, monomial α u * (N : K)⁻¹ * y⁻¹ ^ u * ω⁻¹ ^ (u * v)) *
          ∑ u' ∈ Finset.range N, y ^ u' * ω ^ (u' * v) * P u'
      = ∑ u ∈ Finset.range N, monomial α u * (N : K)⁻¹ * y⁻¹ ^ u *
          ∑ v ∈ Finset.range N, ω⁻¹ ^ (u * v) * ∑ u' ∈ Finset.range N, y ^ u' * ω ^ (u' * v) * P u' := by
        rw [Finset.sum_congr rfl (fun v _ => Finset.sum_mul _ _ _), Finset.sum_comm]
        apply Finset.sum_congr rfl
        intro u _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro v _
        ring
    _ = ∑ u ∈ Finset.range N, monomial α u * P u := by
        apply Finset.sum_congr rfl
        intro u inside
        rw [collapse u (Finset.mem_range.mp inside),
          show monomial α u * (N : K)⁻¹ * y⁻¹ ^ u * ((N : K) * (y ^ u * P u)) =
            monomial α u * P u * ((N : K)⁻¹ * N) * (y⁻¹ ^ u * y ^ u) by ring,
          inv_mul_cancel₀ size, inv_pow, inv_mul_cancel₀ (pow_ne_zero _ y_ne), mul_one, mul_one]
    _ = (univ m (fold α t)).eval (y ^ N) := by
        rw [univ_fold, cubeSum_eq_sum_range, eval_finsetSum]
        apply Finset.sum_congr rfl
        intro u _
        rw [eval_mul, eval_C]

end CompressionLean.Layer1
