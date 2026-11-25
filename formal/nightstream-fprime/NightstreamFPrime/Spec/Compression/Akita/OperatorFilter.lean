import Mathlib.Algebra.Order.Round
import Mathlib.RingTheory.RootsOfUnity.Complex
import NightstreamFPrime.Spec.Compression.Akita.Euclid

/-! The operator-norm filter of the fold challenges of Akita (ePrint 2026/1983, revision of
2026-11-23, Appendix H.1).

- `TrigTable` holds the integer tables `C[t]` and `S[t]` at the scale `Δ` with the error bound
  `ε`. `TrigTable.Accurate` is the enclosure of Appendix H.1, and `TrigTable.nearest` is an
  accurate table with `ε = 1`.
- `TrigTable.Accepts T c` is the fixed-point predicate `U_k ≤ T²·Δ²` of Akita (288).
- `TrigTable.mulBound` is the right inclusion of Akita Theorem H.1 with the spectral formula
  (287): an accurate table accepts only elements with `‖c‖_mul,2 ≤ T`.
- `TrigTable.accepts_of_normOne_le` states that an accurate table with `5·ε ≤ Δ` accepts each
  element with `2·‖c‖_1 ≤ T`.
- `TrigTable.Filtered T keep` states that a filter `keep` accepts only elements that the
  fixed-point predicate of an accurate table accepts, and `TrigTable.Filtered.mulBound` gives
  `‖c‖_mul,2 ≤ T` for them. `TrigTable.filtered_nearest` is an instance. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open ComplexConjugate
open scoped Polynomial

/-- The integer tables of the fixed-point predicate of Akita Appendix H.1: the scale `Δ`, the
error bound `ε`, and the tables `C[t]` and `S[t]`. -/
structure TrigTable where
  scale : ℕ
  error : ℕ
  cos : ℕ → ℤ
  sin : ℕ → ℤ

namespace TrigTable

/-- `|C[t] − Δ·cos(πt/d)| ≤ ε` and `|S[t] − Δ·sin(πt/d)| ≤ ε` for `t < 2d` (Akita Appendix
H.1). -/
def Accurate (table : TrigTable) (d : ℕ) : Prop :=
  ∀ t < 2 * d,
    |(table.cos t : ℝ) - table.scale * Real.cos (Real.pi * t / d)| ≤ table.error ∧
    |(table.sin t : ℝ) - table.scale * Real.sin (Real.pi * t / d)| ≤ table.error

/-- The tables of the nearest integers, with the error bound 1 (Akita Appendix H.1). -/
noncomputable def nearest (scale d : ℕ) : TrigTable where
  scale := scale
  error := 1
  cos t := round ((scale : ℝ) * Real.cos (Real.pi * t / d))
  sin t := round ((scale : ℝ) * Real.sin (Real.pi * t / d))

/-- The tables of the nearest integers are accurate (Akita Appendix H.1). -/
theorem nearest_accurate (scale d : ℕ) : (nearest scale d).Accurate d := by
  intro t _
  simp only [nearest, Nat.cast_one]
  constructor <;> exact (abs_sub_comm _ _).trans_le ((abs_sub_round _).trans (by norm_num))

variable {d : ℕ} [NeZero d]

/-- `R_k = Σ_j c_j·C[(2k+1)·j mod 2d]` of Akita Appendix H.1, with the centered coefficients
of `c`. -/
noncomputable def real (table : TrigTable) (c : Rq d) (k : ℕ) : ℤ :=
  ∑ j ∈ Finset.range d, (coeff c j).valMinAbs * table.cos ((2 * k + 1) * j % (2 * d))

/-- `I_k = Σ_j c_j·S[(2k+1)·j mod 2d]` of Akita Appendix H.1. -/
noncomputable def imaginary (table : TrigTable) (c : Rq d) (k : ℕ) : ℤ :=
  ∑ j ∈ Finset.range d, (coeff c j).valMinAbs * table.sin ((2 * k + 1) * j % (2 * d))

/-- `U_k` of Akita (288), with `r = ‖c‖_1·ε`. -/
noncomputable def upper (table : TrigTable) (c : Rq d) (k : ℕ) : ℤ :=
  table.real c k ^ 2 + table.imaginary c k ^ 2 +
    2 * (|table.real c k| + |table.imaginary c k|) * (normOne c * table.error : ℕ) +
      2 * ((normOne c * table.error : ℕ) : ℤ) ^ 2

/-- The fixed-point predicate of Akita Appendix H.1: `U_k ≤ T²·Δ²` for one frequency of each
conjugate pair. -/
def Accepts (table : TrigTable) (threshold : ℕ) (c : Rq d) : Prop :=
  ∀ k < d / 2, table.upper c k ≤ (threshold : ℤ) ^ 2 * (table.scale : ℤ) ^ 2

private noncomputable def omega (d : ℕ) : ℂ :=
  Complex.exp (2 * Real.pi * Complex.I / (2 * d : ℕ))

private theorem omega_primitive : IsPrimitiveRoot (omega d) (2 * d) :=
  Complex.isPrimitiveRoot_exp (2 * d) (by have := NeZero.ne d; omega)

private theorem omega_pow (t : ℕ) :
    omega d ^ t = Complex.exp ((Real.pi * t / d : ℝ) * Complex.I) := by
  have nonzero : (d : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne d)
  rw [omega, ← Complex.exp_nat_mul]
  congr 1
  push_cast
  field_simp

private theorem omega_pow_re (t : ℕ) : (omega d ^ t).re = Real.cos (Real.pi * t / d) := by
  rw [omega_pow, Complex.exp_ofReal_mul_I_re]

private theorem omega_pow_im (t : ℕ) : (omega d ^ t).im = Real.sin (Real.pi * t / d) := by
  rw [omega_pow, Complex.exp_ofReal_mul_I_im]

private theorem omega_pow_odd (k : ℕ) : (omega d ^ (2 * k + 1)) ^ d = -1 := by
  have nonzero : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne d)
  have half : omega d ^ d = -1 := by
    rw [omega_pow, show Real.pi * d / d = Real.pi by field_simp, Complex.exp_pi_mul_I]
  rw [← pow_mul, mul_comm, pow_mul, half, Odd.neg_one_pow ⟨k, rfl⟩]

private theorem mul_conj_self (a : ℕ) : omega d ^ a * conj (omega d ^ a) = 1 := by
  rw [Complex.mul_conj, map_pow, Complex.normSq_eq_norm_sq,
    omega_primitive.norm'_eq_one (by have := NeZero.ne d; omega)]
  simp

private theorem sum_omega_pow {m : ℕ} (positive : 0 < m) (small : m < d) :
    ∑ k ∈ Finset.range d, omega d ^ ((2 * k + 1) * m) = 0 := by
  have primitive := omega_primitive (d := d)
  have different : omega d ^ (2 * m) ≠ 1 :=
    primitive.pow_ne_one_of_pos_of_lt (by omega) (by omega)
  have cycle : (omega d ^ (2 * m)) ^ d = 1 := by
    rw [← pow_mul, show 2 * m * d = 2 * d * m by ring, pow_mul, primitive.pow_eq_one, one_pow]
  have geometric := geom_sum_mul (omega d ^ (2 * m)) d
  rw [cycle, sub_self] at geometric
  have zero : ∑ k ∈ Finset.range d, (omega d ^ (2 * m)) ^ k = 0 :=
    (mul_eq_zero.mp geometric).resolve_right (sub_ne_zero.mpr different)
  calc ∑ k ∈ Finset.range d, omega d ^ ((2 * k + 1) * m)
      = omega d ^ m * ∑ k ∈ Finset.range d, (omega d ^ (2 * m)) ^ k := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [← pow_mul, ← pow_add]
        congr 1
        ring
    _ = 0 := by rw [zero, mul_zero]

private theorem sum_mul_conj {i j : ℕ} (low : i < d) (high : j < d) :
    ∑ k ∈ Finset.range d, omega d ^ ((2 * k + 1) * i) * conj (omega d ^ ((2 * k + 1) * j)) =
      if i = j then (d : ℂ) else 0 := by
  rcases lt_trichotomy i j with below | rfl | above
  · obtain ⟨m, rfl⟩ : ∃ m, j = i + m := ⟨j - i, by omega⟩
    rw [ite_eq_right (by omega)]
    have each (k : ℕ) :
        omega d ^ ((2 * k + 1) * i) * conj (omega d ^ ((2 * k + 1) * (i + m))) =
          conj (omega d ^ ((2 * k + 1) * m)) := by
      rw [show (2 * k + 1) * (i + m) = (2 * k + 1) * i + (2 * k + 1) * m by ring, pow_add,
        map_mul, ← mul_assoc, mul_conj_self, one_mul]
    simp only [each]
    rw [← map_sum, sum_omega_pow (by omega) (by omega), map_zero]
  · rw [ite_eq_left rfl]
    simp only [mul_conj_self, Finset.sum_const, Finset.card_range, nsmul_eq_mul, mul_one]
  · obtain ⟨m, rfl⟩ : ∃ m, i = j + m := ⟨i - j, by omega⟩
    rw [ite_eq_right (by omega)]
    have each (k : ℕ) :
        omega d ^ ((2 * k + 1) * (j + m)) * conj (omega d ^ ((2 * k + 1) * j)) =
          omega d ^ ((2 * k + 1) * m) := by
      rw [show (2 * k + 1) * (j + m) = (2 * k + 1) * m + (2 * k + 1) * j by ring, pow_add,
        mul_assoc, mul_conj_self, mul_one]
    simp only [each]
    exact sum_omega_pow (by omega) (by omega)

private noncomputable def spectrum (d : ℕ) (f : ℕ → ℝ) (k : ℕ) : ℂ :=
  ∑ i ∈ Finset.range d, (f i : ℂ) * omega d ^ ((2 * k + 1) * i)

private theorem parseval (f : ℕ → ℝ) :
    ∑ k ∈ Finset.range d, Complex.normSq (spectrum d f k) =
      d * ∑ i ∈ Finset.range d, f i ^ 2 := by
  apply Complex.ofReal_injective
  simp only [Complex.ofReal_sum, Complex.ofReal_mul, Complex.ofReal_natCast, Complex.ofReal_pow]
  calc ∑ k ∈ Finset.range d, (Complex.normSq (spectrum d f k) : ℂ)
      = ∑ k ∈ Finset.range d, ∑ i ∈ Finset.range d, ∑ j ∈ Finset.range d, (f i : ℂ) * f j *
          (omega d ^ ((2 * k + 1) * i) * conj (omega d ^ ((2 * k + 1) * j))) := by
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [← Complex.mul_conj, spectrum, map_sum, Finset.sum_mul_sum]
        refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
        rw [map_mul, Complex.conj_ofReal]
        ring
    _ = ∑ i ∈ Finset.range d, ∑ j ∈ Finset.range d, (f i : ℂ) * f j *
          ∑ k ∈ Finset.range d,
            omega d ^ ((2 * k + 1) * i) * conj (omega d ^ ((2 * k + 1) * j)) := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [Finset.sum_comm]
        exact Finset.sum_congr rfl fun j _ => (Finset.mul_sum _ _ _).symm
    _ = ∑ i ∈ Finset.range d, (d : ℂ) * (f i : ℂ) ^ 2 := by
        refine Finset.sum_congr rfl fun i low => ?_
        rw [Finset.sum_congr rfl fun j high => by
          rw [sum_mul_conj (Finset.mem_range.mp low) (Finset.mem_range.mp high)]]
        simp [Finset.mem_range.mp low]
        ring
    _ = d * ∑ i ∈ Finset.range d, (f i : ℂ) ^ 2 := (Finset.mul_sum _ _ _).symm

private theorem spectrum_reflect (f : ℕ → ℝ) {k : ℕ} (small : k < d) :
    spectrum d f (d - 1 - k) = conj (spectrum d f k) := by
  rw [spectrum, spectrum, map_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [map_mul, Complex.conj_ofReal]
  congr 1
  refine mul_right_cancel₀ (pow_ne_zero ((2 * k + 1) * i)
    ((omega_primitive (d := d)).ne_zero (by have := NeZero.ne d; omega))) ?_
  rw [← pow_add, ← add_mul, show 2 * (d - 1 - k) + 1 + (2 * k + 1) = 2 * d by omega,
    pow_mul (omega d) (2 * d) i, omega_primitive.pow_eq_one, one_pow, mul_comm, mul_conj_self]

private noncomputable def centered (x : Rq d) (i : ℕ) : ℝ := (coeff x i).valMinAbs

private noncomputable def lift (x : Rq d) : ℤ[X] :=
  ∑ i ∈ Finset.range d, Polynomial.monomial i (coeff x i).valMinAbs

private theorem coeff_lift (x : Rq d) (i : ℕ) : (lift x).coeff i = (coeff x i).valMinAbs := by
  rw [lift, Polynomial.finsetSum_coeff]
  by_cases small : i < d
  · rw [Finset.sum_eq_single i]
    · exact Finsupp.single_eq_same
    · intro j _ differ
      exact Finsupp.single_eq_of_ne' differ
    · intro absent
      exact absurd (Finset.mem_range.mpr small) absent
  · rw [coeff_eq_zero x (Nat.le_of_not_lt small), ZMod.valMinAbs_zero]
    exact Finset.sum_eq_zero fun j member =>
      Finsupp.single_eq_of_ne' fun equal => small (equal ▸ Finset.mem_range.mp member)

private theorem degree_lift_lt (x : Rq d) : (lift x).degree < d :=
  (Polynomial.degree_lt_iff_coeff_zero _ _).mpr fun m large => by
    rw [coeff_lift, coeff_eq_zero x large, ZMod.valMinAbs_zero]

private theorem map_lift (x : Rq d) : (lift x).map (Int.castRingHom Zq) = rep x := by
  ext i
  rw [Polynomial.coeff_map, coeff_lift, eq_intCast, ZMod.coe_valMinAbs]
  rfl

private theorem negacyclic_monic : (Polynomial.X ^ d + 1 : ℤ[X]).Monic := by
  have monic := Polynomial.monic_X_pow_add_C (1 : ℤ) (NeZero.ne d)
  rwa [Polynomial.C_1] at monic

private noncomputable def product (c v : Rq d) : ℤ[X] :=
  (lift c * lift v) %ₘ (Polynomial.X ^ d + 1)

private theorem coeff_mul_eq (c v : Rq d) (i : ℕ) :
    coeff (c * v) i = ((product c v).coeff i : Zq) := by
  have split : c * v = AdjoinRoot.mk (modulus d) (rep c * rep v) := by
    rw [map_mul, mk_rep, mk_rep]
  have mapped : (product c v).map (Int.castRingHom Zq) = rep (c * v) := by
    rw [product, Polynomial.map_modByMonic _ negacyclic_monic, Polynomial.map_mul, map_lift,
      map_lift, split, rep_mk]
    congr 1
    simp [modulus]
  rw [coeff, ← mapped, Polynomial.coeff_map, eq_intCast]

private theorem degree_product_lt (c v : Rq d) : (product c v).degree < d := by
  have below := Polynomial.degree_modByMonic_lt (lift c * lift v) (negacyclic_monic (d := d))
  have degree := Polynomial.degree_X_pow_add_C (NeZero.pos d) (1 : ℤ)
  rw [Polynomial.C_1] at degree
  rwa [degree] at below

private theorem aeval_product (c v : Rq d) {z : ℂ} (root : z ^ d = -1) :
    Polynomial.aeval z (product c v) =
      Polynomial.aeval z (lift c) * Polynomial.aeval z (lift v) := by
  have whole := congrArg (Polynomial.aeval z)
    (Polynomial.modByMonic_add_div (lift c * lift v) (Polynomial.X ^ d + 1 : ℤ[X]))
  have vanish : Polynomial.aeval z (Polynomial.X ^ d + 1 : ℤ[X]) = 0 := by simp [root]
  rw [map_add, map_mul (Polynomial.aeval z) (Polynomial.X ^ d + 1), vanish, zero_mul, add_zero,
    map_mul] at whole
  exact whole

private theorem aeval_eq_spectrum {p : ℤ[X]} (below : p.degree < d) (k : ℕ) :
    Polynomial.aeval (omega d ^ (2 * k + 1)) p = spectrum d (fun i => (p.coeff i : ℝ)) k := by
  have small : p.natDegree < d := by
    by_cases zero : p = 0
    · rw [zero, Polynomial.natDegree_zero]
      exact NeZero.pos d
    · exact (Polynomial.natDegree_lt_iff_degree_lt zero).mpr below
  rw [Polynomial.aeval_eq_sum_range' small, spectrum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Algebra.smul_def, eq_intCast, Complex.ofReal_intCast, ← pow_mul]

private theorem aeval_lift (x : Rq d) (k : ℕ) :
    Polynomial.aeval (omega d ^ (2 * k + 1)) (lift x) = spectrum d (centered x) k := by
  rw [aeval_eq_spectrum (degree_lift_lt x)]
  congr 1
  funext i
  rw [coeff_lift]
  rfl

private theorem normOne_cast (x : Rq d) :
    (normOne x : ℝ) = ∑ i ∈ Finset.range d, |centered x i| := by
  rw [normOne, Nat.cast_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [magnitude, Nat.cast_natAbs, Int.cast_abs]
  rfl

private theorem normSq_cast (x : Rq d) :
    (normSq x : ℝ) = ∑ i ∈ Finset.range d, centered x i ^ 2 := by
  rw [normSq, Nat.cast_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Nat.cast_pow, magnitude, Nat.cast_natAbs, Int.cast_abs, sq_abs]
  rfl

private theorem spectrum_re (f : ℕ → ℝ) (k : ℕ) :
    (spectrum d f k).re = ∑ j ∈ Finset.range d,
      f j * Real.cos (Real.pi * ((2 * k + 1) * j % (2 * d) : ℕ) / d) := by
  rw [spectrum, Complex.re_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Complex.re_ofReal_mul, pow_eq_pow_mod _ omega_primitive.pow_eq_one, omega_pow_re]

private theorem spectrum_im (f : ℕ → ℝ) (k : ℕ) :
    (spectrum d f k).im = ∑ j ∈ Finset.range d,
      f j * Real.sin (Real.pi * ((2 * k + 1) * j % (2 * d) : ℕ) / d) := by
  rw [spectrum, Complex.im_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [Complex.im_ofReal_mul, pow_eq_pow_mod _ omega_primitive.pow_eq_one, omega_pow_im]

private theorem abs_sum_sub_le (s : Finset ℕ) (a x y : ℕ → ℝ) {scale error : ℝ}
    (close : ∀ j ∈ s, |y j - scale * x j| ≤ error) :
    |scale * ∑ j ∈ s, a j * x j - ∑ j ∈ s, a j * y j| ≤ (∑ j ∈ s, |a j|) * error := by
  rw [Finset.mul_sum, ← Finset.sum_sub_distrib, Finset.sum_mul]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j member => ?_)
  rw [show scale * (a j * x j) - a j * y j = a j * (scale * x j - y j) by ring, abs_mul,
    abs_sub_comm]
  exact mul_le_mul_of_nonneg_left (close j member) (abs_nonneg _)

private theorem real_cast (table : TrigTable) (c : Rq d) (k : ℕ) :
    (table.real c k : ℝ) = ∑ j ∈ Finset.range d,
      centered c j * table.cos ((2 * k + 1) * j % (2 * d)) := by
  simp only [real, Int.cast_sum, Int.cast_mul]
  rfl

private theorem imaginary_cast (table : TrigTable) (c : Rq d) (k : ℕ) :
    (table.imaginary c k : ℝ) = ∑ j ∈ Finset.range d,
      centered c j * table.sin ((2 * k + 1) * j % (2 * d)) := by
  simp only [imaginary, Int.cast_sum, Int.cast_mul]
  rfl

private theorem real_close {table : TrigTable} (accurate : table.Accurate d) (c : Rq d)
    (k : ℕ) :
    |(table.scale : ℝ) * (spectrum d (centered c) k).re - table.real c k| ≤
      normOne c * table.error := by
  rw [spectrum_re, real_cast, normOne_cast]
  exact abs_sum_sub_le _ _ _ _ fun j _ =>
    (accurate _ (Nat.mod_lt _ (by have := NeZero.pos d; omega))).1

private theorem imaginary_close {table : TrigTable} (accurate : table.Accurate d) (c : Rq d)
    (k : ℕ) :
    |(table.scale : ℝ) * (spectrum d (centered c) k).im - table.imaginary c k| ≤
      normOne c * table.error := by
  rw [spectrum_im, imaginary_cast, normOne_cast]
  exact abs_sum_sub_le _ _ _ _ fun j _ =>
    (accurate _ (Nat.mod_lt _ (by have := NeZero.pos d; omega))).2

private theorem sq_add_sq_le {x y a b r : ℝ} (first : |x - a| ≤ r) (second : |y - b| ≤ r) :
    x ^ 2 + y ^ 2 ≤ a ^ 2 + b ^ 2 + 2 * (|a| + |b|) * r + 2 * r ^ 2 := by
  have left := abs_le.mp first
  have right := abs_le.mp second
  have squareX : x ^ 2 ≤ (|a| + r) ^ 2 :=
    sq_le_sq' (by linarith [neg_abs_le a]) (by linarith [le_abs_self a])
  have squareY : y ^ 2 ≤ (|b| + r) ^ 2 :=
    sq_le_sq' (by linarith [neg_abs_le b]) (by linarith [le_abs_self b])
  calc x ^ 2 + y ^ 2 ≤ (|a| + r) ^ 2 + (|b| + r) ^ 2 := add_le_add squareX squareY
    _ = a ^ 2 + b ^ 2 + 2 * (|a| + |b|) * r + 2 * r ^ 2 := by
      rw [add_sq, add_sq, sq_abs, sq_abs]
      ring

private theorem upper_cast (table : TrigTable) (c : Rq d) (k : ℕ) :
    (table.upper c k : ℝ) = (table.real c k : ℝ) ^ 2 + (table.imaginary c k : ℝ) ^ 2 +
      2 * (|(table.real c k : ℝ)| + |(table.imaginary c k : ℝ)|) *
        ((normOne c : ℝ) * table.error) + 2 * ((normOne c : ℝ) * table.error) ^ 2 := by
  rw [upper]
  push_cast
  ring

private theorem normSq_spectrum_le {table : TrigTable} (positive : 0 < table.scale)
    (accurate : table.Accurate d) {threshold : ℕ} {c : Rq d}
    (accepts : table.Accepts threshold c) {k : ℕ} (small : k < d / 2) :
    Complex.normSq (spectrum d (centered c) k) ≤ threshold ^ 2 := by
  have scale : (0 : ℝ) < (table.scale : ℝ) ^ 2 := by positivity
  have enclosed := sq_add_sq_le (real_close accurate c k) (imaginary_close accurate c k)
  have accept : (table.upper c k : ℝ) ≤ (threshold : ℝ) ^ 2 * (table.scale : ℝ) ^ 2 := by
    exact_mod_cast accepts k small
  rw [upper_cast] at accept
  refine le_of_mul_le_mul_left ?_ scale
  rw [Complex.normSq_apply]
  linarith

private theorem normSq_spectrum_le_of_even {table : TrigTable} (even : 2 ∣ d)
    (positive : 0 < table.scale) (accurate : table.Accurate d) {threshold : ℕ} {c : Rq d}
    (accepts : table.Accepts threshold c) {k : ℕ} (small : k < d) :
    Complex.normSq (spectrum d (centered c) k) ≤ threshold ^ 2 := by
  by_cases low : k < d / 2
  · exact normSq_spectrum_le positive accurate accepts low
  · rw [← Complex.normSq_conj, ← spectrum_reflect _ small]
    exact normSq_spectrum_le positive accurate accepts (by obtain ⟨h, rfl⟩ := even; omega)

/-- Akita Theorem H.1, right inclusion, with (287): an element that the fixed-point predicate of
an accurate table accepts has `‖c‖_mul,2 ≤ T`. -/
theorem mulBound {table : TrigTable} (even : 2 ∣ d) (positive : 0 < table.scale)
    (accurate : table.Accurate d) {threshold : ℕ} {c : Rq d}
    (accepts : table.Accepts threshold c) : MulBound threshold c := by
  intro v
  have spectral (k : ℕ) : spectrum d (fun i => ((product c v).coeff i : ℝ)) k =
      spectrum d (centered c) k * spectrum d (centered v) k := by
    rw [← aeval_eq_spectrum (degree_product_lt c v), aeval_product c v (omega_pow_odd k),
      aeval_lift, aeval_lift]
  have coefficients : (normSq (c * v) : ℝ) ≤
      ∑ i ∈ Finset.range d, ((product c v).coeff i : ℝ) ^ 2 := by
    rw [normSq, Nat.cast_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    have bound : (magnitude ((product c v).coeff i : Zq) : ℝ) ≤
        |((product c v).coeff i : ℝ)| := by
      rw [← Int.cast_abs, ← Nat.cast_natAbs]
      exact_mod_cast magnitude_intCast_le _
    rw [coeff_mul_eq, Nat.cast_pow, ← sq_abs ((product c v).coeff i : ℝ)]
    exact pow_le_pow_left₀ (Nat.cast_nonneg _) bound 2
  have energy : (d : ℝ) * ∑ i ∈ Finset.range d, ((product c v).coeff i : ℝ) ^ 2 ≤
      d * (threshold ^ 2 * normSq v) := by
    rw [← parseval, normSq_cast, mul_left_comm (d : ℝ), ← parseval, Finset.mul_sum]
    refine Finset.sum_le_sum fun k member => ?_
    rw [spectral, Complex.normSq_mul]
    exact mul_le_mul_of_nonneg_right
      (normSq_spectrum_le_of_even even positive accurate accepts (Finset.mem_range.mp member))
      (Complex.normSq_nonneg _)
  have bounded := coefficients.trans
    (le_of_mul_le_mul_left energy (Nat.cast_pos.mpr (NeZero.pos d)))
  exact_mod_cast bounded

private theorem abs_le_of_close {y x scale error : ℝ} (close : |y - scale * x| ≤ error)
    (unit : |x| ≤ 1) (nonneg : 0 ≤ scale) : |y| ≤ scale + error := by
  have bounded : |scale * x| ≤ scale := by
    rw [abs_mul, abs_of_nonneg nonneg]
    exact mul_le_of_le_one_right nonneg unit
  calc |y| = |(y - scale * x) + scale * x| := by rw [sub_add_cancel]
    _ ≤ |y - scale * x| + |scale * x| := abs_add_le _ _
    _ ≤ scale + error := by linarith

private theorem abs_sum_mul_le (s : Finset ℕ) (a y : ℕ → ℝ) {bound : ℝ}
    (each : ∀ j ∈ s, |y j| ≤ bound) : |∑ j ∈ s, a j * y j| ≤ (∑ j ∈ s, |a j|) * bound := by
  rw [Finset.sum_mul]
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j member => ?_)
  rw [abs_mul]
  exact mul_le_mul_of_nonneg_left (each j member) (abs_nonneg _)

private theorem upper_le {a b mass scale error threshold : ℝ} (massNonneg : 0 ≤ mass)
    (errorNonneg : 0 ≤ error) (fine : 5 * error ≤ scale) (large : 2 * mass ≤ threshold)
    (first : |a| ≤ mass * (scale + error)) (second : |b| ≤ mass * (scale + error)) :
    a ^ 2 + b ^ 2 + 2 * (|a| + |b|) * (mass * error) + 2 * (mass * error) ^ 2 ≤
      threshold ^ 2 * scale ^ 2 := by
  have radius : 0 ≤ mass * error := mul_nonneg massNonneg errorNonneg
  have shift : mass * (scale + 2 * error) ≤ mass * (7 / 5 * scale) :=
    mul_le_mul_of_nonneg_left (by linarith) massNonneg
  have squareA : (|a| + mass * error) ^ 2 ≤ (mass * (7 / 5 * scale)) ^ 2 :=
    pow_le_pow_left₀ (add_nonneg (abs_nonneg a) radius) (by linarith) 2
  have squareB : (|b| + mass * error) ^ 2 ≤ (mass * (7 / 5 * scale)) ^ 2 :=
    pow_le_pow_left₀ (add_nonneg (abs_nonneg b) radius) (by linarith) 2
  have squareT : (2 * mass) ^ 2 ≤ threshold ^ 2 :=
    pow_le_pow_left₀ (by linarith) large 2
  calc a ^ 2 + b ^ 2 + 2 * (|a| + |b|) * (mass * error) + 2 * (mass * error) ^ 2
      = (|a| + mass * error) ^ 2 + (|b| + mass * error) ^ 2 := by
        rw [add_sq, add_sq, sq_abs, sq_abs]
        ring
    _ ≤ (2 * mass) ^ 2 * scale ^ 2 := by nlinarith [sq_nonneg (mass * scale)]
    _ ≤ threshold ^ 2 * scale ^ 2 := mul_le_mul_of_nonneg_right squareT (sq_nonneg _)

/-- An accurate table with `5·ε ≤ Δ` accepts each element with `2·‖c‖_1 ≤ T`, so the hypothesis
`Accepts` of `mulBound` holds for nonzero elements (Akita Appendix H.1). -/
theorem accepts_of_normOne_le {table : TrigTable} (accurate : table.Accurate d)
    (fine : 5 * table.error ≤ table.scale) {threshold : ℕ} {c : Rq d}
    (mass : 2 * normOne c ≤ threshold) : table.Accepts threshold c := by
  intro k _
  have index (j : ℕ) : (2 * k + 1) * j % (2 * d) < 2 * d :=
    Nat.mod_lt _ (by have := NeZero.pos d; omega)
  have nonneg : (0 : ℝ) ≤ table.scale := Nat.cast_nonneg _
  have first : |(table.real c k : ℝ)| ≤ normOne c * ((table.scale : ℝ) + table.error) := by
    rw [real_cast, normOne_cast]
    exact abs_sum_mul_le _ _ _ fun j _ =>
      abs_le_of_close (accurate _ (index j)).1 (Real.abs_cos_le_one _) nonneg
  have second : |(table.imaginary c k : ℝ)| ≤
      normOne c * ((table.scale : ℝ) + table.error) := by
    rw [imaginary_cast, normOne_cast]
    exact abs_sum_mul_le _ _ _ fun j _ =>
      abs_le_of_close (accurate _ (index j)).2 (Real.abs_sin_le_one _) nonneg
  have fine' : 5 * (table.error : ℝ) ≤ table.scale := by exact_mod_cast fine
  have large : 2 * (normOne c : ℝ) ≤ threshold := by exact_mod_cast mass
  have bounded := upper_le (Nat.cast_nonneg _) (Nat.cast_nonneg _) fine' large first second
  rw [← upper_cast] at bounded
  exact_mod_cast bounded

/-- The filter `keep` of a challenge family is at least as strict as the fixed-point predicate of
an accurate table for the threshold `T` (Akita Appendix H.1). The filter of the pinned Akita code
is this predicate for its tables (`crates/akita-challenges/src/sampler/op_norm.rs`). -/
def Filtered (threshold : ℕ) (keep : Rq d → Prop) : Prop :=
  ∃ table : TrigTable, 0 < table.scale ∧ table.Accurate d ∧
    ∀ c, keep c → table.Accepts threshold c

/-- Akita Theorem H.1, right inclusion: each element that passes a filter with `Filtered` has
`‖c‖_mul,2 ≤ T`. -/
theorem Filtered.mulBound {threshold : ℕ} {keep : Rq d → Prop}
    (filtered : Filtered threshold keep) (even : 2 ∣ d) {c : Rq d} (kept : keep c) :
    MulBound threshold c := by
  obtain ⟨table, positive, accurate, accepts⟩ := filtered
  exact TrigTable.mulBound even positive accurate (accepts c kept)

/-- The fixed-point predicate of the table of the nearest integers at a positive scale is a
filter with `Filtered`. -/
theorem filtered_nearest {scale : ℕ} (positive : 0 < scale) (threshold : ℕ) :
    Filtered (d := d) threshold ((nearest scale d).Accepts threshold) :=
  ⟨nearest scale d, positive, nearest_accurate scale d, fun _ kept => kept⟩

end TrigTable

end NightstreamFPrime.Spec.Compression.Akita
