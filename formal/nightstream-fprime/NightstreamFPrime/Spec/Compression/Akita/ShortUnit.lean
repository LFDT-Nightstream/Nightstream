import Mathlib.FieldTheory.Finite.Basic
import Mathlib.FieldTheory.KummerExtension
import Mathlib.NumberTheory.SumTwoSquares
import Mathlib.Tactic.LinearCombination
import NightstreamFPrime.Spec.FieldPrime
import NightstreamFPrime.Spec.Compression.Akita.Ring

/-! Short elements of `R_{q,d}` are units, for `d` a power of two from 2 (Akita, ePrint 2026/1983,
revision of 2026-11-23, Section 3.2 and (20) with `s_split = 2`; Lyubashevsky and Seiler, ePrint
2017/523, Corollary 1.2 with `k = 2`).

`q = 2^64 − 59` is `5 mod 8`, so `X^d + 1 = (X^{d/2} − r)·(X^{d/2} + r)` for `r² = −1`, and each
factor is irreducible. `isUnit_of_short` states that each `x ≠ 0` with `2·‖x‖_∞² < q` is a unit.
The proof is that of SuperNeo v1.2 Theorem 4 in `Spec/RingStrongSet/LowNormInvertibility.lean`,
for the ring `Rq d` and each degree `2^{n+1}`. -/

universe u

namespace NightstreamFPrime.Spec.Compression.Akita

open Polynomial

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

/-- `X^{2^n} − a` is irreducible over a field in which `−1` is a square and `a` is not a
square. -/
theorem X_pow_two_pow_sub_C_irreducible {K : Type u} [Field K] (n : ℕ) {a : K}
    (negOneSquare : ∃ i : K, i ^ 2 = -1) (notSquare : ∀ b : K, b ^ 2 ≠ a) :
    Irreducible (X ^ 2 ^ n - C a) := by
  induction n generalizing K a with
  | zero => simpa using irreducible_X_sub_C a
  | succ n hypothesis =>
    have quadratic := irreducible_X_pow_sub_C_of_prime Nat.prime_two notSquare
    rw [pow_succ]
    apply irreducible_X_pow_mul_sub_C quadratic
    have := Fact.mk quadratic
    obtain ⟨i, square⟩ := negOneSquare
    apply hypothesis
    · exact ⟨AdjoinRoot.of _ i, by rw [← map_pow, square, map_neg, map_one]⟩
    · intro b rootSquare
      apply notSquare (i * Algebra.norm K b)
      rw [mul_pow, square, ← map_pow, rootSquare, norm_root_X_pow_sub_C two_ne_zero]
      ring

/-- `−1` is a square in `Z_q`, because `q` is `1 mod 4`. -/
theorem exists_sq_eq_neg_one : ∃ r : Zq, r ^ 2 = -1 := by
  obtain ⟨r, square⟩ := ZMod.exists_sq_eq_neg_one_iff.mpr (by decide : fieldModulus % 4 ≠ 3)
  exact ⟨r, by rw [sq, ← square]⟩

/-- A square root of `−1` is not a square in `Z_q`: it has the order 4, and 8 does not divide
`q − 1`. -/
theorem not_square_of_sq_eq_neg_one {r : Zq} (root : r ^ 2 = -1) (value : Zq) :
    value ^ 2 ≠ r := by
  intro square
  have fourth : value ^ 4 = -1 := by
    calc value ^ 4 = (value ^ 2) ^ 2 := by ring
      _ = -1 := by rw [square, root]
  have nonzero : value ≠ 0 := by
    rintro rfl
    rw [zero_pow (by decide)] at fourth
    exact one_ne_zero (neg_eq_zero.mp fourth.symm)
  have eighth : value ^ 8 = 1 := by
    calc value ^ 8 = (value ^ 4) ^ 2 := by ring
      _ = 1 := by rw [fourth, neg_one_sq]
  have reduced : value ^ Nat.gcd 8 (fieldModulus - 1) = 1 :=
    pow_gcd_eq_one.mpr ⟨eighth, ZMod.pow_card_sub_one_eq_one nonzero⟩
  rw [show Nat.gcd 8 (fieldModulus - 1) = 4 by decide, fourth] at reduced
  have : Fact (2 < fieldModulus) := ⟨by decide⟩
  exact ZMod.neg_one_ne_one reduced

/-- A pair `(a, b)` with `a + r·b = 0` for `r² = −1` and magnitudes at most `bound` is zero, when
`2·bound² < q`: `q` divides the integer `a² + b²`, which is below `q`. -/
theorem pair_eq_zero {r a b : Zq} (root : r ^ 2 = -1) {bound : ℕ}
    (small : 2 * bound ^ 2 < fieldModulus) (left : magnitude a ≤ bound)
    (right : magnitude b ≤ bound) (relation : a + r * b = 0) : a = 0 ∧ b = 0 := by
  have vanishes : ((a.valMinAbs ^ 2 + b.valMinAbs ^ 2 : ℤ) : Zq) = 0 := by
    push_cast
    linear_combination (a - r * b) * relation + b ^ 2 * root
  have square (x : Zq) (short : magnitude x ≤ bound) : x.valMinAbs ^ 2 ≤ (bound : ℤ) ^ 2 := by
    rw [← Int.natAbs_sq]
    exact_mod_cast Nat.pow_le_pow_left short 2
  have cap : 2 * (bound : ℤ) ^ 2 < fieldModulus := by exact_mod_cast small
  have zero : a.valMinAbs ^ 2 + b.valMinAbs ^ 2 = 0 := by
    apply Int.eq_zero_of_abs_lt_dvd ((ZMod.intCast_zmod_eq_zero_iff_dvd _ _).mp vanishes)
    rw [abs_lt]
    constructor <;>
      linarith [square a left, square b right, sq_nonneg a.valMinAbs, sq_nonneg b.valMinAbs]
  constructor
  · exact (ZMod.valMinAbs_eq_zero a).mp (pow_eq_zero_iff two_ne_zero |>.mp
      (by linarith [sq_nonneg a.valMinAbs, sq_nonneg b.valMinAbs]))
  · exact (ZMod.valMinAbs_eq_zero b).mp (pow_eq_zero_iff two_ne_zero |>.mp
      (by linarith [sq_nonneg a.valMinAbs, sq_nonneg b.valMinAbs]))

variable {d : ℕ} [NeZero d]

/-- For `d = 2·h` and `r² = −1`, an `x` with `2·‖x‖_∞² < q` whose representative is a multiple
of `X^h − r` is zero: the coefficients `i` and `i + h` of `x` are a pair of `pair_eq_zero`. -/
theorem eq_zero_of_factor_dvd {h : ℕ} (half : h + h = d) {r : Zq} (root : r ^ 2 = -1)
    {bound : ℕ} (small : 2 * bound ^ 2 < fieldModulus) {x : Rq d} (short : normInf x ≤ bound)
    (divides : (X ^ h - C r) ∣ rep x) : x = 0 := by
  obtain ⟨s, product⟩ := divides
  have positive : 0 < h := by
    have := NeZero.pos d
    omega
  have top : ∀ j, h ≤ j → s.coeff j = 0 := by
    intro j large
    by_cases zero : s = 0
    · rw [zero, Polynomial.coeff_zero]
    · have factor : (X ^ h - C r : Zq[X]) ≠ 0 := (monic_X_pow_sub_C r positive.ne').ne_zero
      have degree : (rep x).natDegree = h + s.natDegree := by
        rw [product, natDegree_mul factor zero, natDegree_X_pow_sub_C]
      have below : (rep x).natDegree < d :=
        (natDegree_lt_iff_degree_lt (by rw [product]; exact mul_ne_zero factor zero)).mpr
          (degree_rep_lt x)
      exact coeff_eq_zero_of_natDegree_lt (by omega)
  have pair : ∀ i < h, coeff x i + r * coeff x (i + h) = 0 := by
    intro i low
    show (rep x).coeff i + r * (rep x).coeff (i + h) = 0
    rw [product, sub_mul, Polynomial.coeff_sub, Polynomial.coeff_sub, coeff_X_pow_mul',
      coeff_X_pow_mul', coeff_C_mul, coeff_C_mul, ite_eq_right (by omega), ite_eq_left (by omega),
      Nat.add_sub_cancel, top (i + h) (by omega)]
    ring
  have bounded := (normInf_le_iff x bound).mp short
  refine ext_coeff fun i inside => ?_
  rw [coeff_zero]
  by_cases low : i < h
  · exact (pair_eq_zero root small (bounded i inside) (bounded (i + h) (by omega))
      (pair i low)).1
  · have high := (pair_eq_zero root small (bounded (i - h) (by omega))
      (bounded (i - h + h) (by omega)) (pair (i - h) (by omega))).2
    rwa [Nat.sub_add_cancel (by omega)] at high

/-- Each `x ≠ 0` of `R_{q,d}` with `2·‖x‖_∞² < q` is a unit, for `d = 2^{n+1}` (Lyubashevsky
and Seiler, ePrint 2017/523, Corollary 1.2 with `k = 2`: the bound is `‖x‖_∞ < √(q/2)`). -/
theorem isUnit_of_short (n : ℕ) (power : 2 ^ (n + 1) = d) {x : Rq d} (nonzero : x ≠ 0)
    (short : 2 * normInf x ^ 2 < fieldModulus) : IsUnit x := by
  obtain ⟨r, root⟩ := exists_sq_eq_neg_one
  have half : 2 ^ n + 2 ^ n = d := by
    rw [← power, pow_succ]
    ring
  have coprime : ∀ c : Zq, c ^ 2 = -1 → IsCoprime (X ^ 2 ^ n - C c) (rep x) := fun c square =>
    (X_pow_two_pow_sub_C_irreducible n ⟨c, square⟩
      (not_square_of_sq_eq_neg_one square)).coprime_iff_not_dvd.mpr
      fun divides => nonzero (eq_zero_of_factor_dvd half square short le_rfl divides)
  have lifted : (C r : Zq[X]) ^ 2 = -1 := by rw [← C_pow, root, C_neg, C_1]
  have split : modulus d = (X ^ 2 ^ n - C r) * (X ^ 2 ^ n - C (-r)) := by
    rw [modulus, ← half, C_neg]
    linear_combination lifted
  obtain ⟨a, b, bezout⟩ : IsCoprime (rep x) (modulus d) := by
    rw [split]
    exact (coprime r root).symm.mul_right (coprime (-r) (by rw [neg_sq, root])).symm
  have one := congrArg (AdjoinRoot.mk (modulus d)) bezout
  rw [map_add, map_mul, map_mul, mk_rep, AdjoinRoot.mk_self, mul_zero, add_zero, map_one] at one
  exact isUnit_iff_exists_inv.mpr ⟨_, (mul_comm _ _).trans one⟩

end NightstreamFPrime.Spec.Compression.Akita
