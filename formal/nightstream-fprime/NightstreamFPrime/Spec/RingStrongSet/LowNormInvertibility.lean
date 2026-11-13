import Mathlib.FieldTheory.Finite.Basic
import Mathlib.FieldTheory.KummerExtension
import Mathlib.Tactic.LinearCombination
import NightstreamFPrime.Spec.FieldPrime
import NightstreamFPrime.Spec.RingStrongSet
import NightstreamFPrime.Spec.RingRelation.EvaluationHomomorphism.RingFPolynomial

/-! SuperNeo v1.2 Theorem 4, `z = 4`. The modulus `X^64 + 1` is the product of `X^32 − i` and
`X^32 + i` with `i² = −1`, and each factor is irreducible because `q ≡ 5 (mod 8)`. -/

universe u

namespace NightstreamFPrime.Spec.RingStrongSet

open NightstreamFPrime.Spec
open Polynomial
open RingRelation.EvaluationHomomorphism

attribute [local implicit_reducible] ZMod NightstreamFPrime.Spec.fieldModulus

private abbrev Base := ZMod fieldModulus

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

/-- `X ^ 2 ^ n - a` is irreducible over a field in which `-1` is a square and `a` is not a
square. Mathlib has the odd case, `X_pow_sub_C_irreducible_of_prime_pow_of_ne_two`. -/
private theorem X_pow_two_pow_sub_C_irreducible {K : Type u} [Field K] (n : Nat) {a : K}
    (negOneSquare : ∃ i : K, i ^ 2 = -1) (notSquare : ∀ b : K, b ^ 2 ≠ a) :
    Irreducible (X ^ 2 ^ n - C a) := by
  induction n generalizing K a with
  | zero => simpa using irreducible_X_sub_C a
  | succ n hypothesis =>
    have quadratic := X_pow_sub_C_irreducible_of_prime Nat.prime_two notSquare
    rw [pow_succ]
    apply X_pow_mul_sub_C_irreducible quadratic
    have := Fact.mk quadratic
    obtain ⟨i, square⟩ := negOneSquare
    apply hypothesis
    · exact ⟨AdjoinRoot.of _ i, by rw [← map_pow, square, map_neg, map_one]⟩
    · intro b rootSquare
      apply notSquare (i * Algebra.norm K b)
      rw [mul_pow, square, ← map_pow, rootSquare, norm_root_X_pow_sub_C two_ne_zero]
      ring

/-- `2 ^ ((q − 1) / 4)`, a square root of `-1`. -/
private def fourthRoot : Base := ((2296021864060584341 : Nat) : Base)

private theorem fourthRoot_sq : fourthRoot ^ 2 = -1 := by
  apply eq_neg_of_add_eq_zero_left
  rw [fourthRoot, ← Nat.cast_pow, ← Nat.cast_succ, ZMod.natCast_eq_zero_iff]
  decide

private theorem neg_fourthRoot_sq : (-fourthRoot) ^ 2 = -1 := by
  rw [neg_sq, fourthRoot_sq]

/-- A square root of `-1` has order 4, and 8 does not divide `q − 1`. -/
private theorem not_square_of_sq_eq_neg_one {c : Base} (root : c ^ 2 = -1) (value : Base) :
    value ^ 2 ≠ c := by
  intro square
  have fourth : value ^ 4 = -1 := by
    calc
      value ^ 4 = (value ^ 2) ^ 2 := by ring
      _ = -1 := by rw [square, root]
  have nonzero : value ≠ 0 := by
    rintro rfl
    rw [zero_pow (by decide)] at fourth
    exact one_ne_zero (neg_eq_zero.mp fourth.symm)
  have eighth : value ^ 8 = 1 := by
    calc
      value ^ 8 = (value ^ 4) ^ 2 := by ring
      _ = 1 := by rw [fourth, neg_one_sq]
  have reduced : value ^ Nat.gcd 8 (fieldModulus - 1) = 1 :=
    pow_gcd_eq_one.mpr ⟨eighth, ZMod.pow_card_sub_one_eq_one nonzero⟩
  rw [show Nat.gcd 8 (fieldModulus - 1) = 4 by decide, fourth] at reduced
  exact absurd reduced (by decide)

private theorem factor_irreducible {c : Base} (root : c ^ 2 = -1) :
    Irreducible (X ^ 32 - C c : Polynomial Base) := by
  have irreducible :=
    X_pow_two_pow_sub_C_irreducible 5 ⟨c, root⟩ (not_square_of_sq_eq_neg_one root)
  rwa [show (2 : Nat) ^ 5 = 32 by norm_num] at irreducible

private theorem modulus_eq_product :
    RingFPolynomial.modulus = (X ^ 32 - C fourthRoot) * (X ^ 32 - C (-fourthRoot)) := by
  have expand : (X ^ 32 - C fourthRoot) * (X ^ 32 - C (-fourthRoot)) =
      X ^ 64 - C (fourthRoot ^ 2) := by
    rw [C_neg, C_pow]
    ring
  rw [expand, fourthRoot_sq, C_neg, C_1]
  unfold RingFPolynomial.modulus ringDegree
  ring

private noncomputable def coefficientPolynomial {count : Nat} (coefficient : Fin count → Base) :
    Polynomial Base :=
  ∑ index : Fin count, C (coefficient index) * X ^ index.val

private theorem coeff_coefficientPolynomial {count : Nat} (coefficient : Fin count → Base)
    (index : Nat) :
    (coefficientPolynomial coefficient).coeff index =
      if inside : index < count then coefficient ⟨index, inside⟩ else 0 := by
  classical
  simp only [coefficientPolynomial, finsetSum_coeff, C_mul_X_pow_eq_monomial, coeff_monomial]
  split_ifs with inside
  · rw [Finset.sum_eq_single ⟨index, inside⟩]
    · simp
    · intro other _ different
      exact ite_eq_right (fun equal => different (Fin.ext equal))
    · simp
  · apply Finset.sum_eq_zero
    intro other _
    exact ite_eq_right (by have := other.isLt; omega)

private noncomputable def lowPart (value : RingF) : Polynomial Base :=
  coefficientPolynomial fun index : Fin 32 =>
    value ⟨index.val, by have := index.isLt; change index.val < 64; omega⟩

private noncomputable def highPart (value : RingF) : Polynomial Base :=
  coefficientPolynomial fun index : Fin 32 =>
    value ⟨index.val + 32, by have := index.isLt; change index.val + 32 < 64; omega⟩

private theorem toPolynomial_split (value : RingF) :
    RingFPolynomial.toPolynomial value = lowPart value + X ^ 32 * highPart value := by
  ext index
  have whole : (RingFPolynomial.toPolynomial value).coeff index =
      (coefficientPolynomial (count := ringDegree) value).coeff index := by
    simp only [RingFPolynomial.toPolynomial, coefficientPolynomial]
  rw [whole, coeff_add, coeff_X_pow_mul', lowPart, highPart,
    coeff_coefficientPolynomial, coeff_coefficientPolynomial, coeff_coefficientPolynomial]
  by_cases low : index < 32
  · have inside : index < ringDegree := by change index < 64; omega
    simp [low, inside, show ¬32 ≤ index by omega]
  · by_cases inside : index < ringDegree
    · have high : index - 32 < 32 := by change index < 64 at inside; omega
      simp only [low, inside, high, show 32 ≤ index by omega, dite_eq_left, dite_eq_right,
        ite_true, not_false_eq_true, zero_add]
      congr 2
      omega
    · have high : ¬index - 32 < 32 := by change ¬index < 64 at inside; omega
      simp [low, inside, high]

private def centeredLift (value : Base) : Int :=
  if value.val ≤ fieldModulus - value.val then value.val
  else (value.val : Int) - fieldModulus

private theorem centeredLift_cast (value : Base) : ((centeredLift value : Int) : Base) = value := by
  unfold centeredLift
  split
  · rw [Int.cast_natCast, ZMod.natCast_zmod_val]
  · rw [Int.cast_sub, Int.cast_natCast, Int.cast_natCast, ZMod.natCast_zmod_val,
      ZMod.natCast_self, sub_zero]

private theorem centeredLift_abs_le (value : Base) {bound : Nat}
    (small : min value.val (fieldModulus - value.val) ≤ bound) :
    |centeredLift value| ≤ bound := by
  have below := ZMod.val_lt value
  rw [min_le_iff] at small
  unfold centeredLift
  rw [abs_le]
  split <;> constructor <;> omega

private theorem int_pair_eq_zero {left right : Int} {bound : Nat}
    (small : 2 * bound ^ 2 < fieldModulus)
    (leftBound : |left| ≤ bound) (rightBound : |right| ≤ bound)
    (divides : (fieldModulus : Int) ∣ left ^ 2 + right ^ 2) :
    left = 0 ∧ right = 0 := by
  have cap : 2 * (bound : Int) ^ 2 < fieldModulus := by exact_mod_cast small
  have leftSquare : left ^ 2 ≤ (bound : Int) ^ 2 :=
    sq_le_sq' (abs_le.mp leftBound).1 (abs_le.mp leftBound).2
  have rightSquare : right ^ 2 ≤ (bound : Int) ^ 2 :=
    sq_le_sq' (abs_le.mp rightBound).1 (abs_le.mp rightBound).2
  have normZero : left ^ 2 + right ^ 2 = 0 := by
    apply Int.eq_zero_of_abs_lt_dvd divides
    rw [abs_lt]
    constructor <;> linarith [sq_nonneg left, sq_nonneg right]
  constructor
  · exact pow_eq_zero_iff (two_ne_zero) |>.mp
      (by linarith [sq_nonneg left, sq_nonneg right])
  · exact pow_eq_zero_iff (two_ne_zero) |>.mp
      (by linarith [sq_nonneg left, sq_nonneg right])

private theorem pair_eq_zero {c : Base} (root : c ^ 2 = -1) {bound : Nat}
    (small : 2 * bound ^ 2 < fieldModulus) {left right : Base}
    (leftSmall : min left.val (fieldModulus - left.val) ≤ bound)
    (rightSmall : min right.val (fieldModulus - right.val) ≤ bound)
    (relation : left + c * right = 0) : left = 0 ∧ right = 0 := by
  have vanishes : ((centeredLift left ^ 2 + centeredLift right ^ 2 : Int) : Base) = 0 := by
    push_cast
    rw [centeredLift_cast, centeredLift_cast, eq_neg_of_add_eq_zero_left relation]
    linear_combination right ^ 2 * root
  have zero := int_pair_eq_zero small (centeredLift_abs_le left leftSmall)
    (centeredLift_abs_le right rightSmall)
    ((ZMod.intCast_zmod_eq_zero_iff_dvd _ _).mp vanishes)
  constructor
  · rw [← centeredLift_cast left, zero.1, Int.cast_zero]
  · rw [← centeredLift_cast right, zero.2, Int.cast_zero]

private theorem eq_zero_of_factor_dvd {c : Base} (root : c ^ 2 = -1)
    {bound : Nat} (small : 2 * bound ^ 2 < fieldModulus) {value : RingF}
    (norm : ∀ position, centeredMagnitude (value position) ≤ bound)
    (divides : (X ^ 32 - C c) ∣ RingFPolynomial.toPolynomial value) :
    value = ringFZero := by
  have nonzero : c ≠ 0 := by
    rintro rfl
    rw [zero_pow (by decide)] at root
    exact one_ne_zero (neg_eq_zero.mp root.symm)
  have identity : RingFPolynomial.toPolynomial value =
      (lowPart value + C c * highPart value) + (X ^ 32 - C c) * highPart value := by
    rw [toPolynomial_split]
    ring
  have remainderDivides : (X ^ 32 - C c) ∣ lowPart value + C c * highPart value := by
    have difference := dvd_sub divides (dvd_mul_right (X ^ 32 - C c) (highPart value))
    rwa [identity, add_sub_cancel_right] at difference
  have lowDegree : (lowPart value).degree < ((32 : Nat) : WithBot Nat) :=
    degree_sum_fin_lt _
  have highDegree : (highPart value).degree < ((32 : Nat) : WithBot Nat) :=
    degree_sum_fin_lt _
  have remainderZero : lowPart value + C c * highPart value = 0 := by
    apply eq_zero_of_dvd_of_degree_lt remainderDivides
    rw [degree_X_pow_sub_C (by norm_num) c]
    refine (degree_add_le _ _).trans_lt (max_lt lowDegree ?_)
    rw [degree_C_mul nonzero]
    exact highDegree
  have pair : ∀ index : Fin 32,
      (lowPart value).coeff index.val = 0 ∧ (highPart value).coeff index.val = 0 := by
    intro index
    have relation := congrArg (fun polynomial => polynomial.coeff index.val) remainderZero
    simp only [coeff_add, coeff_C_mul, coeff_zero] at relation
    apply pair_eq_zero root small _ _ relation
    · rw [lowPart, coeff_coefficientPolynomial, dite_eq_left index.isLt]
      exact norm _
    · rw [highPart, coeff_coefficientPolynomial, dite_eq_left index.isLt]
      exact norm _
  have halfZero : ∀ half : Polynomial Base, half.degree < ((32 : Nat) : WithBot Nat) →
      (∀ index : Fin 32, half.coeff index.val = 0) → half = 0 := by
    intro half degree coefficients
    ext index
    rw [coeff_zero]
    by_cases inside : index < 32
    · exact coefficients ⟨index, inside⟩
    · exact coeff_eq_zero_of_degree_lt (degree.trans_le (by exact_mod_cast Nat.le_of_not_gt inside))
  apply RingFPolynomial.toPolynomial_injective
  rw [toPolynomial_split, halfZero _ lowDegree (fun index => (pair index).1),
    halfZero _ highDegree (fun index => (pair index).2), RingFPolynomial.toPolynomial_zero]
  ring

/-- SuperNeo v1.2 Theorem 4, `z = 4`. -/
theorem invertible_of_lowNorm (value : RingF) (bound : Nat)
    (nonzero : value ≠ ringFZero)
    (norm : ∀ position, centeredMagnitude (value position) ≤ bound)
    (small : 2 * bound ^ 2 < fieldModulus) :
    ∃ inverse, ringFMul value inverse = ringFOne ∧ ringFMul inverse value = ringFOne := by
  have first : IsCoprime (X ^ 32 - C fourthRoot) (RingFPolynomial.toPolynomial value) :=
    (factor_irreducible fourthRoot_sq).coprime_iff_not_dvd.mpr
      fun divides => nonzero (eq_zero_of_factor_dvd fourthRoot_sq small norm divides)
  have second : IsCoprime (X ^ 32 - C (-fourthRoot)) (RingFPolynomial.toPolynomial value) :=
    (factor_irreducible neg_fourthRoot_sq).coprime_iff_not_dvd.mpr
      fun divides => nonzero (eq_zero_of_factor_dvd neg_fourthRoot_sq small norm divides)
  have coprime : IsCoprime (RingFPolynomial.toPolynomial value) RingFPolynomial.modulus := by
    rw [modulus_eq_product]
    exact first.symm.mul_right second.symm
  obtain ⟨cofactor, other, bezout⟩ := coprime
  let inverse : RingF := fun index =>
    (cofactor %ₘ RingFPolynomial.modulus).coeff index.val
  have view : RingFPolynomial.toPolynomial inverse = cofactor %ₘ RingFPolynomial.modulus := by
    ext index
    by_cases inside : index < ringDegree
    · rw [RingFPolynomial.coeff_toPolynomial inverse ⟨index, inside⟩]
    · have outside : (ringDegree : WithBot Nat) ≤ index := by
        exact_mod_cast Nat.le_of_not_gt inside
      have inverseDegree :
          (RingFPolynomial.toPolynomial inverse).degree < (ringDegree : WithBot Nat) :=
        degree_sum_fin_lt _
      have remainderDegree :
          (cofactor %ₘ RingFPolynomial.modulus).degree < (ringDegree : WithBot Nat) := by
        rw [← RingFPolynomial.degree_modulus]
        exact degree_modByMonic_lt cofactor RingFPolynomial.modulus_monic
      rw [coeff_eq_zero_of_degree_lt (inverseDegree.trans_le outside),
        coeff_eq_zero_of_degree_lt (remainderDegree.trans_le outside)]
  have left : ringFMul inverse value = ringFOne := by
    apply RingFPolynomial.toPolynomial_injective
    have divides : RingFPolynomial.modulus ∣
        cofactor %ₘ RingFPolynomial.modulus * RingFPolynomial.toPolynomial value - 1 := by
      refine ⟨-(cofactor /ₘ RingFPolynomial.modulus * RingFPolynomial.toPolynomial value) -
        other, ?_⟩
      rw [modByMonic_eq_sub_mul_div]
      linear_combination bezout
    rw [RingFPolynomial.toPolynomial_ringFMul_mod, view, RingFPolynomial.toPolynomial_one,
      modByMonic_eq_of_dvd_sub RingFPolynomial.modulus_monic divides]
    apply (modByMonic_eq_self_iff RingFPolynomial.modulus_monic).mpr
    rw [degree_one, RingFPolynomial.degree_modulus]
    exact_mod_cast (show 0 < ringDegree by decide)
  exact ⟨inverse, by rw [RingFLaws.ringFMul_comm]; exact left, left⟩

theorem lowNormInvertibility : LowNormInvertibility :=
  ⟨fun value bound nonzero norm small => invertible_of_lowNorm value bound nonzero norm small⟩

end NightstreamFPrime.Spec.RingStrongSet
