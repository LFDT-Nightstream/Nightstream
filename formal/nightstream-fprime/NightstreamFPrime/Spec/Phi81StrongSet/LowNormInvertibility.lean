import Mathlib.FieldTheory.Finite.Basic
import Mathlib.FieldTheory.KummerExtension
import Mathlib.Tactic.LinearCombination
import NightstreamFPrime.Spec.GoldilocksPrime
import NightstreamFPrime.Spec.Phi81StrongSet
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial

/-!
Low-norm invertibility in the Goldilocks `Phi81` ring (SuperNeo Theorem 4 with
`z = 3`, so `b_inv = sqrt(q / 3)`).

Owns: the proof that every nonzero `a` in `F_q[X] / Phi81` with centered
coefficients at most `bound` and `3 * bound^2 < q` has a two-sided inverse
for the executed `ringFMul`, and the proof `lowNormInvertibility` of the
`LowNormInvertibility` statement.

Does not own: the challenge set, its difference bound, or Fiat--Shamir.

Import boundary: this module loads Mathlib field theory. The Spec core
(`Phi81StrongSet`, `PiRLCAlgebra.Challenge`, `PiRLCAlgebra.ForkStrongSet`)
does not import it and stays free of Mathlib; only consumers that already
load Mathlib import this module and supply `lowNormInvertibility`.

Proof. `omega = 2^32 - 1` is a primitive cube root of unity in `F_q`. Since
`gcd(9, q - 1) = 3`, `omega` and `omega^2` are not cubes, so both factors of
`Phi81 = (X^27 - omega) (X^27 - omega^2)` are irreducible (Kummer). Write
`a = A + X^27 B` with `deg A, deg B < 27`. If `X^27 - c` divides `a`, then
`A + c B = 0`, so each coefficient pair gives `A_k^2 - A_k B_k + B_k^2 = 0`
mod `q`. Over the integers this value lies in `[0, 3 bound^2]`, below `q`, and
it is zero only for `A_k = B_k = 0`. Therefore `a` is coprime to `Phi81`, and
the Bezout cofactor gives the inverse. No conjecture from LS18 is used.
-/

namespace NightstreamFPrime.Spec.Phi81StrongSet

open NightstreamFPrime.Spec
open Polynomial
open Phi81Relation.EvaluationHomomorphism

private abbrev Base := ZMod goldilocksModulus

local instance : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩

/-! ## A cube root of unity that is not a cube -/

/-- `2^64 mod q`. -/
private def cubeRoot : Base := ((4294967295 : Nat) : Base)

private theorem cubeRoot_cube : cubeRoot ^ 3 = 1 := by
  rw [cubeRoot, ← Nat.cast_pow, ← ZMod.natCast_mod,
    show (4294967295 : Nat) ^ 3 % goldilocksModulus = 1 by decide, Nat.cast_one]

private theorem cubeRoot_ne_one : cubeRoot ≠ 1 := by
  intro equal
  have values := congrArg ZMod.val equal
  rw [cubeRoot, ZMod.val_natCast, ZMod.val_one] at values
  exact absurd values (by decide)

private theorem cubeRoot_root : cubeRoot ^ 2 + cubeRoot + 1 = 0 := by
  have product : (cubeRoot - 1) * (cubeRoot ^ 2 + cubeRoot + 1) = 0 := by
    linear_combination cubeRoot_cube
  rcases mul_eq_zero.mp product with factor | factor
  · exact absurd (sub_eq_zero.mp factor) cubeRoot_ne_one
  · exact factor

private theorem cubeRoot_sq_root : (cubeRoot ^ 2) ^ 2 + cubeRoot ^ 2 + 1 = 0 := by
  linear_combination cubeRoot * cubeRoot_cube + cubeRoot_root

private theorem not_cube_cubeRoot (value : Base) : value ^ 3 ≠ cubeRoot := by
  intro cube
  have nonzero : value ≠ 0 := by
    rintro rfl
    have zero : cubeRoot = 0 := by rw [← cube]; ring
    have one := cubeRoot_cube
    rw [zero, zero_pow (by decide)] at one
    exact zero_ne_one one
  have nine : value ^ 9 = 1 := by
    calc
      value ^ 9 = (value ^ 3) ^ 3 := by ring
      _ = 1 := by rw [cube, cubeRoot_cube]
  have three : value ^ Nat.gcd 9 (goldilocksModulus - 1) = 1 :=
    pow_gcd_eq_one.mpr ⟨nine, ZMod.pow_card_sub_one_eq_one nonzero⟩
  rw [show Nat.gcd 9 (goldilocksModulus - 1) = 3 by decide, cube] at three
  exact cubeRoot_ne_one three

private theorem not_cube_cubeRoot_sq (value : Base) : value ^ 3 ≠ cubeRoot ^ 2 := by
  intro cube
  apply not_cube_cubeRoot (value ^ 2)
  calc
    (value ^ 2) ^ 3 = (value ^ 3) ^ 2 := by ring
    _ = cubeRoot * cubeRoot ^ 3 := by rw [cube]; ring
    _ = cubeRoot := by rw [cubeRoot_cube, mul_one]

private theorem factor_irreducible {c : Base} (notCube : ∀ value : Base, value ^ 3 ≠ c) :
    Irreducible (X ^ 27 - C c : Polynomial Base) := by
  have irreducible :=
    X_pow_sub_C_irreducible_of_prime_pow Nat.prime_three (by decide) 3 notCube
  rwa [show (3 : Nat) ^ 3 = 27 by norm_num] at irreducible

private theorem modulus_eq_product :
    RingFPolynomial.modulus = (X ^ 27 - C cubeRoot) * (X ^ 27 - C (cubeRoot ^ 2)) := by
  have sum : cubeRoot + cubeRoot ^ 2 = -1 := by linear_combination cubeRoot_root
  have product : cubeRoot * cubeRoot ^ 2 = 1 := by linear_combination cubeRoot_cube
  have expand : (X ^ 27 - C cubeRoot) * (X ^ 27 - C (cubeRoot ^ 2)) =
      X ^ 54 - C (cubeRoot + cubeRoot ^ 2) * X ^ 27 + C (cubeRoot * cubeRoot ^ 2) := by
    rw [C_add, C_mul]
    ring
  rw [expand, sum, product, C_neg, C_1]
  unfold RingFPolynomial.modulus ringDegree ringMiddleDegree
  ring

/-! ## Coefficient halves `a = A + X^27 B` -/

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
      exact if_neg (fun equal => different (Fin.ext equal))
    · simp
  · apply Finset.sum_eq_zero
    intro other _
    exact if_neg (by have := other.isLt; omega)

private noncomputable def lowPart (value : RingF) : Polynomial Base :=
  coefficientPolynomial fun index : Fin 27 =>
    value ⟨index.val, by have := index.isLt; change index.val < 54; omega⟩

private noncomputable def highPart (value : RingF) : Polynomial Base :=
  coefficientPolynomial fun index : Fin 27 =>
    value ⟨index.val + 27, by have := index.isLt; change index.val + 27 < 54; omega⟩

private theorem toPolynomial_split (value : RingF) :
    RingFPolynomial.toPolynomial value = lowPart value + X ^ 27 * highPart value := by
  ext index
  have whole : (RingFPolynomial.toPolynomial value).coeff index =
      (coefficientPolynomial (count := ringDegree) value).coeff index := rfl
  rw [whole, coeff_add, coeff_X_pow_mul', lowPart, highPart,
    coeff_coefficientPolynomial, coeff_coefficientPolynomial, coeff_coefficientPolynomial]
  by_cases low : index < 27
  · have inside : index < ringDegree := by change index < 54; omega
    simp [low, inside, show ¬27 ≤ index by omega]
  · by_cases inside : index < ringDegree
    · have high : index - 27 < 27 := by change index < 54 at inside; omega
      simp only [low, inside, high, show 27 ≤ index by omega, dif_pos, dif_neg, if_true,
        not_false_eq_true, zero_add]
      congr 2
      omega
    · have high : ¬index - 27 < 27 := by change ¬index < 54 at inside; omega
      simp [low, inside, high]

/-! ## Norm argument for one coefficient pair -/

/-- Signed representative with absolute value `min(v, q - v)`. -/
private def centeredLift (value : Base) : Int :=
  if value.val ≤ goldilocksModulus - value.val then value.val
  else (value.val : Int) - goldilocksModulus

private theorem centeredLift_cast (value : Base) : ((centeredLift value : Int) : Base) = value := by
  unfold centeredLift
  split
  · rw [Int.cast_natCast, ZMod.natCast_zmod_val]
  · rw [Int.cast_sub, Int.cast_natCast, Int.cast_natCast, ZMod.natCast_zmod_val,
      ZMod.natCast_self, sub_zero]

private theorem centeredLift_abs_le (value : Base) {bound : Nat}
    (small : min value.val (goldilocksModulus - value.val) ≤ bound) :
    |centeredLift value| ≤ bound := by
  have below := ZMod.val_lt value
  rw [min_le_iff] at small
  unfold centeredLift
  rw [abs_le]
  split <;> constructor <;> omega

/-- `A^2 - A B + B^2` is positive definite, so a multiple of `q` below `q`
forces both integers to zero. -/
private theorem int_pair_eq_zero {left right : Int} {bound : Nat}
    (small : 3 * bound ^ 2 < goldilocksModulus)
    (leftBound : |left| ≤ bound) (rightBound : |right| ≤ bound)
    (divides : (goldilocksModulus : Int) ∣ left ^ 2 - left * right + right ^ 2) :
    left = 0 ∧ right = 0 := by
  have cap : 3 * (bound : Int) ^ 2 < goldilocksModulus := by exact_mod_cast small
  obtain ⟨leftLow, leftHigh⟩ := abs_le.mp leftBound
  obtain ⟨rightLow, rightHigh⟩ := abs_le.mp rightBound
  have normZero : left ^ 2 - left * right + right ^ 2 = 0 := by
    apply Int.eq_zero_of_abs_lt_dvd divides
    rw [abs_lt]
    constructor
    · nlinarith [sq_nonneg (2 * left - right), sq_nonneg right]
    · nlinarith
  have rightZero : right = 0 := by
    nlinarith [sq_nonneg (2 * left - right), sq_nonneg right]
  subst right
  exact ⟨by nlinarith [sq_nonneg left], rfl⟩

/-- A small pair cannot satisfy `a + c b = 0` for a root `c` of `X^2 + X + 1`
unless both entries vanish. -/
private theorem pair_eq_zero {c : Base} (root : c ^ 2 + c + 1 = 0) {bound : Nat}
    (small : 3 * bound ^ 2 < goldilocksModulus) {left right : Base}
    (leftSmall : min left.val (goldilocksModulus - left.val) ≤ bound)
    (rightSmall : min right.val (goldilocksModulus - right.val) ≤ bound)
    (relation : left + c * right = 0) : left = 0 ∧ right = 0 := by
  have vanishes : ((centeredLift left ^ 2 - centeredLift left * centeredLift right +
      centeredLift right ^ 2 : Int) : Base) = 0 := by
    push_cast
    rw [centeredLift_cast, centeredLift_cast, eq_neg_of_add_eq_zero_left relation]
    linear_combination right ^ 2 * root
  have zero := int_pair_eq_zero small (centeredLift_abs_le left leftSmall)
    (centeredLift_abs_le right rightSmall)
    ((ZMod.intCast_zmod_eq_zero_iff_dvd _ _).mp vanishes)
  constructor
  · rw [← centeredLift_cast left, zero.1, Int.cast_zero]
  · rw [← centeredLift_cast right, zero.2, Int.cast_zero]

/-! ## Invertibility -/

/-- A small value divisible by one factor `X^27 - c` of `Phi81` is zero. -/
private theorem eq_zero_of_factor_dvd {c : Base} (root : c ^ 2 + c + 1 = 0)
    {bound : Nat} (small : 3 * bound ^ 2 < goldilocksModulus) {value : RingF}
    (norm : ∀ position, centeredMagnitude (value position) ≤ bound)
    (divides : (X ^ 27 - C c) ∣ RingFPolynomial.toPolynomial value) :
    value = ringFZero := by
  have nonzero : c ≠ 0 := by
    rintro rfl
    norm_num at root
  have identity : RingFPolynomial.toPolynomial value =
      (lowPart value + C c * highPart value) + (X ^ 27 - C c) * highPart value := by
    rw [toPolynomial_split]
    ring
  have remainderDivides : (X ^ 27 - C c) ∣ lowPart value + C c * highPart value := by
    have difference := dvd_sub divides (dvd_mul_right (X ^ 27 - C c) (highPart value))
    rwa [identity, add_sub_cancel_right] at difference
  have lowDegree : (lowPart value).degree < ((27 : Nat) : WithBot Nat) :=
    degree_sum_fin_lt _
  have highDegree : (highPart value).degree < ((27 : Nat) : WithBot Nat) :=
    degree_sum_fin_lt _
  have remainderZero : lowPart value + C c * highPart value = 0 := by
    apply eq_zero_of_dvd_of_degree_lt remainderDivides
    rw [degree_X_pow_sub_C (by norm_num) c]
    refine (degree_add_le _ _).trans_lt (max_lt lowDegree ?_)
    rw [degree_C_mul nonzero]
    exact highDegree
  have pair : ∀ index : Fin 27,
      (lowPart value).coeff index.val = 0 ∧ (highPart value).coeff index.val = 0 := by
    intro index
    have relation := congrArg (fun polynomial => polynomial.coeff index.val) remainderZero
    simp only [coeff_add, coeff_C_mul, coeff_zero] at relation
    apply pair_eq_zero root small _ _ relation
    · rw [lowPart, coeff_coefficientPolynomial, dif_pos index.isLt]
      exact norm _
    · rw [highPart, coeff_coefficientPolynomial, dif_pos index.isLt]
      exact norm _
  have halfZero : ∀ half : Polynomial Base, half.degree < ((27 : Nat) : WithBot Nat) →
      (∀ index : Fin 27, half.coeff index.val = 0) → half = 0 := by
    intro half degree coefficients
    ext index
    rw [coeff_zero]
    by_cases inside : index < 27
    · exact coefficients ⟨index, inside⟩
    · exact coeff_eq_zero_of_degree_lt (degree.trans_le (by exact_mod_cast Nat.le_of_not_gt inside))
  apply RingFPolynomial.toPolynomial_injective
  rw [toPolynomial_split, halfZero _ lowDegree (fun index => (pair index).1),
    halfZero _ highDegree (fun index => (pair index).2), RingFPolynomial.toPolynomial_zero]
  ring

/-- SuperNeo Theorem 4 for Goldilocks and `Phi81` (`z = 3`): a nonzero ring
element whose centered coefficients are at most `bound`, with
`3 * bound^2 < q`, is invertible for the executed ring product. -/
theorem invertible_of_lowNorm (value : RingF) (bound : Nat)
    (nonzero : value ≠ ringFZero)
    (norm : ∀ position, centeredMagnitude (value position) ≤ bound)
    (small : 3 * bound ^ 2 < goldilocksModulus) :
    ∃ inverse, ringFMul value inverse = ringFOne ∧ ringFMul inverse value = ringFOne := by
  have first : IsCoprime (X ^ 27 - C cubeRoot) (RingFPolynomial.toPolynomial value) :=
    (factor_irreducible not_cube_cubeRoot).coprime_iff_not_dvd.mpr
      fun divides => nonzero (eq_zero_of_factor_dvd cubeRoot_root small norm divides)
  have second : IsCoprime (X ^ 27 - C (cubeRoot ^ 2)) (RingFPolynomial.toPolynomial value) :=
    (factor_irreducible not_cube_cubeRoot_sq).coprime_iff_not_dvd.mpr
      fun divides => nonzero (eq_zero_of_factor_dvd cubeRoot_sq_root small norm divides)
  have coprime : IsCoprime (RingFPolynomial.toPolynomial value) RingFPolynomial.modulus := by
    rw [modulus_eq_product]
    exact first.symm.mul_right second.symm
  obtain ⟨cofactor, other, bezout⟩ := coprime
  let inverse : RingF := fun index =>
    (cofactor %ₘ RingFPolynomial.modulus).coeff index.val
  have view : RingFPolynomial.toPolynomial inverse = cofactor %ₘ RingFPolynomial.modulus := by
    ext index
    by_cases inside : index < ringDegree
    · exact RingFPolynomial.coeff_toPolynomial inverse ⟨index, inside⟩
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

/-- The `LowNormInvertibility` statement of the Mathlib-free Spec core. -/
theorem lowNormInvertibility : LowNormInvertibility :=
  ⟨fun value bound nonzero norm small => invertible_of_lowNorm value bound nonzero norm small⟩

end NightstreamFPrime.Spec.Phi81StrongSet
