import Mathlib.Algebra.CharP.Algebra
import Mathlib.Algebra.CharP.Frobenius
import Mathlib.FieldTheory.Finite.Basic
import NightstreamFPrime.Spec.GoldilocksPrime
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial

/-!
Frobenius and unit-power identities in the existing Phi81 quotient.
All standard multiplication and powers are in AdjoinRoot, not the
pointwise instances of the RingF coefficient function. These proofs
define no executable inverse and supply no runtime bound.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFFrobenius

open NightstreamFPrime.Spec
open RingFPolynomial

private abbrev Base := ZMod goldilocksModulus

local instance : Fact (Nat.Prime goldilocksModulus) :=
  ⟨GoldilocksPrime.goldilocks_natPrime⟩

local instance : CharP QuotientRing goldilocksModulus :=
  charP_of_injective_ringHom
    (AdjoinRoot.of.injective_of_degree_ne_zero (f := modulus) (by
      rw [degree_modulus]
      decide)) goldilocksModulus

private theorem exponent_mod : (goldilocksModulus ^ 27) % 81 = 1 := by
  have congruence : Nat.ModEq 81 (goldilocksModulus ^ 27) (4 ^ 27) :=
    Nat.ModEq.pow 27 (by decide)
  change (goldilocksModulus ^ 27) % 81 = (4 ^ 27) % 81 at congruence
  exact congruence.trans (by decide)

private theorem frobenius_identity :
    iterateFrobenius QuotientRing goldilocksModulus 27 = RingHom.id QuotientRing := by
  apply AdjoinRoot.ringHom_ext
  · ext coefficient
    change (AdjoinRoot.of modulus (coefficient : Base)) ^ (goldilocksModulus ^ 27) =
      AdjoinRoot.of modulus coefficient
    rw [← map_pow, ZMod.pow_card_pow]
  · change AdjoinRoot.root modulus ^ (goldilocksModulus ^ 27) = AdjoinRoot.root modulus
    rw [root_pow_mod, exponent_mod, pow_one]

/-- Every element is fixed by the 27th iterate of the Goldilocks Frobenius.
The coefficient and root equalities suffice; irreducibility is not needed. -/
theorem quotient_pow_card_pow (value : QuotientRing) :
    value ^ (goldilocksModulus ^ 27) = value := by
  change iterateFrobenius QuotientRing goldilocksModulus 27 value = value
  rw [frobenius_identity, RingHom.id_apply]

/-- Inverse evidence is used only to cancel in this proof. The proposed
power expression receives no inverse data. -/
theorem unit_inverse_product (value inverse : QuotientRing)
    (unit : value * inverse = 1) :
    value ^ (goldilocksModulus ^ 27 - 2) * value = 1 := by
  have lower : 1 < goldilocksModulus ^ 27 :=
    Nat.one_lt_pow (by decide : 27 ≠ 0) (by decide : 1 < goldilocksModulus)
  have predecessor : goldilocksModulus ^ 27 - 1 + 1 = goldilocksModulus ^ 27 := by omega
  have predecessorPower : value ^ (goldilocksModulus ^ 27 - 1) = 1 := by
    calc
      value ^ (goldilocksModulus ^ 27 - 1) =
          value ^ (goldilocksModulus ^ 27 - 1) * (value * inverse) := by rw [unit, mul_one]
      _ = value ^ (goldilocksModulus ^ 27) * inverse := by
        rw [← mul_assoc, ← pow_succ, predecessor]
      _ = 1 := by rw [quotient_pow_card_pow, unit]
  calc
    value ^ (goldilocksModulus ^ 27 - 2) * value =
        value ^ (goldilocksModulus ^ 27 - 1) := by
      rw [← pow_succ]
      congr 1
    _ = 1 := predecessorPower

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFFrobenius
