import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain

/-!
Order-sensitive regression for the shared numeric Boolean-domain bridge.

Owns: one concrete two-bit witness whose distinct coordinates distinguish
little-endian bit order from a reversed interpretation.

Does not own: generic bridge soundness, production arithmetic, Rust, R1CS, or
constraint counts.

| Stage path | Witness | Expected result |
|---|---|---|
| `pi_ccs.domain.numeric.weight.lsb_regression` | point `[3,5]`, index `2 = 0b10` | `(1-3)*5 = -10`, not `3*(1-5) = -12` |
-/

namespace tests.PiCcsPaperJointNumericBooleanDomain

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NumericBooleanDomain

private def integerOps : InterpolationOps Int where
  zero := 0
  one := 1
  add := Int.add
  mul := Int.mul
  neg := Int.neg

private def integerProductLaws : WeightProductLaws integerOps where
  one_mul := by intro value; simp [integerOps]
  mul_one := by intro value; simp [integerOps]
  mul_assoc := by intro left middle right; simp [integerOps, Int.mul_assoc]

private def orderSensitivePoint : CubePoint Int 2 where
  coordinates := [3, 5]
  dimension := by decide

private def binaryTen : Fin (2 ^ 2) := ⟨2, by decide⟩

/-- Index two selects the second coordinate and complements the first because
the head coordinate is bit zero. -/
example :
    tensorWeight integerOps binaryTen orderSensitivePoint = -10 := by
  rfl

/-- The preserved `Nat.testBit` path has exactly the same little-endian value. -/
example :
    testBitWeight integerOps orderSensitivePoint binaryTen = -10 := by
  rfl

/-- Reversing the two bits would produce `3 * (1 - 5) = -12`; the fixture
therefore catches MSB-first drift rather than merely testing equal coordinates. -/
example :
    tensorWeight integerOps binaryTen orderSensitivePoint ≠ -12 := by
  decide

/-- The generic bridge itself specializes to the order-sensitive fixture. -/
example :
    tensorWeight integerOps binaryTen orderSensitivePoint =
      testBitWeight integerOps orderSensitivePoint binaryTen :=
  tensorWeight_eq_testBitWeight integerOps integerProductLaws
    binaryTen orderSensitivePoint

end tests.PiCcsPaperJointNumericBooleanDomain
