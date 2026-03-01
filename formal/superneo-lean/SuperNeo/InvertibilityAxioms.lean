import SuperNeo.Norm

/-!
Low-norm invertibility assumption boundary (Theorem 8 style).

This module stays explicit about trust boundaries:
- `invertibleRq` is the target property for a ring element,
- `invertibilityWindowProp` is the norm-window precondition,
- `lowNormInvertibilityAssumption` is the external theorem/assumption surface.
-/

namespace SuperNeo

/-- Ring-level invertibility witness predicate. -/
def invertibleRq (a : Coeffs) : Prop :=
  ∃ aInv : Coeffs, mulRq a aInv = oneRq

/-- Norm-window precondition used by low-norm invertibility statements. -/
def invertibilityWindowProp (B : Nat) (a : Coeffs) : Prop :=
  normInfCoeffs a ≤ B

/-- External boundary: low-norm elements are invertible in `Rq`. -/
def lowNormInvertibilityAssumption (B : Nat) : Prop :=
  ∀ a : Coeffs, invertibilityWindowProp B a → invertibleRq a

/-- Use the low-norm invertibility boundary to extract an inverse witness. -/
theorem invertibleRq_of_lowNormAssumption
  {B : Nat} {a : Coeffs}
  (hInv : lowNormInvertibilityAssumption B)
  (hWin : invertibilityWindowProp B a) :
  invertibleRq a := by
  exact hInv a hWin


end SuperNeo
