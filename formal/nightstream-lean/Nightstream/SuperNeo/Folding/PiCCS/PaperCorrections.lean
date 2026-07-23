import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetConvention

/-!
Frozen corrections at two inconsistent SuperNeo Section 7.3 / Appendix D.4
formula boundaries.

Owns: only the selected carried-evaluation exponent convention and the
integer root indices dictated by Definition 12's strict centered norm.

Does not own: a joint polynomial implementation, SumCheck, probability
bounds, a concrete field encoding, Rust, R1CS, or costs.

Emits constraints: no.

The displayed `Q` multiplies `Eval` by `gamma^(2K+k)`, while the displayed
target `T` uses unshifted local exponents.  `TargetConvention` proves those
exponents differ whenever a carried coordinate exists.  The frozen target
therefore uses absolute shifted exponents.

The displayed norm-product bounds are also inconsistent across Section 7.3
and Appendix D.4.  The relation itself is unambiguous: `||z||_infinity < b`.
Consequently the frozen roots are the centered integers from `-(b-1)` through
`b-1`; no displayed malformed product range is semantic authority.
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- The carried-evaluation target convention selected by the frozen
specification after the proved exponent obstruction. -/
def carriedTargetConvention : CarriedTargetConvention :=
  .coherentAbsolute

/-- The frozen convention agrees with the absolute coefficient block declared
by the joint polynomial layout. -/
theorem carriedTargetExponent_eq_absolute
    {shape : Shape}
    (coordinate : CarriedCoordinate shape) :
    carriedTargetConvention.exponent coordinate = coordinate.gammaExponent := by
  rfl

/-- The literal local target cannot be substituted for the frozen target. -/
theorem literalTargetExponent_ne_frozen
    {shape : Shape}
    (coordinate : CarriedCoordinate shape) :
    CarriedTargetConvention.literalLocal.exponent coordinate ≠
      carriedTargetConvention.exponent coordinate := by
  exact literalTargetExponent_ne_declaredCarriedExponent coordinate

/-- Integer factors literally suggested by the Section 7.3 display when its
identical lower and upper bounds are read conventionally. -/
def literalSection73NormIndices (b : Nat) : List Int :=
  [Int.ofNat (b - 1)]

/-- Root indices determined by the authoritative strict centered predicate
`|z| < b`, in increasing order. -/
def strictCenteredNormIndices (b : Nat) : List Int :=
  (List.range (2 * b - 1)).map fun offset =>
    Int.ofNat offset - Int.ofNat (b - 1)

/-- At the production-relevant base `b = 2`, the literal display has only the
root `1`, whereas the relation requires `-1`, `0`, and `1`. -/
theorem literalSection73NormIndices_ne_strictCentered_at_two :
    literalSection73NormIndices 2 ≠ strictCenteredNormIndices 2 := by
  decide

end Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections
