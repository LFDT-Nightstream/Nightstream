import SuperNeo.Field

/-!
Equality-polynomial scaffold.

This module keeps a compact executable definition of the `eq` polynomial and a
clear theorem-facing assumption boundary for selector behavior.
-/

namespace SuperNeo

open F

/-- Single-coordinate equality term `x*y + (1-x)*(1-y)`. -/
def eqTerm (x y : F) : F :=
  x * y + (1 - x) * (1 - y)

/-- Product equality polynomial over all coordinates (size-matched inputs only). -/
def eqPoly (x y : Array F) : F :=
  if _h : x.size = y.size then
    (List.range x.size).foldl (fun acc i => acc * eqTerm x[i]! y[i]!) 1
  else
    0

/-- Selector-style proposition for the equality polynomial. -/
def eqPolySelectorProp (x y : Array F) : Prop :=
  (x = y → eqPoly x y = 1) ∧
  (x ≠ y → eqPoly x y = 0)

/-- Theorem-facing boundary: selector behavior on size-compatible vectors. -/
def eqPolyAssumption : Prop :=
  ∀ x y : Array F, x.size = y.size → eqPolySelectorProp x y

theorem eqPoly_eq_zero_of_size_ne
  {x y : Array F}
  (hNe : x.size ≠ y.size) :
  eqPoly x y = 0 := by
  unfold eqPoly
  simp [hNe]


end SuperNeo
