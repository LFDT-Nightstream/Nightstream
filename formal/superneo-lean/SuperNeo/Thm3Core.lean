import SuperNeo.BarLift

/-!
Theorem-3 inner-product transform scaffold.

This file defines a compact inner-product identity boundary and provides the
native proof for the current bar-lift scaffold (`barLiftVector = id`).
-/

namespace SuperNeo

open F

/-- Dot/inner product with an explicit size guard. -/
def innerProduct (a b : Array F) : F :=
  if _h : a.size = b.size then
    (List.range a.size).foldl (fun acc i => acc + a[i]! * b[i]!) 0
  else
    0

/-- Theorem-facing Theorem-3 statement for bar-lifted inner products. -/
def thm3CoreAssumption (bar : Array (Array F)) : Prop :=
  ∀ a b : Array F,
    a.size = b.size →
    innerProduct a b = innerProduct (barLiftVector bar a) (barLiftVector bar b)

/-- Native Theorem-3 proof for the compact scaffold (`barLiftVector = id`). -/
theorem thm3CoreAssumption_native (bar : Array (Array F)) :
  thm3CoreAssumption bar := by
  intro a b hSize
  simp [innerProduct, barLiftVector, hSize]


end SuperNeo
