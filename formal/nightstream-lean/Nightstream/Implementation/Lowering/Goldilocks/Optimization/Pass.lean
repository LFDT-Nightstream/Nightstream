import Nightstream.Implementation.Lowering.Goldilocks.Optimization.Composition

/-!
Contract: the small compositional interface returned by every optimizer pass.

Assurance tier: model-level.

Owns: one target system together with the proof that it replaces the supplied
source, and sequential composition of such results.

Does not own: pass search, cost selection, concrete rows, or a manifest.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization

universe u v

/-- A pass result cannot expose an output without its replacement proof. -/
structure Result
    {Assignment : Type u}
    {Observable : Type v}
    (source : System Assignment Observable)
    (degreeLimit : Nat) where
  target : System Assignment Observable
  replacement : Replacement source target degreeLimit

namespace Result

def identity
    {Assignment : Type u}
    {Observable : Type v}
    (source : System Assignment Observable)
    (degreeLimit : Nat)
    (withinLimit : source.degree <= degreeLimit) :
    Result source degreeLimit where
  target := source
  replacement := Replacement.identity source degreeLimit withinLimit

/-- Apply a second certified result to the target of the first result. -/
def andThen
    {Assignment : Type u}
    {Observable : Type v}
    {source : System Assignment Observable}
    {degreeLimit : Nat}
    (first : Result source degreeLimit)
    (second : Result first.target degreeLimit) :
    Result source degreeLimit where
  target := second.target
  replacement :=
    Replacement.compose first.replacement second.replacement

end Result

end Nightstream.Implementation.Lowering.Goldilocks.Optimization
