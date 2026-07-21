import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits

/-! Focused interface regression for the model-level common-sign radix family. -/

namespace Nightstream.Tests.PiDECUniformSignedDigits

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits

example (parent : F)
    (bounded : centeredMagnitude parent < combinedBound) :
    ∃ sign, Accepted parent sign (splitScalar parent) := by
  exact ⟨honestSign parent, honest_complete parent bounded⟩

example {parent sign : F} {digits : ChildIndex → F}
    (bounded : centeredMagnitude parent < combinedBound)
    (accepted : Accepted parent sign digits) :
    digits = splitScalar parent := by
  exact accepted_digits_eq_splitScalar bounded accepted

example {parent : F} {digits : ChildIndex → F}
    (bounded : centeredMagnitude parent < combinedBound) :
    (∃ sign, Accepted parent sign digits) ↔
      digits = splitScalar parent := by
  exact exists_accepted_iff_exact bounded

example {parent sign : F} {digits : ChildIndex → F}
    (accepted : Accepted parent sign digits) :
    centeredMagnitude parent < combinedBound := by
  exact accepted.parentBounded

example {parent sign : F} {digits : ChildIndex → F}
    (accepted : Accepted parent sign digits) :
    digits = splitScalar parent := by
  exact accepted.digits_eq_splitScalar

end Nightstream.Tests.PiDECUniformSignedDigits
