import Nightstream.Implementation.Lowering.Nebula.ProductSemantics

set_option autoImplicit false

namespace tests.NebulaProductSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Rows.LinearCombination
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ProductSemantics

private def fixtureRows : List Row :=
  extensionRows .readProduct 0
    (constant 1) zero
    (constant 1) zero
    (constant 1) zero
    zero zero zero zero zero

private def fixtureAssignment : Nat → F :=
  fun column => if column = 0 then 1 else 0

/-- A one-factor identity update satisfies both emitted extension rows. -/
theorem honest_extension_update :
    Satisfies fixtureRows fixtureAssignment := by
  intro row member
  simp [fixtureRows, extensionRows] at member
  rcases member with rfl | rfl <;>
    rw [extensionUpdateRow_holds_iff] <;>
    decide

/-- The semantic bridge reads the same identity update as multiplication in
the selected quadratic extension. -/
theorem honest_extension_update_sound :
    evaluatePair fixtureAssignment (constant 1) zero =
      K.mul (evaluatePair fixtureAssignment (constant 1) zero)
        (gatedFactor fixtureAssignment (constant 1) zero
          zero zero zero zero zero) := by
  exact extensionRows_sound .readProduct 0
    (constant 1) zero
    (constant 1) zero
    (constant 1) zero
    zero zero zero zero zero
    fixtureAssignment honest_extension_update

/-- Changing the low output while leaving the two input factors unchanged is
rejected by the emitted rows. -/
theorem mutated_extension_output_rejected :
    ¬ Satisfies
      (extensionRows .readProduct 0
        (constant 2) zero
        (constant 1) zero
        (constant 1) zero
        zero zero zero zero zero)
      fixtureAssignment := by
  intro satisfied
  have low := satisfied
    (extensionUpdateRow (id .readProduct 0 0 0)
      (constant 2) (constant 1) zero (constant 1) zero
      zero zero zero zero zero)
    (by simp [extensionRows, zero, scale])
  rw [extensionUpdateRow_holds_iff] at low
  have notEqual : (-2 + 1 + 0 + -0 + 0 + -0 : F) ≠ 0 := by
    decide
  apply notEqual
  simpa [fixtureAssignment, eval, constant, zero] using low

end tests.NebulaProductSemantics
