import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.FixtureRefinement

/-!
Contract: value-level bridge from one final low-norm assignment to the source
and derived assignments used by the grouped-product rewrite semantics.

Assurance tier: model-level artifact interpreter.

Owns: decoded source and derived assignments and evaluation of source,
output, predecessor, factor, and C-port forms on those same values.

Does not own: a concrete generated row, selector authority, source-row
reconstruction, production coverage, or permission to remove coordinates.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalAssignment

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation

/-- Source assignment decoded from the exact final low-norm slots. -/
def sourceAssignment {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) (assignment : Fin finalColumns → F) :
    Fin sourceColumns → F :=
  decodedSourceAssignment columnsPositive slots definitions fuel assignment

/-- Compiler-owned accumulator assignment decoded from its final low-norm
slots. Missing artifact indices fail closed to zero. -/
def derivedAssignment {finalColumns : Nat}
    (slots : List (DecodedDerivedSlot finalColumns))
    (assignment : Fin finalColumns → F) : Nat → F :=
  fun compilerIndex =>
    match findDerivedSlot slots compilerIndex with
    | some slot => Form.evaluate (derivedSlotForm slot) assignment
    | none => 0

private theorem evaluate_sub {columns : Nat}
    (left right : Form columns) (assignment : Fin columns → F) :
    Form.evaluate (Form.sub left right) assignment =
      Form.evaluate left assignment - Form.evaluate right assignment := by
  simp [Form.sub, Fin.sub_eq_add_neg, Lean.Grind.Fin.neg_mul, Fin.one_mul]

/-- The expanded final image of a source LC is its direct value on the
decoded source assignment when the final constant wire is one. -/
theorem evaluate_sourceLinearForm_eq_linearValue
    {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel : Nat) (value : DecodedSourceLinearCombination sourceColumns)
    (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, columnsPositive⟩ = 1) :
    Form.evaluate
        (sourceLinearForm columnsPositive slots definitions fuel value)
        assignment =
      linearValue value
        (sourceAssignment columnsPositive slots definitions fuel assignment)
        1 := by
  rw [evaluate_sourceLinearForm]
  rw [sourceLinearValue_eq_direct_of_agreement columnsPositive slots
    definitions fuel value assignment
    (sourceAssignment columnsPositive slots definitions fuel assignment)]
  · rw [constantOne]
  · intro term _
    rfl

/-- Evaluation of an output form is the exact source-level output value. -/
theorem evaluate_outputForm_eq_outputValue
    {sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot finalColumns))
    (fuel : Nat) (output : DecodedOutput sourceColumns)
    (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, columnsPositive⟩ = 1) :
    Form.evaluate
        (outputForm columnsPositive slots definitions derived fuel output)
        assignment =
      outputValue
        (sourceAssignment columnsPositive slots definitions fuel assignment)
        1 (derivedAssignment derived assignment) output := by
  cases output with
  | source value =>
      exact evaluate_sourceLinearForm_eq_linearValue columnsPositive slots
        definitions fuel value assignment constantOne
  | derivedProductSum compilerIndex =>
      cases found : findDerivedSlot derived compilerIndex <;>
        simp [outputForm, outputValue, derivedAssignment, found]

/-- Evaluation of a predecessor form is the exact derived predecessor. -/
theorem evaluate_previousForm_eq_previousValue
    {finalColumns : Nat}
    (derived : List (DecodedDerivedSlot finalColumns))
    (previous : Option Nat) (assignment : Fin finalColumns → F) :
    Form.evaluate (previousForm derived previous) assignment =
      previousValue (derivedAssignment derived assignment) previous := by
  cases previous with
  | none => simp [previousForm, previousValue]
  | some compilerIndex =>
      cases found : findDerivedSlot derived compilerIndex <;>
        simp [previousForm, previousValue, derivedAssignment, found]

/-- Exact C-port value `output - base - previous`. -/
theorem evaluate_expectedCForm
    {rows sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (derived : List (DecodedDerivedSlot finalColumns))
    (fuel : Nat) (step : DecodedStep rows sourceColumns)
    (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, columnsPositive⟩ = 1) :
    Form.evaluate
        (expectedCForm columnsPositive slots definitions derived fuel step)
        assignment =
      outputValue
          (sourceAssignment columnsPositive slots definitions fuel assignment)
          1 (derivedAssignment derived assignment) step.output -
        linearValue step.base
          (sourceAssignment columnsPositive slots definitions fuel assignment)
          1 -
        previousValue (derivedAssignment derived assignment) step.previous := by
  unfold expectedCForm
  rw [evaluate_sub, evaluate_sub]
  rw [evaluate_outputForm_eq_outputValue columnsPositive slots definitions
    derived fuel step.output assignment constantOne]
  rw [evaluate_sourceLinearForm_eq_linearValue columnsPositive slots
    definitions fuel step.base assignment constantOne]
  rw [evaluate_previousForm_eq_previousValue]

/-- A present factor's two final forms evaluate to its exact source values. -/
theorem evaluate_factorForms
    {rows sourceColumns finalColumns : Nat}
    (columnsPositive : 0 < finalColumns)
    (slots : List (DecodedSourceSlot sourceColumns finalColumns))
    (definitions : List (DecodedSourceDefinition sourceColumns))
    (fuel index : Nat) (step : DecodedStep rows sourceColumns)
    (factor : DecodedFactor sourceColumns)
    (factorAt : step.factors[index]? = some factor)
    (assignment : Fin finalColumns → F)
    (constantOne : assignment ⟨0, columnsPositive⟩ = 1) :
    Form.evaluate
          (factorFormAt columnsPositive slots definitions fuel index true step)
          assignment =
        factor.coefficient *
          linearValue factor.left
            (sourceAssignment columnsPositive slots definitions fuel assignment)
            1 ∧
      Form.evaluate
          (factorFormAt columnsPositive slots definitions fuel index false step)
          assignment =
        linearValue factor.right
          (sourceAssignment columnsPositive slots definitions fuel assignment)
          1 := by
  constructor
  · simp only [factorFormAt, factorAt, if_pos, factorLeftForm,
      Form.evaluate_scale]
    rw [evaluate_sourceLinearForm_eq_linearValue columnsPositive slots
      definitions fuel factor.left assignment constantOne]
  · simp only [factorFormAt, factorAt, Bool.false_eq_true, if_false,
      factorRightForm]
    rw [evaluate_sourceLinearForm_eq_linearValue columnsPositive slots
      definitions fuel factor.right assignment constantOne]

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalAssignment
