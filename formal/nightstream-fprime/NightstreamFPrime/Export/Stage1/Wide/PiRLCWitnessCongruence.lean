import NightstreamFPrime.Export.Stage1.Wide.PiRLCWitness

/-! The direct PiRLC constructor does not read its overwritten allocation. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiRLCWitness

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation PiRlcWideSampler PiRLCGeometry
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

theorem assignment_eq_of_agrees_outside {columns : Nat} (interface : Interface columns)
    (left right : Assignment F columns) (inputs : InputsBefore interface)
    (agrees : ∀ column, column.val < interface.start ∨
      interface.start + PiRLCGeometry.coordinateCount ≤ column.val → left column = right column) :
    assignment interface left = assignment interface right := by
  have initialEq : initial interface left = initial interface right := by
    funext lane
    exact FieldAssignment.form_eval_eq _ _ _ (fun entry member =>
      agrees entry.column (Or.inl (inputs.sampler.input lane entry member)))
  have valuesEq : inputValues interface left = inputValues interface right := by
    funext ring lane
    exact FieldAssignment.form_eval_eq _ _ _ (fun entry member =>
      agrees entry.column (Or.inl (inputs.value ring lane entry member)))
  have fieldsEq : fieldValues interface left = fieldValues interface right := by
    simp only [fieldValues, initialEq, valuesEq]
  funext column
  unfold assignment Witness.assignment
  rw [initialEq]
  split
  · rfl
  · rename_i outsideSampler
    unfold fieldBase FieldAssignment.write
    rw [fieldsEq]
    split
    · rfl
    · rename_i outsideFields
      apply agrees
      change ¬(interface.start ≤ column.val ∧ column.val < interface.start + 135813) at outsideSampler
      change ¬(interface.start + 135813 ≤ column.val ∧
        column.val < interface.start + 135813 + 104652 * 41) at outsideFields
      rw [PiRLCGeometry.coordinateCount_eq]
      omega

end NightstreamFPrime.Export.Stage1.Wide.PiRLCWitness
