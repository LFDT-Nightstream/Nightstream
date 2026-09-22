import NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixSemantics

/-! Exact decoding and source-row preservation for the four gated state words. -/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixSemantics

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1 NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.MatrixProgram.AffineGrid
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open RunningTransitionReducedMatrixProgram RunningTransitionRetainedBlocks

def initialIndex (index : RunningTransition.StateIndex) :
    Fin RunningTransitionSourceSupport.stateCount :=
  ⟨RunningTransitionInputs.initialStateWordStart - RunningTransitionInputs.iterationWordIndex + index.val,
    by have bound := index.isLt; change index.val < 4 at bound; change 2 + index.val < 11; omega⟩

def currentIndex (index : RunningTransition.StateIndex) :
    Fin RunningTransitionSourceSupport.stateCount :=
  ⟨RunningTransitionInputs.currentStateWordStart - RunningTransitionInputs.iterationWordIndex + index.val,
    by have bound := index.isLt; change index.val < 4 at bound; change 7 + index.val < 11; omega⟩

def stateForms {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (index : RunningTransition.StateIndex) : OrdinaryRow.Forms logicalWidth :=
  { selector := SparseForm.singleton oneColumn 1
    a := SparseForm.add (SparseForm.singleton oneColumn 1)
      (SparseForm.scale (-1) (flagForm geometry))
    b := SparseForm.add
      ((RunningTransitionDirectPlan.Location.state (initialIndex index)).form geometry)
      (SparseForm.scale (-1)
        ((RunningTransitionDirectPlan.Location.state (currentIndex index)).form geometry))
    c := .empty }

theorem stateGrid_row {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (index : RunningTransition.StateIndex) :
    (stateGrid program oneColumn.val).row? logicalWidth index.val =
      some (stateForms geometry oneColumn index) := by
  let selected := region 1 1 0 RunningTransition.stateWordCount
  let major : Fin selected.majorCount := ⟨0, by change 0 < 1; omega⟩
  let middle : Fin selected.middleCount := ⟨0, by change 0 < 1; omega⟩
  let minor : Fin selected.minorCount := index
  have initial := retainedRule_form selected major middle minor (stateWire program)
    (RunningTransitionRetainedGeometry.stateFits geometry) oneColumn.val
    (RunningTransitionInputs.initialStateWordStart - RunningTransitionInputs.iterationWordIndex)
    0 0 1 (1 : F) (by
      change 2 + 0 * 0 + 0 * 0 + index.val * 1 < 11
      have bound := index.isLt; change index.val < 4 at bound; omega)
  have current := retainedRule_form selected major middle minor (stateWire program)
    (RunningTransitionRetainedGeometry.stateFits geometry) oneColumn.val
    (RunningTransitionInputs.currentStateWordStart - RunningTransitionInputs.iterationWordIndex)
    0 0 1 (-1 : F) (by
      change 7 + 0 * 0 + 0 * 0 + index.val * 1 < 11
      have bound := index.isLt; change index.val < 4 at bound; omega)
  have right := AffineGrid.Program.two_form?_of_results _ _ oneColumn.val
    ⟨selected.majorStart + major.val, selected.middleStart + middle.val,
      selected.minorStart + minor.val⟩ _ _ initial current
  have negative : (-1 : F) ≠ 1 := by decide
  have decoded := MultiplicationGrid.Block.row?_of_results (stateGrid program oneColumn.val)
    oneColumn rfl major middle minor
    (stateForms geometry oneColumn index).a (stateForms geometry oneColumn index).b
    (stateForms geometry oneColumn index).c
    (by simpa only [selected, region, Nat.zero_add, stateGrid, stateForms] using
      baseProgram_form geometry selected major middle minor oneColumn) (by
      simpa only [stateGrid, stateForms, selected, region, major, middle, minor,
        Nat.zero_mul, Nat.mul_one, Nat.add_zero, Nat.zero_add, applyCoefficient,
        if_pos rfl, if_neg negative, ↓reduceIte, addSelected, stateWire, wireForm,
        RetainedBlock.ofSemantic, RetainedBlock.semantic,
        RunningTransitionDirectPlan.Location.form, initialIndex, currentIndex,
        SparseForm.add, SparseForm.empty, List.nil_append] using right) rfl
  simpa only [major, middle, minor, Fin.encodeProd, Fin.mkDivMod, Nat.zero_mul,
    Nat.zero_add, Nat.mul_zero, stateForms] using decoded

theorem stateForms_preserve {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment oneColumn = 1)
    (index : RunningTransition.StateIndex) :
    (stateForms geometry oneColumn index).Preserves assignment (decodedEnv geometry assignment)
      (RunningTransitionReducedRows.stateRow index) := by
  have initial := congrArg (fun form : SparseForm logicalWidth => form.eval assignment)
    (sourceForm_state geometry (initialIndex index))
  have current := congrArg (fun form : SparseForm logicalWidth => form.eval assignment)
    (sourceForm_state geometry (currentIndex index))
  have initialAddress : (RunningTransitionDirectPlan.Location.state (initialIndex index)).sourceColumn =
      PilotProduction.priorPreimageStart + RunningTransitionInputs.initialStateWordStart + index.val := by
    change 28 + (2 + index.val) = 0 + 30 + index.val
    omega
  have currentAddress : (RunningTransitionDirectPlan.Location.state (currentIndex index)).sourceColumn =
      PilotProduction.priorPreimageStart + RunningTransitionInputs.currentStateWordStart + index.val := by
    change 28 + (7 + index.val) = 0 + 35 + index.val
    omega
  rw [initialAddress] at initial
  rw [currentAddress] at current
  have binding := bindingForms_preserve geometry oneColumn assignment one
  refine ⟨by simp [stateForms, one], binding.2.2.1, ?_, ?_⟩
  · change (SparseForm.add
        ((RunningTransitionDirectPlan.Location.state (initialIndex index)).form geometry)
        (SparseForm.scale (-1)
          ((RunningTransitionDirectPlan.Location.state (currentIndex index)).form geometry))).eval assignment =
      (R1CS.LinearCombination.add
        (R1CS.LinearCombination.ofVar
          (PilotProduction.priorPreimageStart + RunningTransitionInputs.initialStateWordStart + index.val))
        (R1CS.LinearCombination.scale (-1) (R1CS.LinearCombination.ofVar
          (PilotProduction.priorPreimageStart + RunningTransitionInputs.currentStateWordStart + index.val)))).eval
          (decodedEnv geometry assignment)
    simp only [SparseForm.add_eval, SparseForm.scale_eval, R1CS.LinearCombination.eval_add,
      R1CS.LinearCombination.eval_scale, R1CS.LinearCombination.eval_ofVar]
    exact congrArg₂ (fun left right : F => left + -1 * right) initial.symm current.symm
  · change SparseForm.empty.eval assignment =
      R1CS.LinearCombination.zero.eval (decodedEnv geometry assignment)
    simp only [SparseForm.empty_eval, R1CS.LinearCombination.eval_zero]

end NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixSemantics
