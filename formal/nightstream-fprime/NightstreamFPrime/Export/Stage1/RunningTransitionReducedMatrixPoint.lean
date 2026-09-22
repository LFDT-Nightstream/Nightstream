import NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixSemantics
import NightstreamFPrime.Layout.Stage1.RunningTransitionWordIndexing
import Mathlib.Data.List.Forall2

/-! Exact retained-Poseidon readout for the 56 running-point component rows. -/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixSemantics

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1 NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.MatrixProgram.AffineGrid
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open RunningTransitionReducedMatrixProgram RunningTransitionRetainedBlocks

def pointOutputIndex (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) :
    Fin RunningTransitionSourceSupport.outputCount :=
  ⟨PiCCSInputs.runningPointStart + coordinate.val * 2 + part.val, by
    have coordinateBound := coordinate.isLt
    have partBound := part.isLt
    change coordinate.val < 28 at coordinateBound
    change 40 + coordinate.val * 2 + part.val < 49393
    omega⟩

def pointLane {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) (lane : Fin 8) :
    SparseForm logicalWidth :=
  (PiCCSPoseidonPlan.retainedBlock program).form (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSPoseidonPlan.retainedFits (RunningTransitionRetainedGeometry.poseidonGeometry geometry))
    ⟨40670 + part.val * 86 + coordinate.val * 774 + lane.val, by
      have coordinateBound := coordinate.isLt
      have partBound := part.isLt
      have laneBound := lane.isLt
      change coordinate.val < 28 at coordinateBound
      rw [PiCCSPoseidonPlan.retainedBlock_slotCount]
      omega⟩

theorem pointLane_external {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) :
    SparseLayer.external (pointLane geometry coordinate part) 0 =
      PiCCSTranscriptOutputForms.pointForm
        (RunningTransitionRetainedGeometry.poseidonGeometry geometry) coordinate part := by
  unfold PiCCSTranscriptOutputForms.pointForm PiCCSTranscriptOutputForms.transcriptForm
    PiCCSPoseidonPlan.outputState PoseidonRetainedFamily.outputState
  apply congrArg (fun state => SparseLayer.external state (0 : Fin 8))
  funext lane
  unfold pointLane PoseidonRetainedFamily.form
  apply congrArg ((PiCCSPoseidonPlan.retainedBlock program).form
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSPoseidonPlan.retainedFits (RunningTransitionRetainedGeometry.poseidonGeometry geometry)))
  apply Fin.ext
  simp [PoseidonRetainedFamily.slot, Fin.encodeProd,
    PoseidonRetainedSlots.finalRow_val, PiCCSTranscriptOutputForms.invocation,
    PiCCSTranscriptOutputForms.pointInvocation]
  omega

private def pointTermForm {logicalWidth : Nat} (lanes : Fin 8 → SparseForm logicalWidth)
    (term : Nat × Nat) : SparseForm logicalWidth :=
  if bound : term.1 < 8 then
    applyCoefficient (Spec.Poseidon2.ofNat term.2) (lanes ⟨term.1, bound⟩)
  else .empty

private theorem pointTerm_bound (term : Nat × Nat) (member : term ∈ pointTerms) :
    term.1 < 8 := by
  simp only [pointTerms, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

private theorem pointTerms_external {logicalWidth : Nat}
    (lanes : Fin 8 → SparseForm logicalWidth) :
    combine (pointTerms.map fun term => some (pointTermForm lanes term)) =
      SparseLayer.external lanes 0 := by
  have two : Spec.Poseidon2.ofNat 2 ≠ (1 : F) := by decide
  have three : Spec.Poseidon2.ofNat 3 ≠ (1 : F) := by decide
  have one : Spec.Poseidon2.ofNat 1 = (1 : F) := rfl
  simp [pointTerms, pointTermForm, combine, addSelected, applyCoefficient, two, three, one,
    SparseLayer.external, SparseLayer.block, SparseLayer.mat4, SparseLayer.get,
    SparseLayer.add, SparseLayer.scale, SparseForm.add, SparseForm.empty, List.append_assoc]

theorem pointRule_form {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) (part : Fin 2)
    (oneColumn : Nat) (term : Nat × Nat) (member : term ∈ pointTerms) :
    (retained (region productionShape.cubeVariables 2 0 1) (poseidonWire program)
      ((PiCCSTranscriptOutputForms.pointGrid program 0).slotStart + term.1)
      (PiCCSTranscriptOutputForms.pointGrid program 0).majorSlotStride
      ((PiCCSTranscriptOutputForms.pointGrid program 1).slotStart -
        (PiCCSTranscriptOutputForms.pointGrid program 0).slotStart) 0
      (Spec.Poseidon2.ofNat term.2)).form? logicalWidth oneColumn
        ⟨coordinate.val, part.val, 0⟩ =
      some (some (pointTermForm (pointLane geometry coordinate part) term)) := by
  have laneBound := pointTerm_bound term member
  have coordinateBound := coordinate.isLt
  have partBound := part.isLt
  change coordinate.val < 28 at coordinateBound
  let selected := region productionShape.cubeVariables 2 0 1
  have loaded := retainedRule_form selected coordinate part ⟨0, by change 0 < 1; omega⟩
    (poseidonWire program)
    (PiCCSPoseidonPlan.retainedFits (RunningTransitionRetainedGeometry.poseidonGeometry geometry))
    oneColumn
    ((PiCCSTranscriptOutputForms.pointGrid program 0).slotStart + term.1)
    (PiCCSTranscriptOutputForms.pointGrid program 0).majorSlotStride
    ((PiCCSTranscriptOutputForms.pointGrid program 1).slotStart -
      (PiCCSTranscriptOutputForms.pointGrid program 0).slotStart) 0
    (Spec.Poseidon2.ofNat term.2) (by
      simp only [poseidonWire, RetainedBlock.ofSemantic,
        PiCCSPoseidonPlan.retainedBlock_slotCount, PiCCSTranscriptOutputForms.pointGrid,
        SourceGrid.externalOfSemantic, SourceGrid.ofSemantic, Fin.val_zero, Fin.val_one,
        Nat.zero_mul, Nat.one_mul, Nat.add_zero]
      omega)
  have reorder : 40670 + term.1 + coordinate.val * 774 + part.val * 86 + 0 * 0 =
      40670 + part.val * 86 + coordinate.val * 774 + term.1 := by omega
  simpa only [selected, region, Nat.zero_add, pointTermForm, dif_pos laneBound,
    poseidonWire, wireForm_ofSemantic,
    PiCCSTranscriptOutputForms.pointGrid, SourceGrid.externalOfSemantic,
    SourceGrid.ofSemantic, Fin.val_zero, Fin.val_one, Nat.zero_mul, Nat.one_mul,
    Nat.add_zero, show 40670 + 86 - 40670 = 86 by decide, pointLane, reorder] using loaded

theorem pointRight_form {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) (oneColumn : Nat) :
    (RunningTransitionReducedMatrixProgram.pointGrid program oneColumn).right.form?
      logicalWidth oneColumn ⟨coordinate.val, part.val, 0⟩ =
      some (PiCCSTranscriptOutputForms.pointForm
        (RunningTransitionRetainedGeometry.poseidonGeometry geometry) coordinate part) := by
  have loaded := AffineGrid.Program.form?_of_results
    (RunningTransitionReducedMatrixProgram.pointGrid program oneColumn).right
    oneColumn ⟨coordinate.val, part.val, 0⟩
    (pointTerms.map fun term => some (pointTermForm (pointLane geometry coordinate part) term)) (by
      simp only [RunningTransitionReducedMatrixProgram.pointGrid]
      rw [List.forall₂_map_left_iff, List.forall₂_map_right_iff, List.forall₂_same]
      intro term member
      exact pointRule_form geometry coordinate part oneColumn term member)
  rw [pointTerms_external, pointLane_external] at loaded
  exact loaded

def pointForms {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) :
    OrdinaryRow.Forms logicalWidth :=
  { selector := SparseForm.singleton oneColumn 1
    a := flagForm geometry
    b := PiCCSTranscriptOutputForms.pointForm
      (RunningTransitionRetainedGeometry.poseidonGeometry geometry) coordinate part
    c := (RunningTransitionDirectPlan.Location.output (pointOutputIndex coordinate part)).form geometry }

theorem pointGrid_row {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) :
    (RunningTransitionReducedMatrixProgram.pointGrid program oneColumn.val).row?
      logicalWidth (coordinate.val * 2 + part.val) =
      some (pointForms geometry oneColumn coordinate part) := by
  let selected := region productionShape.cubeVariables 2 0 1
  let minor : Fin selected.minorCount := ⟨0, by change 0 < 1; omega⟩
  have output := retainedRule_form selected coordinate part minor (outputWire program)
    (RunningTransitionRetainedGeometry.outputFits geometry) oneColumn.val
    PiCCSInputs.runningPointStart 2 1 0 (1 : F) (by
      have bound := (pointOutputIndex coordinate part).isLt
      simpa only [pointOutputIndex, minor, Nat.mul_one, Nat.zero_mul, Nat.add_zero,
        outputWire, RetainedBlock.ofSemantic] using bound)
  have decoded := MultiplicationGrid.Block.row?_of_results
    (RunningTransitionReducedMatrixProgram.pointGrid program oneColumn.val)
    oneColumn rfl coordinate part minor
    (pointForms geometry oneColumn coordinate part).a
    (pointForms geometry oneColumn coordinate part).b
    (pointForms geometry oneColumn coordinate part).c
    (by
      change (flagProgram program selected).form? logicalWidth oneColumn.val
        ⟨coordinate.val, part.val, minor.val⟩ = some (flagForm geometry)
      simpa only [selected, region, Nat.zero_add] using
        flagProgram_form geometry selected coordinate part minor oneColumn.val)
    (pointRight_form geometry coordinate part oneColumn.val) (by
      apply AffineGrid.Program.singleton_form?_of_selected
      simpa only [selected, region, minor, Nat.zero_add, Nat.mul_one, Nat.zero_mul,
        Nat.add_zero, applyCoefficient, if_pos rfl, pointForms, pointOutputIndex,
        outputWire, wireForm_ofSemantic,
        RunningTransitionDirectPlan.Location.form] using output)
  have shape : (RunningTransitionReducedMatrixProgram.pointGrid program oneColumn.val).shape =
      ⟨productionShape.cubeVariables, 2, 1⟩ := rfl
  have address :
      (RunningTransitionReducedMatrixProgram.pointGrid program oneColumn.val).shape.middleCount *
          (RunningTransitionReducedMatrixProgram.pointGrid program oneColumn.val).shape.minorCount * coordinate.val +
        (RunningTransitionReducedMatrixProgram.pointGrid program oneColumn.val).shape.minorCount * part.val =
      coordinate.val * 2 + part.val := by
    rw [shape]
    simp only [Nat.mul_one, Nat.one_mul, Nat.mul_comm]
  simp only [minor, Fin.encodeProd, Fin.mkDivMod, Nat.add_zero] at decoded
  rw [address] at decoded
  exact decoded

private theorem recursivePoint_component {sourceWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth sourceWidth}
    (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) :
    RunningTransitionWordIndexing.component
        ((RunningTransitionInputs.recursiveRunningExpr sourceWidth publicFits).point coordinate) part =
      Expr.var (PiCCSTranscriptOutputForms.pointSource coordinate part) := by
  rw [RunningTransitionInputs.recursivePoint_eq_direct]
  by_cases zero : part.val = 0
  · simp only [RunningTransitionWordIndexing.component, zero, ↓reduceIte,
      RunningTransitionInputs.directRoundPoint, PiCCSTranscriptOutputForms.pointSource,
      PiCCSTranscriptOutputForms.pointSourceStart, Nat.zero_mul, Nat.add_zero]
    congr 1
    omega
  · have one : part.val = 1 := by have bound := part.isLt; omega
    simp only [RunningTransitionWordIndexing.component, zero, ↓reduceIte,
      RunningTransitionInputs.directRoundPoint, PiCCSTranscriptOutputForms.pointSource,
      PiCCSTranscriptOutputForms.pointSourceStart, one, Nat.one_mul]
    congr 1
    norm_num [RunningTransitionInputs.roundSampleC0Offset,
      RunningTransitionInputs.roundSampleC1Offset]
    omega

private theorem outputPoint_component {sourceWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth sourceWidth}
    (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) :
    RunningTransitionWordIndexing.component
        ((RunningTransitionInputs.outputRunningExpr sourceWidth publicFits).point coordinate) part =
      Expr.var (RunningTransitionDirectPlan.Location.output
        (pointOutputIndex coordinate part)).sourceColumn := by
  by_cases zero : part.val = 0
  · simp only [RunningTransitionWordIndexing.component, zero, ↓reduceIte,
      RunningTransitionInputs.outputRunningExpr, RunningTransitionInputs.outputPoint,
      RunningTransitionInputs.outputPairAt, RunningTransitionDirectPlan.Location.sourceColumn,
      pointOutputIndex]
    congr 1
  · have one : part.val = 1 := by have bound := part.isLt; omega
    simp only [RunningTransitionWordIndexing.component, zero, ↓reduceIte,
      RunningTransitionInputs.outputRunningExpr, RunningTransitionInputs.outputPoint,
      RunningTransitionInputs.outputPairAt, RunningTransitionDirectPlan.Location.sourceColumn,
      pointOutputIndex, one]
    congr 1

theorem pointForms_preserve {program : ApplicationProgram} {logicalWidth sourceWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth sourceWidth}
    (relation : ProductionKey.LogicalRelation sourceWidth publicFits)
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment oneColumn = 1)
    (coordinate : Fin productionShape.cubeVariables) (part : Fin 2) :
    (pointForms geometry oneColumn coordinate part).Preserves assignment
      (decodedEnv geometry assignment)
      (RunningTransitionReducedRows.muxRow sourceWidth publicFits
        (RunningTransitionWordIndexing.pointIndex coordinate part)) := by
  have values := RunningTransitionReducedRows.muxRow_values relation
    (decodedEnv geometry assignment) (RunningTransitionWordIndexing.pointIndex coordinate part)
  simp only [RunningTransitionWordIndexing.runningWord_point,
    RunningTransitionWordIndexing.defaultWord_point, sub_zero] at values
  refine ⟨by simp [pointForms, one], ?_, ?_, ?_⟩
  · rw [values.1]
    change (flagForm geometry).eval assignment =
      (sourceForm geometry RunningTransitionReducedRows.flagIndex).eval assignment
    rw [sourceForm_flag]
  · rw [values.2.1, recursivePoint_component, Expr.eval_var]
    change (PiCCSTranscriptOutputForms.pointForm
        (RunningTransitionRetainedGeometry.poseidonGeometry geometry) coordinate part).eval assignment =
      (sourceForm geometry (PiCCSTranscriptOutputForms.pointSource coordinate part)).eval assignment
    rw [sourceForm_point]
  · rw [values.2.2, outputPoint_component, Expr.eval_var]
    change ((RunningTransitionDirectPlan.Location.output (pointOutputIndex coordinate part)).form geometry).eval assignment =
      (sourceForm geometry (RunningTransitionDirectPlan.Location.output
        (pointOutputIndex coordinate part)).sourceColumn).eval assignment
    rw [sourceForm_output]

def pointHeaderForms {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth) :
    OrdinaryRow.Forms logicalWidth :=
  ⟨SparseForm.singleton oneColumn 1, flagForm geometry, .empty, .empty⟩

theorem pointHeaderGrid_row {program : ApplicationProgram} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth) :
    (pointHeaderGrid program oneColumn.val).row? logicalWidth 0 =
      some (pointHeaderForms geometry oneColumn) := by
  apply MultiplicationGrid.Block.row?_of_results (pointHeaderGrid program oneColumn.val)
    oneColumn rfl ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩
  · exact flagProgram_form geometry (region 1 1 0 1)
      ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ ⟨0, by change 0 < 1; omega⟩ oneColumn.val
  · rfl
  · rfl

theorem pointHeaderForms_preserve {program : ApplicationProgram} {logicalWidth sourceWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth sourceWidth}
    (relation : ProductionKey.LogicalRelation sourceWidth publicFits)
    (geometry : Geometry program logicalWidth) (oneColumn : Fin logicalWidth)
    (assignment : Assignment F logicalWidth) (one : assignment oneColumn = 1) :
    (pointHeaderForms geometry oneColumn).Preserves assignment (decodedEnv geometry assignment)
      (RunningTransitionReducedRows.muxRow sourceWidth publicFits ⟨0, by decide⟩) := by
  have values := RunningTransitionReducedRows.muxRow_values relation
    (decodedEnv geometry assignment) ⟨0, by decide⟩
  simp only [RunningTransitionWordIndexing.runningWord_pointHeader,
    RunningTransitionWordIndexing.defaultWord_pointHeader, Expr.eval_const, sub_self] at values
  refine ⟨by simp [pointHeaderForms, one], ?_, ?_, ?_⟩
  · rw [values.1]
    change (flagForm geometry).eval assignment =
      (sourceForm geometry RunningTransitionReducedRows.flagIndex).eval assignment
    rw [sourceForm_flag]
  · rw [values.2.1]
    exact SparseForm.empty_eval assignment
  · rw [values.2.2]
    exact SparseForm.empty_eval assignment

end NightstreamFPrime.Export.Stage1.RunningTransitionReducedMatrixSemantics
