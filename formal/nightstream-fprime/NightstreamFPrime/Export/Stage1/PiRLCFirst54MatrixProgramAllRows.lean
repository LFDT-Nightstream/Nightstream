import NightstreamFPrime.Export.Stage1.PiRLCFirst54PositionMatrixProgramSemantics

/-!
Owns the final-pin and four-block row coverage for the compact PiRLC
First54 matrix program.

This module does not compose the preceding PiRLC product family.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCFirst54MatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

theorem finalPin_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (row : Fin PiRLCFirst54DirectSchedule.finalCount) :
    (finalPin geometry).row? logicalWidth row.val =
      some (PinFamilyPlan.forms
        (PiRLCFirst54DirectPlan.finalInterface
          (PiRLCRetainedInputs.first54Inputs geometry)) row) := by
  exact Pin.Block.row?_ofSemantic _ row

theorem positionBlock_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin PiRLCFirst54DirectSchedule.positionCount) :
    (MatrixProgram.Block.multiplicationGrid (positionGrid geometry)).row?
        logicalWidth sourceRow row.val =
      some ((PiRLCFirst54DirectPlan.positionPlan
        (PiRLCRetainedInputs.first54Inputs geometry)).forms row) := by
  let descriptor := PiRLCFirst54DirectSchedule.position row
  have descriptorEq : positionDescriptor descriptor.candidate.source
      descriptor.candidate.round descriptor.slot = descriptor := by
    rcases descriptor with ⟨⟨source, round⟩, slot⟩
    rfl
  have exactRow := positionGrid_row? geometry descriptor.candidate.source
    descriptor.candidate.round descriptor.slot
  have gridToSemantic :
      (Fin.encodeProd (descriptor.candidate.source,
        Fin.encodeProd (descriptor.candidate.round, descriptor.slot))).val =
      (PiRLCFirst54DirectSchedule.positionIndex descriptor).val := by
    rcases descriptor with ⟨⟨source, round⟩, slot⟩
    simp [PiRLCFirst54DirectSchedule.positionIndex,
      PiRLCFirst54DirectSchedule.candidateIndex, Fin.encodeProd, Fin.mkDivMod,
      Gadgets.Sampling.First54Step.slotCount]
    omega
  have semanticToRow := congrArg Fin.val
    (PiRLCFirst54DirectSchedule.positionIndex_position row)
  have encoded :
      (Fin.encodeProd (descriptor.candidate.source,
        Fin.encodeProd (descriptor.candidate.round, descriptor.slot))).val =
      row.val := gridToSemantic.trans (by simpa [descriptor] using semanticToRow)
  have semanticIndex : PiRLCFirst54DirectSchedule.positionIndex
      (positionDescriptor descriptor.candidate.source descriptor.candidate.round
        descriptor.slot) = row := by
    rw [descriptorEq]
    exact PiRLCFirst54DirectSchedule.positionIndex_position row
  change (do
      let forms ← (positionGrid geometry).row? logicalWidth row.val
      pure forms.meaningfulForm) = _
  rw [← encoded]
  have mapped := congrArg
    (fun result : Option (OrdinaryRow.Forms logicalWidth) => do
      let forms ← result
      pure forms.meaningfulForm) exactRow
  exact mapped.trans (by rw [semanticIndex]; rfl)

theorem acceptedBlock_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin PiRLCFirst54DirectSchedule.candidateCount) :
    (MatrixProgram.Block.multiplicationGrid (acceptedGrid geometry)).row?
        logicalWidth sourceRow row.val =
      some ((PiRLCFirst54DirectPlan.acceptedProductPlan
        (PiRLCRetainedInputs.first54Inputs geometry)).forms row) := by
  let descriptor := PiRLCFirst54DirectSchedule.candidate row
  have descriptorEq : candidate descriptor.source descriptor.round =
      descriptor := by
    rcases descriptor with ⟨source, round⟩
    rfl
  have exactRow := acceptedGrid_row? geometry descriptor.source
    descriptor.round
  have gridToSemantic :
      (Fin.encodeProd (descriptor.source,
        Fin.encodeProd (descriptor.round, (0 : Fin 1)))).val =
      (PiRLCFirst54DirectSchedule.candidateIndex descriptor).val := by
    rcases descriptor with ⟨source, round⟩
    simp [PiRLCFirst54DirectSchedule.candidateIndex,
      Fin.encodeProd, Fin.mkDivMod]
  have semanticToRow := congrArg Fin.val
    (PiRLCFirst54DirectSchedule.candidateIndex_candidate row)
  have encoded :
      (Fin.encodeProd (descriptor.source,
        Fin.encodeProd (descriptor.round, (0 : Fin 1)))).val = row.val :=
    gridToSemantic.trans (by simpa [descriptor] using semanticToRow)
  have semanticIndex : PiRLCFirst54DirectSchedule.candidateIndex
      (candidate descriptor.source descriptor.round) = row := by
    rw [descriptorEq]
    exact PiRLCFirst54DirectSchedule.candidateIndex_candidate row
  change (do
      let forms ← (acceptedGrid geometry).row? logicalWidth row.val
      pure forms.meaningfulForm) = _
  rw [← encoded]
  have mapped := congrArg
    (fun result : Option (OrdinaryRow.Forms logicalWidth) => do
      let forms ← result
      pure forms.meaningfulForm) exactRow
  exact mapped.trans (by rw [semanticIndex]; rfl)

theorem valueBlock_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin PiRLCFirst54DirectSchedule.valueCount) :
    (MatrixProgram.Block.multiplicationGrid (valueGrid geometry)).row?
        logicalWidth sourceRow row.val =
      some ((PiRLCFirst54DirectPlan.valuePlan
        (PiRLCRetainedInputs.first54Inputs geometry)).forms row) := by
  let descriptor := PiRLCFirst54DirectSchedule.value row
  have descriptorEq : valueDescriptor descriptor.candidate.source
      descriptor.candidate.round descriptor.slot = descriptor := by
    rcases descriptor with ⟨⟨source, round⟩, slot⟩
    rfl
  have exactRow := valueGrid_row? geometry descriptor.candidate.source
    descriptor.candidate.round descriptor.slot
  have gridToSemantic :
      (Fin.encodeProd (descriptor.candidate.source,
        Fin.encodeProd (descriptor.candidate.round, descriptor.slot))).val =
      (PiRLCFirst54DirectSchedule.valueIndex descriptor).val := by
    rcases descriptor with ⟨⟨source, round⟩, slot⟩
    simp [PiRLCFirst54DirectSchedule.valueIndex,
      PiRLCFirst54DirectSchedule.candidateIndex, Fin.encodeProd, Fin.mkDivMod,
      Gadgets.Sampling.First54ValueStep.outputCount]
    omega
  have semanticToRow := congrArg Fin.val
    (PiRLCFirst54DirectSchedule.valueIndex_value row)
  have encoded :
      (Fin.encodeProd (descriptor.candidate.source,
        Fin.encodeProd (descriptor.candidate.round, descriptor.slot))).val =
      row.val := gridToSemantic.trans (by simpa [descriptor] using semanticToRow)
  have semanticIndex : PiRLCFirst54DirectSchedule.valueIndex
      (valueDescriptor descriptor.candidate.source descriptor.candidate.round
        descriptor.slot) = row := by
    rw [descriptorEq]
    exact PiRLCFirst54DirectSchedule.valueIndex_value row
  change (do
      let forms ← (valueGrid geometry).row? logicalWidth row.val
      pure forms.meaningfulForm) = _
  rw [← encoded]
  have mapped := congrArg
    (fun result : Option (OrdinaryRow.Forms logicalWidth) => do
      let forms ← result
      pure forms.meaningfulForm) exactRow
  exact mapped.trans (by rw [semanticIndex]; rfl)

theorem pinBlock_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin PiRLCFirst54DirectSchedule.finalCount) :
    (MatrixProgram.Block.pin (finalPin geometry)).row? logicalWidth sourceRow
        row.val =
      some ((PiRLCFirst54DirectPlan.finalPlan
        (PiRLCRetainedInputs.first54Inputs geometry)).forms row) := by
  have exactRow := finalPin_row? geometry row
  change (do
      let forms ← (finalPin geometry).row? logicalWidth row.val
      pure forms.meaningfulForm) = _
  have mapped := congrArg
    (fun result : Option (PinRow.Forms logicalWidth) => do
      let forms ← result
      pure forms.meaningfulForm) exactRow
  exact mapped.trans rfl

theorem plan_forms_position
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (row : Fin (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount) :
    (PiRLCFirst54DirectPlan.plan inputs).forms
        (Plan.leftIndex
          (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
          (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount row) =
      (PiRLCFirst54DirectPlan.positionPlan inputs).forms row := by
  funext port
  exact Plan.append_forms_left
    (PiRLCFirst54DirectPlan.positionPlan inputs)
    (PiRLCFirst54DirectPlan.acceptedTailPlan inputs)
    (PiRLCFirst54DirectPlan.totalCount_le inputs) row port

theorem plan_forms_accepted
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (row : Fin (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount) :
    (PiRLCFirst54DirectPlan.plan inputs).forms
        (Plan.rightIndex
          (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
          (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount
          (Plan.leftIndex
            (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
            (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount row)) =
      (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).forms row := by
  funext port
  calc
    _ = (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).forms
        (Plan.leftIndex
          (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
          (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount row) port :=
      Plan.append_forms_right
        (PiRLCFirst54DirectPlan.positionPlan inputs)
        (PiRLCFirst54DirectPlan.acceptedTailPlan inputs)
        (PiRLCFirst54DirectPlan.totalCount_le inputs) _ port
    _ = _ := Plan.append_forms_left
      (PiRLCFirst54DirectPlan.acceptedProductPlan inputs)
      (PiRLCFirst54DirectPlan.valueFinalPlan inputs)
      (PiRLCFirst54DirectPlan.acceptedTailCount_le inputs) row port

theorem plan_forms_value
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (row : Fin (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount) :
    (PiRLCFirst54DirectPlan.plan inputs).forms
        (Plan.rightIndex
          (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
          (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount
          (Plan.rightIndex
            (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
            (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount
            (Plan.leftIndex
              (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount
              (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount row))) =
      (PiRLCFirst54DirectPlan.valuePlan inputs).forms row := by
  funext port
  calc
    _ = (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).forms
        (Plan.rightIndex
          (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
          (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount
          (Plan.leftIndex
            (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount
            (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount row)) port :=
      Plan.append_forms_right
        (PiRLCFirst54DirectPlan.positionPlan inputs)
        (PiRLCFirst54DirectPlan.acceptedTailPlan inputs)
        (PiRLCFirst54DirectPlan.totalCount_le inputs) _ port
    _ = (PiRLCFirst54DirectPlan.valueFinalPlan inputs).forms
        (Plan.leftIndex
          (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount
          (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount row) port :=
      Plan.append_forms_right
        (PiRLCFirst54DirectPlan.acceptedProductPlan inputs)
        (PiRLCFirst54DirectPlan.valueFinalPlan inputs)
        (PiRLCFirst54DirectPlan.acceptedTailCount_le inputs) _ port
    _ = _ := Plan.append_forms_left
      (PiRLCFirst54DirectPlan.valuePlan inputs)
      (PiRLCFirst54DirectPlan.finalPlan inputs)
      (PiRLCFirst54DirectPlan.valueFinalCount_le inputs) row port

theorem plan_forms_final
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (inputs : PiRLCFirst54DirectPlan.Inputs program logicalWidth)
    (row : Fin (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount) :
    (PiRLCFirst54DirectPlan.plan inputs).forms
        (Plan.rightIndex
          (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
          (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount
          (Plan.rightIndex
            (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
            (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount
            (Plan.rightIndex
              (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount
              (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount row))) =
      (PiRLCFirst54DirectPlan.finalPlan inputs).forms row := by
  funext port
  calc
    _ = (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).forms
        (Plan.rightIndex
          (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
          (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount
          (Plan.rightIndex
            (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount
            (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount row)) port :=
      Plan.append_forms_right
        (PiRLCFirst54DirectPlan.positionPlan inputs)
        (PiRLCFirst54DirectPlan.acceptedTailPlan inputs)
        (PiRLCFirst54DirectPlan.totalCount_le inputs) _ port
    _ = (PiRLCFirst54DirectPlan.valueFinalPlan inputs).forms
        (Plan.rightIndex
          (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount
          (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount row) port :=
      Plan.append_forms_right
        (PiRLCFirst54DirectPlan.acceptedProductPlan inputs)
        (PiRLCFirst54DirectPlan.valueFinalPlan inputs)
        (PiRLCFirst54DirectPlan.acceptedTailCount_le inputs) _ port
    _ = _ := Plan.append_forms_right
      (PiRLCFirst54DirectPlan.valuePlan inputs)
      (PiRLCFirst54DirectPlan.finalPlan inputs)
      (PiRLCFirst54DirectPlan.valueFinalCount_le inputs) row port

theorem matrixProgram_position_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin PiRLCFirst54DirectSchedule.positionCount) :
    let inputs := PiRLCRetainedInputs.first54Inputs geometry
    let planRow : Fin (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount :=
      row
    let global := Plan.leftIndex
      (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
      (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount planRow
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiRLCFirst54DirectPlan.plan inputs).forms global) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let planRow : Fin (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount :=
    row
  let global := Plan.leftIndex
    (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
    (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount planRow
  have globalVal : global.val = planRow.val :=
    Plan.leftIndex_val
      (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
      (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount planRow
  have bound : global.val <
      (MatrixProgram.Block.multiplicationGrid
        (positionGrid geometry)).rowCount := by
    rw [globalVal]
    rw [show (MatrixProgram.Block.multiplicationGrid
        (positionGrid geometry)).rowCount =
          (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount by
      exact (positionGrid_rowCount geometry).trans
        (PiRLCFirst54DirectPlan.positionPlan_rowCount inputs).symm]
    exact planRow.isLt
  have selected := MatrixProgram.Program.cons_first_row?
    (.multiplicationGrid (positionGrid geometry))
    [.multiplicationGrid (acceptedGrid geometry),
      .multiplicationGrid (valueGrid geometry), .pin (finalPin geometry)]
    logicalWidth sourceRow global.val bound
  have selectedProgram :
      (matrixProgram geometry).row? logicalWidth sourceRow global.val =
        (MatrixProgram.Block.multiplicationGrid
          (positionGrid geometry)).row? logicalWidth sourceRow global.val := by
    simpa [matrixProgram] using selected
  calc
    _ = (MatrixProgram.Block.multiplicationGrid
        (positionGrid geometry)).row? logicalWidth sourceRow global.val :=
      selectedProgram
    _ = some ((PiRLCFirst54DirectPlan.positionPlan inputs).forms planRow) := by
      rw [globalVal]
      exact positionBlock_row? geometry sourceRow planRow
    _ = _ := congrArg some (plan_forms_position inputs planRow).symm

theorem matrixProgram_accepted_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin (PiRLCFirst54DirectPlan.acceptedProductPlan
      (PiRLCRetainedInputs.first54Inputs geometry)).rowCount) :
    let inputs := PiRLCRetainedInputs.first54Inputs geometry
    let tailIndex := Plan.leftIndex
      (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
      (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount row
    let global := Plan.rightIndex
      (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
      (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount tailIndex
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiRLCFirst54DirectPlan.plan inputs).forms global) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let tailIndex := Plan.leftIndex
    (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
    (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount row
  let global := Plan.rightIndex
    (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
    (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount tailIndex
  have outerBound :
      (MatrixProgram.Block.multiplicationGrid
        (positionGrid geometry)).rowCount ≤ global.val := by
    simp [global, Plan.rightIndex, MatrixProgram.Block.rowCount]
  have outer := MatrixProgram.Program.cons_rest_row?
    (.multiplicationGrid (positionGrid geometry))
    [.multiplicationGrid (acceptedGrid geometry),
      .multiplicationGrid (valueGrid geometry), .pin (finalPin geometry)]
    logicalWidth sourceRow global.val outerBound
  have tailOrdinal : global.val -
      (MatrixProgram.Block.multiplicationGrid
        (positionGrid geometry)).rowCount = tailIndex.val := by
    simp [global, Plan.rightIndex, MatrixProgram.Block.rowCount]
  rw [tailOrdinal] at outer
  have innerBound : tailIndex.val <
      (MatrixProgram.Block.multiplicationGrid
        (acceptedGrid geometry)).rowCount := by
    simpa [tailIndex, Plan.leftIndex, MatrixProgram.Block.rowCount] using row.isLt
  have inner := MatrixProgram.Program.cons_first_row?
    (.multiplicationGrid (acceptedGrid geometry))
    [.multiplicationGrid (valueGrid geometry), .pin (finalPin geometry)]
    logicalWidth sourceRow tailIndex.val innerBound
  have selectedProgram :
      (matrixProgram geometry).row? logicalWidth sourceRow global.val =
        (MatrixProgram.Block.multiplicationGrid
          (acceptedGrid geometry)).row? logicalWidth sourceRow tailIndex.val := by
    calc
      _ = (MatrixProgram.Program.mk [
          .multiplicationGrid (acceptedGrid geometry),
          .multiplicationGrid (valueGrid geometry),
          .pin (finalPin geometry)]).row?
            logicalWidth sourceRow tailIndex.val := by
        simpa [matrixProgram] using outer
      _ = _ := inner
  calc
    _ = (MatrixProgram.Block.multiplicationGrid
        (acceptedGrid geometry)).row? logicalWidth sourceRow tailIndex.val :=
      selectedProgram
    _ = some ((PiRLCFirst54DirectPlan.acceptedProductPlan inputs).forms row) := by
      simpa [tailIndex, Plan.leftIndex, inputs] using
        acceptedBlock_row? geometry sourceRow row
    _ = _ := congrArg some (plan_forms_accepted inputs row).symm

theorem matrixProgram_value_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin PiRLCFirst54DirectSchedule.valueCount) :
    let inputs := PiRLCRetainedInputs.first54Inputs geometry
    let planRow : Fin (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount := row
    let valueFinalIndex := Plan.leftIndex
      (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount
      (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount planRow
    let tailIndex := Plan.rightIndex
      (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
      (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount valueFinalIndex
    let global := Plan.rightIndex
      (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
      (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount tailIndex
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiRLCFirst54DirectPlan.plan inputs).forms global) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let planRow : Fin (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount := row
  let valueFinalIndex := Plan.leftIndex
    (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount
    (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount planRow
  let tailIndex := Plan.rightIndex
    (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
    (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount valueFinalIndex
  let global := Plan.rightIndex
    (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
    (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount tailIndex
  have globalVal : global.val =
      (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount + tailIndex.val :=
    Plan.rightIndex_val _ _ tailIndex
  have tailVal : tailIndex.val =
      (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount +
        valueFinalIndex.val := Plan.rightIndex_val _ _ valueFinalIndex
  have valueFinalVal : valueFinalIndex.val = planRow.val :=
    Plan.leftIndex_val _ _ planRow
  have positionCountEq :
      (MatrixProgram.Block.multiplicationGrid
        (positionGrid geometry)).rowCount =
      (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount :=
    (positionGrid_rowCount geometry).trans
      (PiRLCFirst54DirectPlan.positionPlan_rowCount inputs).symm
  have acceptedCountEq :
      (MatrixProgram.Block.multiplicationGrid
        (acceptedGrid geometry)).rowCount =
      (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount :=
    (acceptedGrid_rowCount geometry).trans
      (PiRLCFirst54DirectPlan.acceptedProductPlan_rowCount inputs).symm
  have valueCountEq :
      (MatrixProgram.Block.multiplicationGrid
        (valueGrid geometry)).rowCount =
      (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount :=
    (valueGrid_rowCount geometry).trans
      (PiRLCFirst54DirectPlan.valuePlan_rowCount inputs).symm
  have outerBound :
      (MatrixProgram.Block.multiplicationGrid
        (positionGrid geometry)).rowCount ≤ global.val := by
    rw [positionCountEq, globalVal]
    omega
  have outerOrdinal : global.val -
      (MatrixProgram.Block.multiplicationGrid
        (positionGrid geometry)).rowCount = tailIndex.val := by
    rw [positionCountEq, globalVal]
    omega
  have acceptedBound :
      (MatrixProgram.Block.multiplicationGrid
        (acceptedGrid geometry)).rowCount ≤ tailIndex.val := by
    rw [acceptedCountEq, tailVal]
    omega
  have acceptedOrdinal : tailIndex.val -
      (MatrixProgram.Block.multiplicationGrid
        (acceptedGrid geometry)).rowCount = valueFinalIndex.val := by
    rw [acceptedCountEq, tailVal]
    omega
  have valueBound : valueFinalIndex.val <
      (MatrixProgram.Block.multiplicationGrid
        (valueGrid geometry)).rowCount := by
    rw [valueFinalVal, valueCountEq]
    exact planRow.isLt
  have outer := MatrixProgram.Program.cons_rest_row?
    (.multiplicationGrid (positionGrid geometry))
    [.multiplicationGrid (acceptedGrid geometry),
      .multiplicationGrid (valueGrid geometry), .pin (finalPin geometry)]
    logicalWidth sourceRow global.val outerBound
  rw [outerOrdinal] at outer
  have acceptedTail := MatrixProgram.Program.cons_rest_row?
    (.multiplicationGrid (acceptedGrid geometry))
    [.multiplicationGrid (valueGrid geometry), .pin (finalPin geometry)]
    logicalWidth sourceRow tailIndex.val acceptedBound
  rw [acceptedOrdinal] at acceptedTail
  have valueSelected := MatrixProgram.Program.cons_first_row?
    (.multiplicationGrid (valueGrid geometry)) [.pin (finalPin geometry)]
    logicalWidth sourceRow valueFinalIndex.val valueBound
  have selectedProgram :
      (matrixProgram geometry).row? logicalWidth sourceRow global.val =
        (MatrixProgram.Block.multiplicationGrid
          (valueGrid geometry)).row? logicalWidth sourceRow
            valueFinalIndex.val := by
    calc
      _ = (MatrixProgram.Program.mk [
          .multiplicationGrid (acceptedGrid geometry),
          .multiplicationGrid (valueGrid geometry),
          .pin (finalPin geometry)]).row?
            logicalWidth sourceRow tailIndex.val := by
        simpa [matrixProgram] using outer
      _ = (MatrixProgram.Program.mk [
          .multiplicationGrid (valueGrid geometry),
          .pin (finalPin geometry)]).row?
            logicalWidth sourceRow valueFinalIndex.val := acceptedTail
      _ = _ := valueSelected
  calc
    _ = (MatrixProgram.Block.multiplicationGrid
        (valueGrid geometry)).row? logicalWidth sourceRow valueFinalIndex.val :=
      selectedProgram
    _ = some ((PiRLCFirst54DirectPlan.valuePlan inputs).forms planRow) := by
      rw [valueFinalVal]
      exact valueBlock_row? geometry sourceRow planRow
    _ = _ := congrArg some (plan_forms_value inputs planRow).symm

theorem matrixProgram_final_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (row : Fin PiRLCFirst54DirectSchedule.finalCount) :
    let inputs := PiRLCRetainedInputs.first54Inputs geometry
    let planRow : Fin (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount := row
    let valueFinalIndex := Plan.rightIndex
      (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount
      (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount planRow
    let tailIndex := Plan.rightIndex
      (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
      (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount valueFinalIndex
    let global := Plan.rightIndex
      (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
      (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount tailIndex
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiRLCFirst54DirectPlan.plan inputs).forms global) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let planRow : Fin (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount := row
  let valueFinalIndex := Plan.rightIndex
    (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount
    (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount planRow
  let tailIndex := Plan.rightIndex
    (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
    (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount valueFinalIndex
  let global := Plan.rightIndex
    (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
    (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount tailIndex
  have globalVal : global.val =
      (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount + tailIndex.val :=
    Plan.rightIndex_val _ _ tailIndex
  have tailVal : tailIndex.val =
      (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount +
        valueFinalIndex.val := Plan.rightIndex_val _ _ valueFinalIndex
  have valueFinalVal : valueFinalIndex.val =
      (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount + planRow.val :=
    Plan.rightIndex_val _ _ planRow
  have positionCountEq :
      (MatrixProgram.Block.multiplicationGrid
        (positionGrid geometry)).rowCount =
      (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount :=
    (positionGrid_rowCount geometry).trans
      (PiRLCFirst54DirectPlan.positionPlan_rowCount inputs).symm
  have acceptedCountEq :
      (MatrixProgram.Block.multiplicationGrid
        (acceptedGrid geometry)).rowCount =
      (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount :=
    (acceptedGrid_rowCount geometry).trans
      (PiRLCFirst54DirectPlan.acceptedProductPlan_rowCount inputs).symm
  have valueCountEq :
      (MatrixProgram.Block.multiplicationGrid
        (valueGrid geometry)).rowCount =
      (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount :=
    (valueGrid_rowCount geometry).trans
      (PiRLCFirst54DirectPlan.valuePlan_rowCount inputs).symm
  have pinCountEq :
      (MatrixProgram.Block.pin (finalPin geometry)).rowCount =
      (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount :=
    (finalPin_rowCount geometry).trans
      (PiRLCFirst54DirectPlan.finalPlan_rowCount inputs).symm
  have outerBound :
      (MatrixProgram.Block.multiplicationGrid
        (positionGrid geometry)).rowCount ≤ global.val := by
    rw [positionCountEq, globalVal]
    omega
  have outerOrdinal : global.val -
      (MatrixProgram.Block.multiplicationGrid
        (positionGrid geometry)).rowCount = tailIndex.val := by
    rw [positionCountEq, globalVal]
    omega
  have acceptedBound :
      (MatrixProgram.Block.multiplicationGrid
        (acceptedGrid geometry)).rowCount ≤ tailIndex.val := by
    rw [acceptedCountEq, tailVal]
    omega
  have acceptedOrdinal : tailIndex.val -
      (MatrixProgram.Block.multiplicationGrid
        (acceptedGrid geometry)).rowCount = valueFinalIndex.val := by
    rw [acceptedCountEq, tailVal]
    omega
  have valueBound :
      (MatrixProgram.Block.multiplicationGrid
        (valueGrid geometry)).rowCount ≤ valueFinalIndex.val := by
    rw [valueCountEq, valueFinalVal]
    omega
  have valueOrdinal : valueFinalIndex.val -
      (MatrixProgram.Block.multiplicationGrid
        (valueGrid geometry)).rowCount = planRow.val := by
    rw [valueCountEq, valueFinalVal]
    omega
  have pinBound : planRow.val <
      (MatrixProgram.Block.pin (finalPin geometry)).rowCount := by
    rw [pinCountEq]
    exact planRow.isLt
  have outer := MatrixProgram.Program.cons_rest_row?
    (.multiplicationGrid (positionGrid geometry))
    [.multiplicationGrid (acceptedGrid geometry),
      .multiplicationGrid (valueGrid geometry), .pin (finalPin geometry)]
    logicalWidth sourceRow global.val outerBound
  rw [outerOrdinal] at outer
  have acceptedTail := MatrixProgram.Program.cons_rest_row?
    (.multiplicationGrid (acceptedGrid geometry))
    [.multiplicationGrid (valueGrid geometry), .pin (finalPin geometry)]
    logicalWidth sourceRow tailIndex.val acceptedBound
  rw [acceptedOrdinal] at acceptedTail
  have valueTail := MatrixProgram.Program.cons_rest_row?
    (.multiplicationGrid (valueGrid geometry)) [.pin (finalPin geometry)]
    logicalWidth sourceRow valueFinalIndex.val valueBound
  rw [valueOrdinal] at valueTail
  have pinSelected := MatrixProgram.Program.cons_first_row?
    (.pin (finalPin geometry)) [] logicalWidth sourceRow planRow.val pinBound
  have selectedProgram :
      (matrixProgram geometry).row? logicalWidth sourceRow global.val =
        (MatrixProgram.Block.pin (finalPin geometry)).row?
          logicalWidth sourceRow planRow.val := by
    calc
      _ = (MatrixProgram.Program.mk [
          .multiplicationGrid (acceptedGrid geometry),
          .multiplicationGrid (valueGrid geometry),
          .pin (finalPin geometry)]).row?
            logicalWidth sourceRow tailIndex.val := by
        simpa [matrixProgram] using outer
      _ = (MatrixProgram.Program.mk [
          .multiplicationGrid (valueGrid geometry),
          .pin (finalPin geometry)]).row?
            logicalWidth sourceRow valueFinalIndex.val := acceptedTail
      _ = (MatrixProgram.Program.mk [.pin (finalPin geometry)]).row?
            logicalWidth sourceRow planRow.val := valueTail
      _ = _ := pinSelected
  calc
    _ = (MatrixProgram.Block.pin (finalPin geometry)).row?
        logicalWidth sourceRow planRow.val := selectedProgram
    _ = some ((PiRLCFirst54DirectPlan.finalPlan inputs).forms planRow) :=
      pinBlock_row? geometry sourceRow planRow
    _ = _ := congrArg some (plan_forms_final inputs planRow).symm

/-- Every compact First54 program row is literally the corresponding row of
the canonical Lean family-major First54 plan. -/
theorem matrixProgram_row?
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCRetainedGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiRLCFirst54DirectPlan.plan
      (PiRLCRetainedInputs.first54Inputs geometry)).rowCount) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      some ((PiRLCFirst54DirectPlan.plan
        (PiRLCRetainedInputs.first54Inputs geometry)).forms global) := by
  let inputs := PiRLCRetainedInputs.first54Inputs geometry
  let positionCount := (PiRLCFirst54DirectPlan.positionPlan inputs).rowCount
  let tailCount := (PiRLCFirst54DirectPlan.acceptedTailPlan inputs).rowCount
  cases outer : Plan.splitIndex positionCount tailCount global with
  | inl positionRow =>
      have indexEq := Plan.leftIndex_of_splitIndex_eq positionCount tailCount
        global positionRow outer
      rw [← indexEq]
      exact matrixProgram_position_row? geometry sourceRow positionRow
  | inr tailRow =>
      have outerEq := Plan.rightIndex_of_splitIndex_eq positionCount tailCount
        global tailRow outer
      let acceptedCount :=
        (PiRLCFirst54DirectPlan.acceptedProductPlan inputs).rowCount
      let valueFinalCount :=
        (PiRLCFirst54DirectPlan.valueFinalPlan inputs).rowCount
      cases middle : Plan.splitIndex acceptedCount valueFinalCount tailRow with
      | inl acceptedRow =>
          have middleEq := Plan.leftIndex_of_splitIndex_eq
            acceptedCount valueFinalCount tailRow acceptedRow middle
          have lifted := congrArg
            (Plan.rightIndex positionCount tailCount) middleEq
          have indexEq := lifted.trans outerEq
          rw [← indexEq]
          exact matrixProgram_accepted_row? geometry sourceRow acceptedRow
      | inr valueFinalRow =>
          have middleEq := Plan.rightIndex_of_splitIndex_eq
            acceptedCount valueFinalCount tailRow valueFinalRow middle
          let valueCount := (PiRLCFirst54DirectPlan.valuePlan inputs).rowCount
          let finalCount := (PiRLCFirst54DirectPlan.finalPlan inputs).rowCount
          cases inner : Plan.splitIndex valueCount finalCount valueFinalRow with
          | inl valueRow =>
              have innerEq := Plan.leftIndex_of_splitIndex_eq
                valueCount finalCount valueFinalRow valueRow inner
              have middleLifted := congrArg
                (Plan.rightIndex acceptedCount valueFinalCount) innerEq
              have tailEq := middleLifted.trans middleEq
              have outerLifted := congrArg
                (Plan.rightIndex positionCount tailCount) tailEq
              have indexEq := outerLifted.trans outerEq
              rw [← indexEq]
              exact matrixProgram_value_row? geometry sourceRow valueRow
          | inr finalRow =>
              have innerEq := Plan.rightIndex_of_splitIndex_eq
                valueCount finalCount valueFinalRow finalRow inner
              have middleLifted := congrArg
                (Plan.rightIndex acceptedCount valueFinalCount) innerEq
              have tailEq := middleLifted.trans middleEq
              have outerLifted := congrArg
                (Plan.rightIndex positionCount tailCount) tailEq
              have indexEq := outerLifted.trans outerEq
              rw [← indexEq]
              exact matrixProgram_final_row? geometry sourceRow finalRow

end NightstreamFPrime.Export.Stage1.PiRLCFirst54MatrixProgram
