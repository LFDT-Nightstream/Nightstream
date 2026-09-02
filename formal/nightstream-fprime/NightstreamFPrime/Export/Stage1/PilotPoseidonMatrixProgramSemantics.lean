import NightstreamFPrime.Export.Stage1.PilotPoseidonMatrixProgram

/-!
Proves that the compact pilot Poseidon2 input programs reconstruct the exact
semantic prior-state and output-state hash inputs.
-/

namespace NightstreamFPrime.Export.Stage1.PilotPoseidonMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

private theorem previousRule_zero
    {sourceWidth logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth 12350)
    (retainedStart oneColumn : Nat) (lane : Fin 8) :
    (previousRule schedule retainedStart).form? logicalWidth oneColumn
        0 lane.val = some none := by
  apply PoseidonInput.Rule.form?_eq_some_none
  simp [previousRule, PoseidonInput.Region.offsets?]

private theorem previousRule_succ
    {sourceWidth logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth 12350)
    (retainedStart oneColumn : Nat)
    (fits : retainedStart + schedule.block.coordinateCount ≤ logicalWidth)
    (invocationOffset : Fin 12349) (lane : Fin 8) :
    (previousRule schedule retainedStart).form? logicalWidth oneColumn
        (1 + invocationOffset.val) lane.val =
      some (some (PoseidonRetainedFamily.outputState schedule retainedStart fits
        ⟨invocationOffset.val, by omega⟩ lane)) := by
  have slotBound : ∀ selected : Fin 8,
      78 + invocationOffset.val * 86 + selected.val <
        schedule.block.slotCount := by
    intro selected
    rw [schedule.slotCount_eq, PoseidonRetainedSlots.rows_length]
    omega
  rw [show (previousRule schedule retainedStart).form? logicalWidth oneColumn
      (1 + invocationOffset.val) lane.val =
        some (some (SparseLayer.external (fun selected : Fin 8 =>
          schedule.block.form retainedStart fits
            ⟨78 + invocationOffset.val * 86 + selected.val,
              slotBound selected⟩) lane)) by
    simpa [previousRule] using
      PoseidonInput.Rule.external_form?_ofSemantic
        (region := PoseidonInput.Region.mk 1 12349 0 8)
        invocationOffset lane lane.isLt schedule.block retainedStart fits
        oneColumn 78 86 slotBound]
  apply congrArg some
  apply congrArg some
  unfold PoseidonRetainedFamily.outputState PoseidonRetainedFamily.form
  apply congrArg (fun state => SparseLayer.external state lane)
  funext selected
  apply congrArg (schedule.block.form retainedStart fits)
  apply Fin.ext
  simp [PoseidonRetainedFamily.slot, Fin.encodeProd,
    PoseidonRetainedSlots.finalRow_val]
  omega

private theorem previousProgram_zero
    {sourceWidth logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth 12350)
    (retainedStart oneColumn : Nat) (lane : Fin 8) :
    (previousProgram schedule retainedStart).form? logicalWidth oneColumn
        0 lane.val = some .empty := by
  apply PoseidonInput.Program.form?_eq_some_empty_of_allOutside
  intro rule member
  simp [previousProgram] at member
  subst rule
  simp [previousRule, PoseidonInput.Region.offsets?]

private theorem previousProgram_succ
    {sourceWidth logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth 12350)
    (retainedStart oneColumn : Nat)
    (fits : retainedStart + schedule.block.coordinateCount ≤ logicalWidth)
    (invocationOffset : Fin 12349) (lane : Fin 8) :
    (previousProgram schedule retainedStart).form? logicalWidth oneColumn
        (1 + invocationOffset.val) lane.val =
      some (PoseidonRetainedFamily.outputState schedule retainedStart fits
        ⟨invocationOffset.val, by omega⟩ lane) := by
  simpa [previousProgram] using
    PoseidonInput.Program.singleton_form?_of_selected
      (previousRule schedule retainedStart) oneColumn
      (1 + invocationOffset.val) lane.val _
      (previousRule_succ schedule retainedStart oneColumn fits
        invocationOffset lane)

private theorem previousRule_result
    {sourceWidth logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth 12350)
    (retainedStart oneColumn : Nat)
    (fits : retainedStart + schedule.block.coordinateCount ≤ logicalWidth)
    (invocation : Fin 12350) (lane : Fin 8) :
    (previousRule schedule retainedStart).form? logicalWidth oneColumn
        invocation.val lane.val =
      some (if first : invocation.val = 0 then none else
        some (PilotPoseidonPlan.previousOutput schedule retainedStart fits
          invocation lane)) := by
  by_cases first : invocation.val = 0
  · rw [dif_pos first]
    have invocationEq : invocation = ⟨0, by omega⟩ := by
      apply Fin.ext
      exact first
    rw [invocationEq]
    exact previousRule_zero schedule retainedStart oneColumn lane
  · rw [dif_neg first]
    let invocationOffset : Fin 12349 :=
      ⟨invocation.val - 1, by omega⟩
    have invocationEq : invocation.val = 1 + invocationOffset.val := by
      dsimp [invocationOffset]
      omega
    rw [invocationEq]
    have selected := previousRule_succ schedule retainedStart oneColumn fits
      invocationOffset lane
    rw [selected]
    apply congrArg some
    apply congrArg some
    unfold PilotPoseidonPlan.previousOutput
    rw [dif_neg first]

private theorem fullInputRule_result
    {sourceWidth logicalWidth : Nat}
    (inputBlock : LowNormBlock.Block sourceWidth) (inputStart : Nat)
    (fits : inputStart + inputBlock.coordinateCount ≤ logicalWidth)
    (slotCount : inputBlock.slotCount = 49393) (oneColumn : Nat)
    (invocation : Fin 12350) (lane : Fin 8) :
    (fullInputRule inputBlock inputStart).form? logicalWidth oneColumn
        invocation.val lane.val =
      some (if invocationBound : invocation.val < 12348 then
        if laneBound : lane.val < 4 then
          some (inputBlock.form inputStart fits
            ⟨invocation.val * 4 + lane.val, by
              rw [slotCount]
              omega⟩)
        else none
      else none) := by
  by_cases invocationBound : invocation.val < 12348
  · rw [dif_pos invocationBound]
    by_cases laneBound : lane.val < 4
    · rw [dif_pos laneBound]
      have selected := PoseidonInput.Rule.retained_form?_ofSemantic
        (region := PoseidonInput.Region.mk 0 12348 0 4)
        ⟨invocation.val, invocationBound⟩ ⟨lane.val, laneBound⟩
        inputBlock inputStart fits oneColumn 0 4 1 (by
          change 0 + invocation.val * 4 + lane.val * 1 <
            inputBlock.slotCount
          rw [slotCount]
          omega)
      simpa [fullInputRule] using selected
    · rw [dif_neg laneBound]
      apply PoseidonInput.Rule.form?_eq_some_none
      simp [fullInputRule, PoseidonInput.Region.offsets?, invocationBound,
        laneBound]
  · rw [dif_neg invocationBound]
    apply PoseidonInput.Rule.form?_eq_some_none
    simp [fullInputRule, PoseidonInput.Region.offsets?, invocationBound]

private theorem tailInputRule_result
    {sourceWidth logicalWidth : Nat}
    (inputBlock : LowNormBlock.Block sourceWidth) (inputStart : Nat)
    (fits : inputStart + inputBlock.coordinateCount ≤ logicalWidth)
    (slotCount : inputBlock.slotCount = 49393) (oneColumn : Nat)
    (invocation : Fin 12350) (lane : Fin 8) :
    (tailInputRule inputBlock inputStart).form? logicalWidth oneColumn
        invocation.val lane.val =
      some (if selected : invocation.val = 12348 ∧ lane.val = 0 then
        some (inputBlock.form inputStart fits ⟨49392, by
          rw [slotCount]
          omega⟩)
      else none) := by
  by_cases selected : invocation.val = 12348 ∧ lane.val = 0
  · rw [dif_pos selected]
    have ruleSelected := PoseidonInput.Rule.retained_form?_ofSemantic
      (region := PoseidonInput.Region.mk 12348 1 0 1)
      (0 : Fin 1) (0 : Fin 1) inputBlock inputStart fits oneColumn
      49392 0 1 (by
        change 49392 + 0 * 0 + 0 * 1 < inputBlock.slotCount
        rw [slotCount]
        omega)
    simpa [tailInputRule, selected.1, selected.2] using ruleSelected
  · rw [dif_neg selected]
    apply PoseidonInput.Rule.form?_eq_some_none
    simp [tailInputRule, PoseidonInput.Region.offsets?]
    omega

private theorem paddingRule_result {logicalWidth : Nat}
    (oneColumn : Fin logicalWidth) (invocation : Fin 12350) (lane : Fin 8) :
    paddingRule.form? logicalWidth oneColumn.val invocation.val lane.val =
      some (if selected : invocation.val = 12349 ∧ lane.val = 0 then
        some (SparseForm.singleton oneColumn 1)
      else none) := by
  by_cases selected : invocation.val = 12349 ∧ lane.val = 0
  · rw [dif_pos selected]
    have ruleSelected := PoseidonInput.Rule.constant_form?
      (region := PoseidonInput.Region.mk 12349 1 0 1)
      (0 : Fin 1) (0 : Fin 1) oneColumn (1 : F)
    simpa [paddingRule, selected.1, selected.2] using ruleSelected
  · rw [dif_neg selected]
    apply PoseidonInput.Rule.form?_eq_some_none
    simp [paddingRule, PoseidonInput.Region.offsets?]
    omega

private theorem chainInputProgram_form?
    {poseidonSourceWidth inputSourceWidth logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule poseidonSourceWidth 12350)
    (retainedStart : Nat)
    (poseidonFits : retainedStart + schedule.block.coordinateCount ≤
      logicalWidth)
    (inputBlock : LowNormBlock.Block inputSourceWidth) (inputStart : Nat)
    (inputFits : inputStart + inputBlock.coordinateCount ≤ logicalWidth)
    (inputSlotCount : inputBlock.slotCount = 49393)
    (oneColumn : Fin logicalWidth) (invocation : Fin 12350) (lane : Fin 8) :
    (chainInputProgram schedule retainedStart inputBlock inputStart).form?
        logicalWidth oneColumn.val invocation.val lane.val =
      some (
        let previous := PilotPoseidonPlan.previousOutput schedule retainedStart
          poseidonFits invocation lane
        if invocation.val < 12349 then
          if lane.val < 4 then
            let offset := invocation.val * 4 + lane.val
            if present : offset < 49393 then
              SparseForm.add previous
                (inputBlock.form inputStart inputFits ⟨offset, by
                  rw [inputSlotCount]
                  exact present⟩)
            else previous
          else previous
        else if lane.val = 0 then
          SparseForm.add previous (SparseForm.singleton oneColumn 1)
        else previous) := by
  have folded := PoseidonInput.Program.four_form?_of_results
    (previousRule schedule retainedStart)
    (fullInputRule inputBlock inputStart)
    (tailInputRule inputBlock inputStart) paddingRule
    oneColumn.val invocation.val lane.val _ _ _ _
    (previousRule_result schedule retainedStart oneColumn.val poseidonFits
      invocation lane)
    (fullInputRule_result inputBlock inputStart inputFits inputSlotCount
      oneColumn.val invocation lane)
    (tailInputRule_result inputBlock inputStart inputFits inputSlotCount
      oneColumn.val invocation lane)
    (paddingRule_result oneColumn invocation lane)
  by_cases first : invocation.val = 0
  · have full : invocation.val < 12348 := by omega
    have absorbing : invocation.val < 12349 := by omega
    by_cases rateLane : lane.val < 4
    · have present : invocation.val * 4 + lane.val < 49393 := by omega
      have firstPresent : lane.val < 49393 := by omega
      simpa [chainInputProgram, first, full, absorbing, rateLane, present,
        firstPresent, PilotPoseidonPlan.previousOutput, SparseForm.add,
        SparseForm.empty] using folded
    · simpa [chainInputProgram, first, full, absorbing, rateLane,
        PilotPoseidonPlan.previousOutput, SparseForm.add,
        SparseForm.empty] using folded
  · by_cases full : invocation.val < 12348
    · have absorbing : invocation.val < 12349 := by omega
      have notTail : invocation.val ≠ 12348 := by omega
      have notPadding : invocation.val ≠ 12349 := by omega
      by_cases rateLane : lane.val < 4
      · have present : invocation.val * 4 + lane.val < 49393 := by omega
        simpa [chainInputProgram, first, full, absorbing, rateLane, present,
          notTail, notPadding, SparseForm.add, SparseForm.empty] using folded
      · simpa [chainInputProgram, first, full, absorbing, rateLane,
          notTail, notPadding, SparseForm.add, SparseForm.empty] using folded
    · have last : invocation.val = 12348 ∨ invocation.val = 12349 := by
        omega
      rcases last with tail | padding
      · have absorbing : invocation.val < 12349 := by omega
        by_cases zeroLane : lane.val = 0
        · have rateLane : lane.val < 4 := by omega
          have present : invocation.val * 4 + lane.val < 49393 := by omega
          have tailPresent : 49392 + lane.val < 49393 := by omega
          simpa [chainInputProgram, first, full, tail, absorbing, zeroLane,
            rateLane, present, tailPresent, SparseForm.add,
            SparseForm.empty] using folded
        · by_cases rateLane : lane.val < 4
          · have absent : ¬invocation.val * 4 + lane.val < 49393 := by
              omega
            have tailAbsent : ¬49392 + lane.val < 49393 := by omega
            simpa [chainInputProgram, first, full, tail, absorbing, zeroLane,
              rateLane, absent, tailAbsent, SparseForm.add,
              SparseForm.empty] using folded
          · simpa [chainInputProgram, first, full, tail, absorbing, zeroLane,
              rateLane, SparseForm.add, SparseForm.empty] using folded
      · have notAbsorbing : ¬invocation.val < 12349 := by omega
        by_cases zeroLane : lane.val = 0
        · simpa [chainInputProgram, first, full, padding, notAbsorbing,
            zeroLane, SparseForm.add, SparseForm.empty] using folded
        · simpa [chainInputProgram, first, full, padding, notAbsorbing,
            zeroLane, SparseForm.add, SparseForm.empty] using folded

theorem priorInputProgram_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (invocation : Fin PilotPoseidonPlan.invocationCount) (lane : Fin 8) :
    (priorInputProgram program).form? logicalWidth
        (PiRLCPoseidonGeometry.oneColumn geometry).val invocation.val lane.val =
      some (PilotPoseidonPlan.priorInputState geometry invocation lane) := by
  have slotCount :
      (PiRLCPoseidonGeometry.priorInputBlock program).slotCount = 49393 := by
    simp [PiRLCPoseidonGeometry.priorInputBlock]
  simpa [priorInputProgram, PilotPoseidonPlan.priorInputState,
    Data.priorChain, Data.liftPilotChain, PilotData.priorChain,
    PilotValues.absorbCount, PilotValues.stateHashWords, Spec.Poseidon2.rate]
    using chainInputProgram_form?
      (PilotPoseidonPlan.priorSchedule program)
      (PiRLCRetainedGeometry.priorPoseidonStart program)
      (PiRLCRetainedGeometry.priorPoseidonFits
        (PiRLCPoseidonGeometry.prefixGeometry geometry))
      (PiRLCPoseidonGeometry.priorInputBlock program)
      (PiRLCPoseidonGeometry.priorInputStart program)
      (PiRLCPoseidonGeometry.priorInputFits geometry) slotCount
      (PiRLCPoseidonGeometry.oneColumn geometry) invocation lane

theorem outputInputProgram_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (invocation : Fin PilotPoseidonPlan.invocationCount) (lane : Fin 8) :
    (outputInputProgram program).form? logicalWidth
        (PiRLCPoseidonGeometry.oneColumn geometry).val invocation.val lane.val =
      some (PilotPoseidonPlan.outputInputState geometry invocation lane) := by
  have slotCount :
      (PiRLCPoseidonGeometry.outputInputBlock program).slotCount = 49393 := by
    simp [PiRLCPoseidonGeometry.outputInputBlock]
  simpa [outputInputProgram, PilotPoseidonPlan.outputInputState,
    Data.outputChain, Data.liftPilotChain, PilotData.outputChain,
    PilotValues.absorbCount, PilotValues.stateHashWords, Spec.Poseidon2.rate]
    using chainInputProgram_form?
      (PilotPoseidonPlan.outputSchedule program)
      (PiRLCRetainedGeometry.outputPoseidonStart program)
      (PiRLCRetainedGeometry.outputPoseidonFits
        (PiRLCPoseidonGeometry.prefixGeometry geometry))
      (PiRLCPoseidonGeometry.outputInputBlock program)
      (PiRLCPoseidonGeometry.outputInputStart program)
      (PiRLCPoseidonGeometry.outputInputFits geometry) slotCount
      (PiRLCPoseidonGeometry.oneColumn geometry) invocation lane

theorem priorInputProgram_state?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (invocation : Fin PilotPoseidonPlan.invocationCount) :
    (priorInputProgram program).state? logicalWidth
        (PiRLCPoseidonGeometry.oneColumn geometry).val invocation.val =
      some (PilotPoseidonPlan.priorInputState geometry invocation) := by
  apply PoseidonInput.Program.state?_eq_some
  · simpa using priorInputProgram_form? geometry invocation (0 : Fin 8)
  · simpa using priorInputProgram_form? geometry invocation (1 : Fin 8)
  · simpa using priorInputProgram_form? geometry invocation (2 : Fin 8)
  · simpa using priorInputProgram_form? geometry invocation (3 : Fin 8)
  · simpa using priorInputProgram_form? geometry invocation (4 : Fin 8)
  · simpa using priorInputProgram_form? geometry invocation (5 : Fin 8)
  · simpa using priorInputProgram_form? geometry invocation (6 : Fin 8)
  · simpa using priorInputProgram_form? geometry invocation (7 : Fin 8)

theorem outputInputProgram_state?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (invocation : Fin PilotPoseidonPlan.invocationCount) :
    (outputInputProgram program).state? logicalWidth
        (PiRLCPoseidonGeometry.oneColumn geometry).val invocation.val =
      some (PilotPoseidonPlan.outputInputState geometry invocation) := by
  apply PoseidonInput.Program.state?_eq_some
  · simpa using outputInputProgram_form? geometry invocation (0 : Fin 8)
  · simpa using outputInputProgram_form? geometry invocation (1 : Fin 8)
  · simpa using outputInputProgram_form? geometry invocation (2 : Fin 8)
  · simpa using outputInputProgram_form? geometry invocation (3 : Fin 8)
  · simpa using outputInputProgram_form? geometry invocation (4 : Fin 8)
  · simpa using outputInputProgram_form? geometry invocation (5 : Fin 8)
  · simpa using outputInputProgram_form? geometry invocation (6 : Fin 8)
  · simpa using outputInputProgram_form? geometry invocation (7 : Fin 8)

theorem priorBlock_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (global : Fin (PilotPoseidonPlan.invocationCount * 94)) :
    (priorBlock geometry).row? logicalWidth global.val =
      let decoded : Fin PilotPoseidonPlan.invocationCount × Fin 94 :=
        Fin.decodeProd global
      some (PoseidonSboxFamilyPlan.rowForms
        (PilotPoseidonPlan.priorInterface geometry) decoded.1 decoded.2) := by
  simpa [priorBlock, PilotPoseidonPlan.priorInterface] using
    Poseidon.Block.row?_ofSemantic
      (PilotPoseidonPlan.priorSchedule program) (by rfl)
      (PiRLCRetainedGeometry.priorPoseidonStart program)
      (PiRLCPoseidonGeometry.oneColumn geometry) (priorInputProgram program)
      (PiRLCRetainedGeometry.priorPoseidonFits
        (PiRLCPoseidonGeometry.prefixGeometry geometry))
      (PilotPoseidonPlan.priorInputState geometry)
      (priorInputProgram_state? geometry) global

theorem outputBlock_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (global : Fin (PilotPoseidonPlan.invocationCount * 94)) :
    (outputBlock geometry).row? logicalWidth global.val =
      let decoded : Fin PilotPoseidonPlan.invocationCount × Fin 94 :=
        Fin.decodeProd global
      some (PoseidonSboxFamilyPlan.rowForms
        (PilotPoseidonPlan.outputInterface geometry) decoded.1 decoded.2) := by
  simpa [outputBlock, PilotPoseidonPlan.outputInterface] using
    Poseidon.Block.row?_ofSemantic
      (PilotPoseidonPlan.outputSchedule program) (by rfl)
      (PiRLCRetainedGeometry.outputPoseidonStart program)
      (PiRLCPoseidonGeometry.oneColumn geometry) (outputInputProgram program)
      (PiRLCRetainedGeometry.outputPoseidonFits
        (PiRLCPoseidonGeometry.prefixGeometry geometry))
      (PilotPoseidonPlan.outputInputState geometry)
      (outputInputProgram_state? geometry) global

theorem priorBlock_plan_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PilotPoseidonPlan.priorPlan geometry).rowCount) :
    (MatrixProgram.Block.poseidon (priorBlock geometry)).row?
        logicalWidth sourceRow global.val =
      some ((PilotPoseidonPlan.priorPlan geometry).forms global) := by
  change (priorBlock geometry).row? logicalWidth global.val = _
  simpa [PilotPoseidonPlan.priorPlan, PoseidonSboxFamilyPlan.plan,
    ProductionRelation.Plan.indexed] using
      priorBlock_row? geometry global

theorem outputBlock_plan_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PilotPoseidonPlan.outputPlan geometry).rowCount) :
    (MatrixProgram.Block.poseidon (outputBlock geometry)).row?
        logicalWidth sourceRow global.val =
      some ((PilotPoseidonPlan.outputPlan geometry).forms global) := by
  change (outputBlock geometry).row? logicalWidth global.val = _
  simpa [PilotPoseidonPlan.outputPlan, PoseidonSboxFamilyPlan.plan,
    ProductionRelation.Plan.indexed] using
      outputBlock_row? geometry global

theorem priorProgram_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PilotPoseidonPlan.priorPlan geometry).rowCount) :
    (MatrixProgram.Program.mk
        [.poseidon (priorBlock geometry)]).row?
        logicalWidth sourceRow global.val =
      some ((PilotPoseidonPlan.priorPlan geometry).forms global) := by
  have bound : global.val <
      (MatrixProgram.Block.poseidon (priorBlock geometry)).rowCount := by
    change global.val < (priorBlock geometry).rowCount
    rw [priorBlock_rowCount]
    simpa [PilotPoseidonPlan.priorPlan,
      PilotPoseidonPlan.invocationCount_eq] using global.isLt
  rw [MatrixProgram.Program.singleton_row?, if_pos bound]
  exact priorBlock_plan_row? geometry sourceRow global

theorem outputProgram_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCPoseidonGeometry.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PilotPoseidonPlan.outputPlan geometry).rowCount) :
    (MatrixProgram.Program.mk
        [.poseidon (outputBlock geometry)]).row?
        logicalWidth sourceRow global.val =
      some ((PilotPoseidonPlan.outputPlan geometry).forms global) := by
  have bound : global.val <
      (MatrixProgram.Block.poseidon (outputBlock geometry)).rowCount := by
    change global.val < (outputBlock geometry).rowCount
    rw [outputBlock_rowCount]
    simpa [PilotPoseidonPlan.outputPlan,
      PilotPoseidonPlan.invocationCount_eq] using global.isLt
  rw [MatrixProgram.Program.singleton_row?, if_pos bound]
  exact outputBlock_plan_row? geometry sourceRow global

end NightstreamFPrime.Export.Stage1.PilotPoseidonMatrixProgram
