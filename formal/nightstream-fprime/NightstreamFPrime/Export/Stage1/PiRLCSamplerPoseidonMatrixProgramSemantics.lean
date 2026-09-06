import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonMatrixProgram

/-!
Proves that the compact PiRLC sampler Poseidon2 matrix program reconstructs
the exact cross-family previous state and verifier-owned entry words.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonMatrixProgram

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec

private theorem piCcsPreviousRule_zero
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (lane : Fin 8) :
    (piCcsPreviousRule program).form? logicalWidth
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val 0 lane.val =
      some (some (PiRLCSamplerPoseidonPlan.piCcsFinalOutput geometry lane)) := by
  have slotBound : ∀ selected : Fin 8,
      piCcsFinalSlotBase + 0 * 0 + selected.val <
        (PiCCSPoseidonPlan.retainedBlock program).slotCount := by
    intro selected
    rw [piCcsFinalSlotBase_eq, PiCCSPoseidonPlan.retainedBlock_slotCount]
    omega
  rw [show (piCcsPreviousRule program).form? logicalWidth
      (PiRLCSamplerPoseidonPlan.oneColumn geometry).val 0 lane.val =
        some (some (SparseLayer.external (fun selected : Fin 8 =>
          (PiCCSPoseidonPlan.retainedBlock program).form
            (PiCCSPoseidonPlan.retainedStart program)
            (PiCCSPoseidonPlan.retainedFits geometry)
            ⟨piCcsFinalSlotBase + 0 * 0 + selected.val,
              slotBound selected⟩) lane)) by
    simpa [piCcsPreviousRule] using
      PoseidonInput.Rule.external_form?_ofSemantic
        (region := PoseidonInput.Region.mk 0 1 0 8)
        (0 : Fin 1) lane lane.isLt
        (PiCCSPoseidonPlan.retainedBlock program)
        (PiCCSPoseidonPlan.retainedStart program)
        (PiCCSPoseidonPlan.retainedFits geometry)
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val
        piCcsFinalSlotBase 0 slotBound]
  apply congrArg some
  apply congrArg some
  unfold PiRLCSamplerPoseidonPlan.piCcsFinalOutput
    PiCCSPoseidonPlan.outputState
    PoseidonRetainedFamily.outputState PoseidonRetainedFamily.form
  apply congrArg (fun state => SparseLayer.external state lane)
  funext selected
  apply congrArg ((PiCCSPoseidonPlan.schedule program).block.form
    (PiCCSPoseidonPlan.retainedStart program)
    (PiCCSPoseidonPlan.retainedFits geometry))
  apply Fin.ext
  simp [piCcsFinalSlotBase, PoseidonRetainedFamily.slot, Fin.encodeProd,
    PoseidonRetainedSlots.finalRow_val, Nat.mul_comm]
  omega

private theorem piCcsPreviousRule_outside
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) (lane : Fin 8)
    (notFirst : current.val ≠ 0) :
    (piCcsPreviousRule program).form? logicalWidth
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val lane.val =
      some none := by
  apply PoseidonInput.Rule.form?_eq_some_none
  simp [piCcsPreviousRule, PoseidonInput.Region.offsets?, notFirst]

private theorem samplerPreviousRule_zero
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (lane : Fin 8) :
    (samplerPreviousRule program).form? logicalWidth
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val 0 lane.val =
      some none := by
  apply PoseidonInput.Rule.form?_eq_some_none
  simp [samplerPreviousRule, PoseidonInput.Region.offsets?]

private theorem samplerPreviousRule_succ
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (invocationOffset : Fin 152) (lane : Fin 8) :
    (samplerPreviousRule program).form? logicalWidth
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val
        (1 + invocationOffset.val) lane.val =
      some (some (PoseidonRetainedFamily.outputState
        (PiRLCSamplerPoseidonPlan.schedule program)
        (PiRLCSamplerPoseidonPlan.retainedStart program)
        (PiRLCSamplerPoseidonPlan.retainedFits geometry)
        ⟨invocationOffset.val, by
          rw [PiRLCSamplerPoseidonPlan.invocationCount_eq]
          omega⟩ lane)) := by
  have slotBound : ∀ selected : Fin 8,
      78 + invocationOffset.val * 86 + selected.val <
        (PiRLCSamplerPoseidonPlan.schedule program).block.slotCount := by
    intro selected
    rw [(PiRLCSamplerPoseidonPlan.schedule program).slotCount_eq,
      PiRLCSamplerPoseidonPlan.invocationCount_eq,
      PoseidonRetainedSlots.rows_length]
    omega
  rw [show (samplerPreviousRule program).form? logicalWidth
      (PiRLCSamplerPoseidonPlan.oneColumn geometry).val
      (1 + invocationOffset.val) lane.val =
        some (some (SparseLayer.external (fun selected : Fin 8 =>
          (PiRLCSamplerPoseidonPlan.schedule program).block.form
            (PiRLCSamplerPoseidonPlan.retainedStart program)
            (PiRLCSamplerPoseidonPlan.retainedFits geometry)
            ⟨78 + invocationOffset.val * 86 + selected.val,
              slotBound selected⟩) lane)) by
    simpa [samplerPreviousRule] using
      PoseidonInput.Rule.external_form?_ofSemantic
        (region := PoseidonInput.Region.mk 1 152 0 8)
        invocationOffset lane lane.isLt
        (PiRLCSamplerPoseidonPlan.schedule program).block
        (PiRLCSamplerPoseidonPlan.retainedStart program)
        (PiRLCSamplerPoseidonPlan.retainedFits geometry)
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val 78 86 slotBound]
  apply congrArg some
  apply congrArg some
  unfold PoseidonRetainedFamily.outputState PoseidonRetainedFamily.form
  apply congrArg (fun state => SparseLayer.external state lane)
  funext selected
  apply congrArg ((PiRLCSamplerPoseidonPlan.schedule program).block.form
    (PiRLCSamplerPoseidonPlan.retainedStart program)
    (PiRLCSamplerPoseidonPlan.retainedFits geometry))
  apply Fin.ext
  simp [PoseidonRetainedFamily.slot, Fin.encodeProd,
    PoseidonRetainedSlots.finalRow_val, Nat.mul_comm]
  omega

@[simp] theorem constantAt_encode
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) (lane : Fin 8) :
    constantAt (Fin.encodeProd (current, lane)) =
      if (PiRLCSamplerPoseidonPlan.descriptor current).2.val = 0 then
        some (PiRLCSamplerPoseidonPlan.entryWord
          (PiRLCSamplerPoseidonPlan.descriptor current).1 lane)
      else none := by
  simp [constantAt]

private theorem entryRule_entry
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) (lane : Fin 8)
    (entry : (PiRLCSamplerPoseidonPlan.descriptor current).2.val = 0) :
    entryRule.form? logicalWidth
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val lane.val =
      some (some (SparseForm.singleton
        (PiRLCSamplerPoseidonPlan.oneColumn geometry)
        (PiRLCSamplerPoseidonPlan.entryWord
          (PiRLCSamplerPoseidonPlan.descriptor current).1 lane))) := by
  let index : Fin (PiRLCSamplerPoseidonPlan.invocationCount * 8) :=
    Fin.encodeProd (current, lane)
  have indexEq : current.val * 8 + lane.val = index.val := by
    simp [index, Fin.encodeProd, Nat.mul_comm]
  have found : constantAt index = some
      (PiRLCSamplerPoseidonPlan.entryWord
        (PiRLCSamplerPoseidonPlan.descriptor current).1 lane) := by
    simpa [index, entry] using constantAt_encode current lane
  simpa [entryRule, constants] using
    PoseidonInput.Rule.optionalConstant_form?_ofSemantic_of_some
      (region := PoseidonInput.Region.mk 0 153 0 8) current lane
      (PiRLCSamplerPoseidonPlan.oneColumn geometry) constantAt 8 index
      indexEq _ found

private theorem entryRule_nonentry
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) (lane : Fin 8)
    (notEntry : (PiRLCSamplerPoseidonPlan.descriptor current).2.val ≠ 0) :
    entryRule.form? logicalWidth
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val lane.val =
      some (some .empty) := by
  let index : Fin (PiRLCSamplerPoseidonPlan.invocationCount * 8) :=
    Fin.encodeProd (current, lane)
  have indexEq : current.val * 8 + lane.val = index.val := by
    simp [index, Fin.encodeProd, Nat.mul_comm]
  have found : constantAt index = none := by
    simpa [index, notEntry] using constantAt_encode current lane
  simpa [entryRule, constants] using
    PoseidonInput.Rule.optionalConstant_form?_ofSemantic_of_none
      (region := PoseidonInput.Region.mk 0 153 0 8) current lane
      (PiRLCSamplerPoseidonPlan.oneColumn geometry).val constantAt 8 index
      indexEq found

theorem inputProgram_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) (lane : Fin 8) :
    (inputProgram program).form? logicalWidth
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val lane.val =
      some (PiRLCSamplerPoseidonPlan.inputState geometry current lane) := by
  by_cases first : current.val = 0
  · have piCcsResult :
        (piCcsPreviousRule program).form? logicalWidth
            (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val
            lane.val =
          some (some (PiRLCSamplerPoseidonPlan.previousOutput geometry current
            lane)) := by
      rw [first]
      simpa [PiRLCSamplerPoseidonPlan.previousOutput, first] using
        piCcsPreviousRule_zero geometry lane
    have samplerResult :
        (samplerPreviousRule program).form? logicalWidth
            (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val
            lane.val = some none := by
      rw [first]
      exact samplerPreviousRule_zero geometry lane
    by_cases entry :
        (PiRLCSamplerPoseidonPlan.descriptor current).2.val = 0
    · have entryResult := entryRule_entry geometry current lane entry
      have folded := PoseidonInput.Program.three_form?_of_results
        (piCcsPreviousRule program) (samplerPreviousRule program) entryRule
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val lane.val
        _ _ _ piCcsResult samplerResult entryResult
      simpa [inputProgram, PiRLCSamplerPoseidonPlan.inputState, entry,
        SparseForm.add, SparseForm.empty] using folded
    · have entryResult := entryRule_nonentry geometry current lane entry
      have folded := PoseidonInput.Program.three_form?_of_results
        (piCcsPreviousRule program) (samplerPreviousRule program) entryRule
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val lane.val
        _ _ _ piCcsResult samplerResult entryResult
      simpa [inputProgram, PiRLCSamplerPoseidonPlan.inputState, entry,
        SparseForm.add, SparseForm.empty] using folded
  · have piCcsResult := piCcsPreviousRule_outside geometry current lane first
    let invocationOffset : Fin 152 :=
      ⟨current.val - 1, by
        have bound : current.val < 153 := by
          simpa only [PiRLCSamplerPoseidonPlan.invocationCount_eq] using
            current.isLt
        omega⟩
    have currentEq : current.val = 1 + invocationOffset.val := by
      dsimp [invocationOffset]
      omega
    have samplerResult :
        (samplerPreviousRule program).form? logicalWidth
            (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val
            lane.val =
          some (some (PiRLCSamplerPoseidonPlan.previousOutput geometry current
            lane)) := by
      rw [currentEq]
      have selected := samplerPreviousRule_succ geometry invocationOffset lane
      rw [selected]
      apply congrArg some
      apply congrArg some
      unfold PiRLCSamplerPoseidonPlan.previousOutput
      rw [dif_neg first]
    by_cases entry :
        (PiRLCSamplerPoseidonPlan.descriptor current).2.val = 0
    · have entryResult := entryRule_entry geometry current lane entry
      have folded := PoseidonInput.Program.three_form?_of_results
        (piCcsPreviousRule program) (samplerPreviousRule program) entryRule
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val lane.val
        _ _ _ piCcsResult samplerResult entryResult
      simpa [inputProgram, PiRLCSamplerPoseidonPlan.inputState, entry,
        SparseForm.add, SparseForm.empty] using folded
    · have entryResult := entryRule_nonentry geometry current lane entry
      have folded := PoseidonInput.Program.three_form?_of_results
        (piCcsPreviousRule program) (samplerPreviousRule program) entryRule
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val lane.val
        _ _ _ piCcsResult samplerResult entryResult
      simpa [inputProgram, PiRLCSamplerPoseidonPlan.inputState, entry,
        SparseForm.add, SparseForm.empty] using folded

theorem inputProgram_state?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (current : Fin PiRLCSamplerPoseidonPlan.invocationCount) :
    (inputProgram program).state? logicalWidth
        (PiRLCSamplerPoseidonPlan.oneColumn geometry).val current.val =
      some (PiRLCSamplerPoseidonPlan.inputState geometry current) := by
  apply PoseidonInput.Program.state?_eq_some
  · simpa using inputProgram_form? geometry current (0 : Fin 8)
  · simpa using inputProgram_form? geometry current (1 : Fin 8)
  · simpa using inputProgram_form? geometry current (2 : Fin 8)
  · simpa using inputProgram_form? geometry current (3 : Fin 8)
  · simpa using inputProgram_form? geometry current (4 : Fin 8)
  · simpa using inputProgram_form? geometry current (5 : Fin 8)
  · simpa using inputProgram_form? geometry current (6 : Fin 8)
  · simpa using inputProgram_form? geometry current (7 : Fin 8)

theorem block_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (global : Fin (PiRLCSamplerPoseidonPlan.invocationCount * 94)) :
    (block geometry).row? logicalWidth global.val =
      let decoded : Fin PiRLCSamplerPoseidonPlan.invocationCount × Fin 94 :=
        Fin.decodeProd global
      some (PoseidonSboxFamilyPlan.rowForms
        (PiRLCSamplerPoseidonPlan.interface geometry) decoded.1 decoded.2) := by
  simpa [block, PiRLCSamplerPoseidonPlan.interface] using
    Poseidon.Block.row?_ofSemantic
      (PiRLCSamplerPoseidonPlan.schedule program) (by rfl)
      (PiRLCSamplerPoseidonPlan.retainedStart program)
      (PiRLCSamplerPoseidonPlan.oneColumn geometry) (inputProgram program)
      (PiRLCSamplerPoseidonPlan.retainedFits geometry)
      (PiRLCSamplerPoseidonPlan.inputState geometry)
      (inputProgram_state? geometry) global

theorem matrixProgram_row?
    {program : Program} {logicalWidth : Nat}
    (geometry : PiCCSPoseidonPlan.Geometry program logicalWidth)
    (sourceRow : Nat → Option R1CS.Row)
    (global : Fin (PiRLCSamplerPoseidonPlan.invocationCount * 94)) :
    (matrixProgram geometry).row? logicalWidth sourceRow global.val =
      let decoded : Fin PiRLCSamplerPoseidonPlan.invocationCount × Fin 94 :=
        Fin.decodeProd global
      some (PoseidonSboxFamilyPlan.rowForms
        (PiRLCSamplerPoseidonPlan.interface geometry) decoded.1 decoded.2) := by
  have bound : global.val <
      (MatrixProgram.Block.poseidon (block geometry)).rowCount := by
    change global.val < (block geometry).rowCount
    rw [show (block geometry).rowCount =
        PiRLCSamplerPoseidonPlan.invocationCount * 94 by
      exact Poseidon.Block.ofSemantic_rowCount
        (PiRLCSamplerPoseidonPlan.schedule program)
        (PiRLCSamplerPoseidonPlan.retainedStart program)
        (PiRLCSamplerPoseidonPlan.oneColumn geometry) (inputProgram program)]
    exact global.isLt
  rw [show matrixProgram geometry =
      MatrixProgram.Program.mk [.poseidon (block geometry)] by rfl]
  rw [MatrixProgram.Program.singleton_row?, if_pos bound]
  exact block_row? geometry global

end NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonMatrixProgram
