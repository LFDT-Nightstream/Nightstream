import NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixProgram

/-! Prove that the application matrix input program reconstructs the placed forms. -/

namespace NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixProgram

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open ApplicationPoseidonRetainedBlock ApplicationPoseidonRetainedGeometry

variable {application : Stage1.Application.Program} {certificate : Certificate application}
  {columns : Nat}

private theorem previousRule_succ (geometry : Geometry application certificate columns)
    (invocation : Fin 2) (lane : Fin 8) :
    (previousRule application certificate).form? columns (oneColumn geometry).val
        (1 + invocation.val) lane.val =
      some (some (Stage1.Poseidon2HashChainCompact.output (interface geometry)
        ⟨invocation.val, by omega⟩ lane)) := by
  have slotBound : ∀ selected : Fin 8,
      78 + invocation.val * 86 + selected.val < (block application certificate).slotCount := by
    intro selected
    rw [block_slotCount]
    omega
  have selected := PoseidonInput.Rule.external_form?_ofSemantic
    (region := ⟨1, 2, 0, 8⟩) invocation lane lane.isLt
    (block application certificate) (localStart application) (localFits geometry)
    (oneColumn geometry).val 78 86 slotBound
  simp only [Nat.zero_add] at selected
  change (previousRule application certificate).form? columns (oneColumn geometry).val
    (1 + invocation.val) lane.val = _ at selected
  rw [selected]
  apply congrArg some
  apply congrArg some
  apply congrArg (fun state => SparseLayer.external state lane)
  funext selected
  apply congrArg ((block application certificate).form (localStart application) (localFits geometry))
  apply Fin.ext
  simp [Fin.encodeProd, PoseidonRetainedSlots.finalRow_val]
  omega

private theorem constantRule_result (geometry : Geometry application certificate columns)
    (invocation : Fin 3) (lane : Fin 8) :
    constantRule.form? columns (oneColumn geometry).val invocation.val lane.val =
      some (some ((constants (Fin.encodeProd (invocation, lane))).elim SparseForm.empty
        (SparseLayer.constant (oneColumn geometry)))) := by
  cases found : constants (Fin.encodeProd (invocation, lane)) with
  | none =>
    simpa only [constantRule, Option.elim_none, Nat.zero_add] using
      PoseidonInput.Rule.optionalConstant_form?_ofSemantic_of_none
        (logicalWidth := columns) (region := ⟨0, 3, 0, 8⟩) invocation lane
        (oneColumn geometry).val constants 8 (Fin.encodeProd (invocation, lane))
        (by simp [Fin.encodeProd]; omega) found
  | some coefficient =>
    simpa only [constantRule, Option.elim_some, SparseLayer.constant, Nat.zero_add] using
      PoseidonInput.Rule.optionalConstant_form?_ofSemantic_of_some
        (region := ⟨0, 3, 0, 8⟩) invocation lane (oneColumn geometry)
        constants 8 (Fin.encodeProd (invocation, lane))
        (by simp [Fin.encodeProd]; omega) coefficient found

private theorem wordRule_result {sourceWidth : Nat}
    (retained : LowNormBlock.Block sourceWidth) (start : Nat)
    (fits : start + retained.coordinateCount ≤ columns) (width : 4 ≤ retained.slotCount)
    (oneColumn atInvocation : Nat) (invocation : Fin 3) (lane : Fin 8) :
    (PoseidonInput.Rule.mk ⟨atInvocation, 1, 0, 4⟩
      (.retained (RetainedBlock.ofSemantic retained start) 0 0 1)).form?
      columns oneColumn invocation.val lane.val =
    some (if selected : invocation.val = atInvocation ∧ lane.val < 4 then
      some (retained.form start fits ⟨lane.val, selected.2.trans_le width⟩) else none) := by
  by_cases selected : invocation.val = atInvocation ∧ lane.val < 4
  · rw [dif_pos selected]
    simpa only [Nat.zero_add, Nat.add_zero, Nat.zero_mul, Nat.mul_one, Fin.val_zero, selected.1] using
      PoseidonInput.Rule.retained_form?_ofSemantic
        (region := ⟨atInvocation, 1, 0, 4⟩) (0 : Fin 1) ⟨lane.val, selected.2⟩
        retained start fits oneColumn 0 0 1 (by simpa using selected.2.trans_le width)
  · rw [dif_neg selected]
    apply PoseidonInput.Rule.form?_eq_some_none
    unfold PoseidonInput.Region.offsets?
    simp only [Nat.zero_le, Nat.sub_zero, ↓reduceIte]
    split_ifs <;> simp_all <;> omega

theorem inputProgram_form? (geometry : Geometry application certificate columns)
    (invocation : Fin 3) (lane : Fin 8) :
    (inputProgram application certificate).form? columns (oneColumn geometry).val
      invocation.val lane.val =
      some (Stage1.Poseidon2HashChainCompact.input (interface geometry) invocation lane) := by
  have previous0 : (previousRule application certificate).form? columns
      (oneColumn geometry).val 0 lane.val = some none := by
    apply PoseidonInput.Rule.form?_eq_some_none
    simp [previousRule, PoseidonInput.Region.offsets?]
  have previous1 := previousRule_succ geometry (0 : Fin 2) lane
  have previous2 := previousRule_succ geometry (1 : Fin 2) lane
  have prior := wordRule_result (ApplicationRetainedBlocks.inputBlock application)
    (inputStart application) (inputFits geometry) (by rfl) (oneColumn geometry).val 0 invocation lane
  have message := wordRule_result (ApplicationRetainedBlocks.witnessBlock application)
    (ApplicationPoseidonRetainedGeometry.witnessStart application) (witnessFits geometry)
    (by change 4 ≤ application.witnessWordCount; rw [certificate.wordCount]; rfl)
    (oneColumn geometry).val 1 invocation lane
  have constant := constantRule_result geometry invocation lane
  have previous : (previousRule application certificate).form? columns
      (oneColumn geometry).val invocation.val lane.val =
      some (if invocation.val = 0 then none else
        some (Stage1.Poseidon2HashChainCompact.output (interface geometry)
          ⟨invocation.val - 1, by omega⟩ lane)) := by
    fin_cases invocation
    · exact previous0
    · exact previous1
    · exact previous2
  have folded := PoseidonInput.Program.four_form?_of_results
    (previousRule application certificate) constantRule (priorRule application)
    (messageRule application) (oneColumn geometry).val invocation.val lane.val
    _ _ _ _ previous constant prior message
  change (inputProgram application certificate).form? columns (oneColumn geometry).val
      invocation.val lane.val = _ at folded
  rw [folded]
  fin_cases invocation <;> fin_cases lane
  all_goals simp [Stage1.Poseidon2HashChainCompact.input,
    Stage1.Poseidon2HashChainCompact.blockLane, constants, Fin.encodeProd,
    SparseLayer.addConstant, SparseLayer.add, SparseLayer.constant,
    SparseForm.add, SparseForm.empty, interface]
  all_goals rfl

end NightstreamFPrime.Export.Stage1.ApplicationPoseidonMatrixProgram
