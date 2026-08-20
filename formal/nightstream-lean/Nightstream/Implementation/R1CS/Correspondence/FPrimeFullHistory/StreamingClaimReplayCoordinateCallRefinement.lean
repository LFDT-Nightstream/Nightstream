import Nightstream.Implementation.Nebula.Commitment.Lanes.ShiftedTernaryEncodingBridge
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataAccumulator
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplaySchema
import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.ShiftedTernarySound

/-!
Contract: structural refinement of one physical claim-coordinate call.

Assurance tier: model-level row refinement.

Owns the proof that one exact shifted-ternary and seeded-Phi81 call, linked to
one verifier-owned claim frame, computes the authoritative contribution for
its map and claim chunk.

Does not own generated call identity, state update or carry rows, phase links,
complete accumulator runs, lifecycle selection, or Module-SIS hardness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateCallRefinement

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.AffinePins
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.SeededPhi81RingRefinement
open Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

/-- Exact trust-boundary link for every active field of one physical call. -/
def CallFrameLinked
    (frame : ClaimFrame) (assignment : Nat → Nat)
    (call : CoordinateCall) : Prop :=
  ∀ field ∈ call.activeFields,
    assignment (call.fieldColumn field) =
      (frame ⟨call.mapKind.framePosition field,
        call.mapKind.framePosition_lt field⟩).val

private theorem opening_block_rows_satisfy
    (call : CoordinateCall) (assignment : Nat → Nat)
    (satisfied : call.Satisfied assignment)
    (field : Fin call.mapKind.fieldCount)
    (active : field ∈ call.activeFields) :
    Satisfies (call.openingBlockRows field) assignment := by
  intro row rowMember
  apply satisfied row
  apply List.mem_append_left
  apply List.mem_append_right
  exact List.mem_flatMap.mpr ⟨field, active, rowMember⟩

private theorem active_opening
    (call : CoordinateCall) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : call.Satisfied assignment)
    (field : Fin call.mapKind.fieldCount)
    (active : field ∈ call.activeFields) :
    Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.CanonicalOpening
      (localAssignment assignment (call.fieldColumn field)
        (call.digitStart field)) := by
  have mapped := opening_block_rows_satisfy call assignment satisfied
    field active
  have localSatisfies :
      Satisfies
        Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.canonicalRows
        (localAssignment assignment (call.fieldColumn field)
          (call.digitStart field)) := by
    exact (Relabel.satisfies_mapped_iff
      Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.canonicalRows
      (OwnerCertificate.shiftedTernaryColumnMap
        (call.fieldColumn field) (call.digitStart field)) assignment).mp mapped
  have localCanonical :
      ∀ column,
        localAssignment assignment (call.fieldColumn field)
            (call.digitStart field) column < goldilocksP := by
    intro column
    exact canonical _
  have localOne :
      localAssignment assignment (call.fieldColumn field)
          (call.digitStart field) 0 = 1 := by
    simpa using one
  exact canonicalOpening_of_canonicalRows goldilocks_euclidPrime
    localCanonical localOne localSatisfies

private theorem zero_rows_satisfy
    (call : CoordinateCall) (assignment : Nat → Nat)
    (satisfied : call.Satisfied assignment) :
    Satisfies call.zeroRows assignment := by
  intro row rowMember
  apply satisfied row
  apply List.mem_append_left
  exact List.mem_append_left _ rowMember

private theorem zeroPins_canonical (call : CoordinateCall) :
    AffinePins.PinsCanonical call.zeroPins := by
  intro pin pinMember
  rcases List.mem_ofFn.mp pinMember with ⟨digit, rfl⟩
  trivial

private theorem inactive_digit_zero
    (call : CoordinateCall) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : call.Satisfied assignment)
    (field : Fin call.mapKind.fieldCount)
    (inactive : field ∉ call.activeFields)
    (digit : Fin digitCount) :
    assignment (call.wordStart field + digit.val) = 0 := by
  have facts := AffinePins.rows_sound (zeroPins_canonical call) canonical one
    (zero_rows_satisfy call assignment satisfied)
  have exact := facts (.zero (call.zeroDigitStart + digit.val))
    (List.mem_ofFn.mpr ⟨digit, rfl⟩)
  simpa [CoordinateCall.wordStart, inactive] using exact

private theorem active_digit_exact
    (frame : ClaimFrame) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (call : CoordinateCall)
    (satisfied : call.Satisfied assignment)
    (linked : CallFrameLinked frame assignment call)
    (chunk : Fin claimChunkCount)
    (callChunk : call.chunk = chunk)
    (field : Fin call.mapKind.fieldCount)
    (active : field ∈ call.activeFields)
    (digit : Fin digitCount) :
    assignment (call.wordStart field + digit.val) =
      selectedDigit frame call.mapKind chunk field digit := by
  have selected : call.mapKind.claimChunk field = chunk := by
    have exact := (call.mapKind.mem_activeFields call.chunk field).mp active
    simpa [callChunk] using exact
  have source := linked field active
  have opening := active_opening call assignment canonical one satisfied
    field active
  rw [CoordinateCall.wordStart, if_pos active]
  unfold selectedDigit
  rw [if_pos selected]
  exact
    Nightstream.Implementation.Nebula.ShiftedTernaryEncodingBridge.productionDigit_eq_protocolDigit
      (frame ⟨call.mapKind.framePosition field,
        call.mapKind.framePosition_lt field⟩)
      source opening digit

private theorem getD_ofFn
    {alpha : Type} {count : Nat} (function : Fin count → alpha)
    (index : Nat) (fallback : alpha) (bound : index < count) :
    (List.ofFn function).getD index fallback =
      function ⟨index, bound⟩ := by
  have listBound : index < (List.ofFn function).length := by
    simpa using bound
  rw [List.getD_eq_getElem _ _ listBound]
  exact List.getElem_ofFn listBound

private theorem wordIndex_quotient
    (kind : MapKind) (field : Fin kind.fieldCount)
    (digit : Fin digitCount) :
    (field.val * digitCount + digit.val) / digitCount = field.val := by
  rw [Nat.mul_comm field.val digitCount]
  rw [Nat.mul_add_div (by decide : 0 < digitCount),
    Nat.div_eq_of_lt digit.isLt, Nat.add_zero]

private theorem wordIndex_remainder
    (kind : MapKind) (field : Fin kind.fieldCount)
    (digit : Fin digitCount) :
    (field.val * digitCount + digit.val) % digitCount = digit.val := by
  exact Nat.mul_add_mod_of_lt digit.isLt

private theorem call_bitColumn
    (call : CoordinateCall) (field : Fin call.mapKind.fieldCount)
    (digit : Fin digitCount) :
    call.block.bitColumn (field.val * digitCount + digit.val) =
      some (call.wordStart field + digit.val) := by
  have selectorBound :
      field.val * digitCount + digit.val <
        (List.ofFn call.wordStart).length * digitCount := by
    simp only [List.length_ofFn]
    have fieldBound := field.isLt
    have digitBound := digit.isLt
    unfold digitCount at digitBound ⊢
    omega
  unfold SeededPhi81.Block.bitColumn
  simp only [CoordinateCall.block, CoordinateCall.wordStarts]
  rw [if_neg (by decide : 41 ≠ 0), if_pos (by
    simpa [digitCount] using selectorBound)]
  simp only [digitCount]
  have quotient := wordIndex_quotient call.mapKind field digit
  have remainder := wordIndex_remainder call.mapKind field digit
  simp only [digitCount] at quotient remainder
  rw [quotient, remainder]
  rw [getD_ofFn call.wordStart field.val 0 field.isLt]

private theorem semantic_bitColumn
    (kind : MapKind) (field : Fin kind.fieldCount)
    (digit : Fin digitCount) :
    (semanticBlock kind).bitColumn
        (field.val * digitCount + digit.val) =
      some (field.val * digitCount + digit.val) := by
  have selectorBound :
      field.val * digitCount + digit.val <
        (List.ofFn fun field : Fin kind.fieldCount =>
          field.val * digitCount).length * digitCount := by
    simp only [List.length_ofFn]
    have fieldBound := field.isLt
    have digitBound := digit.isLt
    unfold digitCount at digitBound ⊢
    omega
  unfold SeededPhi81.Block.bitColumn
  simp only [semanticBlock, semanticWordStarts]
  rw [if_neg (by decide : digitCount ≠ 0), if_pos selectorBound]
  rw [wordIndex_quotient, wordIndex_remainder]
  rw [getD_ofFn
    (fun field : Fin kind.fieldCount => field.val * digitCount)
    field.val 0 field.isLt]

private theorem call_tail_bitColumn_none
    (call : CoordinateCall) (index : Nat)
    (tail : call.mapKind.fieldCount * digitCount ≤ index) :
    call.block.bitColumn index = none := by
  have outside :
      ¬ index < (List.ofFn call.wordStart).length * digitCount := by
    simp only [List.length_ofFn]
    exact Nat.not_lt.mpr tail
  unfold SeededPhi81.Block.bitColumn
  simp only [CoordinateCall.block, CoordinateCall.wordStarts]
  rw [if_neg (by decide : 41 ≠ 0), if_neg (by
    simpa [digitCount] using outside)]

private theorem semantic_tail_bitColumn_none
    (kind : MapKind) (index : Nat)
    (tail : kind.fieldCount * digitCount ≤ index) :
    (semanticBlock kind).bitColumn index = none := by
  have outside :
      ¬ index <
        (List.ofFn fun field : Fin kind.fieldCount =>
          field.val * digitCount).length * digitCount := by
    simp only [List.length_ofFn]
    exact Nat.not_lt.mpr tail
  unfold SeededPhi81.Block.bitColumn
  simp only [semanticBlock, semanticWordStarts]
  rw [if_neg (by decide : digitCount ≠ 0), if_neg outside]

private theorem word_digit_exact
    (frame : ClaimFrame) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (call : CoordinateCall)
    (satisfied : call.Satisfied assignment)
    (linked : CallFrameLinked frame assignment call)
    (chunk : Fin claimChunkCount)
    (callChunk : call.chunk = chunk)
    (field : Fin call.mapKind.fieldCount) (digit : Fin digitCount) :
    assignment (call.wordStart field + digit.val) =
      selectedDigit frame call.mapKind chunk field digit := by
  by_cases active : field ∈ call.activeFields
  · exact active_digit_exact frame assignment canonical one call satisfied
      linked chunk callChunk field active digit
  · have notSelected :
        call.mapKind.claimChunk field ≠ chunk := by
      intro selected
      apply active
      apply (call.mapKind.mem_activeFields call.chunk field).2
      simpa [callChunk] using selected
    calc
      assignment (call.wordStart field + digit.val) = 0 :=
        inactive_digit_zero call assignment canonical one satisfied
          field active digit
      _ = selectedDigit frame call.mapKind chunk field digit := by
        simp [selectedDigit, notSelected]

private theorem call_inputValue_exact
    (frame : ClaimFrame) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (call : CoordinateCall)
    (satisfied : call.Satisfied assignment)
    (linked : CallFrameLinked frame assignment call)
    (chunk : Fin claimChunkCount)
    (callChunk : call.chunk = chunk)
    (messageCol messageRow : Nat) :
    call.block.inputValue assignment messageCol messageRow =
      (semanticBlock call.mapKind).inputValue
        (semanticAssignment frame call.mapKind chunk)
        messageCol messageRow := by
  let index := messageRow * call.mapKind.messageColumnCount + messageCol
  by_cases valid : index < call.mapKind.fieldCount * digitCount
  · let field : Fin call.mapKind.fieldCount :=
      ⟨index / digitCount, by
        have positive : 0 < digitCount := by decide
        exact (Nat.div_lt_iff_lt_mul positive).2 (by
          simpa [Nat.mul_comm] using valid)⟩
    let digit : Fin digitCount :=
      ⟨index % digitCount, Nat.mod_lt _ (by decide)⟩
    have wordEqual :
        field.val * digitCount + digit.val = index := by
      change index / digitCount * digitCount + index % digitCount = index
      simpa [Nat.mul_comm] using Nat.div_add_mod index digitCount
    have physicalBitColumn :
        call.block.bitColumn
            (messageRow * call.block.messageCols + messageCol) =
          some (call.wordStart field + digit.val) := by
      change call.block.bitColumn index = _
      rw [← wordEqual]
      exact call_bitColumn call field digit
    have semanticBitColumn :
        (semanticBlock call.mapKind).bitColumn
            (messageRow * (semanticBlock call.mapKind).messageCols +
              messageCol) = some index := by
      change (semanticBlock call.mapKind).bitColumn index = some index
      rw [← wordEqual]
      exact semantic_bitColumn call.mapKind field digit
    have semanticExact :
        semanticAssignment frame call.mapKind chunk index =
          selectedDigit frame call.mapKind chunk field digit := by
      unfold semanticAssignment
      rw [dif_pos valid]
    rw [SeededPhi81.Block.inputValue_eq_of_bitColumn_some
      physicalBitColumn]
    rw [word_digit_exact frame assignment canonical one call satisfied linked
      chunk callChunk field digit]
    rw [SeededPhi81.Block.inputValue_eq_of_bitColumn_some
      semanticBitColumn]
    exact semanticExact.symm
  · have tail : call.mapKind.fieldCount * digitCount ≤ index :=
      Nat.le_of_not_gt valid
    have physicalNoColumn :
        call.block.bitColumn
            (messageRow * call.block.messageCols + messageCol) = none := by
      change call.block.bitColumn index = none
      exact call_tail_bitColumn_none call index tail
    have semanticNoColumn :
        (semanticBlock call.mapKind).bitColumn
            (messageRow * (semanticBlock call.mapKind).messageCols +
              messageCol) = none := by
      change (semanticBlock call.mapKind).bitColumn index = none
      exact semantic_tail_bitColumn_none call.mapKind index tail
    rw [SeededPhi81.Block.inputValue_eq_zero_of_bitColumn_none
      physicalNoColumn]
    rw [SeededPhi81.Block.inputValue_eq_zero_of_bitColumn_none
      semanticNoColumn]

private theorem call_coefficient_exact
    (call : CoordinateCall) (schedule : call.ScheduleValid)
    (output messageCol messageRow coordinate : Nat) :
    call.block.coefficient output messageCol messageRow coordinate =
      (semanticBlock call.mapKind).coefficient
        output messageCol messageRow coordinate := by
  unfold CoordinateCall.ScheduleValid at schedule
  have scheduleExact :
      { chunkSize := call.chunkSize
        seedsByOutput := call.seedsByOutput
        rejectionFuel := 16 } = call.mapKind.expectedSchedule := by
    simpa only [CoordinateCall.block] using schedule
  unfold SeededPhi81.Block.coefficient SeededPhi81.Block.baseRotations
  simp only [CoordinateCall.block, semanticBlock]
  rw [scheduleExact]

private theorem foldl_eq_of_pointwise
    {alpha beta : Type} (values : List alpha)
    (left right : beta → alpha → beta)
    (same : ∀ accumulated value,
      left accumulated value = right accumulated value)
    (initial : beta) :
    values.foldl left initial = values.foldl right initial := by
  induction values generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [same]
      exact inductionHypothesis _

private theorem call_linearValue_exact
    (frame : ClaimFrame) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (call : CoordinateCall)
    (satisfied : call.Satisfied assignment)
    (linked : CallFrameLinked frame assignment call)
    (chunk : Fin claimChunkCount)
    (callChunk : call.chunk = chunk)
    (schedule : call.ScheduleValid)
    (output coordinate : Nat) :
    call.block.linearValue assignment output coordinate =
      (semanticBlock call.mapKind).linearValue
        (semanticAssignment frame call.mapKind chunk)
        output coordinate := by
  unfold SeededPhi81.Block.linearValue
  rw [show call.block.messageCols =
      call.mapKind.messageColumnCount by rfl]
  rw [show (semanticBlock call.mapKind).messageCols =
      call.mapKind.messageColumnCount by rfl]
  apply congrArg (fun value => value % goldilocksP)
  apply foldl_eq_of_pointwise
  intro outer messageCol
  apply foldl_eq_of_pointwise
  intro inner messageRow
  unfold SeededPhi81.Block.termValue
  rw [call_coefficient_exact call schedule]
  rw [call_inputValue_exact frame assignment canonical one call satisfied
    linked chunk callChunk]

private theorem block_rows_satisfy
    (call : CoordinateCall) (assignment : Nat → Nat)
    (satisfied : call.Satisfied assignment) :
    Satisfies call.block.rows assignment := by
  intro row rowMember
  apply satisfied row
  apply List.mem_append_right
  exact List.mem_append_right _ rowMember

private theorem call_outputColumn_exact
    (call : CoordinateCall) (output : Fin outputWidth) :
    call.block.outputColumns.getD
        ((outputRow output).val * SeededPhi81.dimension +
          (outputCoordinate output).val) 0 =
      call.outputColumn output := by
  have indexEqual :
      (outputRow output).val * SeededPhi81.dimension +
          (outputCoordinate output).val = output.val := by
    simpa [outputRow, outputCoordinate, Nat.mul_comm] using
      Nat.div_add_mod output.val SeededPhi81.dimension
  have outputBound : output.val < 108 := by
    simpa [outputWidth, SeededPhi81.dimension,
      SeededPhi81Sampler.dimension] using output.isLt
  have indexBound :
      (outputRow output).val * SeededPhi81.dimension +
          (outputCoordinate output).val < 108 := by
    rw [indexEqual]
    exact outputBound
  change
    (List.ofFn call.outputColumn).getD
        ((outputRow output).val * SeededPhi81.dimension +
          (outputCoordinate output).val) 0 = _
  rw [getD_ofFn call.outputColumn _ 0 indexBound]
  congr 1
  apply Fin.ext
  exact indexEqual

/-- One exact physical call computes the authoritative contribution for its
map and selected claim chunk. -/
theorem output_eq_chunkContribution
    (frame : ClaimFrame) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (call : CoordinateCall)
    (satisfied : call.Satisfied assignment)
    (linked : CallFrameLinked frame assignment call)
    (chunk : Fin claimChunkCount)
    (callChunk : call.chunk = chunk)
    (schedule : call.ScheduleValid)
    (output : Fin outputWidth) :
    residueNat (assignment (call.outputColumn output)) =
      chunkContribution frame call.mapKind chunk output := by
  have blockHolds := SeededPhi81.sound canonical one
    (block_rows_satisfy call assignment satisfied)
  have outputValue := call.block.output_eq_linearValue blockHolds
    (outputRow output) (outputCoordinate output)
  rw [call_outputColumn_exact call output] at outputValue
  unfold chunkContribution
  rw [outputValue]
  exact congrArg residueNat
    (call_linearValue_exact frame assignment canonical one call satisfied
      linked chunk callChunk schedule (outputRow output).val
      (outputCoordinate output).val)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateCallRefinement
