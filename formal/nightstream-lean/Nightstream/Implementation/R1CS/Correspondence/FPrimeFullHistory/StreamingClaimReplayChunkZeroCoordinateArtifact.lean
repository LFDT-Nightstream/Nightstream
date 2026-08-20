import Nightstream.Implementation.Nebula.Commitment.Lanes.ShiftedTernaryEncodingBridge
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingClaimReplayTransition
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayChunkZeroCoordinateRowCertificate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayExpectedCarryArtifact
import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.ShiftedTernarySound

/-!
Contract: exact v6 generated-row refinement for the three-map coordinate
accumulator in claim chunk zero.

Assurance tier: Rust-conformant phase refinement.

Owns the physical state-column decoder, initial zero state, two active map
updates, inactive running-public carry, and the explicit source-column link
to the verifier-owned claim frame.

Does not own source-frame replay, sampler liveness, other claim chunks,
Poseidon2 execution, lifecycle selection, or Module-SIS hardness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayChunkZeroCoordinateArtifact

open Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayState
open Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayTransition
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.AffinePins
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayChunkZeroCoordinateRowCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExpectedCarryArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateWordLayoutCertificate
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.SeededPhi81RingRefinement
open Nightstream.Implementation.R1CS.ShiftedTernaryCanonicalWord
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.SuperNeo.Concrete
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

def fullGlueRows : List Row := fullArm.glueRows.map IndexedRow.row

private theorem sliced_glue_rows_satisfy
    (assignment : Nat → Nat) (satisfied : fullArm.Satisfied assignment)
    (start count : Nat) :
    Satisfies ((fullGlueRows.drop start).take count) assignment := by
  intro row member
  rcases List.mem_map.mp
      (List.mem_of_mem_drop (List.mem_of_mem_take member)) with
    ⟨indexed, indexedMember, rfl⟩
  exact glue_row_holds fullArm assignment satisfied indexed indexedMember

private theorem structuralColumn_before_middle
    (index : Fin transitionWordCount)
    (lower : 20 ≤ index.val) (upper : index.val < 344) :
    structuralColumn index = index.val := by
  unfold structuralColumn transitionStateWordColumns
  rw [List.getElem_append_left (by
    rw [rotatedRange_length]
    omega)]
  unfold rotatedRange
  rw [List.getElem_append_right (by simp; omega)]
  simp only [List.length_range']
  have offset : index.val - 19 = (index.val - 20) + 1 := by omega
  simp [offset, List.getElem_range']
  omega

private theorem structuralColumn_after_middle
    (index : Fin transitionWordCount)
    (lower : 364 ≤ index.val) (upper : index.val < 688) :
    structuralColumn index = 66 + index.val := by
  unfold structuralColumn transitionStateWordColumns
  rw [List.getElem_append_right (by
    rw [rotatedRange_length]
    omega)]
  simp only [rotatedRange_length]
  unfold rotatedRange
  rw [List.getElem_append_right (by
    simp only [List.length_range']
    omega)]
  simp only [List.length_range']
  have innerOffset : index.val - 344 - 19 =
      (index.val - 364) + 1 := by omega
  simp [innerOffset, List.getElem_range']
  omega

@[simp] theorem transitionColumn_before_coordinate
    (kind : MapKind) (output : Fin outputWidth) :
    transitionColumn .full
        (transitionIndex .before (coordinateIndex kind output)) =
      mapOffset kind + output.val := by
  rw [transitionColumn_eq_structural]
  have outputBound : output.val < 108 := by
    simpa [outputWidth, SeededPhi81.dimension,
      SeededPhi81Sampler.dimension] using output.isLt
  have lower : 20 ≤
      (transitionIndex .before (coordinateIndex kind output)).val := by
    cases kind <;>
      simp only [transitionIndex, sideOffset, coordinateIndex, mapOffset,
        coordinateOffset, frameCursorOffset, cursorWordCount,
        spongeStateWordCount, spongeWidth] <;>
      omega
  have upper :
      (transitionIndex .before (coordinateIndex kind output)).val < 344 := by
    cases kind <;>
      simp only [transitionIndex, sideOffset, coordinateIndex, mapOffset,
        coordinateOffset, frameCursorOffset, cursorWordCount,
        spongeStateWordCount, spongeWidth, outputWidth,
        SeededPhi81.dimension, SeededPhi81Sampler.dimension] <;>
      omega
  calc
    structuralColumn
        (transitionIndex .before (coordinateIndex kind output)) =
        (transitionIndex .before (coordinateIndex kind output)).val :=
      structuralColumn_before_middle _ lower upper
    _ = mapOffset kind + output.val := by
      simp [transitionIndex, sideOffset, coordinateIndex]

@[simp] theorem transitionColumn_after_coordinate
    (kind : MapKind) (output : Fin outputWidth) :
    transitionColumn .full
        (transitionIndex .after (coordinateIndex kind output)) =
      410 + mapOffset kind + output.val := by
  rw [transitionColumn_eq_structural]
  have outputBound : output.val < 108 := by
    simpa [outputWidth, SeededPhi81.dimension,
      SeededPhi81Sampler.dimension] using output.isLt
  have lower : 364 ≤
      (transitionIndex .after (coordinateIndex kind output)).val := by
    cases kind <;>
      simp only [transitionIndex, sideOffset, stateWordCount,
        coordinateMapCount, coordinateIndex, mapOffset, coordinateOffset,
        frameCursorOffset, cursorWordCount, spongeStateWordCount,
        spongeWidth, outputWidth, SeededPhi81.dimension,
        SeededPhi81Sampler.dimension] <;>
      omega
  have upper :
      (transitionIndex .after (coordinateIndex kind output)).val < 688 := by
    have := (transitionIndex .after (coordinateIndex kind output)).isLt
    have count := exact_word_counts.2.2
    omega
  calc
    structuralColumn
        (transitionIndex .after (coordinateIndex kind output)) =
        66 + (transitionIndex .after (coordinateIndex kind output)).val :=
      structuralColumn_after_middle _ lower upper
    _ = 410 + mapOffset kind + output.val := by
      cases kind <;>
        simp [transitionIndex, sideOffset, coordinateIndex, stateWordCount,
          coordinateMapCount, mapOffset, coordinateOffset,
          frameCursorOffset, cursorWordCount, spongeStateWordCount,
          spongeWidth, outputWidth, SeededPhi81.dimension,
          SeededPhi81Sampler.dimension] <;>
        omega

private theorem initial_rows_satisfy
    (assignment : Nat → Nat) (satisfied : fullArm.Satisfied assignment) :
    Satisfies initialRows assignment := by
  rw [← initialRows_exact]
  exact sliced_glue_rows_satisfy assignment satisfied 16 324

private theorem initialPins_canonical :
    AffinePins.PinsCanonical initialPins := by
  unfold initialPins AffinePins.Run.pins
  intro pin member
  rcases List.mem_map.mp member with ⟨offset, _, rfl⟩
  trivial

private theorem initialPin_member
    (kind : MapKind) (output : Fin outputWidth) :
    .zero (beforeColumn kind output) ∈ initialPins := by
  have outputBound : output.val < 108 := by
    simpa [outputWidth, SeededPhi81.dimension,
      SeededPhi81Sampler.dimension] using output.isLt
  unfold initialPins AffinePins.Run.pins
  cases kind with
  | statementFresh =>
      refine List.mem_map.mpr ⟨output.val, List.mem_range.mpr (by omega), ?_⟩
      simp [beforeColumn, mapOffset, coordinateOffset, frameCursorOffset,
        cursorWordCount, spongeStateWordCount, spongeWidth]
  | runningCommitments =>
      refine List.mem_map.mpr
        ⟨108 + output.val, List.mem_range.mpr (by omega), ?_⟩
      simp [beforeColumn, mapOffset, coordinateOffset, frameCursorOffset,
        cursorWordCount, spongeStateWordCount, spongeWidth, outputWidth,
        SeededPhi81.dimension, SeededPhi81Sampler.dimension]
      omega
  | runningPublic =>
      refine List.mem_map.mpr
        ⟨216 + output.val, List.mem_range.mpr (by omega), ?_⟩
      simp [beforeColumn, mapOffset, coordinateOffset, frameCursorOffset,
        cursorWordCount, spongeStateWordCount, spongeWidth, outputWidth,
        SeededPhi81.dimension, SeededPhi81Sampler.dimension]
      omega

theorem before_coordinate_zero
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ∀ kind output, assignment (beforeColumn kind output) = 0 := by
  have facts := AffinePins.rows_sound initialPins_canonical canonical one
    (initial_rows_satisfy assignment satisfied)
  intro kind output
  exact facts (.zero (beforeColumn kind output))
    (initialPin_member kind output)

private theorem rowHolds_of_operand_perms
    (assignment : Nat → Nat) {source target : Row}
    (a : source.a.Perm target.a)
    (b : source.b.Perm target.b)
    (c : source.c.Perm target.c)
    (holds : RowHolds assignment source) :
    RowHolds assignment target := by
  unfold RowHolds at holds ⊢
  calc
    lcEval assignment target.a * lcEval assignment target.b % goldilocksP =
        lcEval assignment source.a * lcEval assignment source.b %
          goldilocksP := by
      rw [Program.lcEval_eq_of_perm assignment a,
        Program.lcEval_eq_of_perm assignment b]
    _ = lcEval assignment source.c := holds
    _ = lcEval assignment target.c :=
      Program.lcEval_eq_of_perm assignment c

private theorem updateRow_perms
    (kind : MapKind) (partialBase : Nat) (output : Fin outputWidth) :
    (updateRow kind partialBase output).a.Perm
        (builderLinearRow (afterColumn kind output)
          [(beforeColumn kind output, 1),
            (partialBase + output.val, 1)]).a ∧
      (updateRow kind partialBase output).b.Perm
        (builderLinearRow (afterColumn kind output)
          [(beforeColumn kind output, 1),
            (partialBase + output.val, 1)]).b ∧
      (updateRow kind partialBase output).c.Perm
        (builderLinearRow (afterColumn kind output)
          [(beforeColumn kind output, 1),
            (partialBase + output.val, 1)]).c := by
  exact ⟨List.Perm.swap _ _ _, List.Perm.refl _, List.Perm.refl _⟩

private theorem update_rows_satisfy
    (assignment : Nat → Nat) (satisfied : fullArm.Satisfied assignment)
    (kind : MapKind) (partialBase start : Nat)
    (exactRows : (fullGlueRows.drop start).take 108 =
      updateRows kind partialBase) :
    Satisfies (updateRows kind partialBase) assignment := by
  rw [← exactRows]
  exact sliced_glue_rows_satisfy assignment satisfied start 108

private theorem update_facts
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (kind : MapKind) (partialBase : Nat)
    (satisfies : Satisfies (updateRows kind partialBase) assignment) :
    ∀ output : Fin outputWidth,
      assignment (afterColumn kind output) =
        (assignment (beforeColumn kind output) +
          assignment (partialBase + output.val)) % goldilocksP := by
  intro output
  have emitted := satisfies (updateRow kind partialBase output)
    (List.mem_ofFn.mpr ⟨output, rfl⟩)
  have builderHolds := rowHolds_of_operand_perms assignment
    (updateRow_perms kind partialBase output).1
    (updateRow_perms kind partialBase output).2.1
    (updateRow_perms kind partialBase output).2.2 emitted
  have defined := builderLinearRow_sound canonical one
    (afterColumn kind output)
    [(beforeColumn kind output, 1), (partialBase + output.val, 1)]
    (by simp [CanonicalTerms]; decide) builderHolds
  simpa [lcEval, Nat.add_comm, Nat.mul_comm] using defined

theorem statementFresh_update
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ∀ output : Fin outputWidth,
      assignment (afterColumn .statementFresh output) =
        (assignment (beforeColumn .statementFresh output) +
          assignment (statementFreshCall.outputColumn output)) % goldilocksP := by
  exact update_facts assignment canonical one .statementFresh 161832
    (update_rows_satisfy assignment satisfied .statementFresh 161832 348
      statementUpdateRows_exact)

theorem runningCommitments_update
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ∀ output : Fin outputWidth,
      assignment (afterColumn .runningCommitments output) =
        (assignment (beforeColumn .runningCommitments output) +
          assignment (runningCommitmentsCall.outputColumn output)) %
            goldilocksP := by
  exact update_facts assignment canonical one .runningCommitments 233841
    (update_rows_satisfy assignment satisfied .runningCommitments 233841 456
      runningCommitmentsUpdateRows_exact)

private theorem carryRow_perms (output : Fin outputWidth) :
    (runningPublicCarryRow output).a.Perm
        (builderLinearRow (afterColumn .runningPublic output)
          [(beforeColumn .runningPublic output, 1)]).a ∧
      (runningPublicCarryRow output).b.Perm
        (builderLinearRow (afterColumn .runningPublic output)
          [(beforeColumn .runningPublic output, 1)]).b ∧
      (runningPublicCarryRow output).c.Perm
        (builderLinearRow (afterColumn .runningPublic output)
          [(beforeColumn .runningPublic output, 1)]).c := by
  exact ⟨List.Perm.swap _ _ _, List.Perm.refl _, List.Perm.refl _⟩

theorem runningPublic_carry
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    ∀ output : Fin outputWidth,
      assignment (afterColumn .runningPublic output) =
        assignment (beforeColumn .runningPublic output) := by
  have rowsSatisfy : Satisfies runningPublicCarryRows assignment := by
    rw [← runningPublicCarryRows_exact]
    exact sliced_glue_rows_satisfy assignment satisfied 564 108
  intro output
  have emitted := rowsSatisfy (runningPublicCarryRow output)
    (List.mem_ofFn.mpr ⟨output, rfl⟩)
  have builderHolds := rowHolds_of_operand_perms assignment
    (carryRow_perms output).1 (carryRow_perms output).2.1
    (carryRow_perms output).2.2 emitted
  have defined := builderLinearRow_sound canonical one
    (afterColumn .runningPublic output)
    [(beforeColumn .runningPublic output, 1)]
    (by simp [CanonicalTerms]; decide) builderHolds
  simpa [lcEval, Nat.mod_eq_of_lt (canonical _)] using defined

/-- Explicit trust-boundary premise for the physical chunk-zero columns.
The Poseidon2 replay bridge must discharge this premise from the same
verifier-owned frame. -/
def FrameLinked
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (assignment : Nat → Nat) : Prop :=
  ∀ offset : Fin (claimChunkFieldCount chunkZero),
    assignment (statementFreshCall.chunkBase + offset.val) =
      (frame (chunkFrameIndex chunkZero offset)).val

def sourceOffset
    (kind : MapKind) (field : Fin kind.fieldCount) :
    Fin (claimChunkFieldCount chunkZero) :=
  ⟨(kind.claimChunkOffset field).val, by
    change (kind.claimChunkOffset field).val < 1024
    exact (kind.claimChunkOffset field).isLt⟩

private theorem sourceOffset_frameIndex
    (kind : MapKind) (field : Fin kind.fieldCount)
    (selected : kind.claimChunk field = chunkZero) :
    chunkFrameIndex chunkZero (sourceOffset kind field) =
      ⟨kind.framePosition field, kind.framePosition_lt field⟩ := by
  apply Fin.ext
  have recompose := kind.framePosition_recompose field
  rw [selected] at recompose
  simpa [chunkFrameIndex, sourceOffset, chunkZero] using recompose

private theorem coordinate_rows_satisfy
    (call : CoordinateCall) (member : call ∈ fullArm.coordinateCalls)
    (assignment : Nat → Nat) (satisfied : fullArm.Satisfied assignment) :
    Satisfies call.rows assignment :=
  coordinate_call_holds fullArm assignment satisfied call member

private theorem opening_block_rows_satisfy
    (call : CoordinateCall) (member : call ∈ fullArm.coordinateCalls)
    (assignment : Nat → Nat) (satisfied : fullArm.Satisfied assignment)
    (field : Fin call.mapKind.fieldCount)
    (active : field ∈ call.activeFields) :
    Satisfies (call.openingBlockRows field) assignment := by
  have callRows := coordinate_rows_satisfy call member assignment satisfied
  intro row rowMember
  apply callRows row
  apply List.mem_append_left
  apply List.mem_append_right
  exact List.mem_flatMap.mpr ⟨field, active, rowMember⟩

private theorem active_opening
    (call : CoordinateCall) (member : call ∈ fullArm.coordinateCalls)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment)
    (field : Fin call.mapKind.fieldCount)
    (active : field ∈ call.activeFields) :
    Nightstream.Implementation.R1CS.ShiftedTernaryCompiler.CanonicalOpening
      (localAssignment assignment (call.fieldColumn field)
        (call.digitStart field)) := by
  have mapped := opening_block_rows_satisfy call member assignment satisfied
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
    (call : CoordinateCall) (member : call ∈ fullArm.coordinateCalls)
    (assignment : Nat → Nat) (satisfied : fullArm.Satisfied assignment) :
    Satisfies call.zeroRows assignment := by
  have callRows := coordinate_rows_satisfy call member assignment satisfied
  intro row rowMember
  apply callRows row
  apply List.mem_append_left
  exact List.mem_append_left _ rowMember

private theorem zeroPins_canonical (call : CoordinateCall) :
    AffinePins.PinsCanonical call.zeroPins := by
  intro pin pinMember
  rcases List.mem_ofFn.mp pinMember with ⟨digit, rfl⟩
  trivial

private theorem inactive_digit_zero
    (call : CoordinateCall) (member : call ∈ fullArm.coordinateCalls)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment)
    (field : Fin call.mapKind.fieldCount)
    (inactive : field ∉ call.activeFields)
    (digit : Fin Nightstream.Protocol.Nebula.ShiftedTernary41V1.digitCount) :
    assignment (call.wordStart field + digit.val) = 0 := by
  have facts := AffinePins.rows_sound (zeroPins_canonical call) canonical one
    (zero_rows_satisfy call member assignment satisfied)
  have exact := facts (.zero (call.zeroDigitStart + digit.val))
    (List.mem_ofFn.mpr ⟨digit, rfl⟩)
  simpa [CoordinateCall.wordStart, inactive] using exact

private theorem active_source_exact
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (assignment : Nat → Nat)
    (linked : FrameLinked frame assignment)
    (call : CoordinateCall) (chunkBase : call.chunkBase = 821)
    (callChunk : call.chunk = chunkZero)
    (field : Fin call.mapKind.fieldCount)
    (active : field ∈ call.activeFields) :
    assignment (call.fieldColumn field) =
      (frame ⟨call.mapKind.framePosition field,
        call.mapKind.framePosition_lt field⟩).val := by
  have selected : call.mapKind.claimChunk field = chunkZero := by
    have exact := (call.mapKind.mem_activeFields call.chunk field).mp active
    simpa [callChunk] using exact
  have source := linked (sourceOffset call.mapKind field)
  rw [sourceOffset_frameIndex call.mapKind field selected] at source
  simpa [CoordinateCall.fieldColumn, sourceOffset, chunkBase] using source

private theorem active_digit_exact
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment)
    (linked : FrameLinked frame assignment)
    (call : CoordinateCall) (member : call ∈ fullArm.coordinateCalls)
    (chunkBase : call.chunkBase = 821)
    (callChunk : call.chunk = chunkZero)
    (field : Fin call.mapKind.fieldCount)
    (active : field ∈ call.activeFields)
    (digit : Fin Nightstream.Protocol.Nebula.ShiftedTernary41V1.digitCount) :
    assignment (call.wordStart field + digit.val) =
      selectedDigit frame call.mapKind chunkZero field digit := by
  have selected : call.mapKind.claimChunk field = chunkZero := by
    have exact := (call.mapKind.mem_activeFields call.chunk field).mp active
    simpa [callChunk] using exact
  have source := active_source_exact frame assignment linked call chunkBase
    callChunk field active
  have opening := active_opening call member assignment canonical one satisfied
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
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment)
    (linked : FrameLinked frame assignment)
    (call : CoordinateCall) (member : call ∈ fullArm.coordinateCalls)
    (chunkBase : call.chunkBase = 821)
    (callChunk : call.chunk = chunkZero)
    (field : Fin call.mapKind.fieldCount) (digit : Fin digitCount) :
    assignment (call.wordStart field + digit.val) =
      selectedDigit frame call.mapKind chunkZero field digit := by
  by_cases active : field ∈ call.activeFields
  · exact active_digit_exact frame assignment canonical one satisfied linked
      call member chunkBase callChunk field active digit
  · have notSelected :
        call.mapKind.claimChunk field ≠ chunkZero := by
      intro selected
      apply active
      apply (call.mapKind.mem_activeFields call.chunk field).2
      simpa [callChunk] using selected
    calc
      assignment (call.wordStart field + digit.val) = 0 :=
        inactive_digit_zero call member assignment canonical one satisfied
          field active digit
      _ = selectedDigit frame call.mapKind chunkZero field digit := by
        simp [selectedDigit, notSelected]

private theorem call_inputValue_exact
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment)
    (linked : FrameLinked frame assignment)
    (call : CoordinateCall) (member : call ∈ fullArm.coordinateCalls)
    (chunkBase : call.chunkBase = 821)
    (callChunk : call.chunk = chunkZero)
    (messageCol messageRow : Nat) :
    call.block.inputValue assignment messageCol messageRow =
      (semanticBlock call.mapKind).inputValue
        (semanticAssignment frame call.mapKind chunkZero)
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
        semanticAssignment frame call.mapKind chunkZero index =
          selectedDigit frame call.mapKind chunkZero field digit := by
      unfold semanticAssignment
      rw [dif_pos valid]
    rw [SeededPhi81.Block.inputValue_eq_of_bitColumn_some
      physicalBitColumn]
    rw [word_digit_exact frame assignment canonical one satisfied linked
      call member chunkBase callChunk field digit]
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
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment)
    (linked : FrameLinked frame assignment)
    (call : CoordinateCall) (member : call ∈ fullArm.coordinateCalls)
    (chunkBase : call.chunkBase = 821)
    (callChunk : call.chunk = chunkZero)
    (schedule : call.ScheduleValid)
    (output coordinate : Nat) :
    call.block.linearValue assignment output coordinate =
      (semanticBlock call.mapKind).linearValue
        (semanticAssignment frame call.mapKind chunkZero)
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
  rw [call_inputValue_exact frame assignment canonical one satisfied linked
    call member chunkBase callChunk]

private theorem block_rows_satisfy
    (call : CoordinateCall) (member : call ∈ fullArm.coordinateCalls)
    (assignment : Nat → Nat) (satisfied : fullArm.Satisfied assignment) :
    Satisfies call.block.rows assignment := by
  have callRows := coordinate_rows_satisfy call member assignment satisfied
  intro row rowMember
  apply callRows row
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

private theorem call_output_eq_chunkContribution
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment)
    (linked : FrameLinked frame assignment)
    (call : CoordinateCall) (member : call ∈ fullArm.coordinateCalls)
    (chunkBase : call.chunkBase = 821)
    (callChunk : call.chunk = chunkZero)
    (schedule : call.ScheduleValid)
    (output : Fin outputWidth) :
    residueNat (assignment (call.outputColumn output)) =
      chunkContribution frame call.mapKind chunkZero output := by
  have blockHolds := SeededPhi81.sound canonical one
    (block_rows_satisfy call member assignment satisfied)
  have outputValue := call.block.output_eq_linearValue blockHolds
    (outputRow output) (outputCoordinate output)
  rw [call_outputColumn_exact call output] at outputValue
  unfold chunkContribution
  rw [outputValue]
  exact congrArg residueNat
    (call_linearValue_exact frame assignment canonical one satisfied linked
      call member chunkBase callChunk schedule
      (outputRow output).val (outputCoordinate output).val)

private theorem runningPublic_not_chunkZero
    (field : Fin MapKind.runningPublic.fieldCount) :
    MapKind.runningPublic.claimChunk field ≠ chunkZero := by
  intro equal
  have values := congrArg Fin.val equal
  change (62_643 + field.val) / 1_024 = 0 at values
  omega

private theorem runningPublic_semanticAssignment_zero
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (column : Nat) :
    semanticAssignment frame .runningPublic chunkZero column = 0 := by
  unfold semanticAssignment
  split
  · simp [selectedDigit, runningPublic_not_chunkZero]
  · rfl

private theorem runningPublic_inputValue_zero
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (messageCol messageRow : Nat) :
    (semanticBlock .runningPublic).inputValue
        (semanticAssignment frame .runningPublic chunkZero)
        messageCol messageRow = 0 := by
  cases selected :
      (semanticBlock .runningPublic).bitColumn
        (messageRow * (semanticBlock .runningPublic).messageCols +
          messageCol) with
  | none =>
      exact SeededPhi81.Block.inputValue_eq_zero_of_bitColumn_none selected
  | some column =>
      rw [SeededPhi81.Block.inputValue_eq_of_bitColumn_some selected]
      exact runningPublic_semanticAssignment_zero frame column

private theorem foldl_identity
    {alpha beta : Type} (values : List alpha) (step : beta → alpha → beta)
    (same : ∀ accumulated value, step accumulated value = accumulated)
    (initial : beta) :
    values.foldl step initial = initial := by
  induction values generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [same]
      exact inductionHypothesis _

private theorem runningPublic_linearValue_zero
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (output coordinate : Nat) :
    (semanticBlock .runningPublic).linearValue
        (semanticAssignment frame .runningPublic chunkZero)
        output coordinate = 0 := by
  unfold SeededPhi81.Block.linearValue
  have folded :
      (List.range (semanticBlock .runningPublic).messageCols).foldl
          (fun outer messageCol =>
            (List.range SeededPhi81.dimension).foldl
              (fun inner messageRow =>
                inner +
                  (semanticBlock .runningPublic).termValue
                    (semanticAssignment frame .runningPublic chunkZero)
                    output coordinate messageCol messageRow)
              outer)
          0 = 0 := by
    apply foldl_identity
    intro outer messageCol
    apply foldl_identity
    intro inner messageRow
    unfold SeededPhi81.Block.termValue
    rw [runningPublic_inputValue_zero]
    simp only [Nat.mul_zero, Nat.add_zero]
  rw [folded]
  rfl

theorem runningPublic_chunkContribution_zero
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (output : Fin outputWidth) :
    chunkContribution frame .runningPublic chunkZero output = 0 := by
  unfold chunkContribution
  rw [runningPublic_linearValue_zero]
  rfl

/-- The exact 324 generated initial rows fix all three coordinate maps to the
authoritative zero state. -/
theorem generated_rows_imply_initial_coordinates_zero
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment) :
    (decodedTransition .full assignment).before.coordinates = zeroState := by
  funext kind output
  change residueNat
      (assignment (transitionColumn .full
        (transitionIndex .before (coordinateIndex kind output)))) = 0
  rw [transitionColumn_before_coordinate]
  change residueNat (assignment (beforeColumn kind output)) = 0
  rw [before_coordinate_zero assignment canonical one satisfied kind output]
  rfl

/-- Exact v6 chunk-zero coordinate-row refinement. The physical source link
is explicit because the separate Poseidon2 replay bridge must derive it from
the same verifier-owned frame. -/
theorem generated_rows_and_frame_link_imply_step
    (frame :
      Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator.ClaimFrame)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArm.Satisfied assignment)
    (linked : FrameLinked frame assignment) :
    Step frame chunkZero
      (decodedTransition .full assignment).before.coordinates
      (decodedTransition .full assignment).after.coordinates := by
  intro kind output
  change residueNat
      (assignment (transitionColumn .full
        (transitionIndex .after (coordinateIndex kind output)))) =
    residueNat
        (assignment (transitionColumn .full
          (transitionIndex .before (coordinateIndex kind output)))) +
      chunkContribution frame kind chunkZero output
  rw [transitionColumn_before_coordinate, transitionColumn_after_coordinate]
  change residueNat (assignment (afterColumn kind output)) =
    residueNat (assignment (beforeColumn kind output)) +
      chunkContribution frame kind chunkZero output
  cases kind with
  | statementFresh =>
      have update := congrArg residueNat
        (statementFresh_update assignment canonical one satisfied output)
      rw [residueNat_mod, residueNat_add] at update
      have contribution := call_output_eq_chunkContribution frame assignment
        canonical one satisfied linked statementFreshCall
        statementFreshCall_member rfl statementFreshCall_chunk
        statementFreshCall_schedule output
      rw [contribution] at update
      exact update
  | runningCommitments =>
      have update := congrArg residueNat
        (runningCommitments_update assignment canonical one satisfied output)
      rw [residueNat_mod, residueNat_add] at update
      have contribution := call_output_eq_chunkContribution frame assignment
        canonical one satisfied linked runningCommitmentsCall
        runningCommitmentsCall_member rfl runningCommitmentsCall_chunk
        runningCommitmentsCall_schedule output
      rw [contribution] at update
      exact update
  | runningPublic =>
      have carry := congrArg residueNat
        (runningPublic_carry assignment canonical one satisfied output)
      rw [runningPublic_chunkContribution_zero]
      simpa only [Fin.add_zero] using carry

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayChunkZeroCoordinateArtifact
