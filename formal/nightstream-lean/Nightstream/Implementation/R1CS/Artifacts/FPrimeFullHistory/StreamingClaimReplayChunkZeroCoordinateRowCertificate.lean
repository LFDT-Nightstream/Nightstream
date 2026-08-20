import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingClaimReplayState
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataAccumulator

/-!
Contract: exact v6 coordinate-call and glue-row identity for claim chunk zero.

Assurance tier: structural Rust-to-Lean artifact certificate.

Owns the two active coordinate calls, all 324 initial zero rows, the two
108-row accumulator updates, and the 108-row inactive running-public carry.
Exact row checks are split into bounded pieces of at most 64 rows.

Does not own coordinate semantics, sampler liveness, source-frame authority,
Poseidon2 replay, other chunks, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayChunkZeroCoordinateRowCertificate

open Nightstream.Implementation.Nebula.ProductionStreamingClaimReplayState
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.AffinePins
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

def chunkZero : Fin claimChunkCount := ⟨0, by decide⟩

def statementFreshCall : CoordinateCall :=
  { mapKind := .statementFresh
    rowStart := 154086
    rowEnd := 160685
    chunkIndex := 0
    chunkBase := 821
    zeroDigitStart := 155445
    activeDigitBase := 155486
    dColumn := 161830
    kappaColumn := 161831
    outputBase := 161832
    seededRowStart := 160577
    chunkSize := statementFreshSchedule.chunkSize
    seedsByOutput := statementFreshSchedule.seedsByOutput }

def runningCommitmentsCall : CoordinateCall :=
  { mapKind := .runningCommitments
    rowStart := 160793
    rowEnd := 233980
    chunkIndex := 0
    chunkBase := 821
    zeroDigitStart := 161940
    activeDigitBase := 161981
    dColumn := 233839
    kappaColumn := 233840
    outputBase := 233841
    seededRowStart := 233872
    chunkSize := runningCommitmentsSchedule.chunkSize
    seedsByOutput := runningCommitmentsSchedule.seedsByOutput }

theorem fullArm_coordinateCalls_exact :
    fullArm.coordinateCalls =
      [statementFreshCall, runningCommitmentsCall] := by
  rfl

theorem statementFreshCall_member :
    statementFreshCall ∈ fullArm.coordinateCalls := by
  rw [fullArm_coordinateCalls_exact]
  simp

theorem runningCommitmentsCall_member :
    runningCommitmentsCall ∈ fullArm.coordinateCalls := by
  rw [fullArm_coordinateCalls_exact]
  simp

@[simp] theorem statementFreshCall_chunk :
    statementFreshCall.chunk = chunkZero := by
  rfl

@[simp] theorem runningCommitmentsCall_chunk :
    runningCommitmentsCall.chunk = chunkZero := by
  rfl

theorem statementFreshCall_schedule :
    statementFreshCall.ScheduleValid := by
  rfl

theorem runningCommitmentsCall_schedule :
    runningCommitmentsCall.ScheduleValid := by
  rfl

def glueRows : List Row := fullArm.glueRows.map IndexedRow.row

def beforeColumn (kind : MapKind) (output : Fin outputWidth) : Nat :=
  mapOffset kind + output.val

def afterColumn (kind : MapKind) (output : Fin outputWidth) : Nat :=
  410 + mapOffset kind + output.val

def initialPins : List AffinePins.Pin :=
  (AffinePins.Run.zero 20 1 324).pins

def initialRows : List Row := AffinePins.rows initialPins

def updateRow
    (kind : MapKind) (partialBase : Nat)
    (output : Fin outputWidth) : Row :=
  ⟨[(beforeColumn kind output, goldilocksP - 1),
      (afterColumn kind output, 1),
      (partialBase + output.val, goldilocksP - 1)],
    [(0, 1)], []⟩

def updateRows (kind : MapKind) (partialBase : Nat) : List Row :=
  List.ofFn (updateRow kind partialBase)

def runningPublicCarryRow (output : Fin outputWidth) : Row :=
  ⟨[(beforeColumn .runningPublic output, goldilocksP - 1),
      (afterColumn .runningPublic output, 1)], [(0, 1)], []⟩

def runningPublicCarryRows : List Row :=
  List.ofFn runningPublicCarryRow

private def initialSource : List Row := (glueRows.drop 16).take 324
private def initialTail0 : List Row := initialSource.drop 64
private def initialTail1 : List Row := initialTail0.drop 64
private def initialTail2 : List Row := initialTail1.drop 64
private def initialTail3 : List Row := initialTail2.drop 64
private def initialTail4 : List Row := initialTail3.drop 64

private def initialExpectedTail0 : List Row := initialRows.drop 64
private def initialExpectedTail1 : List Row := initialExpectedTail0.drop 64
private def initialExpectedTail2 : List Row := initialExpectedTail1.drop 64
private def initialExpectedTail3 : List Row := initialExpectedTail2.drop 64
private def initialExpectedTail4 : List Row := initialExpectedTail3.drop 64

private theorem initialChunk0_exact :
    initialSource.take 64 = initialRows.take 64 := by rfl
private theorem initialChunk1_exact :
    initialTail0.take 64 = initialExpectedTail0.take 64 := by rfl
private theorem initialChunk2_exact :
    initialTail1.take 64 = initialExpectedTail1.take 64 := by rfl
private theorem initialChunk3_exact :
    initialTail2.take 64 = initialExpectedTail2.take 64 := by rfl
private theorem initialChunk4_exact :
    initialTail3.take 64 = initialExpectedTail3.take 64 := by rfl
private theorem initialTail_exact :
    initialTail4 = initialExpectedTail4 := by rfl

private theorem eq_of_five_chunks_and_tail
    {alpha : Type} (left right : List alpha)
    (chunk0 : left.take 64 = right.take 64)
    (chunk1 : (left.drop 64).take 64 = (right.drop 64).take 64)
    (chunk2 : (left.drop 128).take 64 = (right.drop 128).take 64)
    (chunk3 : (left.drop 192).take 64 = (right.drop 192).take 64)
    (chunk4 : (left.drop 256).take 64 = (right.drop 256).take 64)
    (tail : left.drop 320 = right.drop 320) : left = right := by
  rw [← List.take_append_drop 64 left, ← List.take_append_drop 64 right,
    chunk0]
  congr 1
  rw [← List.take_append_drop 64 (left.drop 64),
    ← List.take_append_drop 64 (right.drop 64), chunk1]
  congr 1
  rw [List.drop_drop, List.drop_drop]
  rw [← List.take_append_drop 64 (left.drop 128),
    ← List.take_append_drop 64 (right.drop 128), chunk2]
  congr 1
  rw [List.drop_drop, List.drop_drop]
  rw [← List.take_append_drop 64 (left.drop 192),
    ← List.take_append_drop 64 (right.drop 192), chunk3]
  congr 1
  rw [List.drop_drop, List.drop_drop]
  rw [← List.take_append_drop 64 (left.drop 256),
    ← List.take_append_drop 64 (right.drop 256), chunk4]
  congr 1
  simpa [List.drop_drop, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]
    using tail

theorem initialRows_exact :
    (glueRows.drop 16).take 324 = initialRows := by
  apply eq_of_five_chunks_and_tail
  · exact initialChunk0_exact
  · exact initialChunk1_exact
  · simpa [initialTail0, initialTail1, initialSource,
      initialExpectedTail0, initialExpectedTail1, List.drop_drop,
      Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using initialChunk2_exact
  · simpa [initialTail0, initialTail1, initialTail2, initialSource,
      initialExpectedTail0, initialExpectedTail1, initialExpectedTail2,
      List.drop_drop, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
      initialChunk3_exact
  · simpa [initialTail0, initialTail1, initialTail2, initialTail3,
      initialSource, initialExpectedTail0, initialExpectedTail1,
      initialExpectedTail2, initialExpectedTail3, List.drop_drop,
      Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using initialChunk4_exact
  · simpa [initialTail0, initialTail1, initialTail2, initialTail3,
      initialTail4, initialSource, initialExpectedTail0,
      initialExpectedTail1, initialExpectedTail2, initialExpectedTail3,
      initialExpectedTail4, List.drop_drop, Nat.add_comm,
      Nat.add_left_comm, Nat.add_assoc] using initialTail_exact

private theorem eq_of_head64_and_tail
    {alpha : Type} (left right : List alpha)
    (head : left.take 64 = right.take 64)
    (tail : left.drop 64 = right.drop 64) : left = right := by
  rw [← List.take_append_drop 64 left, ← List.take_append_drop 64 right,
    head, tail]

private theorem statementUpdateHead_exact :
    ((glueRows.drop 348).take 108).take 64 =
      (updateRows .statementFresh 161832).take 64 := by rfl

private theorem statementUpdateTail_exact :
    ((glueRows.drop 348).take 108).drop 64 =
      (updateRows .statementFresh 161832).drop 64 := by rfl

theorem statementUpdateRows_exact :
    (glueRows.drop 348).take 108 =
      updateRows .statementFresh 161832 := by
  exact eq_of_head64_and_tail _ _ statementUpdateHead_exact
    statementUpdateTail_exact

private theorem runningCommitmentsUpdateHead_exact :
    ((glueRows.drop 456).take 108).take 64 =
      (updateRows .runningCommitments 233841).take 64 := by rfl

private theorem runningCommitmentsUpdateTail_exact :
    ((glueRows.drop 456).take 108).drop 64 =
      (updateRows .runningCommitments 233841).drop 64 := by rfl

theorem runningCommitmentsUpdateRows_exact :
    (glueRows.drop 456).take 108 =
      updateRows .runningCommitments 233841 := by
  exact eq_of_head64_and_tail _ _ runningCommitmentsUpdateHead_exact
    runningCommitmentsUpdateTail_exact

private theorem runningPublicCarryHead_exact :
    ((glueRows.drop 564).take 108).take 64 =
      runningPublicCarryRows.take 64 := by rfl

private theorem runningPublicCarryTail_exact :
    ((glueRows.drop 564).take 108).drop 64 =
      runningPublicCarryRows.drop 64 := by rfl

theorem runningPublicCarryRows_exact :
    (glueRows.drop 564).take 108 = runningPublicCarryRows := by
  exact eq_of_head64_and_tail _ _ runningPublicCarryHead_exact
    runningPublicCarryTail_exact

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayChunkZeroCoordinateRowCertificate
