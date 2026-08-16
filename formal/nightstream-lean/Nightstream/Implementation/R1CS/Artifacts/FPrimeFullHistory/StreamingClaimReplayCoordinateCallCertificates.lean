import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataActiveFieldCertificate
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataCoordinateMapCertificates
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayRunningMetadataScheduleCertificate
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplayStatementFreshScheduleCertificate

/-!
Contract: structural validity certificates for the three Rust-emitted
streaming claim-replay coordinate calls.

Assurance tier: Rust-to-Lean artifact geometry certificate.

Owns exact call identity, active-field counts, physical geometry, exact seed
schedule identity, and transfer of the two verifier-owned sampler certificates.

Does not own claim semantics, complete arm validity, or lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateCallCertificates

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataActiveFieldCertificate
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMapCertificates
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingClaimReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

def fullStatementFreshCall : CoordinateCall :=
  { mapKind := .statementFresh
    rowStart := 153978
    rowEnd := 160577
    chunkIndex := 0
    chunkBase := 605
    zeroDigitStart := 155229
    activeDigitBase := 155270
    dColumn := 161614
    kappaColumn := 161615
    outputBase := 161616
    seededRowStart := 160469
    chunkSize := statementFreshSchedule.chunkSize
    seedsByOutput := statementFreshSchedule.seedsByOutput }

def fullRunningMetadataCall : CoordinateCall :=
  { mapKind := .runningMetadata
    rowStart := 160685
    rowEnd := 233872
    chunkIndex := 0
    chunkBase := 605
    zeroDigitStart := 161724
    activeDigitBase := 161765
    dColumn := 233623
    kappaColumn := 233624
    outputBase := 233625
    seededRowStart := 233764
    chunkSize := runningMetadataSchedule.chunkSize
    seedsByOutput := runningMetadataSchedule.seedsByOutput }

def finalStatementFreshCall : CoordinateCall :=
  { mapKind := .statementFresh
    rowStart := 147215
    rowEnd := 269258
    chunkIndex := 85
    chunkBase := 605
    zeroDigitStart := 148629
    activeDigitBase := 148670
    dColumn := 268596
    kappaColumn := 268597
    outputBase := 268598
    seededRowStart := 269150
    chunkSize := statementFreshSchedule.chunkSize
    seedsByOutput := statementFreshSchedule.seedsByOutput }

theorem fullArm_coordinateCalls_exact :
    fullArm.coordinateCalls =
      [fullStatementFreshCall, fullRunningMetadataCall] := by
  rfl

theorem finalArm_coordinateCalls_exact :
    finalArm.coordinateCalls = [finalStatementFreshCall] := by
  rfl

theorem fullStatementFreshCall_activeFields_length :
    fullStatementFreshCall.activeFields.length = 52 := by
  change (MapKind.statementFresh.activeFields firstChunk).length = 52
  exact statementFresh_firstChunk_length

theorem fullRunningMetadataCall_activeFields_length :
    fullRunningMetadataCall.activeFields.length = 589 := by
  change (MapKind.runningMetadata.activeFields firstChunk).length = 589
  exact runningMetadata_firstChunk_length

theorem finalStatementFreshCall_activeFields_length :
    finalStatementFreshCall.activeFields.length = 983 := by
  change (MapKind.statementFresh.activeFields finalChunk).length = 983
  exact statementFresh_finalChunk_length

theorem fullStatementFreshCall_geometry :
    fullStatementFreshCall.GeometryValid 307491 := by
  unfold CoordinateCall.GeometryValid
  rw [fullStatementFreshCall_activeFields_length]
  norm_num [fullStatementFreshCall, CoordinateCall.chunk, claimChunkCount,
    claimChunkFieldCount]

theorem fullRunningMetadataCall_geometry :
    fullRunningMetadataCall.GeometryValid 307491 := by
  unfold CoordinateCall.GeometryValid
  rw [fullRunningMetadataCall_activeFields_length]
  norm_num [fullRunningMetadataCall, CoordinateCall.chunk, claimChunkCount,
    claimChunkFieldCount]

theorem finalStatementFreshCall_geometry :
    finalStatementFreshCall.GeometryValid 342464 := by
  unfold CoordinateCall.GeometryValid
  rw [finalStatementFreshCall_activeFields_length]
  norm_num [finalStatementFreshCall, CoordinateCall.chunk, claimChunkCount,
    claimChunkFieldCount]

theorem fullStatementFreshCall_schedule :
    fullStatementFreshCall.ScheduleValid := by
  change statementFreshSchedule = MapKind.statementFresh.expectedSchedule
  exact
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStatementFreshScheduleCertificate.schedule_exact

theorem fullRunningMetadataCall_schedule :
    fullRunningMetadataCall.ScheduleValid := by
  change runningMetadataSchedule = MapKind.runningMetadata.expectedSchedule
  exact
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayRunningMetadataScheduleCertificate.schedule_exact

theorem finalStatementFreshCall_schedule :
    finalStatementFreshCall.ScheduleValid := by
  change statementFreshSchedule = MapKind.statementFresh.expectedSchedule
  exact
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStatementFreshScheduleCertificate.schedule_exact

theorem fullStatementFreshCall_valid :
    fullStatementFreshCall.Valid 307491 := by
  exact CoordinateCall.valid_of_geometry_schedule_and_certificate
    fullStatementFreshCall_geometry fullStatementFreshCall_schedule
    (certificateBlock_valid .statementFresh)

theorem fullRunningMetadataCall_valid :
    fullRunningMetadataCall.Valid 307491 := by
  exact CoordinateCall.valid_of_geometry_schedule_and_certificate
    fullRunningMetadataCall_geometry fullRunningMetadataCall_schedule
    (certificateBlock_valid .runningMetadata)

theorem finalStatementFreshCall_valid :
    finalStatementFreshCall.Valid 342464 := by
  exact CoordinateCall.valid_of_geometry_schedule_and_certificate
    finalStatementFreshCall_geometry finalStatementFreshCall_schedule
    (certificateBlock_valid .statementFresh)

theorem fullArm_coordinateCalls_valid :
    ∀ call ∈ fullArm.coordinateCalls, call.Valid fullArm.columnCount := by
  intro call member
  rw [fullArm_coordinateCalls_exact] at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · exact fullStatementFreshCall_valid
  · exact fullRunningMetadataCall_valid

theorem finalArm_coordinateCalls_valid :
    ∀ call ∈ finalArm.coordinateCalls, call.Valid finalArm.columnCount := by
  intro call member
  rw [finalArm_coordinateCalls_exact] at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl
  exact finalStatementFreshCall_valid

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateCallCertificates
