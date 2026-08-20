import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyBodyOverlayRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyPhysicalOverlayRows
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingFPrimeProgram
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingFPrimeProgramScheduleCertificate
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLink
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayOverlaySchedule
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPhasedOverlayRelation

/-!
Contract: exact comparison of the Rust-emitted streaming program with the
verifier-owned Lean program.

Assurance tier: Rust-conformant for property
`FPRIME-STREAMING-PROGRAM`.

Owns exact phase codes, phase order, repeated-phase indices, chunk geometry,
and the 436-step production count.

Does not own phase-local constraints, relation rows or columns, recursive
proof integration, same-assignment conformance, or security reduction.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact

open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayOverlaySchedule
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgram.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramLeafCertificateSupport
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingFPrimeProgram

def expectedWorkItems : List (Nat × Nat) :=
  (program productionConfig).map fun item =>
    (item.phase.code.val, item.index)

theorem profile_exact :
    profileId = "nebula-fprime-streaming-program" := by
  decide

theorem artifact_geometry_exact :
    rawProgram.stateChunkFields = 1024 /\
      rawProgram.priorStateFrameFields = 95754 /\
      rawProgram.priorStateChunks = 94 /\
      rawProgram.claimFrameFields = 99903 /\
      rawProgram.claimChunkFields = 1024 /\
      rawProgram.claimChunks = 98 /\
      rawProgram.piCcsRounds = 26 /\
      rawProgram.piRlcFamilies = 110 /\
      rawProgram.firstPiRlcFamilyProgramCursor = 223 /\
      rawProgram.successorPrefixFrameFields = 95636 /\
      rawProgram.successorPrefixChunks = 94 /\
      rawProgram.workItemCount = 436 /\
      rawProgram.lifecycleGroupCount = 2 /\
      rawProgram.circuitKindCount = 23 /\
      rawProgram.claimCoordinateOverlayKindCount = 99 /\
      rawProgram.combinedOverlayKindCount = 209 /\
      rawProgram.piRlcFamilyFirstOverlayKind = 99 /\
      rawProgram.piRlcFamilyEvenPhaseKind = 10 /\
      rawProgram.piRlcFamilyOddPhaseKind = 11 /\
      rawProgram.piRlcFamilyBodySourceRows = 165446 /\
      rawProgram.piRlcFamilyBodyEvenSourceRows = 310646 /\
      rawProgram.piRlcFamilyBodyOddSourceRows = 311846 /\
      rawProgram.piRlcFamilyBodyEvenRows = 1300897 /\
      rawProgram.piRlcFamilyBodyOddRows = 1302097 /\
      rawProgram.piRlcFamilyBodyEvenColumns = 1301126 /\
      rawProgram.piRlcFamilyBodyOddColumns = 1302326 /\
      rawProgram.piRlcFamilyOverlayRows = 108 /\
      rawProgram.piRlcFamilyOverlayColumns = 37788 /\
      rawProgram.piRlcFamilyLinkFieldCount = 37787 /\
      rawProgram.piRlcFamilyTotalLinkFieldCount = 4156570 /\
      rawProgram.phasePublicLogicalColumns = 641 /\
      rawProgram.phasePublicColumns = 648 := by
  decide

theorem artifact_valid : ProgramValid rawProgram := by
  exact Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramScheduleCertificate.rawProgram_valid

/-- Expansion of the compact Rust runs is exactly the Lean phase program.
This compares every phase code and every repeated-phase index. -/
theorem rust_program_exact :
    rawProgram.expanded = expectedWorkItems := by
  exact Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramScheduleCertificate.rust_program_exact

theorem rust_program_length_exact :
    rawProgram.expanded.length = 436 := by
  calc
    rawProgram.expanded.length = expectedWorkItems.length :=
      congrArg List.length rust_program_exact
    _ = (program productionConfig).length := by
      simp [expectedWorkItems]
    _ = 436 := production_program_length

/-- Rust and Lean select the same shared lifecycle circuit for every arm. -/
theorem rust_lifecycle_group_map_exact :
    rawProgram.lifecycleGroupMap = lifecycleCircuitMap productionConfig := by
  exact Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramScheduleCertificate.lifecycle_group_map_exact

/-- Rust and Lean select the same one of 23 stored phase circuits for every
one of the 436 schedule arms. -/
theorem rust_circuit_kind_map_exact :
    rawProgram.circuitKindMap = circuitKindMap productionConfig := by
  exact Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramScheduleCertificate.circuit_kind_map_exact

/-- Rust and Lean select the same coordinate overlay for every production
arm, including the no-op arms and all 98 claim chunks. -/
theorem rust_claim_coordinate_overlay_kind_map_exact :
    rawProgram.claimCoordinateOverlayKindMap = productionOverlayKindMap := by
  exact Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramScheduleCertificate.claim_coordinate_overlay_kind_map_exact

def rustClaimCoordinateOverlayLinkRuns :
    List (Nat × Nat × Nat × Nat × Nat) :=
  rawProgram.claimCoordinateOverlayLinkRuns.map fun run =>
    (run.overlayKind, run.phaseKind, run.chunkIndex, run.activeOffsetStart,
      run.activeFieldCount)

/-- Rust and Lean use the same compact source-field link contract for every
non-no-op coordinate overlay kind. -/
theorem rust_claim_coordinate_overlay_link_runs_exact :
    rustClaimCoordinateOverlayLinkRuns = productionOverlayLinkRuns := by
  exact Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramScheduleCertificate.claim_coordinate_overlay_link_runs_exact

theorem rust_claim_coordinate_overlay_link_census_exact :
    (rustClaimCoordinateOverlayLinkRuns.map fun run => run.2.2.2.2).sum =
        99_520 /\
      (rustClaimCoordinateOverlayLinkRuns.map fun run =>
        432 + run.2.2.2.2).sum = 141_856 := by
  rw [rust_claim_coordinate_overlay_link_runs_exact]
  exact productionOverlayLinkRuns_census

def expectedPiRlcFamilyOverlayKindMap : List Nat :=
  (program productionConfig).map fun item =>
    if item.phase = .piRlcFamily then
      rawProgram.piRlcFamilyFirstOverlayKind + item.index
    else
      0

/-- Rust selects family overlay kinds 99 through 208 on exactly the 110
PiRLC-family work items. Every other work item selects the no-op kind. -/
theorem rust_pi_rlc_family_overlay_kind_map_exact :
    rawProgram.piRlcFamilyOverlayKindMap =
      expectedPiRlcFamilyOverlayKindMap := by
  exact Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramScheduleCertificate.pi_rlc_family_overlay_kind_map_exact

def rustCombinedOverlayKindMap : List Nat :=
  List.zipWith
    (fun claimKind piRlcKind =>
      if piRlcKind = 0 then claimKind else piRlcKind)
    rawProgram.claimCoordinateOverlayKindMap
    rawProgram.piRlcFamilyOverlayKindMap

def expectedCombinedOverlayKindMap : List Nat :=
  (program productionConfig).map fun item =>
    (Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPhasedOverlayRelation.combinedOverlayKindForWorkItem
      item).val

theorem rustCombinedOverlayKindMap_chunked :
    Chunked436Eq rustCombinedOverlayKindMap
      expectedCombinedOverlayKindMap := by
  exact
    { leftLength := rfl
      rightLength := rfl
      chunk0 := rfl
      chunk1 := rfl
      chunk2 := rfl
      chunk3 := rfl
      chunk4 := rfl
      chunk5 := rfl
      remainder := rfl
      leftRemainderLength := rfl
      rightRemainderLength := rfl }

/-- The two Rust overlay maps merge to the exact 209-kind map used by the
Lean semantic relation on all 436 work items. -/
theorem rust_combined_overlay_kind_map_exact :
    rustCombinedOverlayKindMap = expectedCombinedOverlayKindMap := by
  exact rustCombinedOverlayKindMap_chunked.sound

def expectedPiRlcFamilyOverlayLinkRuns : List RawFieldLinkRun :=
  let bodyLayout :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows.layout
  let overlayLayout :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.physicalLayout
  let publicOutputCount :=
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLink.audit.publicOutputCount
  [
    { phaseFieldStart := bodyLayout.input.phase.zeroDigitStart + publicOutputCount
      overlayFieldStart := overlayLayout.zeroDigitStart
      outerCount := 1
      phaseStride := 41
      overlayStride := 41
      fieldCount := 41 },
    { phaseFieldStart :=
        bodyLayout.input.phase.digitStart ⟨0, by decide⟩ ⟨0, by decide⟩ +
          publicOutputCount
      overlayFieldStart :=
        overlayLayout.digitStart ⟨0, by decide⟩ ⟨0, by decide⟩
      outerCount := 918
      phaseStride := 122
      overlayStride := 41
      fieldCount := 41 },
    { phaseFieldStart :=
        bodyLayout.input.phase.outputColumn ⟨0, by decide⟩ + publicOutputCount
      overlayFieldStart := overlayLayout.outputColumn ⟨0, by decide⟩
      outerCount := 1
      phaseStride := 108
      overlayStride := 108
      fieldCount := 108 }
  ]

/-- The three compact Rust link formulas start at the exact normalized Lean
body fields for the zero word, active words, and commitment outputs. -/
theorem rust_pi_rlc_family_overlay_link_runs_exact :
    rawProgram.piRlcFamilyOverlayLinkRuns =
      expectedPiRlcFamilyOverlayLinkRuns := by
  rfl

theorem expected_pi_rlc_family_overlay_link_counts_exact :
    expectedPiRlcFamilyOverlayLinkRuns.map RawFieldLinkRun.linkCount =
      [41, 37_638, 108] := by
  rfl

theorem rust_pi_rlc_family_count_exact :
    rawProgram.piRlcFamilies = 110 :=
  artifact_geometry_exact.2.2.2.2.2.2.2.1

theorem rust_pi_rlc_family_overlay_link_census_exact :
    (rawProgram.piRlcFamilyOverlayLinkRuns.map
        RawFieldLinkRun.linkCount).sum = 37787 /\
      rawProgram.piRlcFamilies *
        (rawProgram.piRlcFamilyOverlayLinkRuns.map
          RawFieldLinkRun.linkCount).sum = 4156570 := by
  rw [rust_pi_rlc_family_overlay_link_runs_exact,
    expected_pi_rlc_family_overlay_link_counts_exact,
    rust_pi_rlc_family_count_exact]
  norm_num

/-- The Rust physical overlay width and compact link census are exactly the
Lean physical layout and its authoritative field-link count. -/
theorem rust_pi_rlc_family_physical_link_contract_exact :
    rawProgram.piRlcFamilyOverlayColumns =
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.physicalLayout.outputColumn
          ⟨107, by decide⟩ + 1 /\
      rawProgram.piRlcFamilyLinkFieldCount =
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.fieldLinkCount /\
      (rawProgram.piRlcFamilyOverlayLinkRuns.map
          RawFieldLinkRun.linkCount).sum =
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.fieldLinkCount /\
      rawProgram.piRlcFamilyTotalLinkFieldCount =
        rawProgram.piRlcFamilies *
          Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.fieldLinkCount := by
  constructor
  · rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.physical_layout_exact.2.2.2.2]
    rfl
  constructor
  · rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.fieldLinkCount_exact]
    rfl
  constructor
  · exact rust_pi_rlc_family_overlay_link_census_exact.1.trans
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.fieldLinkCount_exact.symm
  · rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyPhysicalOverlayRows.fieldLinkCount_exact]
    rfl

/-- The Rust body and overlay row counts equal the Lean source split for
every family and parity. -/
theorem rust_pi_rlc_family_body_overlay_rows_exact
    (setup :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.InputBindingSetup)
    (family : Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows.Family) :
    rawProgram.piRlcFamilyBodySourceRows =
        Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.sourceBodyRows.length /\
      rawProgram.piRlcFamilyBodyEvenSourceRows =
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.bodyRowsForParity
          .even).length /\
      rawProgram.piRlcFamilyBodyOddSourceRows =
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.bodyRowsForParity
          .odd).length /\
      rawProgram.piRlcFamilyOverlayRows =
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.overlayRows
          setup family).length := by
  rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.sourceBodyRows_length]
  rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.bodyRowsForParity_length]
  rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.bodyRowsForParity_length]
  rw [Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyBodyOverlayRows.overlayRows_length]
  decide

/-- Rust and Lean use the same nonredundant shared public phase layout. -/
theorem rust_phase_public_layout_exact :
    rawProgram.phasePublicLogicalColumns = productionPublicLayout.logicalColumns /\
      rawProgram.phasePublicColumns = productionPublicLayout.columns /\
      rawProgram.afterStateDigestStart = productionPublicLayout.afterStateDigestStart /\
      rawProgram.afterStateDigestEnd = productionPublicLayout.afterStateDigestEnd /\
      rawProgram.beforeStateDigestStart = productionPublicLayout.beforeStateDigestStart /\
      rawProgram.beforeStateDigestEnd = productionPublicLayout.beforeStateDigestEnd /\
      rawProgram.beforeCursorStart = productionPublicLayout.beforeCursorStart /\
      rawProgram.beforeCursorEnd = productionPublicLayout.beforeCursorEnd /\
      rawProgram.afterCursorStart = productionPublicLayout.afterCursorStart /\
      rawProgram.afterCursorEnd = productionPublicLayout.afterCursorEnd /\
      rawProgram.phasePublicPaddingStart = productionPublicLayout.paddingStart /\
      rawProgram.phasePublicPaddingEnd = productionPublicLayout.paddingEnd := by
  decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact
