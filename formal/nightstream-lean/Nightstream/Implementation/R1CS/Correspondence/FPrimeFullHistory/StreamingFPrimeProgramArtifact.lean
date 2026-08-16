import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyBodyOverlayRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilyPhysicalOverlayRows
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingFPrimeProgram
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyNormalizedLink
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinateSequence
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPhasedOverlayRelation

/-!
Contract: exact comparison of the Rust-emitted streaming program with the
verifier-owned Lean program.

Assurance tier: Rust-conformant for property
`FPRIME-STREAMING-PROGRAM`.

Owns exact phase codes, phase order, repeated-phase indices, chunk geometry,
and the 400-step production count.

Does not own phase-local constraints, relation rows or columns, recursive
proof integration, same-assignment conformance, or security reduction.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramArtifact

open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateSequence
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgram.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingFPrimeProgram

def expectedWorkItems : List (Nat × Nat) :=
  (program productionConfig).map fun item =>
    (item.phase.code.val, item.index)

theorem profile_exact :
    profileId = "nebula-fprime-streaming-program" := by
  decide

theorem artifact_geometry_exact :
    rawProgram.stateChunkFields = 1024 /\
      rawProgram.priorStateFrameFields = 83874 /\
      rawProgram.priorStateChunks = 82 /\
      rawProgram.claimFrameFields = 88023 /\
      rawProgram.claimChunkFields = 1024 /\
      rawProgram.claimChunks = 86 /\
      rawProgram.piCcsRounds = 26 /\
      rawProgram.piRlcFamilies = 110 /\
      rawProgram.firstPiRlcFamilyProgramCursor = 199 /\
      rawProgram.successorPrefixFrameFields = 83756 /\
      rawProgram.successorPrefixChunks = 82 /\
      rawProgram.workItemCount = 400 /\
      rawProgram.lifecycleGroupCount = 2 /\
      rawProgram.circuitKindCount = 23 /\
      rawProgram.claimCoordinateOverlayKindCount = 26 /\
      rawProgram.combinedOverlayKindCount = 136 /\
      rawProgram.piRlcFamilyFirstOverlayKind = 26 /\
      rawProgram.piRlcFamilyEvenPhaseKind = 10 /\
      rawProgram.piRlcFamilyOddPhaseKind = 11 /\
      rawProgram.piRlcFamilyBodySourceRows = 146006 /\
      rawProgram.piRlcFamilyBodyEvenSourceRows = 275006 /\
      rawProgram.piRlcFamilyBodyOddSourceRows = 276206 /\
      rawProgram.piRlcFamilyBodyEvenRows = 558932 /\
      rawProgram.piRlcFamilyBodyOddRows = 560132 /\
      rawProgram.piRlcFamilyBodyEvenColumns = 559136 /\
      rawProgram.piRlcFamilyBodyOddColumns = 560336 /\
      rawProgram.piRlcFamilyOverlayRows = 108 /\
      rawProgram.piRlcFamilyOverlayColumns = 33360 /\
      rawProgram.piRlcFamilyLinkFieldCount = 33359 /\
      rawProgram.piRlcFamilyTotalLinkFieldCount = 3669490 /\
      rawProgram.phasePublicLogicalColumns = 641 /\
      rawProgram.phasePublicColumns = 648 := by
  decide

theorem artifact_valid : ProgramValid rawProgram := by
  decide

/-- Expansion of the compact Rust runs is exactly the Lean phase program.
This compares every phase code and every repeated-phase index. -/
theorem rust_program_exact :
    rawProgram.expanded = expectedWorkItems := by
  decide

theorem rust_program_length_exact :
    rawProgram.expanded.length = 400 := by
  calc
    rawProgram.expanded.length = expectedWorkItems.length :=
      congrArg List.length rust_program_exact
    _ = (program productionConfig).length := by
      simp [expectedWorkItems]
    _ = 400 := production_program_length

/-- Rust and Lean select the same shared lifecycle circuit for every arm. -/
theorem rust_lifecycle_group_map_exact :
    rawProgram.lifecycleGroupMap = lifecycleCircuitMap productionConfig := by
  decide

/-- Rust and Lean select the same one of 23 stored phase circuits for every
one of the 400 schedule arms. -/
theorem rust_circuit_kind_map_exact :
    rawProgram.circuitKindMap = circuitKindMap productionConfig := by
  decide

/-- Rust and Lean select the same coordinate overlay for every production
arm, including the no-op arms and all 86 claim chunks. -/
theorem rust_claim_coordinate_overlay_kind_map_exact :
    rawProgram.claimCoordinateOverlayKindMap = productionOverlayKindMap := by
  decide

def rustClaimCoordinateOverlayLinkRuns :
    List (Nat × Nat × Nat × Nat × Nat) :=
  rawProgram.claimCoordinateOverlayLinkRuns.map fun run =>
    (run.overlayKind, run.phaseKind, run.chunkIndex, run.activeOffsetStart,
      run.activeFieldCount)

/-- Rust and Lean use the same compact source-field link contract for every
non-no-op coordinate overlay kind. -/
theorem rust_claim_coordinate_overlay_link_runs_exact :
    rustClaimCoordinateOverlayLinkRuns = productionOverlayLinkRuns := by
  decide

theorem rust_claim_coordinate_overlay_link_census_exact :
    (rustClaimCoordinateOverlayLinkRuns.map fun run => run.2.2.2.2).sum =
        21_220 /\
      (rustClaimCoordinateOverlayLinkRuns.map fun run =>
        216 + run.2.2.2.2).sum = 26_620 := by
  rw [rust_claim_coordinate_overlay_link_runs_exact]
  exact productionOverlayLinkRuns_census

def expectedPiRlcFamilyOverlayKindMap : List Nat :=
  (program productionConfig).map fun item =>
    if item.phase = .piRlcFamily then
      rawProgram.piRlcFamilyFirstOverlayKind + item.index
    else
      0

/-- Rust selects family overlay kinds 26 through 135 on exactly the 110
PiRLC-family work items. Every other work item selects the no-op kind. -/
theorem rust_pi_rlc_family_overlay_kind_map_exact :
    rawProgram.piRlcFamilyOverlayKindMap =
      expectedPiRlcFamilyOverlayKindMap := by
  decide

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

/-- The two Rust overlay maps merge to the exact 136-kind map used by the
Lean semantic relation on all 400 work items. -/
theorem rust_combined_overlay_kind_map_exact :
    rustCombinedOverlayKindMap = expectedCombinedOverlayKindMap := by
  native_decide

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
      outerCount := 810
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
  native_decide

theorem rust_pi_rlc_family_overlay_link_census_exact :
    (rawProgram.piRlcFamilyOverlayLinkRuns.map
        RawFieldLinkRun.linkCount).sum = 33359 /\
      rawProgram.piRlcFamilies *
        (rawProgram.piRlcFamilyOverlayLinkRuns.map
          RawFieldLinkRun.linkCount).sum = 3669490 := by
  native_decide

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
  native_decide

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
