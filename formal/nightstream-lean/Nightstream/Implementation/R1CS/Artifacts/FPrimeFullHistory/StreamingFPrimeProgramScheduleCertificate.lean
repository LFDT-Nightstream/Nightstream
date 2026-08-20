import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingProgram
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingFPrimeProgramLeafCertificateSupport
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayOverlaySchedule

/-!
Contract: bounded structural certificates for the Rust-emitted streaming
F-prime program schedule.

Assurance tier: Rust-conformant schedule certificate.

Owns exact 436-entry schedule and selector-map identity, exact 98-entry claim
link identity, and bounded wire-geometry checks. Each large list is checked in
64-entry leaves with an exact final remainder.

Does not own phase-local row semantics, lifecycle relation semantics, or
security reduction.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramScheduleCertificate

open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayOverlaySchedule
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgram.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramLeafCertificateSupport

def expectedWorkItems : List (Nat × Nat) :=
  (program productionConfig).map fun item =>
    (item.phase.code.val, item.index)

def expectedPiRlcFamilyOverlayKindMap : List Nat :=
  (program productionConfig).map fun item =>
    if item.phase = .piRlcFamily then
      rawProgram.piRlcFamilyFirstOverlayKind + item.index
    else
      0

def rustClaimCoordinateOverlayLinkRuns :
    List (Nat × Nat × Nat × Nat × Nat) :=
  rawProgram.claimCoordinateOverlayLinkRuns.map fun run =>
    (run.overlayKind, run.phaseKind, run.chunkIndex, run.activeOffsetStart,
      run.activeFieldCount)

theorem rustProgram_chunked :
    Chunked436Eq rawProgram.expanded expectedWorkItems := by
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

theorem rust_program_exact :
    rawProgram.expanded = expectedWorkItems :=
  rustProgram_chunked.sound

theorem lifecycleGroupMap_chunked :
    Chunked436Eq rawProgram.lifecycleGroupMap
      (lifecycleCircuitMap productionConfig) := by
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

theorem lifecycle_group_map_exact :
    rawProgram.lifecycleGroupMap = lifecycleCircuitMap productionConfig :=
  lifecycleGroupMap_chunked.sound

theorem circuitKindMap_chunked :
    Chunked436Eq rawProgram.circuitKindMap
      (circuitKindMap productionConfig) := by
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

theorem circuit_kind_map_exact :
    rawProgram.circuitKindMap = circuitKindMap productionConfig :=
  circuitKindMap_chunked.sound

theorem claimCoordinateOverlayKindMap_chunked :
    Chunked436Eq rawProgram.claimCoordinateOverlayKindMap
      productionOverlayKindMap := by
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

theorem claim_coordinate_overlay_kind_map_exact :
    rawProgram.claimCoordinateOverlayKindMap = productionOverlayKindMap :=
  claimCoordinateOverlayKindMap_chunked.sound

theorem piRlcFamilyOverlayKindMap_chunked :
    Chunked436Eq rawProgram.piRlcFamilyOverlayKindMap
      expectedPiRlcFamilyOverlayKindMap := by
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

theorem pi_rlc_family_overlay_kind_map_exact :
    rawProgram.piRlcFamilyOverlayKindMap =
      expectedPiRlcFamilyOverlayKindMap :=
  piRlcFamilyOverlayKindMap_chunked.sound

theorem claimCoordinateOverlayLinkRuns_chunked :
    Chunked98Eq rustClaimCoordinateOverlayLinkRuns
      productionOverlayLinkRuns := by
  exact
    { leftLength := rfl
      rightLength := rfl
      chunk := rfl
      remainder := rfl
      leftRemainderLength := rfl
      rightRemainderLength := rfl }

theorem claim_coordinate_overlay_link_runs_exact :
    rustClaimCoordinateOverlayLinkRuns = productionOverlayLinkRuns :=
  claimCoordinateOverlayLinkRuns_chunked.sound

theorem lifecycleGroupMapBounds_chunked :
    Chunked436All rawProgram.lifecycleGroupMap
      (fun group => group < rawProgram.lifecycleGroupCount) := by
  exact
    { length := rfl
      chunk0 := rfl
      chunk1 := rfl
      chunk2 := rfl
      chunk3 := rfl
      chunk4 := rfl
      chunk5 := rfl
      remainder := rfl
      remainderLength := rfl }

theorem lifecycle_group_map_bounds :
    rawProgram.lifecycleGroupMap.all
      (fun group => group < rawProgram.lifecycleGroupCount) = true :=
  lifecycleGroupMapBounds_chunked.sound

theorem circuitKindMapBounds_chunked :
    Chunked436All rawProgram.circuitKindMap
      (fun kind => kind < rawProgram.circuitKindCount) := by
  exact
    { length := rfl
      chunk0 := rfl
      chunk1 := rfl
      chunk2 := rfl
      chunk3 := rfl
      chunk4 := rfl
      chunk5 := rfl
      remainder := rfl
      remainderLength := rfl }

theorem circuit_kind_map_bounds :
    rawProgram.circuitKindMap.all
      (fun kind => kind < rawProgram.circuitKindCount) = true :=
  circuitKindMapBounds_chunked.sound

theorem claimCoordinateOverlayKindMapBounds_chunked :
    Chunked436All rawProgram.claimCoordinateOverlayKindMap
      (fun kind => kind < rawProgram.claimCoordinateOverlayKindCount) := by
  exact
    { length := rfl
      chunk0 := rfl
      chunk1 := rfl
      chunk2 := rfl
      chunk3 := rfl
      chunk4 := rfl
      chunk5 := rfl
      remainder := rfl
      remainderLength := rfl }

theorem claim_coordinate_overlay_kind_map_bounds :
    rawProgram.claimCoordinateOverlayKindMap.all
      (fun kind => kind < rawProgram.claimCoordinateOverlayKindCount) = true :=
  claimCoordinateOverlayKindMapBounds_chunked.sound

theorem piRlcFamilyOverlayKindMapBounds_chunked :
    Chunked436All rawProgram.piRlcFamilyOverlayKindMap
      (fun kind => kind < rawProgram.combinedOverlayKindCount) := by
  exact
    { length := rfl
      chunk0 := rfl
      chunk1 := rfl
      chunk2 := rfl
      chunk3 := rfl
      chunk4 := rfl
      chunk5 := rfl
      remainder := rfl
      remainderLength := rfl }

theorem pi_rlc_family_overlay_kind_map_bounds :
    rawProgram.piRlcFamilyOverlayKindMap.all
      (fun kind => kind < rawProgram.combinedOverlayKindCount) = true :=
  piRlcFamilyOverlayKindMapBounds_chunked.sound

theorem claimCoordinateOverlayLinkRunsBounds_chunked :
    Chunked98All rawProgram.claimCoordinateOverlayLinkRuns
      (RawOverlayLinkRun.valid rawProgram) := by
  exact
    { length := rfl
      chunk := rfl
      remainder := rfl
      remainderLength := rfl }

theorem claim_coordinate_overlay_link_runs_bounds :
    rawProgram.claimCoordinateOverlayLinkRuns.all
      (RawOverlayLinkRun.valid rawProgram) = true :=
  claimCoordinateOverlayLinkRunsBounds_chunked.sound

theorem claimOverlayKinds_chunked :
    Chunked98Eq
      (rawProgram.claimCoordinateOverlayLinkRuns.map
        (fun run => run.overlayKind))
      ((List.range rawProgram.claimCoordinateOverlayKindCount).drop 1) := by
  exact
    { leftLength := rfl
      rightLength := rfl
      chunk := rfl
      remainder := rfl
      leftRemainderLength := rfl
      rightRemainderLength := rfl }

theorem claim_overlay_kinds_exact :
    rawProgram.claimCoordinateOverlayLinkRuns.map
        (fun run => run.overlayKind) =
      (List.range rawProgram.claimCoordinateOverlayKindCount).drop 1 :=
  claimOverlayKinds_chunked.sound

theorem runs_valid : rawProgram.runs.all RawRun.valid = true := by
  rfl

theorem pi_rlc_family_overlay_link_runs_valid :
    rawProgram.piRlcFamilyOverlayLinkRuns.all
      (RawFieldLinkRun.valid rawProgram.piRlcFamilyBodyEvenColumns
        rawProgram.piRlcFamilyOverlayColumns) = true := by
  rfl

theorem pi_rlc_family_overlay_link_sum_exact :
    (rawProgram.piRlcFamilyOverlayLinkRuns.map
        RawFieldLinkRun.linkCount).sum =
      rawProgram.piRlcFamilyLinkFieldCount := by
  rfl

/-- Complete wire validity is composed from small scalar arithmetic and the
bounded schedule leaves above. No complete artifact decision is evaluated. -/
theorem rawProgram_valid : ProgramValid rawProgram := by
  unfold ProgramValid
  exact
    ⟨rfl,
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      runs_valid,
      rustProgram_chunked.leftLength,
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      rfl,
      rfl,
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      rfl,
      rfl,
      rfl,
      rfl,
      rfl,
      rfl,
      rfl,
      rfl,
      rfl,
      rfl,
      rfl,
      by norm_num [rawProgram],
      by norm_num [rawProgram],
      lifecycleGroupMap_chunked.leftLength,
      circuitKindMap_chunked.leftLength,
      claimCoordinateOverlayKindMap_chunked.leftLength,
      piRlcFamilyOverlayKindMap_chunked.leftLength,
      by
        rw [claimCoordinateOverlayLinkRunsBounds_chunked.length]
        rfl,
      claim_overlay_kinds_exact,
      lifecycle_group_map_bounds,
      circuit_kind_map_bounds,
      claim_coordinate_overlay_kind_map_bounds,
      pi_rlc_family_overlay_kind_map_bounds,
      claim_coordinate_overlay_link_runs_bounds,
      pi_rlc_family_overlay_link_runs_valid,
      pi_rlc_family_overlay_link_sum_exact⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgramScheduleCertificate
