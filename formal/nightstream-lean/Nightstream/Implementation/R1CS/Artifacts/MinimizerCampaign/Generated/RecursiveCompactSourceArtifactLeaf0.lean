import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf0 :
    ((rowsChunk wire 0).map (fun row => row.sourceIndex) =
        List.range' 0 65536) ∧
      ((rowsChunk wire 0).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 0).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem presence0 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "fprime.recursive.finalize.application")) = true := by
  native_decide

theorem presence7 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "fprime.recursive.step.prelude")) = true := by
  native_decide

theorem presence11 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "fprime.recursive.step.transcript")) = true := by
  native_decide

theorem presence12 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.allocations")) = true := by
  native_decide

theorem presence13 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.binding")) = true := by
  native_decide

theorem presence14 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.canonicality")) = true := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0
