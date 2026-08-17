import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf168 :
    ((rowsChunk wire 168).map (fun row => row.sourceIndex) =
        List.range' 11010048 65536) ∧
      ((rowsChunk wire 168).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 168).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf169 :
    ((rowsChunk wire 169).map (fun row => row.sourceIndex) =
        List.range' 11075584 65536) ∧
      ((rowsChunk wire 169).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 169).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf170 :
    ((rowsChunk wire 170).map (fun row => row.sourceIndex) =
        List.range' 11141120 46705) ∧
      ((rowsChunk wire 170).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 170).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem presence2 :
    (rowsChunk wire 170).any
      (fun row => decide (row.family = "fprime.recursive.step.accumulator.output_authority.aggregate")) = true := by
  native_decide

theorem presence4 :
    (rowsChunk wire 170).any
      (fun row => decide (row.family = "fprime.recursive.step.counters")) = true := by
  native_decide

theorem presence6 :
    (rowsChunk wire 170).any
      (fun row => decide (row.family = "fprime.recursive.step.output")) = true := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12
