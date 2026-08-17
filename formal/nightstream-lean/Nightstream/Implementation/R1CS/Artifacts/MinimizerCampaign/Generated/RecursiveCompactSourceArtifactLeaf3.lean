import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf3

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf3 :
    ((rowsChunk wire 3).map (fun row => row.sourceIndex) =
        List.range' 196608 65536) ∧
      ((rowsChunk wire 3).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 3).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf4 :
    ((rowsChunk wire 4).map (fun row => row.sourceIndex) =
        List.range' 262144 65536) ∧
      ((rowsChunk wire 4).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 4).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf5 :
    ((rowsChunk wire 5).map (fun row => row.sourceIndex) =
        List.range' 327680 65536) ∧
      ((rowsChunk wire 5).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 5).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf6 :
    ((rowsChunk wire 6).map (fun row => row.sourceIndex) =
        List.range' 393216 65536) ∧
      ((rowsChunk wire 6).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 6).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf3
