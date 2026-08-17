import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf6

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf9 :
    ((rowsChunk wire 9).map (fun row => row.sourceIndex) =
        List.range' 589824 65536) ∧
      ((rowsChunk wire 9).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 9).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf10 :
    ((rowsChunk wire 10).map (fun row => row.sourceIndex) =
        List.range' 655360 65536) ∧
      ((rowsChunk wire 10).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 10).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf11 :
    ((rowsChunk wire 11).map (fun row => row.sourceIndex) =
        List.range' 720896 65536) ∧
      ((rowsChunk wire 11).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 11).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf12 :
    ((rowsChunk wire 12).map (fun row => row.sourceIndex) =
        List.range' 786432 65536) ∧
      ((rowsChunk wire 12).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 12).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf13 :
    ((rowsChunk wire 13).map (fun row => row.sourceIndex) =
        List.range' 851968 65536) ∧
      ((rowsChunk wire 13).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 13).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf14 :
    ((rowsChunk wire 14).map (fun row => row.sourceIndex) =
        List.range' 917504 65536) ∧
      ((rowsChunk wire 14).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 14).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf15 :
    ((rowsChunk wire 15).map (fun row => row.sourceIndex) =
        List.range' 983040 65536) ∧
      ((rowsChunk wire 15).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 15).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf16 :
    ((rowsChunk wire 16).map (fun row => row.sourceIndex) =
        List.range' 1048576 65536) ∧
      ((rowsChunk wire 16).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 16).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf6
