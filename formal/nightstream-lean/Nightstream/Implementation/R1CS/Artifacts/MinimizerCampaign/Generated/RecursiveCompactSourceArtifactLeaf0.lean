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

theorem chunkLeaf1 :
    ((rowsChunk wire 1).map (fun row => row.sourceIndex) =
        List.range' 65536 65536) ∧
      ((rowsChunk wire 1).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 1).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf2 :
    ((rowsChunk wire 2).map (fun row => row.sourceIndex) =
        List.range' 131072 65536) ∧
      ((rowsChunk wire 2).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 2).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

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

theorem chunkLeaf7 :
    ((rowsChunk wire 7).map (fun row => row.sourceIndex) =
        List.range' 458752 65536) ∧
      ((rowsChunk wire 7).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 7).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf8 :
    ((rowsChunk wire 8).map (fun row => row.sourceIndex) =
        List.range' 524288 65536) ∧
      ((rowsChunk wire 8).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 8).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

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
