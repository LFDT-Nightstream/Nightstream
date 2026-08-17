import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf0 :
    ((rowsChunk wire 0).map (fun row => row.sourceIndex) =
        List.range' 0 256) ∧
      ((rowsChunk wire 0).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 0).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 0 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk0) := by
  native_decide

theorem chunkLeaf1 :
    ((rowsChunk wire 1).map (fun row => row.sourceIndex) =
        List.range' 256 256) ∧
      ((rowsChunk wire 1).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 1).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 1 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk1) := by
  native_decide

theorem chunkLeaf2 :
    ((rowsChunk wire 2).map (fun row => row.sourceIndex) =
        List.range' 512 256) ∧
      ((rowsChunk wire 2).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 2).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 2 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk2) := by
  native_decide

theorem chunkLeaf3 :
    ((rowsChunk wire 3).map (fun row => row.sourceIndex) =
        List.range' 768 256) ∧
      ((rowsChunk wire 3).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 3).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 3 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk3) := by
  native_decide

theorem chunkLeaf4 :
    ((rowsChunk wire 4).map (fun row => row.sourceIndex) =
        List.range' 1024 256) ∧
      ((rowsChunk wire 4).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 4).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 4 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk4) := by
  native_decide

theorem chunkLeaf5 :
    ((rowsChunk wire 5).map (fun row => row.sourceIndex) =
        List.range' 1280 256) ∧
      ((rowsChunk wire 5).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 5).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 5 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk5) := by
  native_decide

theorem chunkLeaf6 :
    ((rowsChunk wire 6).map (fun row => row.sourceIndex) =
        List.range' 1536 256) ∧
      ((rowsChunk wire 6).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 6).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 6 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk6) := by
  native_decide

theorem chunkLeaf7 :
    ((rowsChunk wire 7).map (fun row => row.sourceIndex) =
        List.range' 1792 256) ∧
      ((rowsChunk wire 7).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 7).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 7 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk7) := by
  native_decide

theorem chunkLeaf8 :
    ((rowsChunk wire 8).map (fun row => row.sourceIndex) =
        List.range' 2048 256) ∧
      ((rowsChunk wire 8).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 8).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 8 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk8) := by
  native_decide

theorem chunkLeaf9 :
    ((rowsChunk wire 9).map (fun row => row.sourceIndex) =
        List.range' 2304 256) ∧
      ((rowsChunk wire 9).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 9).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 9 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk9) := by
  native_decide

theorem chunkLeaf10 :
    ((rowsChunk wire 10).map (fun row => row.sourceIndex) =
        List.range' 2560 256) ∧
      ((rowsChunk wire 10).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 10).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 10 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk10) := by
  native_decide

theorem chunkLeaf11 :
    ((rowsChunk wire 11).map (fun row => row.sourceIndex) =
        List.range' 2816 256) ∧
      ((rowsChunk wire 11).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 11).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 11 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk11) := by
  native_decide

theorem chunkLeaf12 :
    ((rowsChunk wire 12).map (fun row => row.sourceIndex) =
        List.range' 3072 256) ∧
      ((rowsChunk wire 12).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 12).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 12 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk12) := by
  native_decide

theorem chunkLeaf13 :
    ((rowsChunk wire 13).map (fun row => row.sourceIndex) =
        List.range' 3328 256) ∧
      ((rowsChunk wire 13).all (rowWellFormedAt 39949 38626) = true) ∧
      ((rowsChunk wire 13).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) ∧
      (rowsChunk wire 13 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk13) := by
  native_decide

theorem presence0 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "fprime.base.finalize.application")) = true := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0
