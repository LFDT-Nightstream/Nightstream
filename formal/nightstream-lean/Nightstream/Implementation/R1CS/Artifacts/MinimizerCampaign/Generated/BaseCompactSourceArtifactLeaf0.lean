import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
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
    (chunkFacts (rowsChunk wire 0) 0 256 39949 38626
      wire.completeFamilies
      ["fprime.base.finalize.application"] = true) ∧
      (rowsChunk wire 0 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk0) := by
  native_decide

theorem chunkLeaf1 :
    (chunkFacts (rowsChunk wire 1) 256 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 1 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk1) := by
  native_decide

theorem chunkLeaf2 :
    (chunkFacts (rowsChunk wire 2) 512 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 2 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk2) := by
  native_decide

theorem chunkLeaf3 :
    (chunkFacts (rowsChunk wire 3) 768 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 3 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk3) := by
  native_decide

theorem chunkLeaf4 :
    (chunkFacts (rowsChunk wire 4) 1024 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 4 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk4) := by
  native_decide

theorem chunkLeaf5 :
    (chunkFacts (rowsChunk wire 5) 1280 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 5 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk5) := by
  native_decide

theorem chunkLeaf6 :
    (chunkFacts (rowsChunk wire 6) 1536 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 6 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk6) := by
  native_decide

theorem chunkLeaf7 :
    (chunkFacts (rowsChunk wire 7) 1792 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 7 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk7) := by
  native_decide

theorem chunkLeaf8 :
    (chunkFacts (rowsChunk wire 8) 2048 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 8 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk8) := by
  native_decide

theorem chunkLeaf9 :
    (chunkFacts (rowsChunk wire 9) 2304 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 9 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk9) := by
  native_decide

theorem chunkLeaf10 :
    (chunkFacts (rowsChunk wire 10) 2560 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 10 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk10) := by
  native_decide

theorem chunkLeaf11 :
    (chunkFacts (rowsChunk wire 11) 2816 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 11 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk11) := by
  native_decide

theorem chunkLeaf12 :
    (chunkFacts (rowsChunk wire 12) 3072 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 12 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk12) := by
  native_decide

theorem chunkLeaf13 :
    (chunkFacts (rowsChunk wire 13) 3328 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 13 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk13) := by
  native_decide

theorem presence0 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "fprime.base.finalize.application")) = true :=
  presence_of_chunkFacts (chunkLeaf0).1 (by decide)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf0
