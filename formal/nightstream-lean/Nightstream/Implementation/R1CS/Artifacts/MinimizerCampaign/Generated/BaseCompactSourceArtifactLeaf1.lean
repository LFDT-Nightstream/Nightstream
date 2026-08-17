import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf14 :
    (chunkFacts (rowsChunk wire 14) 3584 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 14 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk14) := by
  native_decide

theorem chunkLeaf15 :
    (chunkFacts (rowsChunk wire 15) 3840 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 15 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk15) := by
  native_decide

theorem chunkLeaf16 :
    (chunkFacts (rowsChunk wire 16) 4096 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 16 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk16) := by
  native_decide

theorem chunkLeaf17 :
    (chunkFacts (rowsChunk wire 17) 4352 256 39949 38626
      wire.completeFamilies
      ["fprime.base.step.prelude"] = true) ∧
      (rowsChunk wire 17 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk17) := by
  native_decide

theorem chunkLeaf18 :
    (chunkFacts (rowsChunk wire 18) 4608 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 18 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk18) := by
  native_decide

theorem chunkLeaf19 :
    (chunkFacts (rowsChunk wire 19) 4864 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 19 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk19) := by
  native_decide

theorem chunkLeaf20 :
    (chunkFacts (rowsChunk wire 20) 5120 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 20 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk20) := by
  native_decide

theorem chunkLeaf21 :
    (chunkFacts (rowsChunk wire 21) 5376 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 21 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk21) := by
  native_decide

theorem chunkLeaf22 :
    (chunkFacts (rowsChunk wire 22) 5632 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 22 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk22) := by
  native_decide

theorem chunkLeaf23 :
    (chunkFacts (rowsChunk wire 23) 5888 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 23 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk23) := by
  native_decide

theorem chunkLeaf24 :
    (chunkFacts (rowsChunk wire 24) 6144 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 24 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk24) := by
  native_decide

theorem chunkLeaf25 :
    (chunkFacts (rowsChunk wire 25) 6400 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 25 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk25) := by
  native_decide

theorem chunkLeaf26 :
    (chunkFacts (rowsChunk wire 26) 6656 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 26 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk26) := by
  native_decide

theorem chunkLeaf27 :
    (chunkFacts (rowsChunk wire 27) 6912 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 27 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk27) := by
  native_decide

theorem presence4 :
    (rowsChunk wire 17).any
      (fun row => decide (row.family = "fprime.base.step.prelude")) = true :=
  presence_of_chunkFacts (chunkLeaf17).1 (by decide)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf1
