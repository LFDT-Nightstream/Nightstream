import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf56 :
    (chunkFacts (rowsChunk wire 56) 14336 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 56 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk56) := by
  native_decide

theorem chunkLeaf57 :
    (chunkFacts (rowsChunk wire 57) 14592 256 39949 38626
      wire.completeFamilies
      ["fprime.base.step.advance",
       "fprime.base.step.initial"] = true) ∧
      (rowsChunk wire 57 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk57) := by
  native_decide

theorem chunkLeaf58 :
    (chunkFacts (rowsChunk wire 58) 14848 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 58 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk58) := by
  native_decide

theorem chunkLeaf59 :
    (chunkFacts (rowsChunk wire 59) 15104 256 39949 38626
      wire.completeFamilies
      ["fprime.base.step.output"] = true) ∧
      (rowsChunk wire 59 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk59) := by
  native_decide

theorem chunkLeaf60 :
    (chunkFacts (rowsChunk wire 60) 15360 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 60 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk60) := by
  native_decide

theorem chunkLeaf61 :
    (chunkFacts (rowsChunk wire 61) 15616 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 61 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk61) := by
  native_decide

theorem chunkLeaf62 :
    (chunkFacts (rowsChunk wire 62) 15872 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 62 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk62) := by
  native_decide

theorem chunkLeaf63 :
    (chunkFacts (rowsChunk wire 63) 16128 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 63 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk63) := by
  native_decide

theorem chunkLeaf64 :
    (chunkFacts (rowsChunk wire 64) 16384 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 64 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk64) := by
  native_decide

theorem chunkLeaf65 :
    (chunkFacts (rowsChunk wire 65) 16640 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 65 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk65) := by
  native_decide

theorem chunkLeaf66 :
    (chunkFacts (rowsChunk wire 66) 16896 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 66 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk66) := by
  native_decide

theorem chunkLeaf67 :
    (chunkFacts (rowsChunk wire 67) 17152 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 67 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk67) := by
  native_decide

theorem chunkLeaf68 :
    (chunkFacts (rowsChunk wire 68) 17408 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 68 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk68) := by
  native_decide

theorem chunkLeaf69 :
    (chunkFacts (rowsChunk wire 69) 17664 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 69 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk69) := by
  native_decide

theorem presence1 :
    (rowsChunk wire 57).any
      (fun row => decide (row.family = "fprime.base.step.advance")) = true :=
  presence_of_chunkFacts (chunkLeaf57).1 (by decide)

theorem presence2 :
    (rowsChunk wire 57).any
      (fun row => decide (row.family = "fprime.base.step.initial")) = true :=
  presence_of_chunkFacts (chunkLeaf57).1 (by decide)

theorem presence3 :
    (rowsChunk wire 59).any
      (fun row => decide (row.family = "fprime.base.step.output")) = true :=
  presence_of_chunkFacts (chunkLeaf59).1 (by decide)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf4
