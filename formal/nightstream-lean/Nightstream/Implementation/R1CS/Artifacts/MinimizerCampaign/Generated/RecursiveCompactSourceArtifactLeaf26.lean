import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf26

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf130 :
    chunkFacts (rowsChunk wire 130) 8519680 65536 11187825 11078210
      wire.completeFamilies
      ["fprime.recursive.step.accumulator.input_link",
       "fprime.recursive.step.accumulator.output_authority.child_digests"] = true := by
  native_decide

theorem presence1 :
    (rowsChunk wire 130).any
      (fun row => decide (row.family = "fprime.recursive.step.accumulator.input_link")) = true :=
  presence_of_chunkFacts chunkLeaf130 (by decide)

theorem presence3 :
    (rowsChunk wire 130).any
      (fun row => decide (row.family = "fprime.recursive.step.accumulator.output_authority.child_digests")) = true :=
  presence_of_chunkFacts chunkLeaf130 (by decide)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf26
