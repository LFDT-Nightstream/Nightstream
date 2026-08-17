import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf16

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf102 :
    chunkFacts (rowsChunk wire 102) 6684672 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_ccs.padded_row.output_digest.claim",
       "nifs.running_parent_pi_dec"] = true := by
  native_decide

theorem presence16 :
    (rowsChunk wire 102).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.claim")) = true :=
  presence_of_chunkFacts chunkLeaf102 (by decide)

theorem presence81 :
    (rowsChunk wire 102).any
      (fun row => decide (row.family = "nifs.running_parent_pi_dec")) = true :=
  presence_of_chunkFacts chunkLeaf102 (by decide)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf16
