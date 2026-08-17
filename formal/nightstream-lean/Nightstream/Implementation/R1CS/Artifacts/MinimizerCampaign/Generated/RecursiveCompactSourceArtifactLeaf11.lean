import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf11

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf57 :
    chunkFacts (rowsChunk wire 57) 3735552 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_ccs.padded_row.prefix"] = true := by
  native_decide

theorem presence22 :
    (rowsChunk wire 57).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.prefix")) = true :=
  presence_of_chunkFacts chunkLeaf57 (by decide)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf11
