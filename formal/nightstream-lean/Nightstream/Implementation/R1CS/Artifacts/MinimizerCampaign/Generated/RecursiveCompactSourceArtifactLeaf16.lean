import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

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
    ((rowsChunk wire 102).map (fun row => row.sourceIndex) =
        List.range' 6684672 65536) ∧
      ((rowsChunk wire 102).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 102).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem presence16 :
    (rowsChunk wire 102).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.claim")) = true := by
  native_decide

theorem presence81 :
    (rowsChunk wire 102).any
      (fun row => decide (row.family = "nifs.running_parent_pi_dec")) = true := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf16
