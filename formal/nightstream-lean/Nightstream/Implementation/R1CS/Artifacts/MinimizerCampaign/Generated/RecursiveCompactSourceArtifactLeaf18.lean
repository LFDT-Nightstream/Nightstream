import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf106 :
    ((rowsChunk wire 106).map (fun row => row.sourceIndex) =
        List.range' 6946816 65536) ∧
      ((rowsChunk wire 106).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 106).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem presence38 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.shape.allocate_parent_and_children")) = true := by
  native_decide

theorem presence39 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.shape.output_parity")) = true := by
  native_decide

theorem presence40 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.shape.parent")) = true := by
  native_decide

theorem presence41 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.consistency.fold_digest")) = true := by
  native_decide

theorem presence67 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.adv")) = true := by
  native_decide

theorem presence68 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.commitment")) = true := by
  native_decide

theorem presence71 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.domain")) = true := by
  native_decide

theorem presence73 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.commitment")) = true := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18
