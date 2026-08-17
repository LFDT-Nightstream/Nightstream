import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf108 :
    ((rowsChunk wire 108).map (fun row => row.sourceIndex) =
        List.range' 7077888 65536) ∧
      ((rowsChunk wire 108).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 108).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem presence69 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.x")) = true := by
  native_decide

theorem presence70 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.y_ring")) = true := by
  native_decide

theorem presence72 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.adv")) = true := by
  native_decide

theorem presence74 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.x")) = true := by
  native_decide

theorem presence75 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.y_ring")) = true := by
  native_decide

theorem presence76 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.sis_digest")) = true := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20
