import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf125 :
    ((rowsChunk wire 125).map (fun row => row.sourceIndex) =
        List.range' 8192000 65536) ∧
      ((rowsChunk wire 125).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 125).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem presence48 :
    (rowsChunk wire 125).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.evaluations.inputs")) = true := by
  native_decide

theorem presence53 :
    (rowsChunk wire 125).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.k_products.rho_times_input")) = true := by
  native_decide

theorem presence77 :
    (rowsChunk wire 125).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.transcript_beta")) = true := by
  native_decide

theorem presence78 :
    (rowsChunk wire 125).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_shared.beta_ladder")) = true := by
  native_decide

theorem presence79 :
    (rowsChunk wire 125).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_shared.rho_evaluations")) = true := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23
