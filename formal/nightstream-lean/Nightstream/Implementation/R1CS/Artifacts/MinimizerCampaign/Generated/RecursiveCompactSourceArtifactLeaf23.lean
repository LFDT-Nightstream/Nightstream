import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

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
    chunkFacts (rowsChunk wire 125) 8192000 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_rlc.verify.identities.commitment.evaluations.inputs",
       "nifs.pi_rlc.verify.identities.commitment.k_products.rho_times_input",
       "nifs.pi_rlc.verify.projection_binding.transcript_beta",
       "nifs.pi_rlc.verify.projection_shared.beta_ladder",
       "nifs.pi_rlc.verify.projection_shared.rho_evaluations"] = true := by
  native_decide

theorem presence48 :
    (rowsChunk wire 125).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.evaluations.inputs")) = true :=
  presence_of_chunkFacts chunkLeaf125 (by decide)

theorem presence53 :
    (rowsChunk wire 125).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.identities.commitment.k_products.rho_times_input")) = true :=
  presence_of_chunkFacts chunkLeaf125 (by decide)

theorem presence77 :
    (rowsChunk wire 125).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.transcript_beta")) = true :=
  presence_of_chunkFacts chunkLeaf125 (by decide)

theorem presence78 :
    (rowsChunk wire 125).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_shared.beta_ladder")) = true :=
  presence_of_chunkFacts chunkLeaf125 (by decide)

theorem presence79 :
    (rowsChunk wire 125).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_shared.rho_evaluations")) = true :=
  presence_of_chunkFacts chunkLeaf125 (by decide)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf23
