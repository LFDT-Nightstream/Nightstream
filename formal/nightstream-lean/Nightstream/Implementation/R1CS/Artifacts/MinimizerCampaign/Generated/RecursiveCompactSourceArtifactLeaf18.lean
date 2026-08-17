import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

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
    chunkFacts (rowsChunk wire 106) 6946816 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_rlc.shape.allocate_parent_and_children",
       "nifs.pi_rlc.shape.output_parity",
       "nifs.pi_rlc.shape.parent",
       "nifs.pi_rlc.verify.consistency.fold_digest",
       "nifs.pi_rlc.verify.projection_binding.combined.adv",
       "nifs.pi_rlc.verify.projection_binding.combined.commitment",
       "nifs.pi_rlc.verify.projection_binding.domain",
       "nifs.pi_rlc.verify.projection_binding.quotient.commitment"] = true := by
  native_decide

theorem presence38 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.shape.allocate_parent_and_children")) = true :=
  presence_of_chunkFacts chunkLeaf106 (by decide)

theorem presence39 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.shape.output_parity")) = true :=
  presence_of_chunkFacts chunkLeaf106 (by decide)

theorem presence40 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.shape.parent")) = true :=
  presence_of_chunkFacts chunkLeaf106 (by decide)

theorem presence41 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.consistency.fold_digest")) = true :=
  presence_of_chunkFacts chunkLeaf106 (by decide)

theorem presence67 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.adv")) = true :=
  presence_of_chunkFacts chunkLeaf106 (by decide)

theorem presence68 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.commitment")) = true :=
  presence_of_chunkFacts chunkLeaf106 (by decide)

theorem presence71 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.domain")) = true :=
  presence_of_chunkFacts chunkLeaf106 (by decide)

theorem presence73 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.commitment")) = true :=
  presence_of_chunkFacts chunkLeaf106 (by decide)

theorem censusGroup :
    ∀ k, 106 ≤ k → k < 107 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is106 : k = 106
  · subst is106
    exact (chunkFacts_split chunkLeaf106).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 106 ≤ k → k < 107 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is106 : k = 106
  · subst is106
    exact (chunkFacts_split chunkLeaf106).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 106 ≤ k → k < 107 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is106 : k = 106
  · subst is106
    exact (chunkFacts_split chunkLeaf106).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf18
