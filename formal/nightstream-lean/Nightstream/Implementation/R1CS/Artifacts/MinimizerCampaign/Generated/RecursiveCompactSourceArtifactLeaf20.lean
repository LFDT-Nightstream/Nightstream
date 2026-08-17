import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

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
    chunkFacts (rowsChunk wire 108) 7077888 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_rlc.verify.projection_binding.combined.x",
       "nifs.pi_rlc.verify.projection_binding.combined.y_ring",
       "nifs.pi_rlc.verify.projection_binding.quotient.adv",
       "nifs.pi_rlc.verify.projection_binding.quotient.x",
       "nifs.pi_rlc.verify.projection_binding.quotient.y_ring",
       "nifs.pi_rlc.verify.projection_binding.sis_digest"] = true := by
  native_decide

theorem presence69 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.x")) = true :=
  presence_of_chunkFacts chunkLeaf108 (by decide)

theorem presence70 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.y_ring")) = true :=
  presence_of_chunkFacts chunkLeaf108 (by decide)

theorem presence72 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.adv")) = true :=
  presence_of_chunkFacts chunkLeaf108 (by decide)

theorem presence74 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.x")) = true :=
  presence_of_chunkFacts chunkLeaf108 (by decide)

theorem presence75 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.y_ring")) = true :=
  presence_of_chunkFacts chunkLeaf108 (by decide)

theorem presence76 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.sis_digest")) = true :=
  presence_of_chunkFacts chunkLeaf108 (by decide)

theorem censusGroup :
    ∀ k, 108 ≤ k → k < 109 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is108 : k = 108
  · subst is108
    exact (chunkFacts_split chunkLeaf108).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 108 ≤ k → k < 109 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is108 : k = 108
  · subst is108
    exact (chunkFacts_split chunkLeaf108).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 108 ≤ k → k < 109 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is108 : k = 108
  · subst is108
    exact (chunkFacts_split chunkLeaf108).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf20
