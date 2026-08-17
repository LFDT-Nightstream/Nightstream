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

theorem censusGroup :
    ∀ k, 102 ≤ k → k < 103 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is102 : k = 102
  · subst is102
    exact (chunkFacts_split chunkLeaf102).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 102 ≤ k → k < 103 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is102 : k = 102
  · subst is102
    exact (chunkFacts_split chunkLeaf102).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 102 ≤ k → k < 103 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is102 : k = 102
  · subst is102
    exact (chunkFacts_split chunkLeaf102).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf16
