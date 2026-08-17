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

theorem censusGroup :
    ∀ k, 57 ≤ k → k < 58 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is57 : k = 57
  · subst is57
    exact (chunkFacts_split chunkLeaf57).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 57 ≤ k → k < 58 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is57 : k = 57
  · subst is57
    exact (chunkFacts_split chunkLeaf57).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 57 ≤ k → k < 58 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is57 : k = 57
  · subst is57
    exact (chunkFacts_split chunkLeaf57).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf11
