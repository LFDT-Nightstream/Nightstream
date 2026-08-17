import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf2

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf2 :
    chunkFacts (rowsChunk wire 2) 131072 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 2 ≤ k → k < 3 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is2 : k = 2
  · subst is2
    exact (chunkFacts_split chunkLeaf2).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 2 ≤ k → k < 3 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is2 : k = 2
  · subst is2
    exact (chunkFacts_split chunkLeaf2).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 2 ≤ k → k < 3 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is2 : k = 2
  · subst is2
    exact (chunkFacts_split chunkLeaf2).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf2
