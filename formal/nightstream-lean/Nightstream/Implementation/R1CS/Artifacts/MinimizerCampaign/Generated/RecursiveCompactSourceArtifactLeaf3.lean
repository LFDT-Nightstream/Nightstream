import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf3

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf3 :
    chunkFacts (rowsChunk wire 3) 196608 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf4 :
    chunkFacts (rowsChunk wire 4) 262144 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf5 :
    chunkFacts (rowsChunk wire 5) 327680 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf6 :
    chunkFacts (rowsChunk wire 6) 393216 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 3 ≤ k → k < 7 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is3 : k = 3
  · subst is3
    exact (chunkFacts_split chunkLeaf3).1
  by_cases is4 : k = 4
  · subst is4
    exact (chunkFacts_split chunkLeaf4).1
  by_cases is5 : k = 5
  · subst is5
    exact (chunkFacts_split chunkLeaf5).1
  by_cases is6 : k = 6
  · subst is6
    exact (chunkFacts_split chunkLeaf6).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 3 ≤ k → k < 7 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is3 : k = 3
  · subst is3
    exact (chunkFacts_split chunkLeaf3).2.1
  by_cases is4 : k = 4
  · subst is4
    exact (chunkFacts_split chunkLeaf4).2.1
  by_cases is5 : k = 5
  · subst is5
    exact (chunkFacts_split chunkLeaf5).2.1
  by_cases is6 : k = 6
  · subst is6
    exact (chunkFacts_split chunkLeaf6).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 3 ≤ k → k < 7 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is3 : k = 3
  · subst is3
    exact (chunkFacts_split chunkLeaf3).2.2.1
  by_cases is4 : k = 4
  · subst is4
    exact (chunkFacts_split chunkLeaf4).2.2.1
  by_cases is5 : k = 5
  · subst is5
    exact (chunkFacts_split chunkLeaf5).2.2.1
  by_cases is6 : k = 6
  · subst is6
    exact (chunkFacts_split chunkLeaf6).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf3
