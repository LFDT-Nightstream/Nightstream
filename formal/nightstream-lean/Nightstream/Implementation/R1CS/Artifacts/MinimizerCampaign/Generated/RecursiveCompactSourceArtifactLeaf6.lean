import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf6

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf9 :
    chunkFacts (rowsChunk wire 9) 589824 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf10 :
    chunkFacts (rowsChunk wire 10) 655360 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf11 :
    chunkFacts (rowsChunk wire 11) 720896 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf12 :
    chunkFacts (rowsChunk wire 12) 786432 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf13 :
    chunkFacts (rowsChunk wire 13) 851968 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf14 :
    chunkFacts (rowsChunk wire 14) 917504 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf15 :
    chunkFacts (rowsChunk wire 15) 983040 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf16 :
    chunkFacts (rowsChunk wire 16) 1048576 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 9 ≤ k → k < 17 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is9 : k = 9
  · subst is9
    exact (chunkFacts_split chunkLeaf9).1
  by_cases is10 : k = 10
  · subst is10
    exact (chunkFacts_split chunkLeaf10).1
  by_cases is11 : k = 11
  · subst is11
    exact (chunkFacts_split chunkLeaf11).1
  by_cases is12 : k = 12
  · subst is12
    exact (chunkFacts_split chunkLeaf12).1
  by_cases is13 : k = 13
  · subst is13
    exact (chunkFacts_split chunkLeaf13).1
  by_cases is14 : k = 14
  · subst is14
    exact (chunkFacts_split chunkLeaf14).1
  by_cases is15 : k = 15
  · subst is15
    exact (chunkFacts_split chunkLeaf15).1
  by_cases is16 : k = 16
  · subst is16
    exact (chunkFacts_split chunkLeaf16).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 9 ≤ k → k < 17 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is9 : k = 9
  · subst is9
    exact (chunkFacts_split chunkLeaf9).2.1
  by_cases is10 : k = 10
  · subst is10
    exact (chunkFacts_split chunkLeaf10).2.1
  by_cases is11 : k = 11
  · subst is11
    exact (chunkFacts_split chunkLeaf11).2.1
  by_cases is12 : k = 12
  · subst is12
    exact (chunkFacts_split chunkLeaf12).2.1
  by_cases is13 : k = 13
  · subst is13
    exact (chunkFacts_split chunkLeaf13).2.1
  by_cases is14 : k = 14
  · subst is14
    exact (chunkFacts_split chunkLeaf14).2.1
  by_cases is15 : k = 15
  · subst is15
    exact (chunkFacts_split chunkLeaf15).2.1
  by_cases is16 : k = 16
  · subst is16
    exact (chunkFacts_split chunkLeaf16).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 9 ≤ k → k < 17 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is9 : k = 9
  · subst is9
    exact (chunkFacts_split chunkLeaf9).2.2.1
  by_cases is10 : k = 10
  · subst is10
    exact (chunkFacts_split chunkLeaf10).2.2.1
  by_cases is11 : k = 11
  · subst is11
    exact (chunkFacts_split chunkLeaf11).2.2.1
  by_cases is12 : k = 12
  · subst is12
    exact (chunkFacts_split chunkLeaf12).2.2.1
  by_cases is13 : k = 13
  · subst is13
    exact (chunkFacts_split chunkLeaf13).2.2.1
  by_cases is14 : k = 14
  · subst is14
    exact (chunkFacts_split chunkLeaf14).2.2.1
  by_cases is15 : k = 15
  · subst is15
    exact (chunkFacts_split chunkLeaf15).2.2.1
  by_cases is16 : k = 16
  · subst is16
    exact (chunkFacts_split chunkLeaf16).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf6
