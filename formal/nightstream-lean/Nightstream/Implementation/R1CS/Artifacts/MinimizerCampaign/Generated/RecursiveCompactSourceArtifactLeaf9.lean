import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf9

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf32 :
    chunkFacts (rowsChunk wire 32) 2097152 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf33 :
    chunkFacts (rowsChunk wire 33) 2162688 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf34 :
    chunkFacts (rowsChunk wire 34) 2228224 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf35 :
    chunkFacts (rowsChunk wire 35) 2293760 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf36 :
    chunkFacts (rowsChunk wire 36) 2359296 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf37 :
    chunkFacts (rowsChunk wire 37) 2424832 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf38 :
    chunkFacts (rowsChunk wire 38) 2490368 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf39 :
    chunkFacts (rowsChunk wire 39) 2555904 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf40 :
    chunkFacts (rowsChunk wire 40) 2621440 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf41 :
    chunkFacts (rowsChunk wire 41) 2686976 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf42 :
    chunkFacts (rowsChunk wire 42) 2752512 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf43 :
    chunkFacts (rowsChunk wire 43) 2818048 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf44 :
    chunkFacts (rowsChunk wire 44) 2883584 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf45 :
    chunkFacts (rowsChunk wire 45) 2949120 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 32 ≤ k → k < 46 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is32 : k = 32
  · subst is32
    exact (chunkFacts_split chunkLeaf32).1
  by_cases is33 : k = 33
  · subst is33
    exact (chunkFacts_split chunkLeaf33).1
  by_cases is34 : k = 34
  · subst is34
    exact (chunkFacts_split chunkLeaf34).1
  by_cases is35 : k = 35
  · subst is35
    exact (chunkFacts_split chunkLeaf35).1
  by_cases is36 : k = 36
  · subst is36
    exact (chunkFacts_split chunkLeaf36).1
  by_cases is37 : k = 37
  · subst is37
    exact (chunkFacts_split chunkLeaf37).1
  by_cases is38 : k = 38
  · subst is38
    exact (chunkFacts_split chunkLeaf38).1
  by_cases is39 : k = 39
  · subst is39
    exact (chunkFacts_split chunkLeaf39).1
  by_cases is40 : k = 40
  · subst is40
    exact (chunkFacts_split chunkLeaf40).1
  by_cases is41 : k = 41
  · subst is41
    exact (chunkFacts_split chunkLeaf41).1
  by_cases is42 : k = 42
  · subst is42
    exact (chunkFacts_split chunkLeaf42).1
  by_cases is43 : k = 43
  · subst is43
    exact (chunkFacts_split chunkLeaf43).1
  by_cases is44 : k = 44
  · subst is44
    exact (chunkFacts_split chunkLeaf44).1
  by_cases is45 : k = 45
  · subst is45
    exact (chunkFacts_split chunkLeaf45).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 32 ≤ k → k < 46 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is32 : k = 32
  · subst is32
    exact (chunkFacts_split chunkLeaf32).2.1
  by_cases is33 : k = 33
  · subst is33
    exact (chunkFacts_split chunkLeaf33).2.1
  by_cases is34 : k = 34
  · subst is34
    exact (chunkFacts_split chunkLeaf34).2.1
  by_cases is35 : k = 35
  · subst is35
    exact (chunkFacts_split chunkLeaf35).2.1
  by_cases is36 : k = 36
  · subst is36
    exact (chunkFacts_split chunkLeaf36).2.1
  by_cases is37 : k = 37
  · subst is37
    exact (chunkFacts_split chunkLeaf37).2.1
  by_cases is38 : k = 38
  · subst is38
    exact (chunkFacts_split chunkLeaf38).2.1
  by_cases is39 : k = 39
  · subst is39
    exact (chunkFacts_split chunkLeaf39).2.1
  by_cases is40 : k = 40
  · subst is40
    exact (chunkFacts_split chunkLeaf40).2.1
  by_cases is41 : k = 41
  · subst is41
    exact (chunkFacts_split chunkLeaf41).2.1
  by_cases is42 : k = 42
  · subst is42
    exact (chunkFacts_split chunkLeaf42).2.1
  by_cases is43 : k = 43
  · subst is43
    exact (chunkFacts_split chunkLeaf43).2.1
  by_cases is44 : k = 44
  · subst is44
    exact (chunkFacts_split chunkLeaf44).2.1
  by_cases is45 : k = 45
  · subst is45
    exact (chunkFacts_split chunkLeaf45).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 32 ≤ k → k < 46 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is32 : k = 32
  · subst is32
    exact (chunkFacts_split chunkLeaf32).2.2.1
  by_cases is33 : k = 33
  · subst is33
    exact (chunkFacts_split chunkLeaf33).2.2.1
  by_cases is34 : k = 34
  · subst is34
    exact (chunkFacts_split chunkLeaf34).2.2.1
  by_cases is35 : k = 35
  · subst is35
    exact (chunkFacts_split chunkLeaf35).2.2.1
  by_cases is36 : k = 36
  · subst is36
    exact (chunkFacts_split chunkLeaf36).2.2.1
  by_cases is37 : k = 37
  · subst is37
    exact (chunkFacts_split chunkLeaf37).2.2.1
  by_cases is38 : k = 38
  · subst is38
    exact (chunkFacts_split chunkLeaf38).2.2.1
  by_cases is39 : k = 39
  · subst is39
    exact (chunkFacts_split chunkLeaf39).2.2.1
  by_cases is40 : k = 40
  · subst is40
    exact (chunkFacts_split chunkLeaf40).2.2.1
  by_cases is41 : k = 41
  · subst is41
    exact (chunkFacts_split chunkLeaf41).2.2.1
  by_cases is42 : k = 42
  · subst is42
    exact (chunkFacts_split chunkLeaf42).2.2.1
  by_cases is43 : k = 43
  · subst is43
    exact (chunkFacts_split chunkLeaf43).2.2.1
  by_cases is44 : k = 44
  · subst is44
    exact (chunkFacts_split chunkLeaf44).2.2.1
  by_cases is45 : k = 45
  · subst is45
    exact (chunkFacts_split chunkLeaf45).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf9
