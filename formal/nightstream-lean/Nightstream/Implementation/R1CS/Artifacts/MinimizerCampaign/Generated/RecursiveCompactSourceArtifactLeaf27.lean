import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf27

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf131 :
    chunkFacts (rowsChunk wire 131) 8585216 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf132 :
    chunkFacts (rowsChunk wire 132) 8650752 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf133 :
    chunkFacts (rowsChunk wire 133) 8716288 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf134 :
    chunkFacts (rowsChunk wire 134) 8781824 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf135 :
    chunkFacts (rowsChunk wire 135) 8847360 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf136 :
    chunkFacts (rowsChunk wire 136) 8912896 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf137 :
    chunkFacts (rowsChunk wire 137) 8978432 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf138 :
    chunkFacts (rowsChunk wire 138) 9043968 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf139 :
    chunkFacts (rowsChunk wire 139) 9109504 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf140 :
    chunkFacts (rowsChunk wire 140) 9175040 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf141 :
    chunkFacts (rowsChunk wire 141) 9240576 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf142 :
    chunkFacts (rowsChunk wire 142) 9306112 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf143 :
    chunkFacts (rowsChunk wire 143) 9371648 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf144 :
    chunkFacts (rowsChunk wire 144) 9437184 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 131 ≤ k → k < 145 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is131 : k = 131
  · subst is131
    exact (chunkFacts_split chunkLeaf131).1
  by_cases is132 : k = 132
  · subst is132
    exact (chunkFacts_split chunkLeaf132).1
  by_cases is133 : k = 133
  · subst is133
    exact (chunkFacts_split chunkLeaf133).1
  by_cases is134 : k = 134
  · subst is134
    exact (chunkFacts_split chunkLeaf134).1
  by_cases is135 : k = 135
  · subst is135
    exact (chunkFacts_split chunkLeaf135).1
  by_cases is136 : k = 136
  · subst is136
    exact (chunkFacts_split chunkLeaf136).1
  by_cases is137 : k = 137
  · subst is137
    exact (chunkFacts_split chunkLeaf137).1
  by_cases is138 : k = 138
  · subst is138
    exact (chunkFacts_split chunkLeaf138).1
  by_cases is139 : k = 139
  · subst is139
    exact (chunkFacts_split chunkLeaf139).1
  by_cases is140 : k = 140
  · subst is140
    exact (chunkFacts_split chunkLeaf140).1
  by_cases is141 : k = 141
  · subst is141
    exact (chunkFacts_split chunkLeaf141).1
  by_cases is142 : k = 142
  · subst is142
    exact (chunkFacts_split chunkLeaf142).1
  by_cases is143 : k = 143
  · subst is143
    exact (chunkFacts_split chunkLeaf143).1
  by_cases is144 : k = 144
  · subst is144
    exact (chunkFacts_split chunkLeaf144).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 131 ≤ k → k < 145 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is131 : k = 131
  · subst is131
    exact (chunkFacts_split chunkLeaf131).2.1
  by_cases is132 : k = 132
  · subst is132
    exact (chunkFacts_split chunkLeaf132).2.1
  by_cases is133 : k = 133
  · subst is133
    exact (chunkFacts_split chunkLeaf133).2.1
  by_cases is134 : k = 134
  · subst is134
    exact (chunkFacts_split chunkLeaf134).2.1
  by_cases is135 : k = 135
  · subst is135
    exact (chunkFacts_split chunkLeaf135).2.1
  by_cases is136 : k = 136
  · subst is136
    exact (chunkFacts_split chunkLeaf136).2.1
  by_cases is137 : k = 137
  · subst is137
    exact (chunkFacts_split chunkLeaf137).2.1
  by_cases is138 : k = 138
  · subst is138
    exact (chunkFacts_split chunkLeaf138).2.1
  by_cases is139 : k = 139
  · subst is139
    exact (chunkFacts_split chunkLeaf139).2.1
  by_cases is140 : k = 140
  · subst is140
    exact (chunkFacts_split chunkLeaf140).2.1
  by_cases is141 : k = 141
  · subst is141
    exact (chunkFacts_split chunkLeaf141).2.1
  by_cases is142 : k = 142
  · subst is142
    exact (chunkFacts_split chunkLeaf142).2.1
  by_cases is143 : k = 143
  · subst is143
    exact (chunkFacts_split chunkLeaf143).2.1
  by_cases is144 : k = 144
  · subst is144
    exact (chunkFacts_split chunkLeaf144).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 131 ≤ k → k < 145 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is131 : k = 131
  · subst is131
    exact (chunkFacts_split chunkLeaf131).2.2.1
  by_cases is132 : k = 132
  · subst is132
    exact (chunkFacts_split chunkLeaf132).2.2.1
  by_cases is133 : k = 133
  · subst is133
    exact (chunkFacts_split chunkLeaf133).2.2.1
  by_cases is134 : k = 134
  · subst is134
    exact (chunkFacts_split chunkLeaf134).2.2.1
  by_cases is135 : k = 135
  · subst is135
    exact (chunkFacts_split chunkLeaf135).2.2.1
  by_cases is136 : k = 136
  · subst is136
    exact (chunkFacts_split chunkLeaf136).2.2.1
  by_cases is137 : k = 137
  · subst is137
    exact (chunkFacts_split chunkLeaf137).2.2.1
  by_cases is138 : k = 138
  · subst is138
    exact (chunkFacts_split chunkLeaf138).2.2.1
  by_cases is139 : k = 139
  · subst is139
    exact (chunkFacts_split chunkLeaf139).2.2.1
  by_cases is140 : k = 140
  · subst is140
    exact (chunkFacts_split chunkLeaf140).2.2.1
  by_cases is141 : k = 141
  · subst is141
    exact (chunkFacts_split chunkLeaf141).2.2.1
  by_cases is142 : k = 142
  · subst is142
    exact (chunkFacts_split chunkLeaf142).2.2.1
  by_cases is143 : k = 143
  · subst is143
    exact (chunkFacts_split chunkLeaf143).2.2.1
  by_cases is144 : k = 144
  · subst is144
    exact (chunkFacts_split chunkLeaf144).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf27
