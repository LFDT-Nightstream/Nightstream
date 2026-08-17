import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf21

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf109 :
    chunkFacts (rowsChunk wire 109) 7143424 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf110 :
    chunkFacts (rowsChunk wire 110) 7208960 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf111 :
    chunkFacts (rowsChunk wire 111) 7274496 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf112 :
    chunkFacts (rowsChunk wire 112) 7340032 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf113 :
    chunkFacts (rowsChunk wire 113) 7405568 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf114 :
    chunkFacts (rowsChunk wire 114) 7471104 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf115 :
    chunkFacts (rowsChunk wire 115) 7536640 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf116 :
    chunkFacts (rowsChunk wire 116) 7602176 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf117 :
    chunkFacts (rowsChunk wire 117) 7667712 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf118 :
    chunkFacts (rowsChunk wire 118) 7733248 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf119 :
    chunkFacts (rowsChunk wire 119) 7798784 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf120 :
    chunkFacts (rowsChunk wire 120) 7864320 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf121 :
    chunkFacts (rowsChunk wire 121) 7929856 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf122 :
    chunkFacts (rowsChunk wire 122) 7995392 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem censusGroup :
    ∀ k, 109 ≤ k → k < 123 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is109 : k = 109
  · subst is109
    exact (chunkFacts_split chunkLeaf109).1
  by_cases is110 : k = 110
  · subst is110
    exact (chunkFacts_split chunkLeaf110).1
  by_cases is111 : k = 111
  · subst is111
    exact (chunkFacts_split chunkLeaf111).1
  by_cases is112 : k = 112
  · subst is112
    exact (chunkFacts_split chunkLeaf112).1
  by_cases is113 : k = 113
  · subst is113
    exact (chunkFacts_split chunkLeaf113).1
  by_cases is114 : k = 114
  · subst is114
    exact (chunkFacts_split chunkLeaf114).1
  by_cases is115 : k = 115
  · subst is115
    exact (chunkFacts_split chunkLeaf115).1
  by_cases is116 : k = 116
  · subst is116
    exact (chunkFacts_split chunkLeaf116).1
  by_cases is117 : k = 117
  · subst is117
    exact (chunkFacts_split chunkLeaf117).1
  by_cases is118 : k = 118
  · subst is118
    exact (chunkFacts_split chunkLeaf118).1
  by_cases is119 : k = 119
  · subst is119
    exact (chunkFacts_split chunkLeaf119).1
  by_cases is120 : k = 120
  · subst is120
    exact (chunkFacts_split chunkLeaf120).1
  by_cases is121 : k = 121
  · subst is121
    exact (chunkFacts_split chunkLeaf121).1
  by_cases is122 : k = 122
  · subst is122
    exact (chunkFacts_split chunkLeaf122).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 109 ≤ k → k < 123 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is109 : k = 109
  · subst is109
    exact (chunkFacts_split chunkLeaf109).2.1
  by_cases is110 : k = 110
  · subst is110
    exact (chunkFacts_split chunkLeaf110).2.1
  by_cases is111 : k = 111
  · subst is111
    exact (chunkFacts_split chunkLeaf111).2.1
  by_cases is112 : k = 112
  · subst is112
    exact (chunkFacts_split chunkLeaf112).2.1
  by_cases is113 : k = 113
  · subst is113
    exact (chunkFacts_split chunkLeaf113).2.1
  by_cases is114 : k = 114
  · subst is114
    exact (chunkFacts_split chunkLeaf114).2.1
  by_cases is115 : k = 115
  · subst is115
    exact (chunkFacts_split chunkLeaf115).2.1
  by_cases is116 : k = 116
  · subst is116
    exact (chunkFacts_split chunkLeaf116).2.1
  by_cases is117 : k = 117
  · subst is117
    exact (chunkFacts_split chunkLeaf117).2.1
  by_cases is118 : k = 118
  · subst is118
    exact (chunkFacts_split chunkLeaf118).2.1
  by_cases is119 : k = 119
  · subst is119
    exact (chunkFacts_split chunkLeaf119).2.1
  by_cases is120 : k = 120
  · subst is120
    exact (chunkFacts_split chunkLeaf120).2.1
  by_cases is121 : k = 121
  · subst is121
    exact (chunkFacts_split chunkLeaf121).2.1
  by_cases is122 : k = 122
  · subst is122
    exact (chunkFacts_split chunkLeaf122).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 109 ≤ k → k < 123 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is109 : k = 109
  · subst is109
    exact (chunkFacts_split chunkLeaf109).2.2.1
  by_cases is110 : k = 110
  · subst is110
    exact (chunkFacts_split chunkLeaf110).2.2.1
  by_cases is111 : k = 111
  · subst is111
    exact (chunkFacts_split chunkLeaf111).2.2.1
  by_cases is112 : k = 112
  · subst is112
    exact (chunkFacts_split chunkLeaf112).2.2.1
  by_cases is113 : k = 113
  · subst is113
    exact (chunkFacts_split chunkLeaf113).2.2.1
  by_cases is114 : k = 114
  · subst is114
    exact (chunkFacts_split chunkLeaf114).2.2.1
  by_cases is115 : k = 115
  · subst is115
    exact (chunkFacts_split chunkLeaf115).2.2.1
  by_cases is116 : k = 116
  · subst is116
    exact (chunkFacts_split chunkLeaf116).2.2.1
  by_cases is117 : k = 117
  · subst is117
    exact (chunkFacts_split chunkLeaf117).2.2.1
  by_cases is118 : k = 118
  · subst is118
    exact (chunkFacts_split chunkLeaf118).2.2.1
  by_cases is119 : k = 119
  · subst is119
    exact (chunkFacts_split chunkLeaf119).2.2.1
  by_cases is120 : k = 120
  · subst is120
    exact (chunkFacts_split chunkLeaf120).2.2.1
  by_cases is121 : k = 121
  · subst is121
    exact (chunkFacts_split chunkLeaf121).2.2.1
  by_cases is122 : k = 122
  · subst is122
    exact (chunkFacts_split chunkLeaf122).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf21
