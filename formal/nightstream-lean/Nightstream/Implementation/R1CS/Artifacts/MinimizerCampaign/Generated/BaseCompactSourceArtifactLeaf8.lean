import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf112 :
    (chunkFacts (rowsChunk wire 112) 28672 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 112 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk112) := by
  native_decide

theorem chunkLeaf113 :
    (chunkFacts (rowsChunk wire 113) 28928 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 113 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk113) := by
  native_decide

theorem chunkLeaf114 :
    (chunkFacts (rowsChunk wire 114) 29184 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 114 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk114) := by
  native_decide

theorem chunkLeaf115 :
    (chunkFacts (rowsChunk wire 115) 29440 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 115 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk115) := by
  native_decide

theorem chunkLeaf116 :
    (chunkFacts (rowsChunk wire 116) 29696 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 116 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk116) := by
  native_decide

theorem chunkLeaf117 :
    (chunkFacts (rowsChunk wire 117) 29952 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 117 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk117) := by
  native_decide

theorem chunkLeaf118 :
    (chunkFacts (rowsChunk wire 118) 30208 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 118 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk118) := by
  native_decide

theorem chunkLeaf119 :
    (chunkFacts (rowsChunk wire 119) 30464 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 119 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk119) := by
  native_decide

theorem chunkLeaf120 :
    (chunkFacts (rowsChunk wire 120) 30720 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 120 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk120) := by
  native_decide

theorem chunkLeaf121 :
    (chunkFacts (rowsChunk wire 121) 30976 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 121 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk121) := by
  native_decide

theorem chunkLeaf122 :
    (chunkFacts (rowsChunk wire 122) 31232 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 122 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk122) := by
  native_decide

theorem chunkLeaf123 :
    (chunkFacts (rowsChunk wire 123) 31488 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 123 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk123) := by
  native_decide

theorem chunkLeaf124 :
    (chunkFacts (rowsChunk wire 124) 31744 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 124 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk124) := by
  native_decide

theorem chunkLeaf125 :
    (chunkFacts (rowsChunk wire 125) 32000 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 125 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk125) := by
  native_decide

theorem censusGroup :
    ∀ k, 112 ≤ k → k < 126 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is112 : k = 112
  · subst is112
    exact (chunkFacts_split (chunkLeaf112).1).1
  by_cases is113 : k = 113
  · subst is113
    exact (chunkFacts_split (chunkLeaf113).1).1
  by_cases is114 : k = 114
  · subst is114
    exact (chunkFacts_split (chunkLeaf114).1).1
  by_cases is115 : k = 115
  · subst is115
    exact (chunkFacts_split (chunkLeaf115).1).1
  by_cases is116 : k = 116
  · subst is116
    exact (chunkFacts_split (chunkLeaf116).1).1
  by_cases is117 : k = 117
  · subst is117
    exact (chunkFacts_split (chunkLeaf117).1).1
  by_cases is118 : k = 118
  · subst is118
    exact (chunkFacts_split (chunkLeaf118).1).1
  by_cases is119 : k = 119
  · subst is119
    exact (chunkFacts_split (chunkLeaf119).1).1
  by_cases is120 : k = 120
  · subst is120
    exact (chunkFacts_split (chunkLeaf120).1).1
  by_cases is121 : k = 121
  · subst is121
    exact (chunkFacts_split (chunkLeaf121).1).1
  by_cases is122 : k = 122
  · subst is122
    exact (chunkFacts_split (chunkLeaf122).1).1
  by_cases is123 : k = 123
  · subst is123
    exact (chunkFacts_split (chunkLeaf123).1).1
  by_cases is124 : k = 124
  · subst is124
    exact (chunkFacts_split (chunkLeaf124).1).1
  by_cases is125 : k = 125
  · subst is125
    exact (chunkFacts_split (chunkLeaf125).1).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 112 ≤ k → k < 126 →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k lower upper
  by_cases is112 : k = 112
  · subst is112
    exact (chunkFacts_split (chunkLeaf112).1).2.1
  by_cases is113 : k = 113
  · subst is113
    exact (chunkFacts_split (chunkLeaf113).1).2.1
  by_cases is114 : k = 114
  · subst is114
    exact (chunkFacts_split (chunkLeaf114).1).2.1
  by_cases is115 : k = 115
  · subst is115
    exact (chunkFacts_split (chunkLeaf115).1).2.1
  by_cases is116 : k = 116
  · subst is116
    exact (chunkFacts_split (chunkLeaf116).1).2.1
  by_cases is117 : k = 117
  · subst is117
    exact (chunkFacts_split (chunkLeaf117).1).2.1
  by_cases is118 : k = 118
  · subst is118
    exact (chunkFacts_split (chunkLeaf118).1).2.1
  by_cases is119 : k = 119
  · subst is119
    exact (chunkFacts_split (chunkLeaf119).1).2.1
  by_cases is120 : k = 120
  · subst is120
    exact (chunkFacts_split (chunkLeaf120).1).2.1
  by_cases is121 : k = 121
  · subst is121
    exact (chunkFacts_split (chunkLeaf121).1).2.1
  by_cases is122 : k = 122
  · subst is122
    exact (chunkFacts_split (chunkLeaf122).1).2.1
  by_cases is123 : k = 123
  · subst is123
    exact (chunkFacts_split (chunkLeaf123).1).2.1
  by_cases is124 : k = 124
  · subst is124
    exact (chunkFacts_split (chunkLeaf124).1).2.1
  by_cases is125 : k = 125
  · subst is125
    exact (chunkFacts_split (chunkLeaf125).1).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 112 ≤ k → k < 126 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is112 : k = 112
  · subst is112
    exact (chunkFacts_split (chunkLeaf112).1).2.2.1
  by_cases is113 : k = 113
  · subst is113
    exact (chunkFacts_split (chunkLeaf113).1).2.2.1
  by_cases is114 : k = 114
  · subst is114
    exact (chunkFacts_split (chunkLeaf114).1).2.2.1
  by_cases is115 : k = 115
  · subst is115
    exact (chunkFacts_split (chunkLeaf115).1).2.2.1
  by_cases is116 : k = 116
  · subst is116
    exact (chunkFacts_split (chunkLeaf116).1).2.2.1
  by_cases is117 : k = 117
  · subst is117
    exact (chunkFacts_split (chunkLeaf117).1).2.2.1
  by_cases is118 : k = 118
  · subst is118
    exact (chunkFacts_split (chunkLeaf118).1).2.2.1
  by_cases is119 : k = 119
  · subst is119
    exact (chunkFacts_split (chunkLeaf119).1).2.2.1
  by_cases is120 : k = 120
  · subst is120
    exact (chunkFacts_split (chunkLeaf120).1).2.2.1
  by_cases is121 : k = 121
  · subst is121
    exact (chunkFacts_split (chunkLeaf121).1).2.2.1
  by_cases is122 : k = 122
  · subst is122
    exact (chunkFacts_split (chunkLeaf122).1).2.2.1
  by_cases is123 : k = 123
  · subst is123
    exact (chunkFacts_split (chunkLeaf123).1).2.2.1
  by_cases is124 : k = 124
  · subst is124
    exact (chunkFacts_split (chunkLeaf124).1).2.2.1
  by_cases is125 : k = 125
  · subst is125
    exact (chunkFacts_split (chunkLeaf125).1).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf8
