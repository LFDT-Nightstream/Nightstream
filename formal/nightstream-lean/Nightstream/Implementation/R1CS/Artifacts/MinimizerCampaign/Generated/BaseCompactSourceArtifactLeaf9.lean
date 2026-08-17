import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf126 :
    (chunkFacts (rowsChunk wire 126) 32256 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 126 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk126) := by
  native_decide

theorem chunkLeaf127 :
    (chunkFacts (rowsChunk wire 127) 32512 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 127 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk127) := by
  native_decide

theorem chunkLeaf128 :
    (chunkFacts (rowsChunk wire 128) 32768 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 128 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk128) := by
  native_decide

theorem chunkLeaf129 :
    (chunkFacts (rowsChunk wire 129) 33024 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 129 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk129) := by
  native_decide

theorem chunkLeaf130 :
    (chunkFacts (rowsChunk wire 130) 33280 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 130 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk130) := by
  native_decide

theorem chunkLeaf131 :
    (chunkFacts (rowsChunk wire 131) 33536 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 131 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk131) := by
  native_decide

theorem chunkLeaf132 :
    (chunkFacts (rowsChunk wire 132) 33792 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 132 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk132) := by
  native_decide

theorem chunkLeaf133 :
    (chunkFacts (rowsChunk wire 133) 34048 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 133 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk133) := by
  native_decide

theorem chunkLeaf134 :
    (chunkFacts (rowsChunk wire 134) 34304 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 134 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk134) := by
  native_decide

theorem chunkLeaf135 :
    (chunkFacts (rowsChunk wire 135) 34560 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 135 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk135) := by
  native_decide

theorem chunkLeaf136 :
    (chunkFacts (rowsChunk wire 136) 34816 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 136 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk136) := by
  native_decide

theorem chunkLeaf137 :
    (chunkFacts (rowsChunk wire 137) 35072 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 137 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk137) := by
  native_decide

theorem chunkLeaf138 :
    (chunkFacts (rowsChunk wire 138) 35328 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 138 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk138) := by
  native_decide

theorem chunkLeaf139 :
    (chunkFacts (rowsChunk wire 139) 35584 256 39949 38626
      wire.completeFamilies
      [] = true) ∧
      (rowsChunk wire 139 = Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact.sourceArtifactRowsChunk139) := by
  native_decide

theorem censusGroup :
    ∀ k, 126 ≤ k → k < 140 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    exact (chunkFacts_split (chunkLeaf126).1).1
  by_cases is127 : k = 127
  · subst is127
    exact (chunkFacts_split (chunkLeaf127).1).1
  by_cases is128 : k = 128
  · subst is128
    exact (chunkFacts_split (chunkLeaf128).1).1
  by_cases is129 : k = 129
  · subst is129
    exact (chunkFacts_split (chunkLeaf129).1).1
  by_cases is130 : k = 130
  · subst is130
    exact (chunkFacts_split (chunkLeaf130).1).1
  by_cases is131 : k = 131
  · subst is131
    exact (chunkFacts_split (chunkLeaf131).1).1
  by_cases is132 : k = 132
  · subst is132
    exact (chunkFacts_split (chunkLeaf132).1).1
  by_cases is133 : k = 133
  · subst is133
    exact (chunkFacts_split (chunkLeaf133).1).1
  by_cases is134 : k = 134
  · subst is134
    exact (chunkFacts_split (chunkLeaf134).1).1
  by_cases is135 : k = 135
  · subst is135
    exact (chunkFacts_split (chunkLeaf135).1).1
  by_cases is136 : k = 136
  · subst is136
    exact (chunkFacts_split (chunkLeaf136).1).1
  by_cases is137 : k = 137
  · subst is137
    exact (chunkFacts_split (chunkLeaf137).1).1
  by_cases is138 : k = 138
  · subst is138
    exact (chunkFacts_split (chunkLeaf138).1).1
  by_cases is139 : k = 139
  · subst is139
    exact (chunkFacts_split (chunkLeaf139).1).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 126 ≤ k → k < 140 →
      (rowsChunk wire k).all (rowWellFormedAt 39949 38626) = true := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    exact (chunkFacts_split (chunkLeaf126).1).2.1
  by_cases is127 : k = 127
  · subst is127
    exact (chunkFacts_split (chunkLeaf127).1).2.1
  by_cases is128 : k = 128
  · subst is128
    exact (chunkFacts_split (chunkLeaf128).1).2.1
  by_cases is129 : k = 129
  · subst is129
    exact (chunkFacts_split (chunkLeaf129).1).2.1
  by_cases is130 : k = 130
  · subst is130
    exact (chunkFacts_split (chunkLeaf130).1).2.1
  by_cases is131 : k = 131
  · subst is131
    exact (chunkFacts_split (chunkLeaf131).1).2.1
  by_cases is132 : k = 132
  · subst is132
    exact (chunkFacts_split (chunkLeaf132).1).2.1
  by_cases is133 : k = 133
  · subst is133
    exact (chunkFacts_split (chunkLeaf133).1).2.1
  by_cases is134 : k = 134
  · subst is134
    exact (chunkFacts_split (chunkLeaf134).1).2.1
  by_cases is135 : k = 135
  · subst is135
    exact (chunkFacts_split (chunkLeaf135).1).2.1
  by_cases is136 : k = 136
  · subst is136
    exact (chunkFacts_split (chunkLeaf136).1).2.1
  by_cases is137 : k = 137
  · subst is137
    exact (chunkFacts_split (chunkLeaf137).1).2.1
  by_cases is138 : k = 138
  · subst is138
    exact (chunkFacts_split (chunkLeaf138).1).2.1
  by_cases is139 : k = 139
  · subst is139
    exact (chunkFacts_split (chunkLeaf139).1).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 126 ≤ k → k < 140 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    exact (chunkFacts_split (chunkLeaf126).1).2.2.1
  by_cases is127 : k = 127
  · subst is127
    exact (chunkFacts_split (chunkLeaf127).1).2.2.1
  by_cases is128 : k = 128
  · subst is128
    exact (chunkFacts_split (chunkLeaf128).1).2.2.1
  by_cases is129 : k = 129
  · subst is129
    exact (chunkFacts_split (chunkLeaf129).1).2.2.1
  by_cases is130 : k = 130
  · subst is130
    exact (chunkFacts_split (chunkLeaf130).1).2.2.1
  by_cases is131 : k = 131
  · subst is131
    exact (chunkFacts_split (chunkLeaf131).1).2.2.1
  by_cases is132 : k = 132
  · subst is132
    exact (chunkFacts_split (chunkLeaf132).1).2.2.1
  by_cases is133 : k = 133
  · subst is133
    exact (chunkFacts_split (chunkLeaf133).1).2.2.1
  by_cases is134 : k = 134
  · subst is134
    exact (chunkFacts_split (chunkLeaf134).1).2.2.1
  by_cases is135 : k = 135
  · subst is135
    exact (chunkFacts_split (chunkLeaf135).1).2.2.1
  by_cases is136 : k = 136
  · subst is136
    exact (chunkFacts_split (chunkLeaf136).1).2.2.1
  by_cases is137 : k = 137
  · subst is137
    exact (chunkFacts_split (chunkLeaf137).1).2.2.1
  by_cases is138 : k = 138
  · subst is138
    exact (chunkFacts_split (chunkLeaf138).1).2.2.1
  by_cases is139 : k = 139
  · subst is139
    exact (chunkFacts_split (chunkLeaf139).1).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactLeaf9
