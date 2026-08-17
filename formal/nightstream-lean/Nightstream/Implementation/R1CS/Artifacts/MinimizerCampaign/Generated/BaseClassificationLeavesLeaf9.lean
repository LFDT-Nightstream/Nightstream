import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf126 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 126) = true := by
  native_decide

theorem classLeaf127 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 127) = true := by
  native_decide

theorem classLeaf128 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 128) = true := by
  native_decide

theorem classLeaf129 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 129) = true := by
  native_decide

theorem classLeaf130 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 130) = true := by
  native_decide

theorem classLeaf131 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 131) = true := by
  native_decide

theorem classLeaf132 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 132) = true := by
  native_decide

theorem classLeaf133 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 133) = true := by
  native_decide

theorem classLeaf134 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 134) = true := by
  native_decide

theorem classLeaf135 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 135) = true := by
  native_decide

theorem classLeaf136 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 136) = true := by
  native_decide

theorem classLeaf137 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 137) = true := by
  native_decide

theorem classLeaf138 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 138) = true := by
  native_decide

theorem classLeaf139 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 139) = true := by
  native_decide

theorem holdsGroup :
    ∀ k, 126 ≤ k → k < 140 →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds
          (backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values) row.row)) = true := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    exact (classFacts_split classLeaf126).1
  by_cases is127 : k = 127
  · subst is127
    exact (classFacts_split classLeaf127).1
  by_cases is128 : k = 128
  · subst is128
    exact (classFacts_split classLeaf128).1
  by_cases is129 : k = 129
  · subst is129
    exact (classFacts_split classLeaf129).1
  by_cases is130 : k = 130
  · subst is130
    exact (classFacts_split classLeaf130).1
  by_cases is131 : k = 131
  · subst is131
    exact (classFacts_split classLeaf131).1
  by_cases is132 : k = 132
  · subst is132
    exact (classFacts_split classLeaf132).1
  by_cases is133 : k = 133
  · subst is133
    exact (classFacts_split classLeaf133).1
  by_cases is134 : k = 134
  · subst is134
    exact (classFacts_split classLeaf134).1
  by_cases is135 : k = 135
  · subst is135
    exact (classFacts_split classLeaf135).1
  by_cases is136 : k = 136
  · subst is136
    exact (classFacts_split classLeaf136).1
  by_cases is137 : k = 137
  · subst is137
    exact (classFacts_split classLeaf137).1
  by_cases is138 : k = 138
  · subst is138
    exact (classFacts_split classLeaf138).1
  by_cases is139 : k = 139
  · subst is139
    exact (classFacts_split classLeaf139).1
  exact absurd upper (by omega)

theorem guardsGroup :
    ∀ k, 126 ≤ k → k < 140 →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k lower upper
  by_cases is126 : k = 126
  · subst is126
    exact (classFacts_split classLeaf126).2
  by_cases is127 : k = 127
  · subst is127
    exact (classFacts_split classLeaf127).2
  by_cases is128 : k = 128
  · subst is128
    exact (classFacts_split classLeaf128).2
  by_cases is129 : k = 129
  · subst is129
    exact (classFacts_split classLeaf129).2
  by_cases is130 : k = 130
  · subst is130
    exact (classFacts_split classLeaf130).2
  by_cases is131 : k = 131
  · subst is131
    exact (classFacts_split classLeaf131).2
  by_cases is132 : k = 132
  · subst is132
    exact (classFacts_split classLeaf132).2
  by_cases is133 : k = 133
  · subst is133
    exact (classFacts_split classLeaf133).2
  by_cases is134 : k = 134
  · subst is134
    exact (classFacts_split classLeaf134).2
  by_cases is135 : k = 135
  · subst is135
    exact (classFacts_split classLeaf135).2
  by_cases is136 : k = 136
  · subst is136
    exact (classFacts_split classLeaf136).2
  by_cases is137 : k = 137
  · subst is137
    exact (classFacts_split classLeaf137).2
  by_cases is138 : k = 138
  · subst is138
    exact (classFacts_split classLeaf138).2
  by_cases is139 : k = 139
  · subst is139
    exact (classFacts_split classLeaf139).2
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9
