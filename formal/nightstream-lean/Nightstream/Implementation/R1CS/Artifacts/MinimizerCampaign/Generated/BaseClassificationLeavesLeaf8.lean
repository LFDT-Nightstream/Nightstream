import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf112 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 112) = true := by
  native_decide

theorem classLeaf113 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 113) = true := by
  native_decide

theorem classLeaf114 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 114) = true := by
  native_decide

theorem classLeaf115 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 115) = true := by
  native_decide

theorem classLeaf116 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 116) = true := by
  native_decide

theorem classLeaf117 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 117) = true := by
  native_decide

theorem classLeaf118 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 118) = true := by
  native_decide

theorem classLeaf119 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 119) = true := by
  native_decide

theorem classLeaf120 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 120) = true := by
  native_decide

theorem classLeaf121 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 121) = true := by
  native_decide

theorem classLeaf122 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 122) = true := by
  native_decide

theorem classLeaf123 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 123) = true := by
  native_decide

theorem classLeaf124 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 124) = true := by
  native_decide

theorem classLeaf125 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 125) = true := by
  native_decide

theorem holdsGroup :
    ∀ k, 112 ≤ k → k < 126 →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds
          (backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values) row.row)) = true := by
  intro k lower upper
  by_cases is112 : k = 112
  · subst is112
    exact (classFacts_split classLeaf112).1
  by_cases is113 : k = 113
  · subst is113
    exact (classFacts_split classLeaf113).1
  by_cases is114 : k = 114
  · subst is114
    exact (classFacts_split classLeaf114).1
  by_cases is115 : k = 115
  · subst is115
    exact (classFacts_split classLeaf115).1
  by_cases is116 : k = 116
  · subst is116
    exact (classFacts_split classLeaf116).1
  by_cases is117 : k = 117
  · subst is117
    exact (classFacts_split classLeaf117).1
  by_cases is118 : k = 118
  · subst is118
    exact (classFacts_split classLeaf118).1
  by_cases is119 : k = 119
  · subst is119
    exact (classFacts_split classLeaf119).1
  by_cases is120 : k = 120
  · subst is120
    exact (classFacts_split classLeaf120).1
  by_cases is121 : k = 121
  · subst is121
    exact (classFacts_split classLeaf121).1
  by_cases is122 : k = 122
  · subst is122
    exact (classFacts_split classLeaf122).1
  by_cases is123 : k = 123
  · subst is123
    exact (classFacts_split classLeaf123).1
  by_cases is124 : k = 124
  · subst is124
    exact (classFacts_split classLeaf124).1
  by_cases is125 : k = 125
  · subst is125
    exact (classFacts_split classLeaf125).1
  exact absurd upper (by omega)

theorem guardsGroup :
    ∀ k, 112 ≤ k → k < 126 →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k lower upper
  by_cases is112 : k = 112
  · subst is112
    exact (classFacts_split classLeaf112).2
  by_cases is113 : k = 113
  · subst is113
    exact (classFacts_split classLeaf113).2
  by_cases is114 : k = 114
  · subst is114
    exact (classFacts_split classLeaf114).2
  by_cases is115 : k = 115
  · subst is115
    exact (classFacts_split classLeaf115).2
  by_cases is116 : k = 116
  · subst is116
    exact (classFacts_split classLeaf116).2
  by_cases is117 : k = 117
  · subst is117
    exact (classFacts_split classLeaf117).2
  by_cases is118 : k = 118
  · subst is118
    exact (classFacts_split classLeaf118).2
  by_cases is119 : k = 119
  · subst is119
    exact (classFacts_split classLeaf119).2
  by_cases is120 : k = 120
  · subst is120
    exact (classFacts_split classLeaf120).2
  by_cases is121 : k = 121
  · subst is121
    exact (classFacts_split classLeaf121).2
  by_cases is122 : k = 122
  · subst is122
    exact (classFacts_split classLeaf122).2
  by_cases is123 : k = 123
  · subst is123
    exact (classFacts_split classLeaf123).2
  by_cases is124 : k = 124
  · subst is124
    exact (classFacts_split classLeaf124).2
  by_cases is125 : k = 125
  · subst is125
    exact (classFacts_split classLeaf125).2
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8
