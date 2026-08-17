import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf140 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 140) = true := by
  native_decide

theorem classLeaf141 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 141) = true := by
  native_decide

theorem classLeaf142 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 142) = true := by
  native_decide

theorem classLeaf143 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 143) = true := by
  native_decide

theorem classLeaf144 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 144) = true := by
  native_decide

theorem classLeaf145 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 145) = true := by
  native_decide

theorem classLeaf146 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 146) = true := by
  native_decide

theorem classLeaf147 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 147) = true := by
  native_decide

theorem classLeaf148 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 148) = true := by
  native_decide

theorem classLeaf149 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 149) = true := by
  native_decide

theorem classLeaf150 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 150) = true := by
  native_decide

theorem classLeaf151 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 151) = true := by
  native_decide

theorem classLeaf152 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 152) = true := by
  native_decide

theorem classLeaf153 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 153) = true := by
  native_decide

theorem holdsGroup :
    ∀ k, 140 ≤ k → k < 154 →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds
          (backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values) row.row)) = true := by
  intro k lower upper
  by_cases is140 : k = 140
  · subst is140
    exact (classFacts_split classLeaf140).1
  by_cases is141 : k = 141
  · subst is141
    exact (classFacts_split classLeaf141).1
  by_cases is142 : k = 142
  · subst is142
    exact (classFacts_split classLeaf142).1
  by_cases is143 : k = 143
  · subst is143
    exact (classFacts_split classLeaf143).1
  by_cases is144 : k = 144
  · subst is144
    exact (classFacts_split classLeaf144).1
  by_cases is145 : k = 145
  · subst is145
    exact (classFacts_split classLeaf145).1
  by_cases is146 : k = 146
  · subst is146
    exact (classFacts_split classLeaf146).1
  by_cases is147 : k = 147
  · subst is147
    exact (classFacts_split classLeaf147).1
  by_cases is148 : k = 148
  · subst is148
    exact (classFacts_split classLeaf148).1
  by_cases is149 : k = 149
  · subst is149
    exact (classFacts_split classLeaf149).1
  by_cases is150 : k = 150
  · subst is150
    exact (classFacts_split classLeaf150).1
  by_cases is151 : k = 151
  · subst is151
    exact (classFacts_split classLeaf151).1
  by_cases is152 : k = 152
  · subst is152
    exact (classFacts_split classLeaf152).1
  by_cases is153 : k = 153
  · subst is153
    exact (classFacts_split classLeaf153).1
  exact absurd upper (by omega)

theorem guardsGroup :
    ∀ k, 140 ≤ k → k < 154 →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k lower upper
  by_cases is140 : k = 140
  · subst is140
    exact (classFacts_split classLeaf140).2
  by_cases is141 : k = 141
  · subst is141
    exact (classFacts_split classLeaf141).2
  by_cases is142 : k = 142
  · subst is142
    exact (classFacts_split classLeaf142).2
  by_cases is143 : k = 143
  · subst is143
    exact (classFacts_split classLeaf143).2
  by_cases is144 : k = 144
  · subst is144
    exact (classFacts_split classLeaf144).2
  by_cases is145 : k = 145
  · subst is145
    exact (classFacts_split classLeaf145).2
  by_cases is146 : k = 146
  · subst is146
    exact (classFacts_split classLeaf146).2
  by_cases is147 : k = 147
  · subst is147
    exact (classFacts_split classLeaf147).2
  by_cases is148 : k = 148
  · subst is148
    exact (classFacts_split classLeaf148).2
  by_cases is149 : k = 149
  · subst is149
    exact (classFacts_split classLeaf149).2
  by_cases is150 : k = 150
  · subst is150
    exact (classFacts_split classLeaf150).2
  by_cases is151 : k = 151
  · subst is151
    exact (classFacts_split classLeaf151).2
  by_cases is152 : k = 152
  · subst is152
    exact (classFacts_split classLeaf152).2
  by_cases is153 : k = 153
  · subst is153
    exact (classFacts_split classLeaf153).2
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10
