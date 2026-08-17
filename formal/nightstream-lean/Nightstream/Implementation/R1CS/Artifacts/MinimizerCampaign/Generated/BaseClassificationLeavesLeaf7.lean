import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf98 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 98) = true := by
  native_decide

theorem classLeaf99 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 99) = true := by
  native_decide

theorem classLeaf100 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 100) = true := by
  native_decide

theorem classLeaf101 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 101) = true := by
  native_decide

theorem classLeaf102 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 102) = true := by
  native_decide

theorem classLeaf103 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 103) = true := by
  native_decide

theorem classLeaf104 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 104) = true := by
  native_decide

theorem classLeaf105 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 105) = true := by
  native_decide

theorem classLeaf106 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 106) = true := by
  native_decide

theorem classLeaf107 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 107) = true := by
  native_decide

theorem classLeaf108 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 108) = true := by
  native_decide

theorem classLeaf109 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 109) = true := by
  native_decide

theorem classLeaf110 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 110) = true := by
  native_decide

theorem classLeaf111 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 111) = true := by
  native_decide

theorem holdsGroup :
    ∀ k, 98 ≤ k → k < 112 →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds
          (backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values) row.row)) = true := by
  intro k lower upper
  by_cases is98 : k = 98
  · subst is98
    exact (classFacts_split classLeaf98).1
  by_cases is99 : k = 99
  · subst is99
    exact (classFacts_split classLeaf99).1
  by_cases is100 : k = 100
  · subst is100
    exact (classFacts_split classLeaf100).1
  by_cases is101 : k = 101
  · subst is101
    exact (classFacts_split classLeaf101).1
  by_cases is102 : k = 102
  · subst is102
    exact (classFacts_split classLeaf102).1
  by_cases is103 : k = 103
  · subst is103
    exact (classFacts_split classLeaf103).1
  by_cases is104 : k = 104
  · subst is104
    exact (classFacts_split classLeaf104).1
  by_cases is105 : k = 105
  · subst is105
    exact (classFacts_split classLeaf105).1
  by_cases is106 : k = 106
  · subst is106
    exact (classFacts_split classLeaf106).1
  by_cases is107 : k = 107
  · subst is107
    exact (classFacts_split classLeaf107).1
  by_cases is108 : k = 108
  · subst is108
    exact (classFacts_split classLeaf108).1
  by_cases is109 : k = 109
  · subst is109
    exact (classFacts_split classLeaf109).1
  by_cases is110 : k = 110
  · subst is110
    exact (classFacts_split classLeaf110).1
  by_cases is111 : k = 111
  · subst is111
    exact (classFacts_split classLeaf111).1
  exact absurd upper (by omega)

theorem guardsGroup :
    ∀ k, 98 ≤ k → k < 112 →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k lower upper
  by_cases is98 : k = 98
  · subst is98
    exact (classFacts_split classLeaf98).2
  by_cases is99 : k = 99
  · subst is99
    exact (classFacts_split classLeaf99).2
  by_cases is100 : k = 100
  · subst is100
    exact (classFacts_split classLeaf100).2
  by_cases is101 : k = 101
  · subst is101
    exact (classFacts_split classLeaf101).2
  by_cases is102 : k = 102
  · subst is102
    exact (classFacts_split classLeaf102).2
  by_cases is103 : k = 103
  · subst is103
    exact (classFacts_split classLeaf103).2
  by_cases is104 : k = 104
  · subst is104
    exact (classFacts_split classLeaf104).2
  by_cases is105 : k = 105
  · subst is105
    exact (classFacts_split classLeaf105).2
  by_cases is106 : k = 106
  · subst is106
    exact (classFacts_split classLeaf106).2
  by_cases is107 : k = 107
  · subst is107
    exact (classFacts_split classLeaf107).2
  by_cases is108 : k = 108
  · subst is108
    exact (classFacts_split classLeaf108).2
  by_cases is109 : k = 109
  · subst is109
    exact (classFacts_split classLeaf109).2
  by_cases is110 : k = 110
  · subst is110
    exact (classFacts_split classLeaf110).2
  by_cases is111 : k = 111
  · subst is111
    exact (classFacts_split classLeaf111).2
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7
