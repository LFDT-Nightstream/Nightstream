import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf70 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 70) = true := by
  native_decide

theorem classLeaf71 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 71) = true := by
  native_decide

theorem classLeaf72 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 72) = true := by
  native_decide

theorem classLeaf73 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 73) = true := by
  native_decide

theorem classLeaf74 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 74) = true := by
  native_decide

theorem classLeaf75 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 75) = true := by
  native_decide

theorem classLeaf76 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 76) = true := by
  native_decide

theorem classLeaf77 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 77) = true := by
  native_decide

theorem classLeaf78 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 78) = true := by
  native_decide

theorem classLeaf79 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 79) = true := by
  native_decide

theorem classLeaf80 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 80) = true := by
  native_decide

theorem classLeaf81 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 81) = true := by
  native_decide

theorem classLeaf82 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 82) = true := by
  native_decide

theorem classLeaf83 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 83) = true := by
  native_decide

theorem holdsGroup :
    ∀ k, 70 ≤ k → k < 84 →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds
          (backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values) row.row)) = true := by
  intro k lower upper
  by_cases is70 : k = 70
  · subst is70
    exact (classFacts_split classLeaf70).1
  by_cases is71 : k = 71
  · subst is71
    exact (classFacts_split classLeaf71).1
  by_cases is72 : k = 72
  · subst is72
    exact (classFacts_split classLeaf72).1
  by_cases is73 : k = 73
  · subst is73
    exact (classFacts_split classLeaf73).1
  by_cases is74 : k = 74
  · subst is74
    exact (classFacts_split classLeaf74).1
  by_cases is75 : k = 75
  · subst is75
    exact (classFacts_split classLeaf75).1
  by_cases is76 : k = 76
  · subst is76
    exact (classFacts_split classLeaf76).1
  by_cases is77 : k = 77
  · subst is77
    exact (classFacts_split classLeaf77).1
  by_cases is78 : k = 78
  · subst is78
    exact (classFacts_split classLeaf78).1
  by_cases is79 : k = 79
  · subst is79
    exact (classFacts_split classLeaf79).1
  by_cases is80 : k = 80
  · subst is80
    exact (classFacts_split classLeaf80).1
  by_cases is81 : k = 81
  · subst is81
    exact (classFacts_split classLeaf81).1
  by_cases is82 : k = 82
  · subst is82
    exact (classFacts_split classLeaf82).1
  by_cases is83 : k = 83
  · subst is83
    exact (classFacts_split classLeaf83).1
  exact absurd upper (by omega)

theorem guardsGroup :
    ∀ k, 70 ≤ k → k < 84 →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k lower upper
  by_cases is70 : k = 70
  · subst is70
    exact (classFacts_split classLeaf70).2
  by_cases is71 : k = 71
  · subst is71
    exact (classFacts_split classLeaf71).2
  by_cases is72 : k = 72
  · subst is72
    exact (classFacts_split classLeaf72).2
  by_cases is73 : k = 73
  · subst is73
    exact (classFacts_split classLeaf73).2
  by_cases is74 : k = 74
  · subst is74
    exact (classFacts_split classLeaf74).2
  by_cases is75 : k = 75
  · subst is75
    exact (classFacts_split classLeaf75).2
  by_cases is76 : k = 76
  · subst is76
    exact (classFacts_split classLeaf76).2
  by_cases is77 : k = 77
  · subst is77
    exact (classFacts_split classLeaf77).2
  by_cases is78 : k = 78
  · subst is78
    exact (classFacts_split classLeaf78).2
  by_cases is79 : k = 79
  · subst is79
    exact (classFacts_split classLeaf79).2
  by_cases is80 : k = 80
  · subst is80
    exact (classFacts_split classLeaf80).2
  by_cases is81 : k = 81
  · subst is81
    exact (classFacts_split classLeaf81).2
  by_cases is82 : k = 82
  · subst is82
    exact (classFacts_split classLeaf82).2
  by_cases is83 : k = 83
  · subst is83
    exact (classFacts_split classLeaf83).2
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5
