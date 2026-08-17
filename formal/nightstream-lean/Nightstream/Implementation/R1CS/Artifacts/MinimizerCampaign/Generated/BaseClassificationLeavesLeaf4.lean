import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf56 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 56) = true := by
  native_decide

theorem classLeaf57 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 57) = true := by
  native_decide

theorem classLeaf58 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 58) = true := by
  native_decide

theorem classLeaf59 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 59) = true := by
  native_decide

theorem classLeaf60 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 60) = true := by
  native_decide

theorem classLeaf61 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 61) = true := by
  native_decide

theorem classLeaf62 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 62) = true := by
  native_decide

theorem classLeaf63 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 63) = true := by
  native_decide

theorem classLeaf64 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 64) = true := by
  native_decide

theorem classLeaf65 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 65) = true := by
  native_decide

theorem classLeaf66 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 66) = true := by
  native_decide

theorem classLeaf67 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 67) = true := by
  native_decide

theorem classLeaf68 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 68) = true := by
  native_decide

theorem classLeaf69 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 69) = true := by
  native_decide

theorem holdsGroup :
    ∀ k, 56 ≤ k → k < 70 →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds
          (backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values) row.row)) = true := by
  intro k lower upper
  by_cases is56 : k = 56
  · subst is56
    exact (classFacts_split classLeaf56).1
  by_cases is57 : k = 57
  · subst is57
    exact (classFacts_split classLeaf57).1
  by_cases is58 : k = 58
  · subst is58
    exact (classFacts_split classLeaf58).1
  by_cases is59 : k = 59
  · subst is59
    exact (classFacts_split classLeaf59).1
  by_cases is60 : k = 60
  · subst is60
    exact (classFacts_split classLeaf60).1
  by_cases is61 : k = 61
  · subst is61
    exact (classFacts_split classLeaf61).1
  by_cases is62 : k = 62
  · subst is62
    exact (classFacts_split classLeaf62).1
  by_cases is63 : k = 63
  · subst is63
    exact (classFacts_split classLeaf63).1
  by_cases is64 : k = 64
  · subst is64
    exact (classFacts_split classLeaf64).1
  by_cases is65 : k = 65
  · subst is65
    exact (classFacts_split classLeaf65).1
  by_cases is66 : k = 66
  · subst is66
    exact (classFacts_split classLeaf66).1
  by_cases is67 : k = 67
  · subst is67
    exact (classFacts_split classLeaf67).1
  by_cases is68 : k = 68
  · subst is68
    exact (classFacts_split classLeaf68).1
  by_cases is69 : k = 69
  · subst is69
    exact (classFacts_split classLeaf69).1
  exact absurd upper (by omega)

theorem guardsGroup :
    ∀ k, 56 ≤ k → k < 70 →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k lower upper
  by_cases is56 : k = 56
  · subst is56
    exact (classFacts_split classLeaf56).2
  by_cases is57 : k = 57
  · subst is57
    exact (classFacts_split classLeaf57).2
  by_cases is58 : k = 58
  · subst is58
    exact (classFacts_split classLeaf58).2
  by_cases is59 : k = 59
  · subst is59
    exact (classFacts_split classLeaf59).2
  by_cases is60 : k = 60
  · subst is60
    exact (classFacts_split classLeaf60).2
  by_cases is61 : k = 61
  · subst is61
    exact (classFacts_split classLeaf61).2
  by_cases is62 : k = 62
  · subst is62
    exact (classFacts_split classLeaf62).2
  by_cases is63 : k = 63
  · subst is63
    exact (classFacts_split classLeaf63).2
  by_cases is64 : k = 64
  · subst is64
    exact (classFacts_split classLeaf64).2
  by_cases is65 : k = 65
  · subst is65
    exact (classFacts_split classLeaf65).2
  by_cases is66 : k = 66
  · subst is66
    exact (classFacts_split classLeaf66).2
  by_cases is67 : k = 67
  · subst is67
    exact (classFacts_split classLeaf67).2
  by_cases is68 : k = 68
  · subst is68
    exact (classFacts_split classLeaf68).2
  by_cases is69 : k = 69
  · subst is69
    exact (classFacts_split classLeaf69).2
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4
