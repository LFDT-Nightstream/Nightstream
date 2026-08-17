import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf42 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 42) = true := by
  native_decide

theorem classLeaf43 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 43) = true := by
  native_decide

theorem classLeaf44 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 44) = true := by
  native_decide

theorem classLeaf45 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 45) = true := by
  native_decide

theorem classLeaf46 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 46) = true := by
  native_decide

theorem classLeaf47 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 47) = true := by
  native_decide

theorem classLeaf48 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 48) = true := by
  native_decide

theorem classLeaf49 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 49) = true := by
  native_decide

theorem classLeaf50 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 50) = true := by
  native_decide

theorem classLeaf51 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 51) = true := by
  native_decide

theorem classLeaf52 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 52) = true := by
  native_decide

theorem classLeaf53 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 53) = true := by
  native_decide

theorem classLeaf54 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 54) = true := by
  native_decide

theorem classLeaf55 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 55) = true := by
  native_decide

theorem holdsGroup :
    ∀ k, 42 ≤ k → k < 56 →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds
          (backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values) row.row)) = true := by
  intro k lower upper
  by_cases is42 : k = 42
  · subst is42
    exact (classFacts_split classLeaf42).1
  by_cases is43 : k = 43
  · subst is43
    exact (classFacts_split classLeaf43).1
  by_cases is44 : k = 44
  · subst is44
    exact (classFacts_split classLeaf44).1
  by_cases is45 : k = 45
  · subst is45
    exact (classFacts_split classLeaf45).1
  by_cases is46 : k = 46
  · subst is46
    exact (classFacts_split classLeaf46).1
  by_cases is47 : k = 47
  · subst is47
    exact (classFacts_split classLeaf47).1
  by_cases is48 : k = 48
  · subst is48
    exact (classFacts_split classLeaf48).1
  by_cases is49 : k = 49
  · subst is49
    exact (classFacts_split classLeaf49).1
  by_cases is50 : k = 50
  · subst is50
    exact (classFacts_split classLeaf50).1
  by_cases is51 : k = 51
  · subst is51
    exact (classFacts_split classLeaf51).1
  by_cases is52 : k = 52
  · subst is52
    exact (classFacts_split classLeaf52).1
  by_cases is53 : k = 53
  · subst is53
    exact (classFacts_split classLeaf53).1
  by_cases is54 : k = 54
  · subst is54
    exact (classFacts_split classLeaf54).1
  by_cases is55 : k = 55
  · subst is55
    exact (classFacts_split classLeaf55).1
  exact absurd upper (by omega)

theorem guardsGroup :
    ∀ k, 42 ≤ k → k < 56 →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k lower upper
  by_cases is42 : k = 42
  · subst is42
    exact (classFacts_split classLeaf42).2
  by_cases is43 : k = 43
  · subst is43
    exact (classFacts_split classLeaf43).2
  by_cases is44 : k = 44
  · subst is44
    exact (classFacts_split classLeaf44).2
  by_cases is45 : k = 45
  · subst is45
    exact (classFacts_split classLeaf45).2
  by_cases is46 : k = 46
  · subst is46
    exact (classFacts_split classLeaf46).2
  by_cases is47 : k = 47
  · subst is47
    exact (classFacts_split classLeaf47).2
  by_cases is48 : k = 48
  · subst is48
    exact (classFacts_split classLeaf48).2
  by_cases is49 : k = 49
  · subst is49
    exact (classFacts_split classLeaf49).2
  by_cases is50 : k = 50
  · subst is50
    exact (classFacts_split classLeaf50).2
  by_cases is51 : k = 51
  · subst is51
    exact (classFacts_split classLeaf51).2
  by_cases is52 : k = 52
  · subst is52
    exact (classFacts_split classLeaf52).2
  by_cases is53 : k = 53
  · subst is53
    exact (classFacts_split classLeaf53).2
  by_cases is54 : k = 54
  · subst is54
    exact (classFacts_split classLeaf54).2
  by_cases is55 : k = 55
  · subst is55
    exact (classFacts_split classLeaf55).2
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3
