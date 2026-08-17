import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf14 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 14) = true := by
  native_decide

theorem classLeaf15 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 15) = true := by
  native_decide

theorem classLeaf16 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 16) = true := by
  native_decide

theorem classLeaf17 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 17) = true := by
  native_decide

theorem classLeaf18 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 18) = true := by
  native_decide

theorem classLeaf19 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 19) = true := by
  native_decide

theorem classLeaf20 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 20) = true := by
  native_decide

theorem classLeaf21 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 21) = true := by
  native_decide

theorem classLeaf22 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 22) = true := by
  native_decide

theorem classLeaf23 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 23) = true := by
  native_decide

theorem classLeaf24 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 24) = true := by
  native_decide

theorem classLeaf25 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 25) = true := by
  native_decide

theorem classLeaf26 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 26) = true := by
  native_decide

theorem classLeaf27 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 27) = true := by
  native_decide

theorem holdsGroup :
    ∀ k, 14 ≤ k → k < 28 →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds
          (backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values) row.row)) = true := by
  intro k lower upper
  by_cases is14 : k = 14
  · subst is14
    exact (classFacts_split classLeaf14).1
  by_cases is15 : k = 15
  · subst is15
    exact (classFacts_split classLeaf15).1
  by_cases is16 : k = 16
  · subst is16
    exact (classFacts_split classLeaf16).1
  by_cases is17 : k = 17
  · subst is17
    exact (classFacts_split classLeaf17).1
  by_cases is18 : k = 18
  · subst is18
    exact (classFacts_split classLeaf18).1
  by_cases is19 : k = 19
  · subst is19
    exact (classFacts_split classLeaf19).1
  by_cases is20 : k = 20
  · subst is20
    exact (classFacts_split classLeaf20).1
  by_cases is21 : k = 21
  · subst is21
    exact (classFacts_split classLeaf21).1
  by_cases is22 : k = 22
  · subst is22
    exact (classFacts_split classLeaf22).1
  by_cases is23 : k = 23
  · subst is23
    exact (classFacts_split classLeaf23).1
  by_cases is24 : k = 24
  · subst is24
    exact (classFacts_split classLeaf24).1
  by_cases is25 : k = 25
  · subst is25
    exact (classFacts_split classLeaf25).1
  by_cases is26 : k = 26
  · subst is26
    exact (classFacts_split classLeaf26).1
  by_cases is27 : k = 27
  · subst is27
    exact (classFacts_split classLeaf27).1
  exact absurd upper (by omega)

theorem guardsGroup :
    ∀ k, 14 ≤ k → k < 28 →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k lower upper
  by_cases is14 : k = 14
  · subst is14
    exact (classFacts_split classLeaf14).2
  by_cases is15 : k = 15
  · subst is15
    exact (classFacts_split classLeaf15).2
  by_cases is16 : k = 16
  · subst is16
    exact (classFacts_split classLeaf16).2
  by_cases is17 : k = 17
  · subst is17
    exact (classFacts_split classLeaf17).2
  by_cases is18 : k = 18
  · subst is18
    exact (classFacts_split classLeaf18).2
  by_cases is19 : k = 19
  · subst is19
    exact (classFacts_split classLeaf19).2
  by_cases is20 : k = 20
  · subst is20
    exact (classFacts_split classLeaf20).2
  by_cases is21 : k = 21
  · subst is21
    exact (classFacts_split classLeaf21).2
  by_cases is22 : k = 22
  · subst is22
    exact (classFacts_split classLeaf22).2
  by_cases is23 : k = 23
  · subst is23
    exact (classFacts_split classLeaf23).2
  by_cases is24 : k = 24
  · subst is24
    exact (classFacts_split classLeaf24).2
  by_cases is25 : k = 25
  · subst is25
    exact (classFacts_split classLeaf25).2
  by_cases is26 : k = 26
  · subst is26
    exact (classFacts_split classLeaf26).2
  by_cases is27 : k = 27
  · subst is27
    exact (classFacts_split classLeaf27).2
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1
