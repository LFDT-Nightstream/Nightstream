import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf28 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 28) = true := by
  native_decide

theorem classLeaf29 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 29) = true := by
  native_decide

theorem classLeaf30 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 30) = true := by
  native_decide

theorem classLeaf31 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 31) = true := by
  native_decide

theorem classLeaf32 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 32) = true := by
  native_decide

theorem classLeaf33 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 33) = true := by
  native_decide

theorem classLeaf34 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 34) = true := by
  native_decide

theorem classLeaf35 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 35) = true := by
  native_decide

theorem classLeaf36 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 36) = true := by
  native_decide

theorem classLeaf37 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 37) = true := by
  native_decide

theorem classLeaf38 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 38) = true := by
  native_decide

theorem classLeaf39 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 39) = true := by
  native_decide

theorem classLeaf40 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 40) = true := by
  native_decide

theorem classLeaf41 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 41) = true := by
  native_decide

theorem holdsGroup :
    ∀ k, 28 ≤ k → k < 42 →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds
          (backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values) row.row)) = true := by
  intro k lower upper
  by_cases is28 : k = 28
  · subst is28
    exact (classFacts_split classLeaf28).1
  by_cases is29 : k = 29
  · subst is29
    exact (classFacts_split classLeaf29).1
  by_cases is30 : k = 30
  · subst is30
    exact (classFacts_split classLeaf30).1
  by_cases is31 : k = 31
  · subst is31
    exact (classFacts_split classLeaf31).1
  by_cases is32 : k = 32
  · subst is32
    exact (classFacts_split classLeaf32).1
  by_cases is33 : k = 33
  · subst is33
    exact (classFacts_split classLeaf33).1
  by_cases is34 : k = 34
  · subst is34
    exact (classFacts_split classLeaf34).1
  by_cases is35 : k = 35
  · subst is35
    exact (classFacts_split classLeaf35).1
  by_cases is36 : k = 36
  · subst is36
    exact (classFacts_split classLeaf36).1
  by_cases is37 : k = 37
  · subst is37
    exact (classFacts_split classLeaf37).1
  by_cases is38 : k = 38
  · subst is38
    exact (classFacts_split classLeaf38).1
  by_cases is39 : k = 39
  · subst is39
    exact (classFacts_split classLeaf39).1
  by_cases is40 : k = 40
  · subst is40
    exact (classFacts_split classLeaf40).1
  by_cases is41 : k = 41
  · subst is41
    exact (classFacts_split classLeaf41).1
  exact absurd upper (by omega)

theorem guardsGroup :
    ∀ k, 28 ≤ k → k < 42 →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k lower upper
  by_cases is28 : k = 28
  · subst is28
    exact (classFacts_split classLeaf28).2
  by_cases is29 : k = 29
  · subst is29
    exact (classFacts_split classLeaf29).2
  by_cases is30 : k = 30
  · subst is30
    exact (classFacts_split classLeaf30).2
  by_cases is31 : k = 31
  · subst is31
    exact (classFacts_split classLeaf31).2
  by_cases is32 : k = 32
  · subst is32
    exact (classFacts_split classLeaf32).2
  by_cases is33 : k = 33
  · subst is33
    exact (classFacts_split classLeaf33).2
  by_cases is34 : k = 34
  · subst is34
    exact (classFacts_split classLeaf34).2
  by_cases is35 : k = 35
  · subst is35
    exact (classFacts_split classLeaf35).2
  by_cases is36 : k = 36
  · subst is36
    exact (classFacts_split classLeaf36).2
  by_cases is37 : k = 37
  · subst is37
    exact (classFacts_split classLeaf37).2
  by_cases is38 : k = 38
  · subst is38
    exact (classFacts_split classLeaf38).2
  by_cases is39 : k = 39
  · subst is39
    exact (classFacts_split classLeaf39).2
  by_cases is40 : k = 40
  · subst is40
    exact (classFacts_split classLeaf40).2
  by_cases is41 : k = 41
  · subst is41
    exact (classFacts_split classLeaf41).2
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2
