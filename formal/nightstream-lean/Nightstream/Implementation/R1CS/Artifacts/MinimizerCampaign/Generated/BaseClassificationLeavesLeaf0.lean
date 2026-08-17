import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf0 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 0) = true := by
  native_decide

theorem classLeaf1 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 1) = true := by
  native_decide

theorem classLeaf2 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 2) = true := by
  native_decide

theorem classLeaf3 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 3) = true := by
  native_decide

theorem classLeaf4 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 4) = true := by
  native_decide

theorem classLeaf5 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 5) = true := by
  native_decide

theorem classLeaf6 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 6) = true := by
  native_decide

theorem classLeaf7 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 7) = true := by
  native_decide

theorem classLeaf8 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 8) = true := by
  native_decide

theorem classLeaf9 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 9) = true := by
  native_decide

theorem classLeaf10 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 10) = true := by
  native_decide

theorem classLeaf11 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 11) = true := by
  native_decide

theorem classLeaf12 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 12) = true := by
  native_decide

theorem classLeaf13 :
    classFacts Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values overridePairs
      (rowsChunk wire 13) = true := by
  native_decide

theorem holdsGroup :
    ∀ k, 0 ≤ k → k < 14 →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds
          (backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values) row.row)) = true := by
  intro k lower upper
  by_cases is0 : k = 0
  · subst is0
    exact (classFacts_split classLeaf0).1
  by_cases is1 : k = 1
  · subst is1
    exact (classFacts_split classLeaf1).1
  by_cases is2 : k = 2
  · subst is2
    exact (classFacts_split classLeaf2).1
  by_cases is3 : k = 3
  · subst is3
    exact (classFacts_split classLeaf3).1
  by_cases is4 : k = 4
  · subst is4
    exact (classFacts_split classLeaf4).1
  by_cases is5 : k = 5
  · subst is5
    exact (classFacts_split classLeaf5).1
  by_cases is6 : k = 6
  · subst is6
    exact (classFacts_split classLeaf6).1
  by_cases is7 : k = 7
  · subst is7
    exact (classFacts_split classLeaf7).1
  by_cases is8 : k = 8
  · subst is8
    exact (classFacts_split classLeaf8).1
  by_cases is9 : k = 9
  · subst is9
    exact (classFacts_split classLeaf9).1
  by_cases is10 : k = 10
  · subst is10
    exact (classFacts_split classLeaf10).1
  by_cases is11 : k = 11
  · subst is11
    exact (classFacts_split classLeaf11).1
  by_cases is12 : k = 12
  · subst is12
    exact (classFacts_split classLeaf12).1
  by_cases is13 : k = 13
  · subst is13
    exact (classFacts_split classLeaf13).1
  exact absurd upper (by omega)

theorem guardsGroup :
    ∀ k, 0 ≤ k → k < 14 →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k lower upper
  by_cases is0 : k = 0
  · subst is0
    exact (classFacts_split classLeaf0).2
  by_cases is1 : k = 1
  · subst is1
    exact (classFacts_split classLeaf1).2
  by_cases is2 : k = 2
  · subst is2
    exact (classFacts_split classLeaf2).2
  by_cases is3 : k = 3
  · subst is3
    exact (classFacts_split classLeaf3).2
  by_cases is4 : k = 4
  · subst is4
    exact (classFacts_split classLeaf4).2
  by_cases is5 : k = 5
  · subst is5
    exact (classFacts_split classLeaf5).2
  by_cases is6 : k = 6
  · subst is6
    exact (classFacts_split classLeaf6).2
  by_cases is7 : k = 7
  · subst is7
    exact (classFacts_split classLeaf7).2
  by_cases is8 : k = 8
  · subst is8
    exact (classFacts_split classLeaf8).2
  by_cases is9 : k = 9
  · subst is9
    exact (classFacts_split classLeaf9).2
  by_cases is10 : k = 10
  · subst is10
    exact (classFacts_split classLeaf10).2
  by_cases is11 : k = 11
  · subst is11
    exact (classFacts_split classLeaf11).2
  by_cases is12 : k = 12
  · subst is12
    exact (classFacts_split classLeaf12).2
  by_cases is13 : k = 13
  · subst is13
    exact (classFacts_split classLeaf13).2
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0
