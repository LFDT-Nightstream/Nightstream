import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment
import Nightstream.Assurance.ChunkLeaves
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11

/-!
GENERATED FILE - do not edit by hand.

Dispatchers over the shared classification leaves.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

def background : Nat → Field := backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem holdsAll :
    ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).all
        (fun row => decide (Algebraic.Holds background row.row)) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  by_cases group0 : k < 14
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.holdsGroup k (by omega) group0
  by_cases group1 : k < 28
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.holdsGroup k (by omega) group1
  by_cases group2 : k < 42
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.holdsGroup k (by omega) group2
  by_cases group3 : k < 56
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.holdsGroup k (by omega) group3
  by_cases group4 : k < 70
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.holdsGroup k (by omega) group4
  by_cases group5 : k < 84
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.holdsGroup k (by omega) group5
  by_cases group6 : k < 98
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.holdsGroup k (by omega) group6
  by_cases group7 : k < 112
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.holdsGroup k (by omega) group7
  by_cases group8 : k < 126
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.holdsGroup k (by omega) group8
  by_cases group9 : k < 140
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.holdsGroup k (by omega) group9
  by_cases group10 : k < 154
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.holdsGroup k (by omega) group10
  by_cases group11 : k < 157
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11.holdsGroup k (by omega) group11
  exact absurd bound (by omega)

theorem guardsAll :
    ∀ k, k < wire.chunkCount →
      chunkGuardsOverrides overridePairs (rowsChunk wire k) = true := by
  intro k bound
  rw [chunkCount_eq] at bound
  by_cases group0 : k < 14
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf0.guardsGroup k (by omega) group0
  by_cases group1 : k < 28
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1.guardsGroup k (by omega) group1
  by_cases group2 : k < 42
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2.guardsGroup k (by omega) group2
  by_cases group3 : k < 56
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf3.guardsGroup k (by omega) group3
  by_cases group4 : k < 70
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf4.guardsGroup k (by omega) group4
  by_cases group5 : k < 84
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5.guardsGroup k (by omega) group5
  by_cases group6 : k < 98
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf6.guardsGroup k (by omega) group6
  by_cases group7 : k < 112
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf7.guardsGroup k (by omega) group7
  by_cases group8 : k < 126
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8.guardsGroup k (by omega) group8
  by_cases group9 : k < 140
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf9.guardsGroup k (by omega) group9
  by_cases group10 : k < 154
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf10.guardsGroup k (by omega) group10
  by_cases group11 : k < 157
  · exact Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11.guardsGroup k (by omega) group11
  exact absurd bound (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeaves
