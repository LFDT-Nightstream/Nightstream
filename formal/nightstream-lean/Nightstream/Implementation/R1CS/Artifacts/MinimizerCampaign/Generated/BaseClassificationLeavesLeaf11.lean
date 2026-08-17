import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def background : Nat → Field := backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf154 :
    ((rowsChunk wire 154).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 154) = true) := by
  native_decide

theorem classLeaf155 :
    ((rowsChunk wire 155).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 155) = true) := by
  native_decide

theorem classLeaf156 :
    ((rowsChunk wire 156).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 156) = true) := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf11
