import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment

/-!
GENERATED FILE - do not edit by hand.

Shared classification leaves for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

def background : Nat → Field := backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf112 :
    ((rowsChunk wire 112).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 112) = true) := by
  native_decide

theorem classLeaf113 :
    ((rowsChunk wire 113).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 113) = true) := by
  native_decide

theorem classLeaf114 :
    ((rowsChunk wire 114).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 114) = true) := by
  native_decide

theorem classLeaf115 :
    ((rowsChunk wire 115).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 115) = true) := by
  native_decide

theorem classLeaf116 :
    ((rowsChunk wire 116).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 116) = true) := by
  native_decide

theorem classLeaf117 :
    ((rowsChunk wire 117).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 117) = true) := by
  native_decide

theorem classLeaf118 :
    ((rowsChunk wire 118).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 118) = true) := by
  native_decide

theorem classLeaf119 :
    ((rowsChunk wire 119).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 119) = true) := by
  native_decide

theorem classLeaf120 :
    ((rowsChunk wire 120).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 120) = true) := by
  native_decide

theorem classLeaf121 :
    ((rowsChunk wire 121).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 121) = true) := by
  native_decide

theorem classLeaf122 :
    ((rowsChunk wire 122).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 122) = true) := by
  native_decide

theorem classLeaf123 :
    ((rowsChunk wire 123).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 123) = true) := by
  native_decide

theorem classLeaf124 :
    ((rowsChunk wire 124).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 124) = true) := by
  native_decide

theorem classLeaf125 :
    ((rowsChunk wire 125).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 125) = true) := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf8
