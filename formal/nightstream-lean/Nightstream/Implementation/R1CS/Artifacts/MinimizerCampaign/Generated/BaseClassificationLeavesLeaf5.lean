import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment

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

def background : Nat → Field := backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf70 :
    ((rowsChunk wire 70).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 70) = true) := by
  native_decide

theorem classLeaf71 :
    ((rowsChunk wire 71).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 71) = true) := by
  native_decide

theorem classLeaf72 :
    ((rowsChunk wire 72).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 72) = true) := by
  native_decide

theorem classLeaf73 :
    ((rowsChunk wire 73).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 73) = true) := by
  native_decide

theorem classLeaf74 :
    ((rowsChunk wire 74).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 74) = true) := by
  native_decide

theorem classLeaf75 :
    ((rowsChunk wire 75).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 75) = true) := by
  native_decide

theorem classLeaf76 :
    ((rowsChunk wire 76).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 76) = true) := by
  native_decide

theorem classLeaf77 :
    ((rowsChunk wire 77).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 77) = true) := by
  native_decide

theorem classLeaf78 :
    ((rowsChunk wire 78).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 78) = true) := by
  native_decide

theorem classLeaf79 :
    ((rowsChunk wire 79).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 79) = true) := by
  native_decide

theorem classLeaf80 :
    ((rowsChunk wire 80).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 80) = true) := by
  native_decide

theorem classLeaf81 :
    ((rowsChunk wire 81).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 81) = true) := by
  native_decide

theorem classLeaf82 :
    ((rowsChunk wire 82).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 82) = true) := by
  native_decide

theorem classLeaf83 :
    ((rowsChunk wire 83).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 83) = true) := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf5
