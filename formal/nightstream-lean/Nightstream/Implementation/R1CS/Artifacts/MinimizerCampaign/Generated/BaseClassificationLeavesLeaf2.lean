import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment

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

def background : Nat → Field := backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf28 :
    ((rowsChunk wire 28).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 28) = true) := by
  native_decide

theorem classLeaf29 :
    ((rowsChunk wire 29).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 29) = true) := by
  native_decide

theorem classLeaf30 :
    ((rowsChunk wire 30).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 30) = true) := by
  native_decide

theorem classLeaf31 :
    ((rowsChunk wire 31).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 31) = true) := by
  native_decide

theorem classLeaf32 :
    ((rowsChunk wire 32).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 32) = true) := by
  native_decide

theorem classLeaf33 :
    ((rowsChunk wire 33).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 33) = true) := by
  native_decide

theorem classLeaf34 :
    ((rowsChunk wire 34).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 34) = true) := by
  native_decide

theorem classLeaf35 :
    ((rowsChunk wire 35).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 35) = true) := by
  native_decide

theorem classLeaf36 :
    ((rowsChunk wire 36).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 36) = true) := by
  native_decide

theorem classLeaf37 :
    ((rowsChunk wire 37).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 37) = true) := by
  native_decide

theorem classLeaf38 :
    ((rowsChunk wire 38).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 38) = true) := by
  native_decide

theorem classLeaf39 :
    ((rowsChunk wire 39).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 39) = true) := by
  native_decide

theorem classLeaf40 :
    ((rowsChunk wire 40).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 40) = true) := by
  native_decide

theorem classLeaf41 :
    ((rowsChunk wire 41).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 41) = true) := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf2
