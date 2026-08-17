import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifactWire
import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment

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

def background : Nat → Field := backgroundFn Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCampaignAssignment.values

def overridePairs : List (Nat × String) :=
  [(3811, "fprime.base.step.initial")]

theorem classLeaf14 :
    ((rowsChunk wire 14).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 14) = true) := by
  native_decide

theorem classLeaf15 :
    ((rowsChunk wire 15).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 15) = true) := by
  native_decide

theorem classLeaf16 :
    ((rowsChunk wire 16).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 16) = true) := by
  native_decide

theorem classLeaf17 :
    ((rowsChunk wire 17).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 17) = true) := by
  native_decide

theorem classLeaf18 :
    ((rowsChunk wire 18).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 18) = true) := by
  native_decide

theorem classLeaf19 :
    ((rowsChunk wire 19).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 19) = true) := by
  native_decide

theorem classLeaf20 :
    ((rowsChunk wire 20).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 20) = true) := by
  native_decide

theorem classLeaf21 :
    ((rowsChunk wire 21).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 21) = true) := by
  native_decide

theorem classLeaf22 :
    ((rowsChunk wire 22).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 22) = true) := by
  native_decide

theorem classLeaf23 :
    ((rowsChunk wire 23).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 23) = true) := by
  native_decide

theorem classLeaf24 :
    ((rowsChunk wire 24).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 24) = true) := by
  native_decide

theorem classLeaf25 :
    ((rowsChunk wire 25).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 25) = true) := by
  native_decide

theorem classLeaf26 :
    ((rowsChunk wire 26).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 26) = true) := by
  native_decide

theorem classLeaf27 :
    ((rowsChunk wire 27).all
        (fun row => decide (Algebraic.Holds background row.row)) = true) ∧
      (chunkGuardsOverrides overridePairs (rowsChunk wire 27) = true) := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseClassificationLeavesLeaf1
