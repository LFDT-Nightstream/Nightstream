import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf26

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf130 :
    ((rowsChunk wire 130).map (fun row => row.sourceIndex) =
        List.range' 8519680 65536) ∧
      ((rowsChunk wire 130).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 130).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem presence1 :
    (rowsChunk wire 130).any
      (fun row => decide (row.family = "fprime.recursive.step.accumulator.input_link")) = true := by
  native_decide

theorem presence3 :
    (rowsChunk wire 130).any
      (fun row => decide (row.family = "fprime.recursive.step.accumulator.output_authority.child_digests")) = true := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf26
