import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

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
    chunkFacts (rowsChunk wire 130) 8519680 65536 11187825 11078210
      wire.completeFamilies
      ["fprime.recursive.step.accumulator.input_link",
       "fprime.recursive.step.accumulator.output_authority.child_digests"] = true := by
  native_decide

theorem presence1 :
    (rowsChunk wire 130).any
      (fun row => decide (row.family = "fprime.recursive.step.accumulator.input_link")) = true :=
  presence_of_chunkFacts chunkLeaf130 (by decide)

theorem presence3 :
    (rowsChunk wire 130).any
      (fun row => decide (row.family = "fprime.recursive.step.accumulator.output_authority.child_digests")) = true :=
  presence_of_chunkFacts chunkLeaf130 (by decide)

theorem censusGroup :
    ∀ k, 130 ≤ k → k < 131 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is130 : k = 130
  · subst is130
    exact (chunkFacts_split chunkLeaf130).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 130 ≤ k → k < 131 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is130 : k = 130
  · subst is130
    exact (chunkFacts_split chunkLeaf130).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 130 ≤ k → k < 131 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is130 : k = 130
  · subst is130
    exact (chunkFacts_split chunkLeaf130).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf26
