import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf31

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf170 :
    chunkFacts (rowsChunk wire 170) 11141120 46705 11187825 11078210
      wire.completeFamilies
      ["fprime.recursive.step.accumulator.output_authority.aggregate",
       "fprime.recursive.step.counters",
       "fprime.recursive.step.output"] = true := by
  native_decide

theorem presence2 :
    (rowsChunk wire 170).any
      (fun row => decide (row.family = "fprime.recursive.step.accumulator.output_authority.aggregate")) = true :=
  presence_of_chunkFacts chunkLeaf170 (by decide)

theorem presence4 :
    (rowsChunk wire 170).any
      (fun row => decide (row.family = "fprime.recursive.step.counters")) = true :=
  presence_of_chunkFacts chunkLeaf170 (by decide)

theorem presence6 :
    (rowsChunk wire 170).any
      (fun row => decide (row.family = "fprime.recursive.step.output")) = true :=
  presence_of_chunkFacts chunkLeaf170 (by decide)

theorem censusGroup :
    ∀ k, 170 ≤ k → k < 171 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is170 : k = 170
  · subst is170
    exact (chunkFacts_split chunkLeaf170).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 170 ≤ k → k < 171 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is170 : k = 170
  · subst is170
    exact (chunkFacts_split chunkLeaf170).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 170 ≤ k → k < 171 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is170 : k = 170
  · subst is170
    exact (chunkFacts_split chunkLeaf170).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf31
