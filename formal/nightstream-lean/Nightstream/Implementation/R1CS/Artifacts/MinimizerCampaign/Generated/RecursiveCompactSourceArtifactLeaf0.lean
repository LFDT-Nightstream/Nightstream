import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf0 :
    chunkFacts (rowsChunk wire 0) 0 65536 11187825 11078210
      wire.completeFamilies
      ["fprime.recursive.finalize.application",
       "fprime.recursive.step.prelude",
       "fprime.recursive.step.transcript",
       "nifs.pi_ccs.padded_row.allocations",
       "nifs.pi_ccs.padded_row.binding",
       "nifs.pi_ccs.padded_row.canonicality"] = true := by
  native_decide

theorem presence0 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "fprime.recursive.finalize.application")) = true :=
  presence_of_chunkFacts chunkLeaf0 (by decide)

theorem presence7 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "fprime.recursive.step.prelude")) = true :=
  presence_of_chunkFacts chunkLeaf0 (by decide)

theorem presence11 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "fprime.recursive.step.transcript")) = true :=
  presence_of_chunkFacts chunkLeaf0 (by decide)

theorem presence12 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.allocations")) = true :=
  presence_of_chunkFacts chunkLeaf0 (by decide)

theorem presence13 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.binding")) = true :=
  presence_of_chunkFacts chunkLeaf0 (by decide)

theorem presence14 :
    (rowsChunk wire 0).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.canonicality")) = true :=
  presence_of_chunkFacts chunkLeaf0 (by decide)

theorem censusGroup :
    ∀ k, 0 ≤ k → k < 1 →
      (rowsChunk wire k).map (fun row => row.sourceIndex) =
        List.range' (wire.chunkStart k) (wire.chunkLength k) := by
  intro k lower upper
  by_cases is0 : k = 0
  · subst is0
    exact (chunkFacts_split chunkLeaf0).1
  exact absurd upper (by omega)

theorem wfGroup :
    ∀ k, 0 ≤ k → k < 1 →
      (rowsChunk wire k).all (rowWellFormedAt 11187825 11078210) = true := by
  intro k lower upper
  by_cases is0 : k = 0
  · subst is0
    exact (chunkFacts_split chunkLeaf0).2.1
  exact absurd upper (by omega)

theorem coverGroup :
    ∀ k, 0 ≤ k → k < 1 →
      (rowsChunk wire k).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true := by
  intro k lower upper
  by_cases is0 : k = 0
  · subst is0
    exact (chunkFacts_split chunkLeaf0).2.2.1
  exact absurd upper (by omega)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0
