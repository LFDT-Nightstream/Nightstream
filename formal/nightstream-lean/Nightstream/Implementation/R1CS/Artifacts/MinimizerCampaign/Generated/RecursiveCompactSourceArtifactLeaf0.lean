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

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf0
