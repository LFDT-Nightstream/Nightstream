import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf58 :
    chunkFacts (rowsChunk wire 58) 3801088 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf59 :
    chunkFacts (rowsChunk wire 59) 3866624 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf60 :
    chunkFacts (rowsChunk wire 60) 3932160 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_ccs.padded_row.challenges"] = true := by
  native_decide

theorem chunkLeaf61 :
    chunkFacts (rowsChunk wire 61) 3997696 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_ccs.padded_row.sumcheck"] = true := by
  native_decide

theorem chunkLeaf62 :
    chunkFacts (rowsChunk wire 62) 4063232 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf63 :
    chunkFacts (rowsChunk wire 63) 4128768 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_ccs.padded_row.terminal"] = true := by
  native_decide

theorem chunkLeaf64 :
    chunkFacts (rowsChunk wire 64) 4194304 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_ccs.padded_row.output_digest.preimage.outer_header",
       "nifs.pi_ccs.padded_row.output_digest.preimage.source_headers",
       "nifs.pi_ccs.padded_row.output_digest.preimage.y_ring",
       "nifs.pi_ccs.padded_row.output_digest.sis",
       "nifs.pi_ccs.padded_row.output_transcript"] = true := by
  native_decide

theorem chunkLeaf65 :
    chunkFacts (rowsChunk wire 65) 4259840 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf66 :
    chunkFacts (rowsChunk wire 66) 4325376 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf67 :
    chunkFacts (rowsChunk wire 67) 4390912 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf68 :
    chunkFacts (rowsChunk wire 68) 4456448 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf69 :
    chunkFacts (rowsChunk wire 69) 4521984 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf70 :
    chunkFacts (rowsChunk wire 70) 4587520 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf71 :
    chunkFacts (rowsChunk wire 71) 4653056 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem presence15 :
    (rowsChunk wire 60).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.challenges")) = true :=
  presence_of_chunkFacts chunkLeaf60 (by decide)

theorem presence17 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.preimage.outer_header")) = true :=
  presence_of_chunkFacts chunkLeaf64 (by decide)

theorem presence18 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.preimage.source_headers")) = true :=
  presence_of_chunkFacts chunkLeaf64 (by decide)

theorem presence19 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.preimage.y_ring")) = true :=
  presence_of_chunkFacts chunkLeaf64 (by decide)

theorem presence20 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.sis")) = true :=
  presence_of_chunkFacts chunkLeaf64 (by decide)

theorem presence21 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_transcript")) = true :=
  presence_of_chunkFacts chunkLeaf64 (by decide)

theorem presence23 :
    (rowsChunk wire 61).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.sumcheck")) = true :=
  presence_of_chunkFacts chunkLeaf61 (by decide)

theorem presence24 :
    (rowsChunk wire 63).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.terminal")) = true :=
  presence_of_chunkFacts chunkLeaf63 (by decide)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12
