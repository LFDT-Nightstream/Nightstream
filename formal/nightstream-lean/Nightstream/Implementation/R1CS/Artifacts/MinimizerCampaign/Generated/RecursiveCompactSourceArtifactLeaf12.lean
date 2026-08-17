import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

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
    ((rowsChunk wire 58).map (fun row => row.sourceIndex) =
        List.range' 3801088 65536) ∧
      ((rowsChunk wire 58).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 58).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf59 :
    ((rowsChunk wire 59).map (fun row => row.sourceIndex) =
        List.range' 3866624 65536) ∧
      ((rowsChunk wire 59).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 59).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf60 :
    ((rowsChunk wire 60).map (fun row => row.sourceIndex) =
        List.range' 3932160 65536) ∧
      ((rowsChunk wire 60).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 60).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf61 :
    ((rowsChunk wire 61).map (fun row => row.sourceIndex) =
        List.range' 3997696 65536) ∧
      ((rowsChunk wire 61).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 61).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf62 :
    ((rowsChunk wire 62).map (fun row => row.sourceIndex) =
        List.range' 4063232 65536) ∧
      ((rowsChunk wire 62).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 62).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf63 :
    ((rowsChunk wire 63).map (fun row => row.sourceIndex) =
        List.range' 4128768 65536) ∧
      ((rowsChunk wire 63).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 63).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf64 :
    ((rowsChunk wire 64).map (fun row => row.sourceIndex) =
        List.range' 4194304 65536) ∧
      ((rowsChunk wire 64).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 64).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf65 :
    ((rowsChunk wire 65).map (fun row => row.sourceIndex) =
        List.range' 4259840 65536) ∧
      ((rowsChunk wire 65).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 65).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf66 :
    ((rowsChunk wire 66).map (fun row => row.sourceIndex) =
        List.range' 4325376 65536) ∧
      ((rowsChunk wire 66).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 66).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf67 :
    ((rowsChunk wire 67).map (fun row => row.sourceIndex) =
        List.range' 4390912 65536) ∧
      ((rowsChunk wire 67).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 67).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf68 :
    ((rowsChunk wire 68).map (fun row => row.sourceIndex) =
        List.range' 4456448 65536) ∧
      ((rowsChunk wire 68).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 68).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf69 :
    ((rowsChunk wire 69).map (fun row => row.sourceIndex) =
        List.range' 4521984 65536) ∧
      ((rowsChunk wire 69).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 69).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf70 :
    ((rowsChunk wire 70).map (fun row => row.sourceIndex) =
        List.range' 4587520 65536) ∧
      ((rowsChunk wire 70).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 70).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf71 :
    ((rowsChunk wire 71).map (fun row => row.sourceIndex) =
        List.range' 4653056 65536) ∧
      ((rowsChunk wire 71).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 71).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem presence15 :
    (rowsChunk wire 60).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.challenges")) = true := by
  native_decide

theorem presence17 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.preimage.outer_header")) = true := by
  native_decide

theorem presence18 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.preimage.source_headers")) = true := by
  native_decide

theorem presence19 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.preimage.y_ring")) = true := by
  native_decide

theorem presence20 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.sis")) = true := by
  native_decide

theorem presence21 :
    (rowsChunk wire 64).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_transcript")) = true := by
  native_decide

theorem presence23 :
    (rowsChunk wire 61).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.sumcheck")) = true := by
  native_decide

theorem presence24 :
    (rowsChunk wire 63).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.terminal")) = true := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf12
