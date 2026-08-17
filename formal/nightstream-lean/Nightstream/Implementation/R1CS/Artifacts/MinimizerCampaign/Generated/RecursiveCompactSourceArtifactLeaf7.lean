import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf7

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

theorem chunkLeaf98 :
    ((rowsChunk wire 98).map (fun row => row.sourceIndex) =
        List.range' 6422528 65536) ∧
      ((rowsChunk wire 98).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 98).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf99 :
    ((rowsChunk wire 99).map (fun row => row.sourceIndex) =
        List.range' 6488064 65536) ∧
      ((rowsChunk wire 99).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 99).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf100 :
    ((rowsChunk wire 100).map (fun row => row.sourceIndex) =
        List.range' 6553600 65536) ∧
      ((rowsChunk wire 100).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 100).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf101 :
    ((rowsChunk wire 101).map (fun row => row.sourceIndex) =
        List.range' 6619136 65536) ∧
      ((rowsChunk wire 101).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 101).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf102 :
    ((rowsChunk wire 102).map (fun row => row.sourceIndex) =
        List.range' 6684672 65536) ∧
      ((rowsChunk wire 102).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 102).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf103 :
    ((rowsChunk wire 103).map (fun row => row.sourceIndex) =
        List.range' 6750208 65536) ∧
      ((rowsChunk wire 103).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 103).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf104 :
    ((rowsChunk wire 104).map (fun row => row.sourceIndex) =
        List.range' 6815744 65536) ∧
      ((rowsChunk wire 104).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 104).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf105 :
    ((rowsChunk wire 105).map (fun row => row.sourceIndex) =
        List.range' 6881280 65536) ∧
      ((rowsChunk wire 105).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 105).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf106 :
    ((rowsChunk wire 106).map (fun row => row.sourceIndex) =
        List.range' 6946816 65536) ∧
      ((rowsChunk wire 106).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 106).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf107 :
    ((rowsChunk wire 107).map (fun row => row.sourceIndex) =
        List.range' 7012352 65536) ∧
      ((rowsChunk wire 107).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 107).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf108 :
    ((rowsChunk wire 108).map (fun row => row.sourceIndex) =
        List.range' 7077888 65536) ∧
      ((rowsChunk wire 108).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 108).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf109 :
    ((rowsChunk wire 109).map (fun row => row.sourceIndex) =
        List.range' 7143424 65536) ∧
      ((rowsChunk wire 109).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 109).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf110 :
    ((rowsChunk wire 110).map (fun row => row.sourceIndex) =
        List.range' 7208960 65536) ∧
      ((rowsChunk wire 110).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 110).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem chunkLeaf111 :
    ((rowsChunk wire 111).map (fun row => row.sourceIndex) =
        List.range' 7274496 65536) ∧
      ((rowsChunk wire 111).all (rowWellFormedAt 11187825 11078210) = true) ∧
      ((rowsChunk wire 111).all
        (fun row => decide (row.family ∈ wire.completeFamilies)) = true) := by
  native_decide

theorem presence16 :
    (rowsChunk wire 102).any
      (fun row => decide (row.family = "nifs.pi_ccs.padded_row.output_digest.claim")) = true := by
  native_decide

theorem presence26 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.acceptance_bound")) = true := by
  native_decide

theorem presence27 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.chunk.accept")) = true := by
  native_decide

theorem presence28 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.chunk.mod5")) = true := by
  native_decide

theorem presence29 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.chunk.symbol_and_prefix")) = true := by
  native_decide

theorem presence30 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.initialize")) = true := by
  native_decide

theorem presence31 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.selection.initialize")) = true := by
  native_decide

theorem presence32 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.selection.one_hot")) = true := by
  native_decide

theorem presence33 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.selection.products")) = true := by
  native_decide

theorem presence34 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.transcript.bind_outputs_digest")) = true := by
  native_decide

theorem presence35 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.transcript.digest_rounds")) = true := by
  native_decide

theorem presence36 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.transcript.lane_bit_decomposition")) = true := by
  native_decide

theorem presence37 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.transcript.rho_domain_separator")) = true := by
  native_decide

theorem presence38 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.shape.allocate_parent_and_children")) = true := by
  native_decide

theorem presence39 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.shape.output_parity")) = true := by
  native_decide

theorem presence40 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.shape.parent")) = true := by
  native_decide

theorem presence41 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.consistency.fold_digest")) = true := by
  native_decide

theorem presence67 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.adv")) = true := by
  native_decide

theorem presence68 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.commitment")) = true := by
  native_decide

theorem presence69 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.x")) = true := by
  native_decide

theorem presence70 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.combined.y_ring")) = true := by
  native_decide

theorem presence71 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.domain")) = true := by
  native_decide

theorem presence72 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.adv")) = true := by
  native_decide

theorem presence73 :
    (rowsChunk wire 106).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.commitment")) = true := by
  native_decide

theorem presence74 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.x")) = true := by
  native_decide

theorem presence75 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.quotient.y_ring")) = true := by
  native_decide

theorem presence76 :
    (rowsChunk wire 108).any
      (fun row => decide (row.family = "nifs.pi_rlc.verify.projection_binding.sis_digest")) = true := by
  native_decide

theorem presence81 :
    (rowsChunk wire 102).any
      (fun row => decide (row.family = "nifs.running_parent_pi_dec")) = true := by
  native_decide

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf7
