import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

/-!
GENERATED FILE - do not edit by hand.

Bounded per-chunk leaf certificates for one slice of the artifact.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17

open Nightstream.Assurance.CompactSourceArtifact
open Nightstream.Assurance.ConstraintMinimization
open Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire

set_option maxHeartbeats 2000000
set_option maxRecDepth 65536

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

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17
