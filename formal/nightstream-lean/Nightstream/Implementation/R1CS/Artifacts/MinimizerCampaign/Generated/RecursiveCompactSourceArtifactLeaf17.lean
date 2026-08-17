import Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactWire
import Nightstream.Assurance.ChunkLeaves

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
    chunkFacts (rowsChunk wire 103) 6750208 65536 11187825 11078210
      wire.completeFamilies
      ["nifs.pi_rlc.challenge.sampler.acceptance_bound",
       "nifs.pi_rlc.challenge.sampler.chunk.accept",
       "nifs.pi_rlc.challenge.sampler.chunk.mod5",
       "nifs.pi_rlc.challenge.sampler.chunk.symbol_and_prefix",
       "nifs.pi_rlc.challenge.sampler.initialize",
       "nifs.pi_rlc.challenge.sampler.selection.initialize",
       "nifs.pi_rlc.challenge.sampler.selection.one_hot",
       "nifs.pi_rlc.challenge.sampler.selection.products",
       "nifs.pi_rlc.challenge.transcript.bind_outputs_digest",
       "nifs.pi_rlc.challenge.transcript.digest_rounds",
       "nifs.pi_rlc.challenge.transcript.lane_bit_decomposition",
       "nifs.pi_rlc.challenge.transcript.rho_domain_separator"] = true := by
  native_decide

theorem chunkLeaf104 :
    chunkFacts (rowsChunk wire 104) 6815744 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem chunkLeaf105 :
    chunkFacts (rowsChunk wire 105) 6881280 65536 11187825 11078210
      wire.completeFamilies
      [] = true := by
  native_decide

theorem presence26 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.acceptance_bound")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

theorem presence27 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.chunk.accept")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

theorem presence28 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.chunk.mod5")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

theorem presence29 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.chunk.symbol_and_prefix")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

theorem presence30 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.initialize")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

theorem presence31 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.selection.initialize")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

theorem presence32 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.selection.one_hot")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

theorem presence33 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.sampler.selection.products")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

theorem presence34 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.transcript.bind_outputs_digest")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

theorem presence35 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.transcript.digest_rounds")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

theorem presence36 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.transcript.lane_bit_decomposition")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

theorem presence37 :
    (rowsChunk wire 103).any
      (fun row => decide (row.family = "nifs.pi_rlc.challenge.transcript.rho_domain_separator")) = true :=
  presence_of_chunkFacts chunkLeaf103 (by decide)

end Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifactLeaf17
