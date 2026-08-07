import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityCodec
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityPoseidon2
import Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaddedRowIdentity.Generated.Layout

/-!
Bounded Rust-to-Lean conformance for the selected one-joint PiCCS layout.

Owns: every bounded gamma slot, the joint-domain maximum in both rectangular
directions, the transcript tag order, one exact Rust proof-codec vector, the
complete output order check, and the selected production output field count.

Does not own: Rust matrix evaluation, arbitrary-size transcript refinement,
the complete NIFS codec, R1CS lowering, or a production matrix artifact.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaddedRowIdentity

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs

namespace Artifact

def shape : Shape where
  cubeVariables := Generated.rowVariablesWhenRowsLtAssignment
  freshCount := Generated.freshCount
  runningCount := Generated.runningCount
  matrixCount := Generated.matrixCount
  coefficientCount := Generated.coefficientCount

def expectedFreshGammaExponents : List Nat :=
  (canonicalFinIndices shape.freshCount).map Fin.val

def expectedNormGammaExponents : List Nat :=
  (canonicalFinIndices shape.sourceCount).map fun source =>
    shape.normOffset + source.val

def expectedCarriedGammaExponents : List Nat :=
  (canonicalCarriedCoordinates shape).map CarriedCoordinate.gammaExponent

theorem carried_input_length :
    Generated.carriedGammaExponents.length = Generated.carriedCount := by
  set_option maxRecDepth 2000 in
    decide

theorem carried_count_eq_shape :
    Generated.carriedCount = shape.carriedEvaluationCount := by
  decide

theorem fresh_gamma_slots_match :
    Generated.freshGammaExponents = expectedFreshGammaExponents := by
  decide

theorem norm_gamma_slots_match :
    Generated.normGammaExponents = expectedNormGammaExponents := by
  decide

theorem carried_gamma_slots_match :
    Generated.carriedGammaExponents = expectedCarriedGammaExponents := by
  set_option maxRecDepth 2000 in
    decide

theorem joint_domain_is_max_in_both_directions :
    Generated.rowVariablesWhenRowsLtAssignment = max 5 6 ∧
      Generated.rowVariablesWhenRowsGtAssignment = max 7 6 := by
  decide

def expectedTranscriptTags : List Nat :=
  [ PaddedRowIdentityPoseidon2.publicInputTag,
    41, 42, 43, 45, 46, 47 ]

theorem transcript_tags_match :
    Generated.transcriptTags = expectedTranscriptTags := by
  decide

def sampleRounds : List (List (Nat × Nat)) :=
  [[(1, 101), (2, 102), (3, 103)],
   [(4, 104), (5, 105), (6, 106)]]

def encodePair (value : Nat × Nat) : List Nat :=
  [value.1, value.2]

def expectedSampleProofWords : List Nat :=
  [1102, 1, sampleRounds.length,
    sampleRounds.head?.map List.length |>.getD 0] ++
    sampleRounds.flatMap fun round => round.flatMap encodePair

theorem sample_proof_codec_matches :
    Generated.sampleProofWords = expectedSampleProofWords := by
  decide

theorem sample_output_field_count_matches :
    Generated.sampleOutputFieldCount =
      20 + shape.sourceCount * shape.matrixCount * shape.coefficientCount * 2 := by
  decide

/-- The Rust generator compares every sampled output field against the exact
source-major, matrix-major, coefficient-major, low-limb/high-limb sequence
before it emits this artifact. -/
theorem sample_output_order_matches :
    Generated.sampleOutputOrderMatches = true := by
  rfl

theorem production_output_field_count_matches
    (message : FullOutputCoordinates.FullOutput K
      PaddedRowIdentity.shape) :
    Generated.productionOutputFieldCount =
      (PaddedRowIdentityPoseidon2.outputFields message).length := by
  rw [PaddedRowIdentityPoseidon2.outputFields_length]
  decide

theorem piCcs_tag_is_distinct_from_full_nifs_proof_tag :
    1102 ≠ PaddedRowIdentityCodec.proofEnvelopeTag := by
  decide

end Artifact

end Nightstream.Implementation.Rust.CanonicalConformance.PiCcsPaddedRowIdentity
