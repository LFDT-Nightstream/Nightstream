import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalSourceBinding

/-!
Contract: exact terminal source-binding rows recover every decoded XOut,
Nebula-lane, local-state, and delayed-payload field from the final selective
assignment.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBindingRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBinding.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalSourceBinding

private theorem decoder_terms_canonical
    (start width radix scale : Nat)
    (canonical : CanonicalTerms (decoderTerms 0 width radix scale)) :
    CanonicalTerms (decoderTerms start width radix scale) := by
  intro term member
  rcases List.mem_map.mp member with ⟨offset, offsetMember, rfl⟩
  apply canonical (offset, (scale * radix ^ offset) % goldilocksP)
  exact List.mem_map.mpr ⟨offset, offsetMember, by simp⟩

private theorem composite_terms_canonical
    (decoder : CompositeDecoder) (canonical : decoder.Canonical) :
    CanonicalTerms decoder.terms := by
  intro term member
  rcases List.mem_flatMap.mp member with ⟨segment, segmentMember, termMember⟩
  exact canonical segment segmentMember term termMember

private theorem decoder_row_sound
    {assignment : Nat → Nat}
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (output : Nat) (terms : List (Nat × Nat))
    (canonicalTerms : CanonicalTerms terms)
    (holds : RowHolds assignment (decoderRow output terms)) :
    assignment output = lcEval assignment terms := by
  have permutation :
      (negateTerms terms ++ [(output, 1)]).Perm
        ((output, 1) :: negateTerms terms) := by
    simpa using List.Perm.append_comm (negateTerms terms) [(output, 1)]
  have evalPermutation := lcEval_eq_of_perm assignment permutation
  have builderHolds : RowHolds assignment (builderLinearRow output terms) := by
    simpa [decoderRow, builderLinearRow, RowHolds, evalPermutation] using holds
  exact builderLinearRow_sound canonicalAssignment one output terms
    canonicalTerms builderHolds

private theorem block_row_member
    (artifact : RawArtifact) (block : DecoderBlock)
    (groupMember : DecoderGroup.block block ∈ artifact.decoderGroups)
    (index : Nat) (indexLt : index < block.count) :
    decoderRow (block.decodedColumns.start + index) (block.termsAt index) ∈
      artifact.rows := by
  apply List.mem_flatMap.mpr
  refine ⟨DecoderGroup.block block, groupMember, ?_⟩
  exact List.mem_map.mpr ⟨index, List.mem_range.mpr indexLt, rfl⟩

private theorem composite_row_member
    (artifact : RawArtifact) (decoder : CompositeDecoder)
    (groupMember : DecoderGroup.composite decoder ∈ artifact.decoderGroups) :
    decoderRow decoder.decodedColumn decoder.terms ∈ artifact.rows := by
  apply List.mem_flatMap.mpr
  exact ⟨DecoderGroup.composite decoder, groupMember, by simp [DecoderGroup.rows]⟩

/-- Satisfaction of the exact compactly certified Rust family recovers every
decoded source field as its complete affine final-assignment decoder. -/
theorem rows_imply_decoder_groups
    (assignment : Nat → Nat)
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment) :
    ∀ group ∈ rawArtifact.decoderGroups, group.Holds assignment := by
  intro group groupMember
  have groupCanonical := rawArtifact_valid.groupsCanonical group groupMember
  cases group with
  | block item =>
      intro index indexLt
      apply decoder_row_sound canonicalAssignment one
      · exact decoder_terms_canonical
          (item.finalColumns.start + index * item.width)
          item.width item.radix item.scale groupCanonical
      · exact satisfied _ (block_row_member rawArtifact item groupMember index indexLt)
  | composite decoder =>
      apply decoder_row_sound canonicalAssignment one
      · exact composite_terms_canonical decoder groupCanonical
      · exact satisfied _ (composite_row_member rawArtifact decoder groupMember)

/-- Every delayed-payload alias is the exact one-bit Rust source coordinate
recorded by the terminal source-binding artifact. -/
theorem rows_imply_delayedPayloadSource
    (assignment : Nat → Nat)
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment)
    (index : Nat) (bounded : index < delayedPayloadBlock.count) :
    assignment (28041985 + index) = assignment (22126657 + index) := by
  have groupHolds := rows_imply_decoder_groups assignment canonicalAssignment
    one satisfied (DecoderGroup.block delayedPayloadBlock)
      delayedPayloadBlock_member
  change delayedPayloadBlock.Holds assignment at groupHolds
  have openHolds := groupHolds index bounded
  rw [delayedPayloadDecodedColumn, delayedPayloadTermsAt] at openHolds
  simpa [lcEval, Nat.mod_eq_of_lt
    (canonicalAssignment (22126657 + index))] using openHolds

/-- Exact source-binding rows identify the terminal finalizer's delayed-open
column with the one verifier-assignment bit that Rust emitted for it. -/
theorem rows_imply_delayedOpenSource
    (assignment : Nat → Nat)
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment) :
    assignment 28043385 = assignment delayedOpenSourceColumn := by
  simpa [delayedOpenOffset, delayedOpenSourceColumn] using
    rows_imply_delayedPayloadSource assignment canonicalAssignment one
      satisfied delayedOpenOffset delayedOpenOffset_bound

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBindingRowSound
