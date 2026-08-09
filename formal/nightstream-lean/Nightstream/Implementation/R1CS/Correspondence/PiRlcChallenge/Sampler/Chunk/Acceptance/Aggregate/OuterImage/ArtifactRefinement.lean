import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.AggregateAcceptanceOuterImage
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage.Semantics

/-!
Kernel-checked census of the fixed recursive aggregate-acceptance outer image.

Owns: exact protocol dimensions, direct-decoder and Boolean-owner counts, and
unique source/physical row reconciliation for the generated 960-chunk
artifact.

Does not own: semantic equations (in `OuterImage.Semantics`), Rust extraction,
complete R1CS satisfaction, global F′ soundness, cost estimates outside this
subtree, or permission to remove constraints.

Emits constraints: no. All theorems compute over generated evidence and are
checked by Lean's kernel.

| Stage path | Generated leaves | Exact obligation |
|---|---:|---|
| `nifs.pi_rlc.challenge.sampler.chunk.bits.decoder.singleton` | 15,360 | direct encoded coordinate |
| `nifs.pi_rlc.challenge.sampler.chunk.bits.boolean_owner` | 7,680 unique rows | 7,680 left and 7,680 aliased right |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | 8,640 rows | nine active rows × 960 chunks |
| complete selected outer image | 19,200 source / 16,320 physical rows | no unexplained or multiply counted unique row |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage

open AggregateAcceptanceOuterImageArtifact
open AggregateAcceptanceOuterImageData
open Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.Generated.AggregateAcceptanceOuterImage

def allBits : List BitOuterImage :=
  chunks.flatMap (fun chunk => chunk.bits)

def ownerCount (predicate : BooleanOwner → Bool) : Nat :=
  (allBits.filter fun bit => predicate bit.owner).length

def selectedSourceRows : List Nat :=
  chunks.flatMap (fun chunk =>
    chunk.sourceRows ++ chunk.bits.map (fun bit => bit.sourceBooleanRow))

def selectedBooleanRows : List Nat :=
  allBits.map (fun bit => bit.owner.encodedRow)

def selectedActiveRows : List Nat :=
  chunks.flatMap (fun chunk => chunk.activeRows)

def selectedPhysicalRows : List Nat :=
  selectedBooleanRows ++ selectedActiveRows

/-- Fixed relation dimensions and top-level phase counts. -/
theorem generated_outer_image_shape_exact :
    Shape.schemaVersion = 2 ∧
      Shape.sourceRowCount = 7_169_252 ∧
      Shape.sourceColumnCount = 7_100_181 ∧
      Shape.encodedRowCount = 7_253_817 ∧
      Shape.encodedColumnCount = 9_820_662 ∧
      Shape.matrixArity = 56 ∧
      AggregateAcceptanceOuterImageData.challenges.length = 15 ∧
      (∀ challenge ∈ AggregateAcceptanceOuterImageData.challenges,
        challenge.length = 64) ∧
      chunks.length = 960 ∧
      allBits.length = 15_360 := by
  native_decide

/-- Every chunk has the fixed direct-decoder leaf shape. -/
theorem generated_decoder_tree_exact :
    Shape.directDecoderCount = 15_360 ∧
      (∀ chunk ∈ chunks, chunk.bits.length = 16) ∧
      (∀ chunk ∈ chunks, chunk.encodedOutputs.length = 14) ∧
      (∀ chunk ∈ chunks, chunk.activeRows.length = 9) := by
  native_decide

/-- Exact owner-family and active-row reconciliation. Right-owner entries
alias the matching left physical row; unique-row counts expose that sharing. -/
theorem generated_physical_row_tree_exact :
    ownerCount BooleanOwner.isPairLeft = 7_680 ∧
      ownerCount BooleanOwner.isPairRight = 7_680 ∧
      selectedBooleanRows.eraseDups.length = 7_680 ∧
      selectedActiveRows.length = 8_640 ∧
      selectedActiveRows.eraseDups.length = 8_640 ∧
      selectedPhysicalRows.eraseDups.length = 16_320 := by
  native_decide

/-- Exact source-row reconciliation. Equality of raw and deduplicated counts
proves that every selected source row has one leaf owner in this tree. -/
theorem generated_source_row_tree_exact :
    selectedSourceRows.length = 19_200 ∧
      selectedSourceRows.eraseDups.length = 19_200 := by
  native_decide

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage
