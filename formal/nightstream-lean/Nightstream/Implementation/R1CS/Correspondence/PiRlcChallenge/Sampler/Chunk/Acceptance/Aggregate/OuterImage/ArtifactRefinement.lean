import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.AggregateAcceptanceOuterImage
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage.Semantics

/-!
Kernel-checked census of the fixed recursive aggregate-acceptance outer image.

Owns: exact protocol/phase/family/leaf counts, decoder-definition coverage,
Boolean-owner census, and unique source/physical row reconciliation for the
generated 960-chunk artifact.

Does not own: semantic equations (in `OuterImage.Semantics`), Rust extraction,
complete R1CS satisfaction, global F′ soundness, cost estimates outside this
subtree, or permission to remove constraints.

Emits constraints: no. All theorems compute over generated evidence and are
checked by Lean's kernel.

| Stage path | Generated leaves | Exact obligation |
|---|---:|---|
| `nifs.pi_rlc.challenge.sampler.chunk.bits.decoder.singleton` | 15,120 | direct encoded coordinate |
| `nifs.pi_rlc.challenge.sampler.chunk.bits.decoder.sparse_linear` | 240 × 391 terms | composed encoded LC plus three source definitions |
| `nifs.pi_rlc.challenge.sampler.chunk.bits.boolean_owner` | 7,920 unique rows | 7,680 left, 7,440 aliased right, 240 translated |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | 8,640 rows | nine active rows × 960 chunks |
| complete selected outer image | 19,920 source / 16,560 physical rows | no unexplained or multiply counted unique row |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage

open AggregateAcceptanceOuterImageArtifact
open AggregateAcceptanceOuterImageData

def allBits : List BitOuterImage :=
  chunks.flatMap (fun chunk => chunk.bits)

def sparseDecoderCountFor (pattern : Nat) : Nat :=
  (allBits.filter fun bit =>
    match bit.decoded with
    | .sparseLinear candidate _ => candidate == pattern
    | .singleton _ => false).length

def ownerCount (predicate : BooleanOwner → Bool) : Nat :=
  (allBits.filter fun bit => predicate bit.owner).length

def decoderWellFormed (bit : BitOuterImage) : Bool :=
  match bit.decoded with
  | .singleton _ => bit.definitionColumns.isEmpty
  | .sparseLinear pattern _ =>
      pattern < sparseLinearPatterns.length &&
        bit.definitionColumns.length == 3

def definitionWidthCount (width : Nat) : Nat :=
  (linearDefinitions.filter fun definition =>
    definition.terms.length == width).length

def selectedSourceRows : List Nat :=
  linearDefinitions.map (fun definition => definition.sourceRow) ++
    chunks.flatMap (fun chunk =>
      chunk.sourceRows ++ chunk.bits.map (fun bit => bit.sourceBooleanRow))

def selectedBooleanRows : List Nat :=
  allBits.map (fun bit => bit.owner.encodedRow)

def selectedActiveRows : List Nat :=
  chunks.flatMap (fun chunk => chunk.activeRows)

def selectedPhysicalRows : List Nat :=
  selectedBooleanRows ++ selectedActiveRows

def referencedDefinitionColumns : List Nat :=
  allBits.flatMap (fun bit => bit.definitionColumns)

def definedSourceColumns : List Nat :=
  linearDefinitions.map (fun definition => definition.sourceColumn)

/-- Fixed relation dimensions and top-level phase counts. -/
theorem generated_outer_image_shape_exact :
    AggregateAcceptanceOuterImageData.challenges.length = 15 ∧
      definitionShards.length = 15 ∧
      (∀ shard ∈ definitionShards, shard.length = 48) ∧
      (∀ challenge ∈ AggregateAcceptanceOuterImageData.challenges,
        challenge.length = 64) ∧
      chunks.length = 960 ∧
      allBits.length = 15_360 ∧
      linearDefinitions.length = 720 := by
  native_decide

/-- Every leaf has the fixed profile shape, and every sparse decoder names
exactly one of the four complete 391-term patterns. -/
theorem generated_decoder_tree_exact :
    sparseLinearPatterns.length = 4 ∧
      (∀ pattern ∈ sparseLinearPatterns, pattern.length = 391) ∧
      (allBits.filter BitOuterImage.isSingleton).length = 15_120 ∧
      (allBits.filter BitOuterImage.isSparse).length = 240 ∧
      sparseDecoderCountFor 0 = 60 ∧
      sparseDecoderCountFor 1 = 60 ∧
      sparseDecoderCountFor 2 = 60 ∧
      sparseDecoderCountFor 3 = 60 ∧
      allBits.all decoderWellFormed = true := by
  native_decide

/-- The 720 source-definition leaves reconcile exactly with the 240 sparse
decoder leaves and their 1/8/64-width provenance layers. -/
theorem generated_definition_tree_exact :
    definitionWidthCount 1 = 240 ∧
      definitionWidthCount 8 = 240 ∧
      definitionWidthCount 64 = 240 ∧
      referencedDefinitionColumns.length = 720 ∧
      referencedDefinitionColumns.eraseDups.length = 720 ∧
      referencedDefinitionColumns.all
        (fun column => definedSourceColumns.contains column) = true := by
  native_decide

/-- Exact owner-family and active-row reconciliation. Right-owner entries
alias the matching left physical row; unique-row counts expose that sharing. -/
theorem generated_physical_row_tree_exact :
    ownerCount BooleanOwner.isPairLeft = 7_680 ∧
      ownerCount BooleanOwner.isPairRight = 7_440 ∧
      ownerCount BooleanOwner.isTranslated = 240 ∧
      selectedBooleanRows.eraseDups.length = 7_920 ∧
      selectedActiveRows.length = 8_640 ∧
      selectedActiveRows.eraseDups.length = 8_640 ∧
      selectedPhysicalRows.eraseDups.length = 16_560 := by
  native_decide

/-- Exact source-row reconciliation. Equality of raw and deduplicated counts
proves that every selected source row has one leaf owner in this tree. -/
theorem generated_source_row_tree_exact :
    selectedSourceRows.length = 19_920 ∧
      selectedSourceRows.eraseDups.length = 19_920 := by
  native_decide

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage
