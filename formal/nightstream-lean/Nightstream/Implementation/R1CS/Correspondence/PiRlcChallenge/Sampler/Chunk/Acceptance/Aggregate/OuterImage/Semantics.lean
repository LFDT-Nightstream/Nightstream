import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.AggregateAcceptanceOuterImageSchema
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.ArtifactRefinement

/-!
Semantic interpretation of one aggregate-acceptance outer image.

Owns: evaluation of direct decoded images, Boolean-owner equations, physical
output placement, and the conditional soundness/completeness bridge from
active rows to independent source acceptance semantics.

Does not own: generated coordinates, proof that the fixed artifact has the
required shape, extraction from Rust, satisfaction of the complete R1CS,
cost totals, or permission to remove constraints.

Emits constraints: no. This module assigns independent mathematical meaning
to the production evidence.

| Stage path | Mathematical obligation | Authority class | Principal result |
|---|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.bits.decoder.singleton` | source bit equals its encoded coordinate | checked | `DecoderAgreement` |
| `nifs.pi_rlc.challenge.sampler.chunk.bits.boolean_owner` | decoded value is a field bit | checked | `booleanOwner_holds_iff` |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | nine active rows prove tree and accept meaning | checked | `activeRowsHold_iff_sourceMeaning` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage

open Nightstream.Implementation.R1CS
open Mod5
open AggregateAcceptanceOuterImageArtifact

abbrev ColumnAssignment := Nat → GateField

def bitValue
    (assignment : ColumnAssignment) (bit : BitOuterImage) : GateField :=
  assignment bit.encodedColumn

/-- Decode the sixteen input positions. Missing positions evaluate to zero;
the Boolean-owner obligation below separately requires every position to have
an artifact record. -/
def decodedChunkBits
    (assignment : ColumnAssignment) (chunk : ChunkOuterImage) :
    Fin 16 → GateField :=
  fun index =>
    match chunk.bits[index.val]? with
    | some bit => bitValue assignment bit
    | none => 0

/-- Read the corresponding authoritative source columns. -/
def sourceChunkBits
    (sourceAssignment : ColumnAssignment) (chunk : ChunkOuterImage) :
    Fin 16 → GateField :=
  fun index =>
    match chunk.bits[index.val]? with
    | some bit => sourceAssignment bit.sourceColumn
    | none => 0

/-- Every decoded physical input equals the source value named by its record. -/
def DecoderAgreement
    (sourceAssignment encodedAssignment : ColumnAssignment)
    (chunk : ChunkOuterImage) : Prop :=
  ∀ index,
    decodedChunkBits encodedAssignment chunk index =
      sourceChunkBits sourceAssignment chunk index

/-- Interpret the exact fourteen-output physical interval. -/
def encodedOutputs
    (assignment : ColumnAssignment) (chunk : ChunkOuterImage) :
    ProductTreeOutputs :=
  fun index => assignment (chunk.encodedOutputStart + index.val)

/-- The mathematical equation owned by one physical Boolean row. Pair-right
reverses the row orientation. Row numbers remain placement evidence, not
semantic authority. -/
def booleanOwnerHolds
    (assignment : ColumnAssignment) (value : GateField) :
    BooleanOwner → Prop
  | .pairLeft _ pairedColumn =>
      QuadraticZeroPair (bitResidual value)
        (bitResidual (assignment pairedColumn))
  | .pairRight _ pairedColumn =>
      QuadraticZeroPair (bitResidual (assignment pairedColumn))
        (bitResidual value)

/-- Boolean facts needed for completeness of one owner equation. -/
def booleanOwnerContextBoolean
    (assignment : ColumnAssignment) (value : GateField) :
    BooleanOwner → Prop
  | .pairLeft _ pairedColumn | .pairRight _ pairedColumn =>
      FieldBit value ∧ FieldBit (assignment pairedColumn)

theorem booleanOwner_holds_iff
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    (assignment : ColumnAssignment) (value : GateField)
    (owner : BooleanOwner) :
    booleanOwnerHolds assignment value owner ↔
      booleanOwnerContextBoolean assignment value owner := by
  cases owner with
  | pairLeft row pairedColumn =>
      change
        QuadraticZeroPair (bitResidual value)
            (bitResidual (assignment pairedColumn)) ↔
          FieldBit value ∧ FieldBit (assignment pairedColumn)
      rw [quadraticZeroPair_iff nonresidue,
        bitResidual_zero_iff prime, bitResidual_zero_iff prime]
  | pairRight row pairedColumn =>
      change
        QuadraticZeroPair (bitResidual (assignment pairedColumn))
            (bitResidual value) ↔
          FieldBit value ∧ FieldBit (assignment pairedColumn)
      rw [quadraticZeroPair_iff nonresidue,
        bitResidual_zero_iff prime, bitResidual_zero_iff prime]
      constructor <;> rintro ⟨left, right⟩ <;> exact ⟨right, left⟩

theorem booleanOwnerHolds_fieldBit
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    (assignment : ColumnAssignment) (value : GateField)
    (owner : BooleanOwner)
    (holds : booleanOwnerHolds assignment value owner) :
    FieldBit value := by
  have context :=
    (booleanOwner_holds_iff prime nonresidue assignment value owner).mp holds
  cases owner with
  | pairLeft row pairedColumn => exact context.1
  | pairRight row pairedColumn => exact context.1

/-- Every input position exists and its exact physical owner equation holds. -/
def BooleanOwnersHold
    (assignment : ColumnAssignment) (chunk : ChunkOuterImage) : Prop :=
  ∀ index : Fin 16, ∃ bit,
    chunk.bits[index.val]? = some bit ∧
      booleanOwnerHolds assignment (bitValue assignment bit) bit.owner

theorem decodedChunkBits_are_boolean
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    {assignment : ColumnAssignment} {chunk : ChunkOuterImage}
    (rows : BooleanOwnersHold assignment chunk) :
    ∀ index, FieldBit (decodedChunkBits assignment chunk index) := by
  intro index
  rcases rows index with ⟨bit, present, holds⟩
  have bitBoolean :=
    booleanOwnerHolds_fieldBit prime nonresidue assignment
      (bitValue assignment bit) bit.owner holds
  simpa [decodedChunkBits, present] using bitBoolean

/-- The exact nine-row role-normalized leaf evaluated at this chunk's physical
columns. -/
def ActiveRowsHold
    (assignment : ColumnAssignment) (chunk : ChunkOuterImage) : Prop :=
  GeneratedAggregateAcceptanceRows
    (decodedChunkBits assignment chunk)
    (encodedOutputs assignment chunk)
    (assignment chunk.encodedAccept)

/-- Once decoder and Boolean-owner obligations are discharged, the nine
physical active rows are sound and complete for the independent source
product-tree and acceptance meanings. -/
theorem activeRowsHold_iff_sourceMeaning
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    {sourceAssignment encodedAssignment : ColumnAssignment}
    {chunk : ChunkOuterImage}
    (booleanRows : BooleanOwnersHold encodedAssignment chunk)
    (decoder : DecoderAgreement sourceAssignment encodedAssignment chunk) :
    ActiveRowsHold encodedAssignment chunk ↔
      ProductTreeMeaning
          (candidateBits (sourceChunkBits sourceAssignment chunk))
          (encodedOutputs encodedAssignment chunk) ∧
        SourceAcceptanceMeaning
          (candidateBits (sourceChunkBits sourceAssignment chunk))
          (encodedAssignment chunk.encodedAccept) := by
  have decodedBoolean :=
    decodedChunkBits_are_boolean prime nonresidue booleanRows
  have decodedExact :
      decodedChunkBits encodedAssignment chunk =
        sourceChunkBits sourceAssignment chunk := by
    funext index
    exact decoder index
  unfold ActiveRowsHold
  rw [generatedAggregateAcceptanceRows_iff_sourceMeaning
    prime nonresidue _ _ _ decodedBoolean]
  rw [decodedExact]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage
