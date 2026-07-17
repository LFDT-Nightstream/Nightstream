import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.AggregateAcceptanceOuterImageSchema
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.ArtifactRefinement

/-!
Semantic interpretation of one aggregate-acceptance outer image.

Owns: evaluation of singleton and sparse decoded images, removed source-linear
definitions, Boolean-owner equations, physical output placement, and the
conditional soundness/completeness bridge from active rows to independent
source acceptance semantics.

Does not own: generated coordinates, proof that the fixed artifact has the
required shape, extraction from Rust, satisfaction of the complete R1CS,
cost totals, or permission to remove constraints.

Emits constraints: no. This module assigns independent mathematical meaning
to the production evidence.

| Stage path | Mathematical obligation | Authority class | Principal result |
|---|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.bits.decoder` | source bit equals singleton or exact sparse LC | checked | `DecoderAgreement` |
| source linear schedule | removed source value equals its defining LC | checked provenance | `linearDefinitionHolds` |
| `nifs.pi_rlc.challenge.sampler.chunk.bits.boolean_owner` | decoded value is a field bit | checked | `booleanOwner_holds_iff` |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | nine active rows prove tree and accept meaning | checked | `activeRowsHold_iff_sourceMeaning` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage

open Nightstream.Implementation.R1CS
open Mod5
open AggregateAcceptanceArtifact
open AggregateAcceptanceOuterImageArtifact

abbrev ColumnAssignment := Nat → GateField

/-- Evaluate an exact list of source or encoded linear terms. -/
def evalLinearTerms
    (assignment : ColumnAssignment) (base : Nat)
    (terms : List SourceLinearTerm) : GateField :=
  (terms.map fun term =>
    coefficient term.coefficient * assignment (base + term.column)).sum

/-- Interpret one generated decoder using handwritten field arithmetic. -/
def decodedImageValue
    (patterns : List (List SourceLinearTerm))
    (assignment : ColumnAssignment) : DecodedImage → GateField
  | .singleton column => assignment column
  | .sparseLinear pattern encodedStart =>
      match patterns[pattern]? with
      | some terms => evalLinearTerms assignment encodedStart terms
      | none => 0

def bitValue
    (patterns : List (List SourceLinearTerm))
    (assignment : ColumnAssignment) (bit : BitOuterImage) : GateField :=
  decodedImageValue patterns assignment bit.decoded

/-- One removed generic-linear source row retains its exact equation. -/
def linearDefinitionHolds
    (assignment : ColumnAssignment) (definition : LinearDefinition) : Prop :=
  assignment definition.sourceColumn =
    evalLinearTerms assignment 0 definition.terms

def LinearDefinitionsHold
    (assignment : ColumnAssignment) (definitions : List LinearDefinition) : Prop :=
  ∀ definition ∈ definitions, linearDefinitionHolds assignment definition

/-- Decode the sixteen input positions. Missing positions evaluate to zero;
the Boolean-owner obligation below separately requires every position to have
an artifact record. -/
def decodedChunkBits
    (patterns : List (List SourceLinearTerm))
    (assignment : ColumnAssignment) (chunk : ChunkOuterImage) :
    Fin 16 → GateField :=
  fun index =>
    match chunk.bits[index.val]? with
    | some bit => bitValue patterns assignment bit
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
    (patterns : List (List SourceLinearTerm))
    (sourceAssignment encodedAssignment : ColumnAssignment)
    (chunk : ChunkOuterImage) : Prop :=
  ∀ index,
    decodedChunkBits patterns encodedAssignment chunk index =
      sourceChunkBits sourceAssignment chunk index

/-- Interpret the exact fourteen-output physical interval. -/
def encodedOutputs
    (assignment : ColumnAssignment) (chunk : ChunkOuterImage) :
    ProductTreeOutputs :=
  fun index => assignment (chunk.encodedOutputStart + index.val)

/-- The mathematical equation owned by one physical Boolean row. Pair-right
reverses the row orientation; translated rows directly constrain the decoded
sparse value. Row numbers remain placement evidence, not semantic authority. -/
def booleanOwnerHolds
    (assignment : ColumnAssignment) (value : GateField) :
    BooleanOwner → Prop
  | .pairLeft _ pairedColumn =>
      QuadraticZeroPair (bitResidual value)
        (bitResidual (assignment pairedColumn))
  | .pairRight _ pairedColumn =>
      QuadraticZeroPair (bitResidual (assignment pairedColumn))
        (bitResidual value)
  | .translatedSource _ _ => bitResidual value = 0

/-- Boolean facts needed for completeness of one owner equation. -/
def booleanOwnerContextBoolean
    (assignment : ColumnAssignment) (value : GateField) :
    BooleanOwner → Prop
  | .pairLeft _ pairedColumn | .pairRight _ pairedColumn =>
      FieldBit value ∧ FieldBit (assignment pairedColumn)
  | .translatedSource _ _ => FieldBit value

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
  | translatedSource sourceRow encodedRow =>
      change bitResidual value = 0 ↔ FieldBit value
      exact bitResidual_zero_iff prime value

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
  | translatedSource sourceRow encodedRow => exact context

/-- Every input position exists and its exact physical owner equation holds. -/
def BooleanOwnersHold
    (patterns : List (List SourceLinearTerm))
    (assignment : ColumnAssignment) (chunk : ChunkOuterImage) : Prop :=
  ∀ index : Fin 16, ∃ bit,
    chunk.bits[index.val]? = some bit ∧
      booleanOwnerHolds assignment (bitValue patterns assignment bit) bit.owner

theorem decodedChunkBits_are_boolean
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    {patterns : List (List SourceLinearTerm)}
    {assignment : ColumnAssignment} {chunk : ChunkOuterImage}
    (rows : BooleanOwnersHold patterns assignment chunk) :
    ∀ index, FieldBit (decodedChunkBits patterns assignment chunk index) := by
  intro index
  rcases rows index with ⟨bit, present, holds⟩
  have bitBoolean :=
    booleanOwnerHolds_fieldBit prime nonresidue assignment
      (bitValue patterns assignment bit) bit.owner holds
  simpa [decodedChunkBits, present] using bitBoolean

/-- The exact nine-row role-normalized leaf evaluated at this chunk's physical
columns. -/
def ActiveRowsHold
    (patterns : List (List SourceLinearTerm))
    (assignment : ColumnAssignment) (chunk : ChunkOuterImage) : Prop :=
  GeneratedAggregateAcceptanceRows
    (decodedChunkBits patterns assignment chunk)
    (encodedOutputs assignment chunk)
    (assignment chunk.encodedAccept)

/-- Once decoder and Boolean-owner obligations are discharged, the nine
physical active rows are sound and complete for the independent source
product-tree and acceptance meanings. -/
theorem activeRowsHold_iff_sourceMeaning
    (prime : EuclidPrime goldilocksP) (nonresidue : SevenNonresidue)
    {patterns : List (List SourceLinearTerm)}
    {sourceAssignment encodedAssignment : ColumnAssignment}
    {chunk : ChunkOuterImage}
    (booleanRows : BooleanOwnersHold patterns encodedAssignment chunk)
    (decoder : DecoderAgreement patterns sourceAssignment encodedAssignment chunk) :
    ActiveRowsHold patterns encodedAssignment chunk ↔
      ProductTreeMeaning
          (sourceChunkBits sourceAssignment chunk)
          (encodedOutputs encodedAssignment chunk) ∧
        SourceAcceptanceMeaning
          (sourceChunkBits sourceAssignment chunk)
          (encodedAssignment chunk.encodedAccept) := by
  have decodedBoolean :=
    decodedChunkBits_are_boolean prime nonresidue booleanRows
  have decodedExact :
      decodedChunkBits patterns encodedAssignment chunk =
        sourceChunkBits sourceAssignment chunk := by
    funext index
    exact decoder index
  unfold ActiveRowsHold
  rw [generatedAggregateAcceptanceRows_iff_sourceMeaning
    prime nonresidue _ _ _ decodedBoolean]
  rw [decodedExact]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.Aggregate.OuterImage
