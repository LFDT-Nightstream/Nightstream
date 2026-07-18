import Nightstream.Implementation.R1CS.Core.Program

/-!
Typed artifact language for the fixed recursive aggregate-acceptance image.

Owns: exact finite records for decoded source bits, removed linear-definition
provenance, Boolean-row owners, chunk placement and global source/encoded
dimensions.

Does not own: generated values, mathematical acceptance semantics, row
evaluation, Rust conformance, or permission to remove constraints.

Emits constraints: no.

Authority boundary: these structures describe non-authoritative production
evidence. Handwritten correspondence must interpret every record against
independent semantics.

| Stage path | Record branch | Mathematical obligation |
|---|---|---|
| `nifs.pi_rlc.challenge.sampler.chunk.bits.decoder` | `DecodedImage` | source bit equals its exact encoded singleton or sparse LC |
| `nifs.pi_rlc.challenge.sampler.chunk.bits.boolean_owner` | `BooleanOwner` | one physical row proves decoded bit membership |
| `nifs.pi_rlc.challenge.sampler.chunk.accept.packed` | `ChunkOuterImage` | sixteen inputs and nine active rows share one exact placement |
| source linear schedule | `LinearDefinition` | every removed definition retains its exact source-row provenance |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceOuterImageArtifact

/-- A decoded bit is either one retained Boolean coordinate or one generated
391-term pattern translated by `encodedStart`. -/
inductive DecodedImage where
  | singleton (encodedColumn : Nat)
  | sparseLinear (pattern encodedStart : Nat)
deriving DecidableEq, Repr

/-- Exact physical row that owns Booleanity for one decoded source bit. -/
inductive BooleanOwner where
  | pairLeft (encodedRow pairedColumn : Nat)
  | pairRight (encodedRow pairedColumn : Nat)
  | translatedSource (sourceRow encodedRow : Nat)
deriving DecidableEq, Repr

def BooleanOwner.encodedRow : BooleanOwner → Nat
  | .pairLeft row _ | .pairRight row _ | .translatedSource _ row => row

def BooleanOwner.isPairLeft : BooleanOwner → Bool
  | .pairLeft .. => true
  | _ => false

def BooleanOwner.isPairRight : BooleanOwner → Bool
  | .pairRight .. => true
  | _ => false

def BooleanOwner.isTranslated : BooleanOwner → Bool
  | .translatedSource .. => true
  | _ => false

/-- One exact source-to-source term in a removed generic linear definition. -/
structure SourceLinearTerm where
  column : Nat
  coefficient : Int
deriving DecidableEq, Repr

/-- One removed source definition and its original source-row owner. -/
structure LinearDefinition where
  sourceColumn : Nat
  sourceRow : Nat
  terms : List SourceLinearTerm
deriving DecidableEq, Repr

/-- Complete outer image for one of the sixteen source bits. -/
structure BitOuterImage where
  sourceColumn : Nat
  sourceBooleanRow : Nat
  decoded : DecodedImage
  definitionColumns : List Nat
  owner : BooleanOwner
deriving DecidableEq, Repr

def BitOuterImage.isSingleton (bit : BitOuterImage) : Bool :=
  match bit.decoded with
  | .singleton _ => true
  | .sparseLinear _ _ => false

def BitOuterImage.isSparse (bit : BitOuterImage) : Bool :=
  !bit.isSingleton

/-- Physical/source placement of one 16-input, 14-output, nine-row active
aggregate-acceptance chunk. Fixed lengths are checked by the generated-data
certificate rather than hidden in the record type. -/
structure ChunkOuterImage where
  sourceRowStart : Nat
  sourceAcceptColumn : Nat
  sourceInverseColumn : Nat
  bits : List BitOuterImage
  encodedAccept : Nat
  encodedOutputStart : Nat
  activeRowStart : Nat
deriving DecidableEq, Repr

def ChunkOuterImage.sourceRows (chunk : ChunkOuterImage) : List Nat :=
  List.range' chunk.sourceRowStart 4

def ChunkOuterImage.encodedOutputs (chunk : ChunkOuterImage) : List Nat :=
  List.range' chunk.encodedOutputStart 14

def ChunkOuterImage.activeRows (chunk : ChunkOuterImage) : List Nat :=
  List.range' chunk.activeRowStart 9

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceOuterImageArtifact
