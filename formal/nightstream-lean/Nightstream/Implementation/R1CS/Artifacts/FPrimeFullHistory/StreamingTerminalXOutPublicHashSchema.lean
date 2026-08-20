import Nightstream.Implementation.Nebula.Core.FieldCodec
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-!
Contract: compact source-row schema for the recursive-terminal XOut public
Poseidon2 hash.

Rust owns the source rows and final selective-row projection. This schema
reconstructs one exact nine-round sponge, four canonical-u64 blocks, and the
256 equality pins to verifier-owned public bits.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact

open Nightstream.Implementation.R1CS

structure Range where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

def Range.Valid (range : Range) : Prop :=
  range.start ≤ range.stop

instance (range : Range) : Decidable range.Valid := by
  unfold Range.Valid
  infer_instance

structure PublicWord where
  fieldColumn : Nat
  canonicalBitColumns : List Nat
  highIsMaxColumn : Nat
  inverseColumn : Nat
  publicBitColumns : List Nat
  canonicalRows : Range
  equalityRows : List Nat
deriving DecidableEq, Repr

def PublicWord.columnMap (word : PublicWord) : List Nat :=
  [0, word.fieldColumn] ++ word.canonicalBitColumns ++
    [word.highIsMaxColumn, word.inverseColumn]

def PublicWord.linkPairs (word : PublicWord) : List (Nat × Nat) :=
  (List.range 64).map fun index =>
    (word.publicBitColumns.getD index 0,
      word.canonicalBitColumns.getD index 0)

def PublicWord.canonicalProgram (word : PublicWord) : List Row :=
  Nightstream.Implementation.R1CS.CanonicalU64.rows.map
    (Relabel.row word.columnMap)

def PublicWord.linkProgram (word : PublicWord) : List Row :=
  EqualityPins.rows word.linkPairs

def PublicWord.rows (word : PublicWord) : List Row :=
  word.canonicalProgram ++ word.linkProgram

def PublicWord.Valid (word : PublicWord) : Prop :=
  word.canonicalBitColumns.length = 64 ∧
    word.publicBitColumns.length = 64 ∧
    word.equalityRows.length = 64 ∧
    word.canonicalRows.Valid ∧
    word.canonicalRows.stop - word.canonicalRows.start = 69

instance (word : PublicWord) : Decidable word.Valid := by
  unfold PublicWord.Valid
  infer_instance

structure FirstLeafPlacement where
  rewriteId : Nat
  sourceRows : Range
  finalRows : Range
  finalColumns : Nat
  selectorColumn : Nat
  externalSlotStarts : List Nat
  localSlotStart : Nat
  slotWidth : Nat
  localSlotCount : Nat
deriving DecidableEq, Repr

def FirstLeafPlacement.Valid (placement : FirstLeafPlacement) : Prop :=
  placement.sourceRows.Valid ∧
    placement.finalRows.Valid ∧
    placement.sourceRows.stop - placement.sourceRows.start = 600 ∧
    placement.finalRows.stop - placement.finalRows.start = 86 ∧
    placement.selectorColumn < placement.finalColumns ∧
    placement.externalSlotStarts.length = 3 ∧
    placement.slotWidth = 41 ∧
    placement.localSlotCount = 86 ∧
    (∀ start ∈ placement.externalSlotStarts,
      start + placement.slotWidth ≤ placement.finalColumns) ∧
    placement.localSlotStart +
      placement.localSlotCount * placement.slotWidth ≤ placement.finalColumns

instance (placement : FirstLeafPlacement) : Decidable placement.Valid := by
  unfold FirstLeafPlacement.Valid
  infer_instance

structure AbsoluteTerm where
  column : Nat
  coefficient : Nat
deriving DecidableEq, Repr

structure AbsoluteGeometricRun where
  columnStart : Nat
  length : Nat
  initial : Nat
  ratio : Nat
deriving DecidableEq, Repr

structure AbsolutePort where
  explicit : List AbsoluteTerm
  geometric : List AbsoluteGeometricRun
deriving DecidableEq, Repr

structure SourceImage where
  sourceColumn : Nat
  port : AbsolutePort
deriving DecidableEq, Repr

/-- Compact source-to-final placement for one 86-row Poseidon2 call. The
eight images align with `round.call.inputColumns`. The 86 local slots own
the S-box outputs. All other trace values are derived linear forms. -/
structure PoseidonCallPlacement where
  roundIndex : Nat
  rewriteId : Nat
  sourceRows : Range
  finalRows : Range
  finalColumns : Nat
  selectorColumn : Nat
  localSlotStart : Nat
  slotWidth : Nat
  localSlotCount : Nat
  inputSourceColumns : List Nat
  inputImages : List SourceImage
deriving DecidableEq, Repr

def PoseidonCallPlacement.Valid
    (placement : PoseidonCallPlacement) : Prop :=
  placement.sourceRows.Valid ∧
    placement.finalRows.Valid ∧
    placement.sourceRows.stop - placement.sourceRows.start = 600 ∧
    placement.finalRows.stop - placement.finalRows.start = 86 ∧
    placement.selectorColumn < placement.finalColumns ∧
    placement.slotWidth = 41 ∧
    placement.localSlotCount = 86 ∧
    placement.localSlotStart +
      placement.localSlotCount * placement.slotWidth ≤ placement.finalColumns ∧
    placement.inputSourceColumns.length = 8 ∧
    placement.inputImages.length = 8

instance (placement : PoseidonCallPlacement) : Decidable placement.Valid := by
  unfold PoseidonCallPlacement.Valid
  infer_instance

/-- Exact source and final ownership for one retained Poseidon2 output-copy
row. `finalPorts` are the thirteen Rust-emitted selective matrix ports. -/
structure OutputCopyPlacement where
  lane : Nat
  rewriteId : Nat
  sourceRows : Range
  finalRow : Nat
  finalRows : Nat
  finalColumns : Nat
  selectorColumn : Nat
  outputSourceColumn : Nat
  linearFormConstant : Nat
  linearFormTerms : List AbsoluteTerm
  finalPorts : List AbsolutePort
deriving DecidableEq, Repr

def OutputCopyPlacement.Valid (placement : OutputCopyPlacement) : Prop :=
  placement.lane < 4 ∧
    placement.sourceRows.Valid ∧
    placement.finalRow < placement.finalRows ∧
    placement.selectorColumn < placement.finalColumns ∧
    placement.finalPorts.length = 13

instance (placement : OutputCopyPlacement) : Decidable placement.Valid := by
  unfold OutputCopyPlacement.Valid
  infer_instance

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceArtifactIdentity : String
  finalArtifactIdentity : String
  lifecycleScope : String
  rowFamily : String
  sourceHashRows : Range
  sourceRowRuns : List Range
  finalRowRuns : List Range
  firstLeafPlacement : FirstLeafPlacement
  callPlacements : List PoseidonCallPlacement
  outputCopies : List OutputCopyPlacement
  xOutImages : List SourceImage
  outputImages : List SourceImage
  trace : Poseidon2Sponge.Trace
  publicWords : List PublicWord
deriving DecidableEq, Repr

def RawArtifact.rows (artifact : RawArtifact) : List Row :=
  artifact.trace.rows ++ artifact.publicWords.flatMap PublicWord.rows

def RawArtifact.Satisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.rows assignment

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 2 ∧
    artifact.profileId =
      "nightstream/goldilocks/b2-k16/streaming-terminal-x-out-public-hash/v2" ∧
    artifact.sourceArtifactIdentity =
      "rust:nightstream/streaming-terminal-lifecycle/source-rows/v1" ∧
    artifact.finalArtifactIdentity =
      "rust:nightstream/streaming-selective-ccs/final-rows/v1" ∧
    artifact.lifecycleScope = "recursive-terminal-arm-435" ∧
    artifact.rowFamily = "terminal.streaming.x_out.public_hash" ∧
    artifact.sourceHashRows.Valid ∧
    (∀ range ∈ artifact.sourceRowRuns, range.Valid) ∧
    (∀ range ∈ artifact.finalRowRuns, range.Valid) ∧
    artifact.firstLeafPlacement.Valid ∧
    artifact.callPlacements.length = 9 ∧
    (∀ placement ∈ artifact.callPlacements, placement.Valid) ∧
    List.Forall₂
      (fun lane placement =>
        placement.lane = lane ∧
          placement.outputSourceColumn =
            artifact.trace.outputColumns.getD lane 0 ∧
          placement.Valid)
      [0, 1, 2, 3] artifact.outputCopies ∧
    artifact.xOutImages.length = 32 ∧
    artifact.outputImages.length = 4 ∧
    artifact.trace.inputColumns.length = 32 ∧
    artifact.trace.rounds.length = 9 ∧
    artifact.trace.outputColumns.length = 4 ∧
    artifact.publicWords.length = 4 ∧
    List.Forall₂
      (fun lane word =>
        word.fieldColumn = artifact.trace.outputColumns.getD lane 0 ∧
          word.Valid)
      (List.range 4) artifact.publicWords

instance (artifact : RawArtifact) : Decidable artifact.Valid := by
  unfold RawArtifact.Valid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact
