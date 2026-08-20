import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingArtifactLeafSchema

/-!
Contract: compact schema for the public-binding suffix of the two PiRLC
family body shapes.

Owns the exact 1,045-field local before and after state-column lists, both
32-field full XOut preimages and four-field outputs, the two derived
program-cursor words, 11 canonical-u64 calls, 544 public-state Poseidon2 calls,
and the small glue-row set after the established algebra-and-replay source
prefix. One exact delegated range belongs to the phase-envelope artifact.

Does not own the source prefix semantics, low-norm slot projection, lifecycle
integration, or Poseidon2 collision resistance.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

inductive OwnerKind where
  | canonical
  | poseidon2
  | glue
  | phaseEnvelope
deriving DecidableEq, Repr, Inhabited

structure Owner where
  rowStart : Nat
  rowEnd : Nat
  kind : OwnerKind
  index : Nat
deriving DecidableEq, Repr, Inhabited

inductive HashRoundKind where
  | absorb
  | pad
deriving DecidableEq, Repr, Inhabited

structure RawHashRound where
  kind : HashRoundKind
  chunkColumns : List Nat
  stateBeforeColumns : List Nat
  permutationInputColumns : List Nat
  definingRows : List Nat
  permutationOutputColumns : List Nat
deriving DecidableEq, Repr, Inhabited

structure RawHash where
  rowStart : Nat
  rowEnd : Nat
  inputColumns : List Nat
  zeroColumn : Nat
  zeroRow : Nat
  permutationCallStart : Nat
  rounds : List RawHashRound
  outputColumns : List Nat
deriving DecidableEq, Repr, Inhabited

structure RawArm where
  sourceRowCount : Nat
  rowCount : Nat
  columnCount : Nat
  publicColumnCount : Nat
  replayPoseidon2CallCount : Nat
  publicPoseidon2CallCount : Nat
  phaseEnvelopeRowStart : Nat
  phaseEnvelopeRowEnd : Nat
  beforeFamilyCursorColumn : Nat
  afterFamilyCursorColumn : Nat
  beforeStateColumns : List Nat
  afterStateColumns : List Nat
  afterXOutPreimageColumns : List Nat
  beforeXOutPreimageColumns : List Nat
  afterXOutDigestColumns : List Nat
  beforeXOutDigestColumns : List Nat
  afterXOutHash : RawHash
  beforeXOutHash : RawHash
  publicWordCallIndices : List Nat
  afterDigestPinColumns : List Nat
  beforeDigestPinColumns : List Nat
  canonicalCalls : List CanonicalCall
  poseidon2Calls : List Poseidon2Call.Call
  glueRows : List IndexedRow
  owners : List Owner
deriving DecidableEq, Repr

/-- Run-compressed source ownership for the final selective public prefix.
Column zero is the affine constant, the middle interval copies source fields
at the same indices, and the last interval is verifier-pinned zero padding. -/
structure RawPublicDecoder where
  constantOneColumn : Nat
  sourceFieldStart : Nat
  sourceFieldEnd : Nat
  paddingStart : Nat
  paddingEnd : Nat
deriving DecidableEq, Repr

def RawPublicDecoder.Valid
    (decoder : RawPublicDecoder) (logicalColumns publicColumns : Nat) : Prop :=
  decoder.constantOneColumn = 0 ∧
    decoder.sourceFieldStart = 1 ∧
    decoder.sourceFieldEnd = logicalColumns ∧
    decoder.paddingStart = logicalColumns ∧
    decoder.paddingEnd = publicColumns

def Owner.Matches (arm : RawArm) (owner : Owner) : Prop :=
  match owner.kind with
  | .canonical =>
      ∃ call, arm.canonicalCalls[owner.index]? = some call ∧
        owner.rowStart = call.rowStart ∧ owner.rowEnd = call.rowEnd
  | .poseidon2 =>
      ∃ call, arm.poseidon2Calls[owner.index]? = some call ∧
        owner.rowStart = call.rowStart ∧ owner.rowEnd = call.rowEnd
  | .glue =>
      ∃ indexed, arm.glueRows[owner.index]? = some indexed ∧
        owner.rowStart = indexed.index ∧ owner.rowEnd = indexed.index + 1
  | .phaseEnvelope =>
      owner.index = 0 ∧ owner.rowStart = arm.phaseEnvelopeRowStart ∧
        owner.rowEnd = arm.phaseEnvelopeRowEnd

instance (arm : RawArm) (owner : Owner) : Decidable (owner.Matches arm) := by
  unfold Owner.Matches
  split <;> infer_instance

def exactOwnerChainFrom (arm : RawArm) : Nat → List Owner → Bool
  | cursor, [] => cursor == arm.rowCount
  | cursor, owner :: rest =>
      owner.rowStart == cursor && decide (owner.rowStart < owner.rowEnd) &&
        decide (owner.Matches arm) && exactOwnerChainFrom arm owner.rowEnd rest

def ownerIndices (kind : OwnerKind) (owners : List Owner) : List Nat :=
  (owners.filter fun owner => owner.kind = kind).map Owner.index

def columnsValid (columnCount expectedLength : Nat)
    (columns : List Nat) : Prop :=
  columns.length = expectedLength ∧ columns.Nodup ∧
    ∀ column ∈ columns, column < columnCount

instance (columnCount expectedLength : Nat) (columns : List Nat) :
    Decidable (columnsValid columnCount expectedLength columns) := by
  unfold columnsValid
  infer_instance

def columnsBelow (columnCount expectedLength : Nat)
    (columns : List Nat) : Prop :=
  columns.length = expectedLength ∧
    ∀ column ∈ columns, column < columnCount

instance (columnCount expectedLength : Nat) (columns : List Nat) :
    Decidable (columnsBelow columnCount expectedLength columns) := by
  unfold columnsBelow
  infer_instance

def poseidon2OutputColumns (call : Poseidon2Call.Call) : List Nat :=
  (List.range 8).map fun lane => call.columnMap (601 + lane)

def hasGlueIndex (arm : RawArm) (index : Nat) : Bool :=
  arm.glueRows.any fun indexed => indexed.index == index

def RawHash.roundValid
    (arm : RawArm) (hash : RawHash) (index : Nat) : Bool :=
  match hash.rounds[index]? with
  | none => false
  | some round =>
      let expectedKind := if index < 8 then HashRoundKind.absorb else .pad
      let expectedChunk :=
        if index < 8 then (hash.inputColumns.drop (4 * index)).take 4 else []
      let expectedState :=
        if index = 0 then List.replicate 8 hash.zeroColumn
        else (hash.rounds.getD (index - 1) default).permutationOutputColumns
      let expectedDefinitionCount := if index < 8 then 4 else 1
      let callValid :=
        match arm.poseidon2Calls[hash.permutationCallStart + index]? with
        | none => false
        | some call =>
            decide (call.inputColumns = round.permutationInputColumns ∧
              poseidon2OutputColumns call = round.permutationOutputColumns)
      decide (round.kind = expectedKind ∧
          round.chunkColumns = expectedChunk ∧
          round.stateBeforeColumns = expectedState ∧
          round.definingRows.length = expectedDefinitionCount ∧
          columnsBelow arm.columnCount 8 round.stateBeforeColumns ∧
          columnsBelow arm.columnCount 8 round.permutationInputColumns ∧
          columnsBelow arm.columnCount 8 round.permutationOutputColumns ∧
          ∀ row ∈ round.definingRows,
            row < arm.rowCount ∧ hasGlueIndex arm row = true) &&
        callValid

def RawHash.Valid
    (arm : RawArm) (expectedInput expectedOutput : List Nat)
    (expectedCallStart : Nat) (hash : RawHash) : Prop :=
  arm.sourceRowCount ≤ hash.rowStart ∧ hash.rowStart < hash.rowEnd ∧
    hash.rowEnd ≤ arm.rowCount ∧
    hash.inputColumns = expectedInput ∧ hash.outputColumns = expectedOutput ∧
    hash.permutationCallStart = expectedCallStart ∧
    columnsValid arm.columnCount 32 hash.inputColumns ∧
    columnsValid arm.columnCount 4 hash.outputColumns ∧
    hash.zeroColumn < arm.columnCount ∧ hash.zeroRow = hash.rowStart ∧
    hasGlueIndex arm hash.zeroRow = true ∧
    hash.rounds.length = 9 ∧
    (List.range 9).all (fun index => RawHash.roundValid arm hash index) = true ∧
    hash.outputColumns =
      (hash.rounds.getD 8 default).permutationOutputColumns.take 4 ∧
    hash.rowEnd =
      (arm.poseidon2Calls.getD (expectedCallStart + 8) default).rowEnd

instance (arm : RawArm) (expectedInput expectedOutput : List Nat)
    (expectedCallStart : Nat) (hash : RawHash) :
    Decidable (hash.Valid arm expectedInput expectedOutput expectedCallStart) := by
  unfold RawHash.Valid
  infer_instance

def RawArm.ScalarValid (arm : RawArm) : Prop :=
  0 < arm.sourceRowCount ∧ arm.sourceRowCount < arm.rowCount ∧
    arm.publicColumnCount = 641 ∧ arm.publicColumnCount ≤ arm.columnCount ∧
    arm.replayPoseidon2CallCount > 0 ∧
    arm.publicPoseidon2CallCount = 544 ∧
    arm.poseidon2Calls.length = arm.publicPoseidon2CallCount ∧
    arm.sourceRowCount ≤ arm.phaseEnvelopeRowStart ∧
    arm.phaseEnvelopeRowStart < arm.phaseEnvelopeRowEnd ∧
    arm.phaseEnvelopeRowEnd ≤ arm.rowCount ∧
    arm.beforeFamilyCursorColumn < arm.columnCount ∧
    arm.afterFamilyCursorColumn < arm.columnCount

def RawArm.StateColumnLayoutValid (arm : RawArm) : Prop :=
    columnsValid arm.columnCount 1045 arm.beforeStateColumns ∧
    columnsValid arm.columnCount 1045 arm.afterStateColumns

def RawArm.XOutColumnLayoutValid (arm : RawArm) : Prop :=
    columnsValid arm.columnCount 32 arm.afterXOutPreimageColumns ∧
    columnsValid arm.columnCount 32 arm.beforeXOutPreimageColumns ∧
    columnsValid arm.columnCount 4 arm.afterXOutDigestColumns ∧
    columnsValid arm.columnCount 4 arm.beforeXOutDigestColumns

def RawArm.HashLayoutValid (arm : RawArm) : Prop :=
    RawHash.Valid arm arm.afterXOutPreimageColumns
      arm.afterXOutDigestColumns 526 arm.afterXOutHash ∧
    RawHash.Valid arm arm.beforeXOutPreimageColumns
      arm.beforeXOutDigestColumns 535 arm.beforeXOutHash

def RawArm.PublicAndPinLayoutValid (arm : RawArm) : Prop :=
    arm.publicWordCallIndices = [3, 4, 5, 6, 7, 8, 9, 10, 0, 1] ∧
    columnsValid arm.columnCount 13 arm.afterDigestPinColumns ∧
    columnsValid arm.columnCount 13 arm.beforeDigestPinColumns

def RawArm.CanonicalCallsValid (arm : RawArm) : Prop :=
    arm.canonicalCalls.length = 11 ∧
    ∀ call ∈ arm.canonicalCalls,
      call.Valid arm.columnCount ∧ arm.sourceRowCount ≤ call.rowStart

def RawArm.Poseidon2CallsValid (arm : RawArm) : Prop :=
    ∀ call ∈ arm.poseidon2Calls,
      PoseidonCallValid arm.columnCount call ∧
        arm.sourceRowCount ≤ call.rowStart

def RawArm.GlueRowsValid (arm : RawArm) : Prop :=
    ∀ indexed ∈ arm.glueRows,
      arm.sourceRowCount ≤ indexed.index ∧ indexed.index < arm.rowCount ∧
        rowColumnsBelow arm.columnCount indexed.row

def RawArm.LeafGeometryValid (arm : RawArm) : Prop :=
  arm.CanonicalCallsValid ∧ arm.Poseidon2CallsValid ∧ arm.GlueRowsValid

def RawArm.OwnershipValid (arm : RawArm) : Prop :=
    ownerIndices .canonical arm.owners = List.range arm.canonicalCalls.length ∧
    ownerIndices .poseidon2 arm.owners = List.range arm.poseidon2Calls.length ∧
    ownerIndices .glue arm.owners = List.range arm.glueRows.length ∧
    ownerIndices .phaseEnvelope arm.owners = [0] ∧
    exactOwnerChainFrom arm arm.sourceRowCount arm.owners = true

def RawArm.Valid (arm : RawArm) : Prop :=
  arm.ScalarValid ∧ arm.StateColumnLayoutValid ∧
    arm.XOutColumnLayoutValid ∧ arm.HashLayoutValid ∧
    arm.PublicAndPinLayoutValid ∧ arm.LeafGeometryValid ∧
    arm.OwnershipValid

def RawArm.Satisfied (arm : RawArm) (assignment : Nat → Nat) : Prop :=
  (∀ call ∈ arm.canonicalCalls, call.Satisfied assignment) ∧
    (∀ call ∈ arm.poseidon2Calls, Satisfies call.rows assignment) ∧
    ∀ indexed ∈ arm.glueRows, RowHolds assignment indexed.row

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  familyStateFields : Nat
  sharedPublicWords : Nat
  publicBitsPerWord : Nat
  firstFamilyProgramCursor : Nat
  lowNormRows : Nat
  lowNormColumns : Nat
  lowNormPublicColumns : Nat
  publicDecoder : RawPublicDecoder
  even : RawArm
  odd : RawArm
deriving DecidableEq, Repr

def RawArtifact.MetadataValid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 4 ∧
    artifact.profileId =
      "nebula-f-prime-streaming-pi-rlc-family-public-v4" ∧
    artifact.familyStateFields = 1045 ∧
    artifact.sharedPublicWords = 10 ∧
    artifact.publicBitsPerWord = 64 ∧
    artifact.firstFamilyProgramCursor = 223 ∧
    artifact.lowNormRows = 491046 ∧
    artifact.lowNormColumns = 8858862 ∧
    artifact.lowNormPublicColumns = 648 ∧
    artifact.publicDecoder.Valid artifact.even.publicColumnCount
      artifact.lowNormPublicColumns ∧
    artifact.even.sourceRowCount = 310646 ∧
    artifact.even.rowCount = 1300897 ∧
    artifact.even.columnCount = 1301126 ∧
    artifact.even.replayPoseidon2CallCount = 242 ∧
    artifact.odd.sourceRowCount = 311846 ∧
    artifact.odd.rowCount = 1302097 ∧
    artifact.odd.columnCount = 1302326 ∧
    artifact.odd.replayPoseidon2CallCount = 244

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.MetadataValid ∧ artifact.even.Valid ∧ artifact.odd.Valid

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
