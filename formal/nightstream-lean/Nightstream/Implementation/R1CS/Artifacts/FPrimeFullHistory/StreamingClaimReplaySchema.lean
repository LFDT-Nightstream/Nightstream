import Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
import Nightstream.Implementation.R1CS.Core.Poseidon2Call
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingClaimSchedule

/-!
Contract: compact schema for one Rust-emitted streaming claim-replay arm.

The large repeated blocks are references to the exact 69-row canonical-u64
recipe and exact 600-row Poseidon2 permutation. Only the small glue-row set is
stored directly. The owner list must cover every Rust row once.

Assurance tier: artifact schema. Generated data and its Rust drift owner are
separate obligations.

Emits constraints: no. It describes existing constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows

structure CanonicalCall where
  rowStart : Nat
  rowEnd : Nat
  fieldColumn : Nat
  bitBase : Nat
  highFlagColumn : Nat
  inverseColumn : Nat
deriving DecidableEq, Repr, Inhabited

def CanonicalCall.layout (call : CanonicalCall) : CanonicalU64Recipe.Layout where
  base := call.bitBase
  input := [(call.fieldColumn, 1)]

def CanonicalCall.Valid (columnCount : Nat) (call : CanonicalCall) : Prop :=
  call.rowEnd = call.rowStart + 69 ∧
    0 < call.bitBase ∧
    call.highFlagColumn = call.bitBase + 64 ∧
    call.inverseColumn = call.bitBase + 65 ∧
    call.fieldColumn < columnCount ∧
    call.inverseColumn < columnCount

instance (columnCount : Nat) (call : CanonicalCall) :
    Decidable (call.Valid columnCount) := by
  unfold CanonicalCall.Valid
  infer_instance

structure IndexedRow where
  index : Nat
  row : Row
deriving DecidableEq, Repr

/-- One compact fixed-position Ajtai call inside a selected claim arm. The
21,220-entry selector is reconstructed from `chunkIndex`; generated data
stores only physical bases and the verifier-owned seed schedule. -/
structure CoordinateCall where
  rowStart : Nat
  rowEnd : Nat
  chunkIndex : Nat
  chunkBase : Nat
  zeroDigitStart : Nat
  activeDigitBase : Nat
  dColumn : Nat
  kappaColumn : Nat
  outputBase : Nat
  seededRowStart : Nat
  chunkSize : Nat
  seedsByOutput : List (List (List Nat))
deriving DecidableEq, Repr, Inhabited

def CoordinateCall.chunk (call : CoordinateCall) : Fin claimChunkCount :=
  ⟨call.chunkIndex % claimChunkCount, Nat.mod_lt _ (by decide)⟩

def CoordinateCall.layout (call : CoordinateCall) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows.Layout where
  activeFields := activeFields call.chunk
  activeFieldsNodup := activeFields_nodup call.chunk
  fieldColumn := fun field => call.chunkBase + (claimChunkOffset field).val
  digitStart := fun field =>
    call.activeDigitBase + (activeFields call.chunk).idxOf field * 122
  zeroDigitStart := call.zeroDigitStart
  dColumn := call.dColumn
  kappaColumn := call.kappaColumn
  outputColumn := fun output => call.outputBase + output.val
  seededRowStart := call.seededRowStart

def CoordinateCall.block (call : CoordinateCall) : SeededPhi81.Block where
  rowStart := call.seededRowStart
  wordStarts := call.layout.wordStarts
  wordWidth := 41
  kappa := 2
  messageCols := 16112
  outputColumns := List.ofFn call.layout.outputColumn
  superneoTransformedColumns := false
  schedule := {
    chunkSize := call.chunkSize
    seedsByOutput := call.seedsByOutput
    rejectionFuel := 16 }

def CoordinateCall.rows (call : CoordinateCall) : List Row :=
  sourceRows call.layout ++
    (shapeRows call.layout ++ call.block.rows)

def CoordinateCall.Valid (columnCount : Nat) (call : CoordinateCall) : Prop :=
  call.chunkIndex < claimChunkCount ∧
    (activeFields call.chunk).length > 0 ∧
    call.activeDigitBase = call.zeroDigitStart + 41 ∧
    call.dColumn =
      call.activeDigitBase + (activeFields call.chunk).length * 122 ∧
    call.kappaColumn = call.dColumn + 1 ∧
    call.outputBase = call.kappaColumn + 1 ∧
    call.seededRowStart =
      call.rowStart + 41 + (activeFields call.chunk).length * 124 + 2 ∧
    call.rowEnd = call.seededRowStart + 108 ∧
    call.chunkBase + claimChunkWidth ≤ columnCount ∧
    call.zeroDigitStart + 41 ≤ columnCount ∧
    call.outputBase + 108 ≤ columnCount ∧
    call.block.Valid

instance (columnCount : Nat) (call : CoordinateCall) :
    Decidable (call.Valid columnCount) := by
  unfold CoordinateCall.Valid
  infer_instance

def CoordinateCall.Satisfied
    (assignment : Nat → Nat) (call : CoordinateCall) : Prop :=
  Satisfies call.rows assignment

inductive OwnerKind where
  | canonical
  | poseidon2
  | coordinate
  | glue
deriving DecidableEq, Repr, Inhabited

structure Owner where
  rowStart : Nat
  rowEnd : Nat
  kind : OwnerKind
  index : Nat
deriving DecidableEq, Repr, Inhabited

structure RawArm where
  rowCount : Nat
  columnCount : Nat
  publicColumnCount : Nat
  activeFields : Nat
  replayPoseidon2CallCount : Nat
  stateDigestPoseidon2CallCount : Nat
  stateWordColumns : List Nat
  publicWordCallIndices : List Nat
  afterDigestPinColumns : List Nat
  beforeDigestPinColumns : List Nat
  canonicalCalls : List CanonicalCall
  poseidon2Calls : List Poseidon2Call.Call
  coordinateCalls : List CoordinateCall
  glueRows : List IndexedRow
  owners : List Owner
deriving DecidableEq, Repr

def rowColumnsBelow (columnCount : Nat) (row : Row) : Prop :=
  (∀ term ∈ row.a, term.1 < columnCount) ∧
    (∀ term ∈ row.b, term.1 < columnCount) ∧
    ∀ term ∈ row.c, term.1 < columnCount

instance (columnCount : Nat) (row : Row) :
    Decidable (rowColumnsBelow columnCount row) := by
  unfold rowColumnsBelow
  infer_instance

def PoseidonCallValid (columnCount : Nat) (call : Poseidon2Call.Call) : Prop :=
  call.rowEnd = call.rowStart + 600 ∧
    call.inputColumns.length = 8 ∧
    (∀ column ∈ call.inputColumns, column < columnCount) ∧
    call.firstAllocatedColumn + 600 ≤ columnCount

instance (columnCount : Nat) (call : Poseidon2Call.Call) :
    Decidable (PoseidonCallValid columnCount call) := by
  unfold PoseidonCallValid
  infer_instance

def Owner.Matches (arm : RawArm) (owner : Owner) : Prop :=
  match owner.kind with
  | .canonical =>
      ∃ call, arm.canonicalCalls[owner.index]? = some call ∧
        owner.rowStart = call.rowStart ∧ owner.rowEnd = call.rowEnd
  | .poseidon2 =>
      ∃ call, arm.poseidon2Calls[owner.index]? = some call ∧
        owner.rowStart = call.rowStart ∧ owner.rowEnd = call.rowEnd
  | .coordinate =>
      ∃ call, arm.coordinateCalls[owner.index]? = some call ∧
        owner.rowStart = call.rowStart ∧ owner.rowEnd = call.rowEnd
  | .glue =>
      ∃ indexed, arm.glueRows[owner.index]? = some indexed ∧
        owner.rowStart = indexed.index ∧ owner.rowEnd = indexed.index + 1

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

def RawArm.Valid (arm : RawArm) : Prop :=
  0 < arm.rowCount ∧ 0 < arm.columnCount ∧
    arm.publicColumnCount ≤ arm.columnCount ∧
    arm.publicColumnCount = 641 ∧
    arm.replayPoseidon2CallCount = arm.activeFields / 4 ∧
    arm.stateDigestPoseidon2CallCount = 68 ∧
    arm.poseidon2Calls.length =
      arm.replayPoseidon2CallCount + arm.stateDigestPoseidon2CallCount ∧
    arm.stateWordColumns.length = 256 ∧
    arm.stateWordColumns.Nodup ∧
    (∀ column ∈ arm.stateWordColumns, column < arm.columnCount) ∧
    arm.publicWordCallIndices.length = 10 ∧
    arm.publicWordCallIndices.Nodup ∧
    (∀ index ∈ arm.publicWordCallIndices,
      index < arm.canonicalCalls.length) ∧
    arm.afterDigestPinColumns.length = 13 ∧
    arm.afterDigestPinColumns.Nodup ∧
    (∀ column ∈ arm.afterDigestPinColumns, column < arm.columnCount) ∧
    arm.beforeDigestPinColumns.length = 13 ∧
    arm.beforeDigestPinColumns.Nodup ∧
    (∀ column ∈ arm.beforeDigestPinColumns, column < arm.columnCount) ∧
    arm.canonicalCalls.length = 10 ∧
    (∀ call ∈ arm.canonicalCalls, call.Valid arm.columnCount) ∧
    (∀ call ∈ arm.poseidon2Calls, PoseidonCallValid arm.columnCount call) ∧
    (∀ call ∈ arm.coordinateCalls, call.Valid arm.columnCount) ∧
    (∀ indexed ∈ arm.glueRows,
      indexed.index < arm.rowCount ∧ rowColumnsBelow arm.columnCount indexed.row) ∧
    ownerIndices .canonical arm.owners = List.range arm.canonicalCalls.length ∧
    ownerIndices .poseidon2 arm.owners = List.range arm.poseidon2Calls.length ∧
    ownerIndices .coordinate arm.owners =
      List.range arm.coordinateCalls.length ∧
    ownerIndices .glue arm.owners = List.range arm.glueRows.length ∧
    exactOwnerChainFrom arm 0 arm.owners = true

instance (arm : RawArm) : Decidable arm.Valid := by
  unfold RawArm.Valid
  infer_instance

def CanonicalCall.Satisfied (assignment : Nat → Nat)
    (call : CanonicalCall) : Prop :=
  Satisfies (CanonicalU64Recipe.rows call.layout) assignment

def RawArm.Satisfied (arm : RawArm) (assignment : Nat → Nat) : Prop :=
  (∀ call ∈ arm.canonicalCalls, call.Satisfied assignment) ∧
    (∀ call ∈ arm.poseidon2Calls, Satisfies call.rows assignment) ∧
    (∀ call ∈ arm.coordinateCalls, call.Satisfied assignment) ∧
    ∀ indexed ∈ arm.glueRows, RowHolds assignment indexed.row

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  frameFields : Nat
  chunkFields : Nat
  finalChunkFields : Nat
  fullChunks : Nat
  transitionStateWords : Nat
  stateDigestWords : Nat
  sharedPublicWords : Nat
  publicBitsPerWord : Nat
  sharedPrivateFields : Nat
  lowNormRows : Nat
  lowNormColumns : Nat
  lowNormPublicColumns : Nat
  lowNormTotalCoordinates : Nat
  lowNormArity : Nat
  lowNormDegree : Nat
  lowNormSharedPrivateCoordinates : Nat
  lowNormFullBranchCoordinates : Nat
  lowNormFinalBranchCoordinates : Nat
  lowNormFullPoseidon2Coordinates : Nat
  lowNormFinalPoseidon2Coordinates : Nat
  full : RawArm
  finalChunk : RawArm
deriving DecidableEq, Repr

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 3 ∧
    artifact.profileId = "nebula-f-prime-streaming-claim-replay-v3" ∧
    artifact.frameFields = 88023 ∧
    artifact.chunkFields = 1024 ∧
    artifact.finalChunkFields = 983 ∧
    artifact.fullChunks = 85 ∧
    artifact.transitionStateWords = 256 ∧
    artifact.stateDigestWords = 8 ∧
    artifact.sharedPublicWords = 10 ∧
    artifact.publicBitsPerWord = 64 ∧
    artifact.full.activeFields = artifact.chunkFields ∧
    artifact.finalChunk.activeFields = artifact.finalChunkFields ∧
    artifact.full.publicColumnCount = 641 ∧
    artifact.finalChunk.publicColumnCount = 641 ∧
    artifact.full.publicWordCallIndices =
      [2, 3, 4, 5, 6, 7, 8, 9, 0, 1] ∧
    artifact.finalChunk.publicWordCallIndices =
      [2, 3, 4, 5, 6, 7, 8, 9, 0, 1] ∧
    artifact.full.coordinateCalls.length = 1 ∧
    artifact.finalChunk.coordinateCalls.length = 0 ∧
    artifact.sharedPrivateFields = 260 ∧
    artifact.lowNormRows = 61034 ∧
    artifact.lowNormColumns = 673866 ∧
    artifact.lowNormPublicColumns = 648 ∧
    artifact.lowNormTotalCoordinates = 673865 ∧
    artifact.lowNormArity = 13 ∧ artifact.lowNormDegree = 8 ∧
    artifact.lowNormSharedPrivateCoordinates = 260 ∧
    artifact.lowNormFullBranchCoordinates = 667145 ∧
    artifact.lowNormFinalBranchCoordinates = 642358 ∧
    artifact.lowNormFullPoseidon2Coordinates = 641384 ∧
    artifact.lowNormFinalPoseidon2Coordinates = 619626 ∧
    artifact.full.Valid ∧ artifact.finalChunk.Valid

instance (artifact : RawArtifact) : Decidable artifact.Valid := by
  unfold RawArtifact.Valid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
