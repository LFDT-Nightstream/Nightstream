import Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
import Nightstream.Implementation.R1CS.Core.Poseidon2Call

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

inductive OwnerKind where
  | canonical
  | poseidon2
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
  canonicalCalls : List CanonicalCall
  poseidon2Calls : List Poseidon2Call.Call
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
    arm.canonicalCalls.length = 40 ∧
    arm.poseidon2Calls.length = arm.activeFields / 4 ∧
    (∀ call ∈ arm.canonicalCalls, call.Valid arm.columnCount) ∧
    (∀ call ∈ arm.poseidon2Calls, PoseidonCallValid arm.columnCount call) ∧
    (∀ indexed ∈ arm.glueRows,
      indexed.index < arm.rowCount ∧ rowColumnsBelow arm.columnCount indexed.row) ∧
    ownerIndices .canonical arm.owners = List.range arm.canonicalCalls.length ∧
    ownerIndices .poseidon2 arm.owners = List.range arm.poseidon2Calls.length ∧
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
    ∀ indexed ∈ arm.glueRows, RowHolds assignment indexed.row

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  frameFields : Nat
  chunkFields : Nat
  finalChunkFields : Nat
  fullChunks : Nat
  transitionPublicWords : Nat
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
  artifact.schemaVersion = 1 ∧
    artifact.profileId = "nebula-f-prime-streaming-claim-replay-v1" ∧
    artifact.frameFields = 88023 ∧
    artifact.chunkFields = 1024 ∧
    artifact.finalChunkFields = 983 ∧
    artifact.fullChunks = 85 ∧
    artifact.transitionPublicWords = 40 ∧
    artifact.publicBitsPerWord = 64 ∧
    artifact.full.activeFields = artifact.chunkFields ∧
    artifact.finalChunk.activeFields = artifact.finalChunkFields ∧
    artifact.full.publicColumnCount = 2561 ∧
    artifact.finalChunk.publicColumnCount = 2561 ∧
    artifact.sharedPrivateFields = 1103 ∧
    artifact.lowNormRows = 51338 ∧
    artifact.lowNormColumns = 536112 ∧
    artifact.lowNormPublicColumns = 2592 ∧
    artifact.lowNormTotalCoordinates = 536086 ∧
    artifact.lowNormArity = 13 ∧ artifact.lowNormDegree = 8 ∧
    artifact.lowNormSharedPrivateCoordinates = 1103 ∧
    artifact.lowNormFullBranchCoordinates = 507311 ∧
    artifact.lowNormFinalBranchCoordinates = 484610 ∧
    artifact.lowNormFullPoseidon2Coordinates = 506368 ∧
    artifact.lowNormFinalPoseidon2Coordinates = 484610 ∧
    artifact.full.Valid ∧ artifact.finalChunk.Valid

instance (artifact : RawArtifact) : Decidable artifact.Valid := by
  unfold RawArtifact.Valid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
