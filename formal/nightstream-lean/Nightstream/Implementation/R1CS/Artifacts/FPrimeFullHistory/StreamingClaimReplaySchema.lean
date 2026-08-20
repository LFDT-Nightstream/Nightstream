import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingArtifactLeafSchema
import Nightstream.Implementation.R1CS.Core.AffinePins
import Nightstream.Implementation.R1CS.Ownership.Core.OwnerCertificate
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataCoordinateMaps

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
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler

/-- One compact fixed-position Ajtai call inside a selected claim arm. Its
map kind and chunk index reconstruct the complete verifier-owned selector.
Generated data stores only physical bases and the exact seed schedule. -/
structure CoordinateCall where
  mapKind : MapKind
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

def CoordinateCall.activeFields (call : CoordinateCall) :
    List (Fin call.mapKind.fieldCount) :=
  call.mapKind.activeFields call.chunk

def CoordinateCall.fieldColumn
    (call : CoordinateCall) (field : Fin call.mapKind.fieldCount) : Nat :=
  call.chunkBase + (call.mapKind.claimChunkOffset field).val

def CoordinateCall.digitStart
    (call : CoordinateCall) (field : Fin call.mapKind.fieldCount) : Nat :=
  call.activeDigitBase + call.activeFields.idxOf field * 122

def CoordinateCall.wordStart
    (call : CoordinateCall) (field : Fin call.mapKind.fieldCount) : Nat :=
  if field ∈ call.activeFields then call.digitStart field
  else call.zeroDigitStart

def CoordinateCall.wordStarts (call : CoordinateCall) : List Nat :=
  List.ofFn call.wordStart

def CoordinateCall.outputColumn (call : CoordinateCall)
    (output : Fin 108) : Nat :=
  call.outputBase + output.val

def CoordinateCall.zeroPins (call : CoordinateCall) : List AffinePins.Pin :=
  List.ofFn fun digit : Fin 41 =>
    .zero (call.zeroDigitStart + digit.val)

def CoordinateCall.zeroRows (call : CoordinateCall) : List Row :=
  AffinePins.rows call.zeroPins

def CoordinateCall.openingBlockRows
    (call : CoordinateCall) (field : Fin call.mapKind.fieldCount) : List Row :=
  canonicalRows.map (Relabel.row
    (OwnerCertificate.shiftedTernaryColumnMap
      (call.fieldColumn field) (call.digitStart field)))

def CoordinateCall.openingRows (call : CoordinateCall) : List Row :=
  call.activeFields.flatMap call.openingBlockRows

def CoordinateCall.sourceRows (call : CoordinateCall) : List Row :=
  call.zeroRows ++ call.openingRows

def CoordinateCall.shapePins (call : CoordinateCall) : List AffinePins.Pin :=
  [.constant call.dColumn 54, .constant call.kappaColumn 2]

def CoordinateCall.shapeRows (call : CoordinateCall) : List Row :=
  AffinePins.rows call.shapePins

def CoordinateCall.block (call : CoordinateCall) : SeededPhi81.Block where
  rowStart := call.seededRowStart
  wordStarts := call.wordStarts
  wordWidth := 41
  kappa := 2
  messageCols := call.mapKind.messageColumnCount
  outputColumns := List.ofFn call.outputColumn
  superneoTransformedColumns := false
  schedule := {
    chunkSize := call.chunkSize
    seedsByOutput := call.seedsByOutput
    rejectionFuel := 16 }

def CoordinateCall.rows (call : CoordinateCall) : List Row :=
  call.sourceRows ++ (call.shapeRows ++ call.block.rows)

def CoordinateCall.GeometryValid
    (columnCount : Nat) (call : CoordinateCall) : Prop :=
  call.chunkIndex < claimChunkCount ∧
    call.activeFields.length > 0 ∧
    call.activeDigitBase = call.zeroDigitStart + 41 ∧
    call.dColumn =
      call.activeDigitBase + call.activeFields.length * 122 ∧
    call.kappaColumn = call.dColumn + 1 ∧
    call.outputBase = call.kappaColumn + 1 ∧
    call.seededRowStart =
      call.rowStart + 41 + call.activeFields.length * 124 + 2 ∧
    call.rowEnd = call.seededRowStart + 108 ∧
    call.chunkBase + claimChunkFieldCount call.chunk ≤ columnCount ∧
    call.zeroDigitStart + 41 ≤ columnCount ∧
    call.outputBase + 108 ≤ columnCount

instance (columnCount : Nat) (call : CoordinateCall) :
    Decidable (call.GeometryValid columnCount) := by
  unfold CoordinateCall.GeometryValid
  infer_instance

def CoordinateCall.ScheduleValid (call : CoordinateCall) : Prop :=
  call.block.schedule = call.mapKind.expectedSchedule

instance (call : CoordinateCall) : Decidable call.ScheduleValid := by
  unfold CoordinateCall.ScheduleValid
  infer_instance

def CoordinateCall.Valid (columnCount : Nat) (call : CoordinateCall) : Prop :=
  call.GeometryValid columnCount ∧ call.ScheduleValid ∧ call.block.Valid

instance (columnCount : Nat) (call : CoordinateCall) :
    Decidable (call.Valid columnCount) := by
  unfold CoordinateCall.Valid
  infer_instance

/-- A complete verifier-owned sampler certificate transfers to any physical
block that uses the exact same map profile. -/
theorem CoordinateCall.blockValid_of_certificate
    (call : CoordinateCall)
    (schedule : call.ScheduleValid)
    (certificate : call.mapKind.certificateBlock.Valid) :
    call.block.Valid := by
  apply certificate.transfer
  · rfl
  · rfl
  · simpa [MapKind.certificateBlock] using schedule
  · rfl
  · change
      (List.ofFn call.wordStart).length =
        (List.replicate call.mapKind.fieldCount 0).length
    simp only [List.length_ofFn, List.length_replicate]
  · change
      (List.ofFn call.outputColumn).length = (List.range 108).length
    simp only [List.length_ofFn, List.length_range]
  · rfl

theorem CoordinateCall.valid_of_geometry_schedule_and_certificate
    {columnCount : Nat} {call : CoordinateCall}
    (geometry : call.GeometryValid columnCount)
    (schedule : call.ScheduleValid)
    (certificate : call.mapKind.certificateBlock.Valid) :
    call.Valid columnCount := by
  exact ⟨geometry, schedule,
    call.blockValid_of_certificate schedule certificate⟩

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

def RawArm.ScalarValid (arm : RawArm) : Prop :=
  0 < arm.rowCount ∧ 0 < arm.columnCount ∧
    arm.publicColumnCount ≤ arm.columnCount ∧
    arm.publicColumnCount = 641 ∧
    arm.replayPoseidon2CallCount = arm.activeFields / 4 ∧
    arm.stateDigestPoseidon2CallCount = 176 ∧
    arm.poseidon2Calls.length =
      arm.replayPoseidon2CallCount + arm.stateDigestPoseidon2CallCount

instance (arm : RawArm) : Decidable arm.ScalarValid := by
  unfold RawArm.ScalarValid
  infer_instance

def RawArm.StateWordLayoutValid (arm : RawArm) : Prop :=
  arm.stateWordColumns.length = 688 ∧
    arm.stateWordColumns.Nodup ∧
    ∀ column ∈ arm.stateWordColumns, column < arm.columnCount

instance (arm : RawArm) : Decidable arm.StateWordLayoutValid := by
  unfold RawArm.StateWordLayoutValid
  infer_instance

def RawArm.PublicWordLayoutValid (arm : RawArm) : Prop :=
  arm.publicWordCallIndices.length = 10 ∧
    arm.publicWordCallIndices.Nodup ∧
    (∀ index ∈ arm.publicWordCallIndices,
      index < arm.canonicalCalls.length)

instance (arm : RawArm) : Decidable arm.PublicWordLayoutValid := by
  unfold RawArm.PublicWordLayoutValid
  infer_instance

def RawArm.DigestPinLayoutValid (arm : RawArm) : Prop :=
  arm.afterDigestPinColumns.length = 13 ∧
    arm.afterDigestPinColumns.Nodup ∧
    (∀ column ∈ arm.afterDigestPinColumns, column < arm.columnCount) ∧
    arm.beforeDigestPinColumns.length = 13 ∧
    arm.beforeDigestPinColumns.Nodup ∧
    ∀ column ∈ arm.beforeDigestPinColumns, column < arm.columnCount

instance (arm : RawArm) : Decidable arm.DigestPinLayoutValid := by
  unfold RawArm.DigestPinLayoutValid
  infer_instance

def RawArm.CanonicalCallsValid (arm : RawArm) : Prop :=
  arm.canonicalCalls.length = 10 ∧
    ∀ call ∈ arm.canonicalCalls, call.Valid arm.columnCount

instance (arm : RawArm) : Decidable arm.CanonicalCallsValid := by
  unfold RawArm.CanonicalCallsValid
  infer_instance

def RawArm.Poseidon2CallsValid (arm : RawArm) : Prop :=
  ∀ call ∈ arm.poseidon2Calls, PoseidonCallValid arm.columnCount call

instance (arm : RawArm) : Decidable arm.Poseidon2CallsValid := by
  unfold RawArm.Poseidon2CallsValid
  infer_instance

def RawArm.GlueRowsValid (arm : RawArm) : Prop :=
  ∀ indexed ∈ arm.glueRows,
    indexed.index < arm.rowCount ∧ rowColumnsBelow arm.columnCount indexed.row

instance (arm : RawArm) : Decidable arm.GlueRowsValid := by
  unfold RawArm.GlueRowsValid
  infer_instance

def RawArm.LeafGeometryValid (arm : RawArm) : Prop :=
  arm.CanonicalCallsValid ∧ arm.Poseidon2CallsValid ∧ arm.GlueRowsValid

instance (arm : RawArm) : Decidable arm.LeafGeometryValid := by
  unfold RawArm.LeafGeometryValid
  infer_instance

def RawArm.OwnershipValid (arm : RawArm) : Prop :=
  ownerIndices .canonical arm.owners = List.range arm.canonicalCalls.length ∧
    ownerIndices .poseidon2 arm.owners = List.range arm.poseidon2Calls.length ∧
    ownerIndices .coordinate arm.owners =
      List.range arm.coordinateCalls.length ∧
    ownerIndices .glue arm.owners = List.range arm.glueRows.length ∧
    exactOwnerChainFrom arm 0 arm.owners = true

instance (arm : RawArm) : Decidable arm.OwnershipValid := by
  unfold RawArm.OwnershipValid
  infer_instance

def RawArm.ValidWithoutCoordinates (arm : RawArm) : Prop :=
  arm.ScalarValid ∧ arm.StateWordLayoutValid ∧
    arm.PublicWordLayoutValid ∧ arm.DigestPinLayoutValid ∧
    arm.LeafGeometryValid ∧ arm.OwnershipValid

instance (arm : RawArm) : Decidable arm.ValidWithoutCoordinates := by
  unfold RawArm.ValidWithoutCoordinates
  infer_instance

def RawArm.Valid (arm : RawArm) : Prop :=
  arm.ValidWithoutCoordinates ∧
    ∀ call ∈ arm.coordinateCalls, call.Valid arm.columnCount

instance (arm : RawArm) : Decidable arm.Valid := by
  unfold RawArm.Valid
  infer_instance

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

def RawArtifact.MetadataValid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 5 ∧
    artifact.profileId = "nebula-f-prime-streaming-claim-replay-goldilocks-b2-k16-v6" ∧
    artifact.frameFields = 99903 ∧
    artifact.chunkFields = 1024 ∧
    artifact.finalChunkFields = 575 ∧
    artifact.fullChunks = 97 ∧
    artifact.transitionStateWords = 688 ∧
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
    artifact.full.coordinateCalls.map CoordinateCall.mapKind =
      [.statementFresh, .runningCommitments] ∧
    artifact.finalChunk.coordinateCalls.map CoordinateCall.mapKind =
      [.statementFresh] ∧
    artifact.sharedPrivateFields = 692 ∧
    artifact.lowNormRows = 118213 ∧
    artifact.lowNormColumns = 1608012 ∧
    artifact.lowNormPublicColumns = 648 ∧
    artifact.lowNormTotalCoordinates = 1608006 ∧
    artifact.lowNormArity = 13 ∧ artifact.lowNormDegree = 8 ∧
    artifact.lowNormSharedPrivateCoordinates = 692 ∧
    artifact.lowNormFullBranchCoordinates = 1578966 ∧
    artifact.lowNormFinalBranchCoordinates = 1160758 ∧
    artifact.lowNormFullPoseidon2Coordinates = 1523744 ∧
    artifact.lowNormFinalPoseidon2Coordinates = 1125306

instance (artifact : RawArtifact) : Decidable artifact.MetadataValid := by
  unfold RawArtifact.MetadataValid
  infer_instance

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.MetadataValid ∧
    artifact.full.Valid ∧ artifact.finalChunk.Valid

instance (artifact : RawArtifact) : Decidable artifact.Valid := by
  unfold RawArtifact.Valid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
