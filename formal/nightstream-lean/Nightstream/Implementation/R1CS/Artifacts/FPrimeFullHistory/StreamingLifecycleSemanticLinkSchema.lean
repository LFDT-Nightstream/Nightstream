import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCPhaseEnvelopeValidityCertificate
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.PureSponge

/-!
Contract: compact exact-row schema for the lifecycle semantic-link family.

Owns two independent delayed-payload slices, their Boolean rows, two
Poseidon2 phase-envelope hashes, and the eight links from hash outputs to the
outer semantic lanes. Rust checks every represented source row.

Does not own phase-local semantics, generated values, lifecycle transitions,
or Poseidon2 collision resistance.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact

def payloadRowCount : Nat := 2 * payloadFields
def equalityRowCount : Nat := 2 * digestFields
def totalRows : Nat := payloadRowCount + 2 * hashTotalRows + equalityRowCount

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceIdentity : String
  sourceRowsSha256 : String
  rowCount : Nat
  columnCount : Nat
  constantValues : List Nat
  beforeSemanticColumns : List Nat
  afterSemanticColumns : List Nat
  beforeLocalColumns : List Nat
  afterLocalColumns : List Nat
  beforePayloadStartColumn : Nat
  afterPayloadStartColumn : Nat
  beforeHashConstantStartColumn : Nat
  afterHashConstantStartColumn : Nat
  beforeHashOutputColumns : List Nat
  afterHashOutputColumns : List Nat
  equalityRowStart : Nat
deriving DecidableEq, Repr

def RawArtifact.semanticColumns (artifact : RawArtifact) : StateSide → List Nat
  | .before => artifact.beforeSemanticColumns
  | .after => artifact.afterSemanticColumns

def RawArtifact.localColumns (artifact : RawArtifact) : StateSide → List Nat
  | .before => artifact.beforeLocalColumns
  | .after => artifact.afterLocalColumns

def RawArtifact.payloadStartColumn (artifact : RawArtifact) : StateSide → Nat
  | .before => artifact.beforePayloadStartColumn
  | .after => artifact.afterPayloadStartColumn

def RawArtifact.payloadColumns
    (artifact : RawArtifact) (side : StateSide) : List Nat :=
  List.range' (artifact.payloadStartColumn side) payloadFields

def RawArtifact.hashConstantStartColumn
    (artifact : RawArtifact) : StateSide → Nat
  | .before => artifact.beforeHashConstantStartColumn
  | .after => artifact.afterHashConstantStartColumn

def RawArtifact.hashOutputColumns
    (artifact : RawArtifact) : StateSide → List Nat
  | .before => artifact.beforeHashOutputColumns
  | .after => artifact.afterHashOutputColumns

def RawArtifact.hashRecipe
    (artifact : RawArtifact) (side : StateSide) : HashRecipe where
  constantValues := artifact.constantValues
  constantStartColumn := artifact.hashConstantStartColumn side
  localColumns := artifact.localColumns side
  payloadColumns := artifact.payloadColumns side
  outputColumns := artifact.hashOutputColumns side

def RawArtifact.payloadRows (artifact : RawArtifact) : List Row :=
  (artifact.payloadColumns .before ++ artifact.payloadColumns .after).map bitRow

def RawArtifact.equalityRows (artifact : RawArtifact) : List Row :=
  (List.range digestFields).flatMap fun lane =>
    [builderLinearRow (artifact.beforeSemanticColumns.getD lane 0)
        [(artifact.beforeHashOutputColumns.getD lane 0, 1)],
      builderLinearRow (artifact.afterSemanticColumns.getD lane 0)
        [(artifact.afterHashOutputColumns.getD lane 0, 1)]]

def RawArtifact.programPieces (artifact : RawArtifact) : List (List Row) :=
  let before := artifact.hashRecipe .before
  let after := artifact.hashRecipe .after
  [artifact.payloadRows,
    constantRows before,
    before.trace.rows,
    constantRows after,
    after.trace.rows,
    artifact.equalityRows]

def RawArtifact.program (artifact : RawArtifact) : List Row :=
  artifact.programPieces.flatten

def RawArtifact.Satisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.program assignment

instance (artifact : RawArtifact) (assignment : Nat → Nat) :
    Decidable (artifact.Satisfied assignment) := by
  unfold RawArtifact.Satisfied
  infer_instance

def expectedConstantValues : List Nat :=
  [57, 30521782141150574, 31069335676202596,
    27422324158721583, 30796712690673199, 27414614995316581,
    29396737889036653, 30792317818729313, 33266151269363297,
    49, 2169]

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 1 ∧
    artifact.profileId =
      "nightstream/goldilocks/streaming-lifecycle-semantic-link/v1" ∧
    artifact.sourceIdentity = "rust:streaming-lifecycle-semantic-link/v1" ∧
    artifact.sourceRowsSha256.length = 64 ∧
    artifact.rowCount = totalRows ∧
    artifact.constantValues = expectedConstantValues ∧
    artifact.beforeSemanticColumns = [1, 2, 3, 4] ∧
    artifact.afterSemanticColumns = [5, 6, 7, 8] ∧
    artifact.beforeLocalColumns = [9, 10, 11, 12] ∧
    artifact.afterLocalColumns = [13, 14, 15, 16] ∧
    artifact.beforePayloadStartColumn = 17 ∧
    artifact.afterPayloadStartColumn = 17 + payloadFields ∧
    artifact.beforeHashConstantStartColumn = 17 + 2 * payloadFields ∧
    artifact.afterHashConstantStartColumn =
      artifact.beforeHashConstantStartColumn + hashTotalRows ∧
    artifact.columnCount = artifact.afterHashConstantStartColumn + hashTotalRows ∧
    artifact.equalityRowStart = payloadRowCount + 2 * hashTotalRows ∧
    artifact.beforeHashOutputColumns =
      ((artifact.hashRecipe .before).callOutputColumns absorbRounds).take 4 ∧
    artifact.afterHashOutputColumns =
      ((artifact.hashRecipe .after).callOutputColumns absorbRounds).take 4

instance (artifact : RawArtifact) : Decidable artifact.Valid := by
  unfold RawArtifact.Valid
  infer_instance

def phasePreimage
    (artifact : RawArtifact) (side : StateSide)
    (assignment : Nat → Nat) : List Nat :=
  artifact.constantValues ++
    (artifact.localColumns side).map assignment ++
    (artifact.payloadColumns side).map assignment

def phaseChunks
    (artifact : RawArtifact) (side : StateSide)
    (assignment : Nat → Nat) :=
  Nightstream.Implementation.R1CS.Poseidon2PureSponge.fullRateChunks
    (phasePreimage artifact side assignment) absorbRounds

/-- Independent semantic target recomputed from the exact local digest and
payload columns. -/
def phaseEnvelopeDigest
    (artifact : RawArtifact) (side : StateSide)
    (assignment : Nat → Nat) (lane : Fin 4) : Nat :=
  Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge.digest
    Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants.selected
    (phaseChunks artifact side assignment) lane

structure SemanticLink
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop where
  payloadBinary : ∀ side column,
    column ∈ artifact.payloadColumns side → assignment column ≤ 1
  semanticExact : ∀ side lane,
    assignment ((artifact.semanticColumns side).getD lane.val 0) =
      phaseEnvelopeDigest artifact side assignment lane

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact
