import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleSemanticLinkSchema

/-!
Contract: compact exact-row schema for the base and recursive lifecycle
semantic-link source stages.

Base fixes the before payload to zero. Recursive reuses its separately checked
private delayed-input domain. Both check the after payload, recompute both
Poseidon2 semantic digests, and bind the outer semantic lanes.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCPhaseEnvelope.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact

inductive SourceScope
  | base
  | recursive
deriving DecidableEq, Repr

def SourceScope.beforePayloadRowCount : SourceScope → Nat
  | .base => payloadFields
  | .recursive => 0

def SourceScope.rowCount (scope : SourceScope) : Nat :=
  scope.beforePayloadRowCount + hashTotalRows + payloadFields +
    hashTotalRows + 2 * digestFields

structure SourceArtifact where
  scope : SourceScope
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
  beforePayloadRowStart : Nat
  beforeHashConstantRowStart : Nat
  afterPayloadRowStart : Nat
  afterHashConstantRowStart : Nat
  equalityRowStart : Nat
deriving DecidableEq, Repr

def SourceArtifact.semanticColumns
    (artifact : SourceArtifact) : StateSide → List Nat
  | .before => artifact.beforeSemanticColumns
  | .after => artifact.afterSemanticColumns

def SourceArtifact.localColumns
    (artifact : SourceArtifact) : StateSide → List Nat
  | .before => artifact.beforeLocalColumns
  | .after => artifact.afterLocalColumns

def SourceArtifact.payloadStartColumn
    (artifact : SourceArtifact) : StateSide → Nat
  | .before => artifact.beforePayloadStartColumn
  | .after => artifact.afterPayloadStartColumn

def SourceArtifact.payloadColumns
    (artifact : SourceArtifact) (side : StateSide) : List Nat :=
  List.range' (artifact.payloadStartColumn side) payloadFields

def SourceArtifact.hashConstantStartColumn
    (artifact : SourceArtifact) : StateSide → Nat
  | .before => artifact.beforeHashConstantStartColumn
  | .after => artifact.afterHashConstantStartColumn

def SourceArtifact.hashOutputColumns
    (artifact : SourceArtifact) : StateSide → List Nat
  | .before => artifact.beforeHashOutputColumns
  | .after => artifact.afterHashOutputColumns

def SourceArtifact.hashRecipe
    (artifact : SourceArtifact) (side : StateSide) : HashRecipe where
  constantValues := artifact.constantValues
  constantStartColumn := artifact.hashConstantStartColumn side
  localColumns := artifact.localColumns side
  payloadColumns := artifact.payloadColumns side
  outputColumns := artifact.hashOutputColumns side

def SourceArtifact.beforePayloadRows (artifact : SourceArtifact) : List Row :=
  match artifact.scope with
  | .base => (artifact.payloadColumns .before).map fun column =>
      builderLinearRow column []
  | .recursive => []

def SourceArtifact.afterPayloadRows (artifact : SourceArtifact) : List Row :=
  (artifact.payloadColumns .after).map bitRow

def SourceArtifact.equalityRows (artifact : SourceArtifact) : List Row :=
  (List.range digestFields).map (fun lane =>
      builderLinearRow (artifact.beforeSemanticColumns.getD lane 0)
        [(artifact.beforeHashOutputColumns.getD lane 0, 1)]) ++
    (List.range digestFields).map (fun lane =>
      builderLinearRow (artifact.afterSemanticColumns.getD lane 0)
        [(artifact.afterHashOutputColumns.getD lane 0, 1)])

def SourceArtifact.programPieces (artifact : SourceArtifact) : List (List Row) :=
  let before := artifact.hashRecipe .before
  let after := artifact.hashRecipe .after
  [artifact.beforePayloadRows,
    constantRows before,
    before.trace.rows,
    artifact.afterPayloadRows,
    constantRows after,
    after.trace.rows,
    artifact.equalityRows]

def SourceArtifact.program (artifact : SourceArtifact) : List Row :=
  artifact.programPieces.flatten

def SourceArtifact.Satisfied
    (artifact : SourceArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.program assignment

instance (artifact : SourceArtifact) (assignment : Nat → Nat) :
    Decidable (artifact.Satisfied assignment) := by
  unfold SourceArtifact.Satisfied
  infer_instance

def SourceArtifact.semanticGeometry
    (artifact : SourceArtifact) :
    Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.RawArtifact where
  schemaVersion := artifact.schemaVersion
  profileId := artifact.profileId
  sourceIdentity := artifact.sourceIdentity
  sourceRowsSha256 := artifact.sourceRowsSha256
  rowCount := artifact.rowCount
  columnCount := artifact.columnCount
  constantValues := artifact.constantValues
  beforeSemanticColumns := artifact.beforeSemanticColumns
  afterSemanticColumns := artifact.afterSemanticColumns
  beforeLocalColumns := artifact.beforeLocalColumns
  afterLocalColumns := artifact.afterLocalColumns
  beforePayloadStartColumn := artifact.beforePayloadStartColumn
  afterPayloadStartColumn := artifact.afterPayloadStartColumn
  beforeHashConstantStartColumn := artifact.beforeHashConstantStartColumn
  afterHashConstantStartColumn := artifact.afterHashConstantStartColumn
  beforeHashOutputColumns := artifact.beforeHashOutputColumns
  afterHashOutputColumns := artifact.afterHashOutputColumns
  equalityRowStart := artifact.equalityRowStart

def SourceArtifact.SemanticLink
    (artifact : SourceArtifact) (assignment : Nat → Nat) : Prop :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSemanticLink.Artifact.SemanticLink
    artifact.semanticGeometry assignment

def expectedProfileId : SourceScope → String
  | .base =>
      "nightstream/goldilocks/streaming-lifecycle-source-semantic-link/base/v1"
  | .recursive =>
      "nightstream/goldilocks/streaming-lifecycle-source-semantic-link/recursive/v1"

def expectedSourceIdentity : SourceScope → String
  | .base => "rust:streaming-lifecycle-source-semantic-link/base/v1"
  | .recursive => "rust:streaming-lifecycle-source-semantic-link/recursive/v1"

def SourceArtifact.Valid (artifact : SourceArtifact) : Prop :=
  artifact.schemaVersion = 1 ∧
    artifact.profileId = expectedProfileId artifact.scope ∧
    artifact.sourceIdentity = expectedSourceIdentity artifact.scope ∧
    artifact.sourceRowsSha256.length = 64 ∧
    artifact.rowCount = artifact.scope.rowCount ∧
    artifact.constantValues = expectedConstantValues ∧
    artifact.beforeSemanticColumns = [1, 2, 3, 4] ∧
    artifact.afterSemanticColumns = [5, 6, 7, 8] ∧
    artifact.beforeLocalColumns = [9, 10, 11, 12] ∧
    artifact.beforePayloadStartColumn = 13 ∧
    artifact.afterLocalColumns =
      List.range' (13 + payloadFields) digestFields ∧
    artifact.afterPayloadStartColumn = 13 + payloadFields + digestFields ∧
    artifact.beforeHashConstantStartColumn = 13 + 2 * payloadFields + digestFields ∧
    artifact.afterHashConstantStartColumn =
      artifact.beforeHashConstantStartColumn + hashTotalRows ∧
    artifact.columnCount = artifact.afterHashConstantStartColumn + hashTotalRows ∧
    artifact.beforePayloadRowStart = 0 ∧
    artifact.beforeHashConstantRowStart = artifact.scope.beforePayloadRowCount ∧
    artifact.afterPayloadRowStart =
      artifact.scope.beforePayloadRowCount + hashTotalRows ∧
    artifact.afterHashConstantRowStart =
      artifact.afterPayloadRowStart + payloadFields ∧
    artifact.equalityRowStart =
      artifact.afterHashConstantRowStart + hashTotalRows ∧
    artifact.beforeHashOutputColumns =
      ((artifact.hashRecipe .before).callOutputColumns absorbRounds).take 4 ∧
    artifact.afterHashOutputColumns =
      ((artifact.hashRecipe .after).callOutputColumns absorbRounds).take 4

instance (artifact : SourceArtifact) : Decidable artifact.Valid := by
  unfold SourceArtifact.Valid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleSourceSemanticLink.Artifact
