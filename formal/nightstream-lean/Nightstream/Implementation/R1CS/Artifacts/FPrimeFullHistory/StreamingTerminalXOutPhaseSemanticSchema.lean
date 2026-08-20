import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeSchema
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: compact exact-row schema for the terminal XOut phase-semantic family.

Owns one Poseidon2 recomputation from the exact phase-local state and delayed
payload columns, followed by four links to the terminal XOut semantic lanes.
Rust checks every represented source row against this structural recipe.

Does not own phase-local semantics, generated values, lifecycle transitions,
or Poseidon2 collision resistance.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact

def digestFields : Nat := 4
def payloadFields : Nat := 2169
def constantFields : Nat := 11
def hashInputFields : Nat := constantFields + digestFields + payloadFields
def absorbRounds : Nat :=
  (hashInputFields + (rate - 1)) / rate
def hashTotalRows : Nat :=
  constantFields + 1 + hashInputFields +
    absorbRounds * permutationRows + 1 + permutationRows
def equalityRowCount : Nat := 4
def totalRows : Nat := hashTotalRows + equalityRowCount

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceIdentity : String
  sourceRowsSha256 : String
  rowCount : Nat
  columnCount : Nat
  sourceRowStart : Nat
  finalRowStart : Nat
  constantValues : List Nat
  constantStartColumn : Nat
  localColumns : List Nat
  payloadColumns : List Nat
  hashOutputColumns : List Nat
  xOutSemanticColumns : List Nat
  baselineDigestValue : Nat
  equalityRowStart : Nat
deriving DecidableEq, Repr

def RawArtifact.hashRecipe (artifact : RawArtifact) : VariableHashRecipe where
  constantValues := artifact.constantValues
  constantStartColumn := artifact.constantStartColumn
  localColumns := artifact.localColumns
  payloadColumns := artifact.payloadColumns
  orderedInputColumns :=
    List.range' artifact.constantStartColumn artifact.constantValues.length ++
      artifact.localColumns ++ artifact.payloadColumns
  outputColumns := artifact.hashOutputColumns

def RawArtifact.equalityRows (artifact : RawArtifact) : List Row :=
  (List.range digestFields).map fun lane =>
    builderLinearRow (artifact.xOutSemanticColumns.getD lane 0)
      [(artifact.hashOutputColumns.getD lane 0, 1)]

def RawArtifact.programPieces (artifact : RawArtifact) : List (List Row) :=
  [constantRows artifact.hashRecipe,
    artifact.hashRecipe.trace.rows,
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
    49, payloadFields]

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 2 ∧
    artifact.profileId =
      "nightstream/goldilocks/streaming-terminal-phase-semantic/v2" ∧
    artifact.sourceIdentity = "rust:streaming-terminal-phase-semantic/v2" ∧
    artifact.sourceRowsSha256 =
      "89aae9a5eb9aa1f455cb97d60b648c7fdd03d729935d6d6cc87fe5419773173d" ∧
    artifact.rowCount = totalRows ∧
    artifact.columnCount = 352017 ∧
    artifact.sourceRowStart = 24 ∧
    artifact.finalRowStart = artifact.sourceRowStart ∧
    artifact.constantValues = expectedConstantValues ∧
    artifact.constantValues.length = constantFields ∧
    (∀ value ∈ artifact.constantValues, value < goldilocksP) ∧
    artifact.constantStartColumn = 2272 ∧
    artifact.localColumns = [33, 34, 35, 36] ∧
    artifact.payloadColumns = List.range' 37 payloadFields ∧
    artifact.hashOutputColumns = [332661, 332662, 332663, 332664] ∧
    artifact.xOutSemanticColumns = [20, 21, 22, 23] ∧
    artifact.baselineDigestValue < goldilocksP ∧
    artifact.equalityRowStart = hashTotalRows

instance (artifact : RawArtifact) : Decidable artifact.Valid := by
  unfold RawArtifact.Valid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact
