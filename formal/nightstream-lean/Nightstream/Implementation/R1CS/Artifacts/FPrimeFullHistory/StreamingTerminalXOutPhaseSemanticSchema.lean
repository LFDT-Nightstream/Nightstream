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
def payloadFields : Nat := 4
def constantFields : Nat := 11
def hashInputFields : Nat := 19
def absorbRounds : Nat := 5
def hashTotalRows : Nat := 3632
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
    49, 4]

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 1 ∧
    artifact.profileId =
      "nightstream/goldilocks/streaming-terminal-phase-semantic/v1" ∧
    artifact.sourceIdentity = "rust:streaming-terminal-phase-semantic/v1" ∧
    artifact.sourceRowsSha256.length = 64 ∧
    artifact.rowCount = totalRows ∧
    artifact.columnCount = 23087 ∧
    artifact.sourceRowStart = 24 ∧
    artifact.finalRowStart = artifact.sourceRowStart ∧
    artifact.constantValues = expectedConstantValues ∧
    artifact.constantValues.length = constantFields ∧
    (∀ value ∈ artifact.constantValues, value < goldilocksP) ∧
    artifact.constantStartColumn = 107 ∧
    artifact.localColumns = [33, 34, 35, 36] ∧
    artifact.payloadColumns = [37, 38, 39, 40] ∧
    artifact.hashOutputColumns = [3731, 3732, 3733, 3734] ∧
    artifact.xOutSemanticColumns = [20, 21, 22, 23] ∧
    artifact.baselineDigestValue < goldilocksP ∧
    artifact.equalityRowStart = hashTotalRows

instance (artifact : RawArtifact) : Decidable artifact.Valid := by
  unfold RawArtifact.Valid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact
