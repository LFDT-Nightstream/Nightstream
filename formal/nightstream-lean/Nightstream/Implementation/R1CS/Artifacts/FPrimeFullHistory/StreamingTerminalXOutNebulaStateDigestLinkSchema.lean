import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeSchema
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: compact exact-row schema for the terminal Nebula-state-digest family.

Owns the absent and present Poseidon2 recipes, the open-bit check, the selected
digest mux, and four final XOut links. Rust checks this structural recipe
against all 19,353 source rows.

Does not own the authority of the lane fields or Poseidon2 collision resistance.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact

def digestFields : Nat := 4
def bitRowCount : Nat := 1
def muxRowCount : Nat := digestFields
def equalityRowCount : Nat := digestFields
def absentConstantFields : Nat := 13
def absentInputFields : Nat := 58
def presentConstantFields : Nat := 10
def presentInputFields : Nat := 59
def hashRowCount (constantFields inputFields : Nat) : Nat :=
  constantFields + 1 + inputFields +
    ((inputFields + (rate - 1)) / rate) * permutationRows +
    1 + permutationRows
def absentHashRows : Nat :=
  hashRowCount absentConstantFields absentInputFields
def presentHashRows : Nat :=
  hashRowCount presentConstantFields presentInputFields
def familyRows : Nat := 19353

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceIdentity : String
  sourceRowsSha256 : String
  rowCount : Nat
  columnCount : Nat
  sourceRowStart : Nat
  finalRowStart : Nat
  openColumn : Nat
  absentConstantValues : List Nat
  absentConstantStartColumn : Nat
  absentInputColumns : List Nat
  absentOutputColumns : List Nat
  presentConstantValues : List Nat
  presentConstantStartColumn : Nat
  presentInputColumns : List Nat
  presentOutputColumns : List Nat
  hashOutputColumns : List Nat
  xOutStateColumns : List Nat
  baselineDigestValue : Nat
  absentRowStart : Nat
  presentRowStart : Nat
  muxRowStart : Nat
  equalityRowStart : Nat
  selectedSourceRow : Nat
deriving DecidableEq, Repr

def RawArtifact.absentRecipe (artifact : RawArtifact) : VariableHashRecipe where
  constantValues := artifact.absentConstantValues
  constantStartColumn := artifact.absentConstantStartColumn
  localColumns := []
  payloadColumns := []
  orderedInputColumns := artifact.absentInputColumns
  outputColumns := artifact.absentOutputColumns

def RawArtifact.presentRecipe (artifact : RawArtifact) : VariableHashRecipe where
  constantValues := artifact.presentConstantValues
  constantStartColumn := artifact.presentConstantStartColumn
  localColumns := []
  payloadColumns := []
  orderedInputColumns := artifact.presentInputColumns
  outputColumns := artifact.presentOutputColumns

def RawArtifact.bitRow (artifact : RawArtifact) : Row :=
  ⟨[(artifact.openColumn, 1)],
    [(0, negCoeff 1), (artifact.openColumn, 1)], []⟩

def selectedMuxRow
    (selector present absent output : Nat) : Row :=
  ⟨[(selector, 1)],
    [(absent, negCoeff 1), (present, 1)],
    [(absent, negCoeff 1), (output, 1)]⟩

def RawArtifact.muxRows (artifact : RawArtifact) : List Row :=
  (List.range digestFields).map fun lane =>
    selectedMuxRow artifact.openColumn
      (artifact.presentOutputColumns.getD lane 0)
      (artifact.absentOutputColumns.getD lane 0)
      (artifact.hashOutputColumns.getD lane 0)

def RawArtifact.equalityRows (artifact : RawArtifact) : List Row :=
  (List.range digestFields).map fun lane =>
    builderLinearRow (artifact.xOutStateColumns.getD lane 0)
      [(artifact.hashOutputColumns.getD lane 0, 1)]

def RawArtifact.programPieces (artifact : RawArtifact) : List (List Row) :=
  [[artifact.bitRow],
    constantRows artifact.absentRecipe,
    artifact.absentRecipe.trace.rows,
    constantRows artifact.presentRecipe,
    artifact.presentRecipe.trace.rows,
    artifact.muxRows,
    artifact.equalityRows]

def RawArtifact.program (artifact : RawArtifact) : List Row :=
  artifact.programPieces.flatten

def RawArtifact.Satisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  Satisfies artifact.program assignment

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 2 ∧
    artifact.profileId =
      "nightstream/goldilocks/streaming-terminal-nebula-state-digest/v2" ∧
    artifact.sourceIdentity =
      "rust:streaming-terminal-nebula-state-digest/v2" ∧
    artifact.sourceRowsSha256 =
      "89aae9a5eb9aa1f455cb97d60b648c7fdd03d729935d6d6cc87fe5419773173d" ∧
    artifact.rowCount = familyRows ∧
    artifact.columnCount = 352017 ∧
    artifact.sourceRowStart = 330425 ∧
    artifact.finalRowStart = artifact.sourceRowStart ∧
    artifact.openColumn = 2210 ∧
    artifact.absentConstantValues.length = absentConstantFields ∧
    (∀ value ∈ artifact.absentConstantValues, value < goldilocksP) ∧
    artifact.absentConstantStartColumn = 332669 ∧
    artifact.absentInputColumns.length = absentInputFields ∧
    artifact.absentOutputColumns = [342334, 342335, 342336, 342337] ∧
    artifact.presentConstantValues.length = presentConstantFields ∧
    (∀ value ∈ artifact.presentConstantValues, value < goldilocksP) ∧
    artifact.presentConstantStartColumn = 342342 ∧
    artifact.presentInputColumns.length = presentInputFields ∧
    artifact.presentOutputColumns = [352005, 352006, 352007, 352008] ∧
    artifact.hashOutputColumns.length = digestFields ∧
    (∀ column ∈ artifact.hashOutputColumns, column < artifact.columnCount) ∧
    artifact.xOutStateColumns = [29, 30, 31, 32] ∧
    artifact.baselineDigestValue < goldilocksP ∧
    artifact.absentRowStart = bitRowCount ∧
    artifact.presentRowStart = artifact.absentRowStart + absentHashRows ∧
    artifact.muxRowStart = artifact.presentRowStart + presentHashRows ∧
    artifact.equalityRowStart = artifact.muxRowStart + muxRowCount ∧
    artifact.equalityRowStart + equalityRowCount = artifact.rowCount ∧
    artifact.selectedSourceRow =
      artifact.sourceRowStart + artifact.equalityRowStart ∧
    artifact.selectedSourceRow = 349774

instance (artifact : RawArtifact) : Decidable artifact.Valid := by
  unfold RawArtifact.Valid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact
