import Nightstream.Implementation.R1CS.Core.Poseidon2Call

/-!
Contract: compact schema for the two production PiRLC family replay shapes.

The input replay uses the exact 918 columns read by the PiRLC algebra. The
output replay uses the same 54 algebra output columns. Each compact call is an
exact column renaming of the 600-row production Poseidon2 artifact.

Assurance tier: artifact schema. Generated data and its Rust drift owner are
separate obligations.

Emits constraints: no. It describes existing constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplay.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

structure RawArm where
  rowCount : Nat
  columnCount : Nat
  beforeAbsorbed : Nat
  afterAbsorbed : Nat
  inputPoseidon2CallCount : Nat
  outputPoseidon2CallCount : Nat
  inputColumns : List Nat
  outputColumns : List Nat
  inputBeforeColumns : List Nat
  inputAfterColumns : List Nat
  outputBeforeColumns : List Nat
  outputAfterColumns : List Nat
  poseidon2Calls : List Poseidon2Call.Call
deriving DecidableEq, Repr

def PoseidonCallValid (columnCount : Nat) (call : Poseidon2Call.Call) : Prop :=
  call.rowEnd = call.rowStart + 600 ∧
    call.inputColumns.length = 8 ∧
    (∀ column ∈ call.inputColumns, column < columnCount) ∧
    call.firstAllocatedColumn + 600 ≤ columnCount

instance (columnCount : Nat) (call : Poseidon2Call.Call) :
    Decidable (PoseidonCallValid columnCount call) := by
  unfold PoseidonCallValid
  infer_instance

def exactCallChainFrom : Nat → Nat → List Poseidon2Call.Call → Bool
  | _, _, [] => true
  | row, column, call :: rest =>
      call.rowStart == row &&
        call.rowEnd == row + 600 &&
        call.firstAllocatedColumn == column &&
        exactCallChainFrom (row + 600) (column + 600) rest

def ExactCallChain
    (rowCount columnCount : Nat) (calls : List Poseidon2Call.Call) : Prop :=
  exactCallChainFrom 0 165680 calls = true ∧
    rowCount = calls.length * 600 ∧
    columnCount = 165680 + rowCount

instance (rowCount columnCount : Nat) (calls : List Poseidon2Call.Call) :
    Decidable (ExactCallChain rowCount columnCount calls) := by
  unfold ExactCallChain
  infer_instance

def columnsValid (columnCount expectedLength : Nat) (columns : List Nat) : Prop :=
  columns.length = expectedLength ∧ columns.Nodup ∧
    ∀ column ∈ columns, column < columnCount

instance (columnCount expectedLength : Nat) (columns : List Nat) :
    Decidable (columnsValid columnCount expectedLength columns) := by
  unfold columnsValid
  infer_instance

def RawArm.Valid
    (arm : RawArm) (beforeAbsorbed afterAbsorbed inputCalls outputCalls : Nat) : Prop :=
  arm.rowCount > 0 ∧ arm.columnCount > 165680 ∧
    arm.beforeAbsorbed = beforeAbsorbed ∧
    arm.afterAbsorbed = afterAbsorbed ∧
    arm.inputPoseidon2CallCount = inputCalls ∧
    arm.outputPoseidon2CallCount = outputCalls ∧
    arm.poseidon2Calls.length = inputCalls + outputCalls ∧
    arm.inputColumns = List.range' 919 918 ∧
    arm.outputColumns = List.range' 1837 54 ∧
    arm.inputBeforeColumns = List.range' 165664 8 ∧
    arm.outputBeforeColumns = List.range' 165672 8 ∧
    columnsValid arm.columnCount 8 arm.inputAfterColumns ∧
    columnsValid arm.columnCount 8 arm.outputAfterColumns ∧
    (∀ call ∈ arm.poseidon2Calls, PoseidonCallValid arm.columnCount call) ∧
    ExactCallChain arm.rowCount arm.columnCount arm.poseidon2Calls

instance
    (arm : RawArm) (beforeAbsorbed afterAbsorbed inputCalls outputCalls : Nat) :
    Decidable (arm.Valid beforeAbsorbed afterAbsorbed inputCalls outputCalls) := by
  unfold RawArm.Valid
  infer_instance

def RawArm.Satisfied (arm : RawArm) (assignment : Nat → Nat) : Prop :=
  ∀ call ∈ arm.poseidon2Calls, Satisfies call.rows assignment

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceColumns : Nat
  even : RawArm
  odd : RawArm
deriving DecidableEq, Repr

def RawArtifact.Valid (artifact : RawArtifact) : Prop :=
  artifact.schemaVersion = 1 ∧
    artifact.profileId =
      "nebula-f-prime-streaming-pi-rlc-family-replay-v1" ∧
    artifact.sourceColumns = 165664 ∧
    artifact.even.rowCount = 145200 ∧
    artifact.even.columnCount = 310880 ∧
    artifact.odd.rowCount = 146400 ∧
    artifact.odd.columnCount = 312080 ∧
    artifact.even.Valid 0 2 229 13 ∧
    artifact.odd.Valid 2 0 230 14

instance (artifact : RawArtifact) : Decidable artifact.Valid := by
  unfold RawArtifact.Valid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplay.Artifact
