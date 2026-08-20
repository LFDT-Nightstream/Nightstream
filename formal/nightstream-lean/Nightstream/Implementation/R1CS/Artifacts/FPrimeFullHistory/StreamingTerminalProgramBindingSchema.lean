import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeSchema
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: compact exact-row schema for the terminal Nebula program-binding
family. It owns one Poseidon2 recipe and four links to the carried lane.

It does not own the source of the 12 verifier configuration constants or
Poseidon2 collision resistance.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProgramBinding.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact

def digestFields : Nat := 4
def constantFields : Nat := 19
def inputFields : Nat := 19
def absorbRounds : Nat := (inputFields + (rate - 1)) / rate
def traceRows : Nat :=
  1 + inputFields + absorbRounds * permutationRows + 1 + permutationRows
def equalityRowsCount : Nat := digestFields
def totalRows : Nat := constantFields + traceRows + equalityRowsCount

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
  inputColumns : List Nat
  hashOutputColumns : List Nat
  carriedBindingColumns : List Nat
  equalityRowStart : Nat
deriving DecidableEq, Repr

def RawArtifact.hashRecipe (artifact : RawArtifact) : VariableHashRecipe where
  constantValues := artifact.constantValues
  constantStartColumn := artifact.constantStartColumn
  localColumns := []
  payloadColumns := []
  orderedInputColumns := artifact.inputColumns
  outputColumns := artifact.hashOutputColumns

def RawArtifact.equalityRows (artifact : RawArtifact) : List Row :=
  (List.range digestFields).map fun lane =>
    builderLinearRow (artifact.carriedBindingColumns.getD lane 0)
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

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProgramBinding.Artifact
