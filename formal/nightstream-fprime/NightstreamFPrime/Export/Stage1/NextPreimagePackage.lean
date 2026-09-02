import NightstreamFPrime.Export.RowSemantics
import NightstreamFPrime.Export.Stage1.Rows
import NightstreamFPrime.Layout.Stage1.NextPreimageInputs

/-!
Owns the five ordinary package rows for HyperNova Construction 2's next
state-hash preimage. These rows read existing Spartan columns, allocate no
private value, and belong only to the per-application Stage 1 suffix.
-/

namespace NightstreamFPrime.Export.Stage1.NextPreimagePackage

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle.Stage1

def privateStart : Nat := Spartan.spartanColumnCount

def operations : List Op :=
  Circuit.ops (NextPreimage.main NextPreimageInputs.spartanInterface) privateStart

def constraints : List Expr := flatConstraints operations

def lowered : R1CS.LoweredConstraints :=
  R1CS.lowerConstraints constraints privateStart

def compiledRows (rowStart : Nat) : List Rows.CompiledRow :=
  Rows.compileRowsTR privateStart rowStart lowered.rows

def witnessInstructions (rowStart : Nat) : List WitnessInstruction :=
  Rows.witnessInstructionsTR (compiledRows rowStart)

def assertionRows (rowStart : Nat) : List SparseRow :=
  Rows.assertionRowsTR (compiledRows rowStart)

def sourceRows : List R1CS.Row :=
  (compiledRows 0).map Rows.CompiledRow.toR1CS

theorem operations_eq : operations =
    NextPreimage.opsAt NextPreimageInputs.spartanInterface privateStart := by
  rfl

theorem constraints_eq : constraints =
    NextPreimage.assertions NextPreimageInputs.spartanInterface privateStart := by
  unfold constraints operations
  rw [NextPreimage.main_ops, NextPreimage.flatConstraints_opsAt]

theorem lowered_rows_eq : lowered.rows =
    (R1CS.lowerConstraints constraints privateStart).rows := by
  rfl

theorem compiledRows_toR1CS (rowStart : Nat) :
    (compiledRows rowStart).map Rows.CompiledRow.toR1CS = lowered.rows := by
  unfold compiledRows
  rw [Rows.compileRowsTR_toR1CS]

theorem sourceRows_eq : sourceRows = lowered.rows := by
  unfold sourceRows
  exact compiledRows_toR1CS 0

theorem lowered_rowCount : lowered.rows.length = 5 := by
  rfl

theorem compiledRows_length (rowStart : Nat) :
    (compiledRows rowStart).length = 5 := by
  rw [← List.length_map, compiledRows_toR1CS, lowered_rowCount]

theorem compiledRows_rowIndices (rowStart : Nat) :
    (compiledRows rowStart).map Rows.CompiledRow.rowIndex =
      List.range' rowStart 5 := by
  unfold compiledRows
  rw [Rows.compileRowsTR_rowIndices, lowered_rowCount]

theorem sourceRows_length : sourceRows.length = 5 := by
  rw [sourceRows_eq, lowered_rowCount]

theorem witnessInstructions_eq_nil (rowStart : Nat) :
    witnessInstructions rowStart = [] := by
  rfl

theorem assertionRows_length (rowStart : Nat) :
    (assertionRows rowStart).length = 5 := by
  have classified :=
    Rows.witnessInstructionsTR_length_add_assertionRowsTR_length
      (compiledRows rowStart)
  have witnessLength := congrArg List.length
    (witnessInstructions_eq_nil rowStart)
  unfold witnessInstructions at witnessLength
  unfold assertionRows
  rw [witnessLength, List.length_nil, zero_add,
    compiledRows_length] at classified
  exact classified

/-- Satisfaction of the exact five source rows implies the typed wiring
predicate in the Spartan environment. -/
theorem sourceRows_imply_spec (env : Env)
    (rows : R1CS.RowsHold env sourceRows) :
    NextPreimage.SpecHolds NextPreimageInputs.spartanInterface privateStart env := by
  have loweredRows : R1CS.RowsHold env lowered.rows := by
    rw [← sourceRows_eq]
    exact rows
  have logical : ConstraintsHold env constraints :=
    R1CS.lowerConstraints_sound env constraints privateStart loweredRows
  have flat : holdsFlat env operations := by
    simpa [constraints] using logical
  exact NextPreimage.soundness NextPreimageInputs.spartanInterface env
    privateStart (holdsFlat_implies_holds env operations flat)

theorem sourceRows_varsBelow :
    ∀ row ∈ sourceRows, row.VarsBelow Spartan.spartanColumnCount := by
  rw [sourceRows_eq]
  apply R1CS.lowerConstraints_rows_varsBelow constraints privateStart
  rw [constraints_eq]
  exact NextPreimage.flatConstraints_varsBelow
    NextPreimageInputs.spartanInterface privateStart (fun _ => 0)
    (NextPreimageInputs.spartanAssumptions privateStart (fun _ => 0)
      (Nat.le_refl _))

end NightstreamFPrime.Export.Stage1.NextPreimagePackage
