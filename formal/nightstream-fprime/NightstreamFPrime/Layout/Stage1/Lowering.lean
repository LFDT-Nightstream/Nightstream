import NightstreamFPrime.Layout.Stage1.ApplicationInputs
import NightstreamFPrime.Layout.Stage1.NextPreimageInputs
import NightstreamFPrime.Layout.R1CS

/-!
Owns the application and NextPreimage lowering, final column counts, and
assignment maps for a verifier-selected application.

Application-private columns occupy the old constant/public boundary, so the
old constant and public suffix moves by one exact displacement. `LoweringRows`
owns the complete physical rows and logical Stage 1 circuit instantiation.
-/

namespace NightstreamFPrime.Layout.Stage1.Lowering

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle

/-! ## Selected application lowering -/

def applicationOperations
    (program : Lifecycle.Stage1.Application.Program) : List Op :=
  Circuit.ops
    (program.circuit (ApplicationInputs.interface program)).main
    (ApplicationInputs.localStart program)

def applicationConstraints
    (program : Lifecycle.Stage1.Application.Program) : List Expr :=
  flatConstraints (applicationOperations program)

def applicationFirstFresh
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  ApplicationInputs.localStart program +
    localLength (applicationOperations program)

def applicationPlan
    (program : Lifecycle.Stage1.Application.Program) : R1CS.LoweringPlan where
  constraints := applicationConstraints program
  firstFresh := applicationFirstFresh program

def applicationRows
    (program : Lifecycle.Stage1.Application.Program) : List R1CS.Row :=
  (applicationPlan program).rows

def applicationPrivateCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  localLength (applicationOperations program) +
    (applicationPlan program).freshColumnCount

/-- New caller-owned application witness words plus all application logical
and R1CS fresh columns. -/
def addedPrivateColumnCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  program.witnessWordCount + applicationPrivateCount program

theorem applicationRows_length
    (program : Lifecycle.Stage1.Application.Program) :
    (applicationRows program).length =
      R1CS.totalRowCount (applicationConstraints program) := by
  exact R1CS.LoweringPlan.rowCount_eq (applicationPlan program)

theorem applicationPlan_next
    (program : Lifecycle.Stage1.Application.Program) :
    (applicationPlan program).next =
      Spartan.privateColumnCount + addedPrivateColumnCount program := by
  rw [R1CS.LoweringPlan.next_eq]
  change applicationFirstFresh program +
      (applicationPlan program).freshColumnCount =
    Spartan.privateColumnCount + addedPrivateColumnCount program
  unfold applicationFirstFresh addedPrivateColumnCount
    applicationPrivateCount ApplicationInputs.localStart
    ApplicationInputs.witnessStart
  omega

/-! ## Final suffix relocation -/

/-- Existing private columns stay fixed. The old constant and all public
columns move after the exact application-private suffix. -/
def shiftColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Nat) : Nat :=
  if column < Spartan.constantColumn then column
  else column + addedPrivateColumnCount program

@[simp] theorem shiftColumn_private
    (program : Lifecycle.Stage1.Application.Program) (column : Nat)
    (bound : column < Spartan.constantColumn) :
    shiftColumn program column = column := by
  simp [shiftColumn, bound]

@[simp] theorem shiftColumn_constantOrPublic
    (program : Lifecycle.Stage1.Application.Program) (column : Nat)
    (bound : Spartan.constantColumn ≤ column) :
    shiftColumn program column = column + addedPrivateColumnCount program := by
  simp [shiftColumn, Nat.not_lt.mpr bound]

def shiftCombination (program : Lifecycle.Stage1.Application.Program)
    (combination : R1CS.LinearCombination) : R1CS.LinearCombination :=
  ⟨combination.constant,
    combination.terms.map fun term => (shiftColumn program term.1, term.2)⟩

def shiftRow (program : Lifecycle.Stage1.Application.Program)
    (row : R1CS.Row) : R1CS.Row :=
  ⟨shiftCombination program row.a, shiftCombination program row.b,
    shiftCombination program row.c⟩

def shiftRows (program : Lifecycle.Stage1.Application.Program)
    (rows : List R1CS.Row) : List R1CS.Row :=
  rows.map (shiftRow program)

def basePullback (program : Lifecycle.Stage1.Application.Program)
    (env : Env) : Env :=
  fun column => env (shiftColumn program column)

theorem shiftCombination_eval
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (combination : R1CS.LinearCombination) :
    (shiftCombination program combination).eval env =
      combination.eval (basePullback program env) := by
  unfold shiftCombination R1CS.LinearCombination.eval basePullback
  rw [List.map_map]
  rfl

theorem shiftRow_holds
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (row : R1CS.Row) :
    (shiftRow program row).Holds env ↔
      row.Holds (basePullback program env) := by
  simp [R1CS.Row.Holds, shiftRow, shiftCombination_eval]

theorem shiftRows_hold
    (program : Lifecycle.Stage1.Application.Program) (env : Env)
    (rows : List R1CS.Row) :
    R1CS.RowsHold env (shiftRows program rows) ↔
      R1CS.RowsHold (basePullback program env) rows := by
  constructor
  · intro holds row member
    exact (shiftRow_holds program env row).mp
      (holds (shiftRow program row) (by
        exact List.mem_map.mpr ⟨row, member, rfl⟩))
  · intro holds row member
    rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
    exact (shiftRow_holds program env source).mpr
      (holds source sourceMember)

/-! ## Next-preimage lowering -/

def nextPreimagePrivateStart : Nat := Spartan.spartanColumnCount

def nextPreimageOperations : List Op :=
  Circuit.ops
    (Lifecycle.Stage1.NextPreimage.main NextPreimageInputs.spartanInterface)
    nextPreimagePrivateStart

def nextPreimageConstraints : List Expr :=
  flatConstraints nextPreimageOperations

def nextPreimagePlan : R1CS.LoweringPlan where
  constraints := nextPreimageConstraints
  firstFresh := nextPreimagePrivateStart

def nextPreimageRows : List R1CS.Row := nextPreimagePlan.rows

theorem nextPreimageRows_length : nextPreimageRows.length = 5 := by
  rfl

theorem nextPreimage_noFresh : nextPreimagePlan.freshColumnCount = 0 := by
  rfl

/-! ## Final column counts -/

def privateColumnCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  Spartan.privateColumnCount + addedPrivateColumnCount program

def constantColumn
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  Spartan.constantColumn + addedPrivateColumnCount program

def publicColumnCount : Nat := Spartan.publicColumnCount

def totalColumnCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  Spartan.spartanColumnCount + addedPrivateColumnCount program

theorem constantColumn_eq_privateColumnCount
    (program : Lifecycle.Stage1.Application.Program) :
    constantColumn program = privateColumnCount program := by
  rfl

theorem totalColumnCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    totalColumnCount program =
      privateColumnCount program + 1 + publicColumnCount := by
  unfold totalColumnCount privateColumnCount publicColumnCount
    addedPrivateColumnCount
  norm_num [Spartan.spartanColumnCount, Spartan.privateColumnCount,
    Spartan.publicColumnCount]
  omega

end NightstreamFPrime.Layout.Stage1.Lowering
