import NightstreamFPrime.Layout.Stage1.AssemblerApplicationCompleteness
import NightstreamFPrime.Layout.Stage1.NextPreimageInputs
import NightstreamFPrime.Layout.Stage1.Spartan
import NightstreamFPrime.Layout.R1CS

/-!
Owns the one physical Stage 1 row order for a verifier-selected application.

The validated prefix is first mapped to Spartan order. Application-private
columns then occupy the old constant/public boundary, so the old constant and
public suffix moves by one exact displacement. The selected Lean application
rows and the five NextPreimage rows follow. No file boundary adds a row.
-/

namespace NightstreamFPrime.Layout.Stage1.Lowering

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

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

/-! ## Complete physical layout -/

def physicalRows
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) : List R1CS.Row :=
  (shiftRows program (Spartan.remappedRows relation) ++
    applicationRows program) ++ nextPreimageRows

def physicalRowCount
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (physicalRows relation program).length

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

def jointDomain
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  max (physicalRowCount relation program) (totalColumnCount program)

theorem physicalRowCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    physicalRowCount relation program =
      29218024 + R1CS.totalRowCount (applicationConstraints program) + 5 := by
  have prefixLength : (Spartan.remappedRows relation).length = 29218024 := by
    unfold Spartan.remappedRows Spartan.remapRows
    rw [List.length_map]
    exact Spartan.sourceRowCount_eq relation
  unfold physicalRowCount physicalRows shiftRows
  rw [List.length_append, List.length_append, List.length_map,
    prefixLength, applicationRows_length,
    nextPreimageRows_length]

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

/-! ## Sole logical circuit instantiation -/

/-- The canonical layout instantiates the sole logical Stage 1 circuit at its
verifier-owned compact root. -/
noncomputable def logicalCircuit
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) : FormalCircuit :=
  Lifecycle.Stage1.circuit relation ajtai program
    (AssemblerInputs.interface relation program) template
    (AssemblerInputs.rootOffset program)
    (AssemblerApplicationCompleteness.rootCompleteness relation ajtai program
      template)

theorem logicalCircuit_coverage
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) :
    (Circuit.ops (logicalCircuit relation ajtai program template).main
      (AssemblerInputs.rootOffset program)).length = 8 := by
  exact Lifecycle.Stage1.circuit_coverage relation ajtai program
    (AssemblerInputs.interface relation program) template
    (AssemblerInputs.rootOffset program)
    (AssemblerApplicationCompleteness.rootCompleteness relation ajtai program
      template)

end NightstreamFPrime.Layout.Stage1.Lowering
