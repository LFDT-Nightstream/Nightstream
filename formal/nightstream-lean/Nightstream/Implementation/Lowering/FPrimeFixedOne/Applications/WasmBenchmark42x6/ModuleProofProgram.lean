import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Module
import Nightstream.Implementation.Lowering.Goldilocks.InstructionReceipts
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsCompiler
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsManifest

/-!
Contract: native four-matrix CCS proof program for the exact 42-times-6
module.

Assurance tier: model-level.

Owns: public binding of every module byte, one private product, one public
result, exact selected-CCS rows, finite compilation, soundness, honest
completeness, ownership, and cost.

Does not own: arbitrary WASM, Rust parsing, Spartan, WHIR, recursive F-prime,
or a cryptographic reduction.

Emits constraints: sixty-one byte pins, one multiplication, and one output
link. The selector is the verifier-fixed constant one and adds no column.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofProgram

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm
open Nightstream.SuperNeo.Concrete

private abbrev Field := Nightstream.SuperNeo.Concrete.F

def moduleByteCount : Nat := certifiedModule.bytes.length

theorem moduleByteCount_exact : moduleByteCount = 61 := by
  decide

def byteOwner : PhysicalOwner := .typed (.input 0)

def productOwner : PhysicalOwner :=
  .typed (.instruction .root)

def outputOwner : PhysicalOwner :=
  .typed (.instruction (.rest .root))

def moduleByteColumn (index : Fin moduleByteCount) : ColumnId where
  owner := byteOwner
  bundleIndex := 0
  coordinateIndex := index.val

def productColumn : ColumnId where
  owner := productOwner
  bundleIndex := 0
  coordinateIndex := 0

def outputColumn : ColumnId where
  owner := outputOwner
  bundleIndex := 0
  coordinateIndex := 0

def byteField (value : Byte) : Field :=
  ⟨value.val, Nat.lt_trans value.isLt (by decide)⟩

def moduleByteValue (index : Fin moduleByteCount) : Field :=
  byteField (certifiedModule.bytes.get index)

def moduleByteAllocations : List OwnedColumn :=
  List.ofFn fun index : Fin moduleByteCount =>
    { id := moduleByteColumn index, ownership := .publicColumn }

def moduleByteRow (index : Fin moduleByteCount) : OwnedRow where
  id := { owner := byteOwner, ordinal := index.val }
  row := {
    a := singleton (moduleByteColumn index) 1
    b := singleton oneColumn 1
    c := singleton oneColumn (moduleByteValue index)
  }

def moduleByteRows : List OwnedRow :=
  List.ofFn moduleByteRow

def moduleByteReceipt : InstructionReceipt where
  owner := byteOwner
  kind := .literal
  allocations := moduleByteAllocations
  rows := moduleByteRows
  allocationsOwned := by
    intro column member
    rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
    rfl
  rowsOwned := by
    intro row member
    rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
    rfl

def productRow : OwnedRow where
  id := { owner := productOwner, ordinal := 0 }
  row := {
    a := singleton oneColumn 42
    b := singleton oneColumn 6
    c := singleton productColumn 1
  }

def productReceipt : InstructionReceipt where
  owner := productOwner
  kind := .product
  allocations :=
    [{ id := productColumn, ownership := .auxiliaryColumn }]
  rows := [productRow]
  allocationsOwned := by simp [productColumn, productOwner]
  rowsOwned := by simp [productRow, productOwner]

def outputRow : OwnedRow where
  id := { owner := outputOwner, ordinal := 0 }
  row := {
    a := singleton productColumn 1
    b := singleton oneColumn 1
    c := singleton outputColumn 1
  }

def outputReceipt : InstructionReceipt where
  owner := outputOwner
  kind := .affine
  allocations :=
    [{ id := outputColumn, ownership := .publicColumn }]
  rows := [outputRow]
  allocationsOwned := by simp [outputColumn, outputOwner]
  rowsOwned := by simp [outputRow, outputOwner]

def selected (receipt : InstructionReceipt) : SelectedReceipt where
  receipt := receipt
  selector := oneColumn

/-- The module proof is a native selected-CCS program. All rows are active
because their selector is the verifier-fixed constant-one column. -/
def program : NativeCcsProgram.Program where
  one := oneColumn
  receipts :=
    [ selected InstructionReceipt.prelude
    , selected moduleByteReceipt
    , selected productReceipt
    , selected outputReceipt
    ]

@[simp] theorem program_rows_length : program.rows.length = 63 := by
  simp [program, selected, NativeCcsProgram.Program.rows,
    SelectedReceipt.rows, NativeCcsSelector.select, moduleByteReceipt,
    moduleByteRows, productReceipt, outputReceipt, moduleByteCount_exact]

@[simp] theorem program_cost_exact :
    program.cost = ⟨63, 0, 63, 1⟩ := by
  decide

theorem program_columnIds_nodup : program.columnIds.Nodup := by
  decide

theorem program_rowIds_nodup : program.rowIds.Nodup := by
  decide

theorem oneColumn_mem : oneColumn ∈ program.columnIds := by
  simp [program, selected, NativeCcsProgram.Program.columnIds,
    NativeCcsProgram.Program.allocations, SelectedReceipt.allocations,
    InstructionReceipt.prelude, preludeColumns]

theorem moduleByteColumn_mem (index : Fin moduleByteCount) :
    moduleByteColumn index ∈ program.columnIds := by
  simp [program, selected, NativeCcsProgram.Program.columnIds,
    NativeCcsProgram.Program.allocations, SelectedReceipt.allocations,
    InstructionReceipt.prelude, preludeColumns, moduleByteReceipt,
    moduleByteAllocations, productReceipt, outputReceipt]

theorem productColumn_mem : productColumn ∈ program.columnIds := by
  simp [program, selected, NativeCcsProgram.Program.columnIds,
    NativeCcsProgram.Program.allocations, SelectedReceipt.allocations,
    InstructionReceipt.prelude, preludeColumns, moduleByteReceipt,
    moduleByteAllocations, productReceipt, outputReceipt]

theorem outputColumn_mem : outputColumn ∈ program.columnIds := by
  simp [program, selected, NativeCcsProgram.Program.columnIds,
    NativeCcsProgram.Program.allocations, SelectedReceipt.allocations,
    InstructionReceipt.prelude, preludeColumns, moduleByteReceipt,
    moduleByteAllocations, productReceipt, outputReceipt]

private theorem sourceSatisfies
    (receipt : InstructionReceipt)
    (member : selected receipt ∈ program.receipts)
    (assignment : ColumnId → Field)
    (satisfied : program.Satisfies assignment) :
    Goldilocks.Satisfies receipt.rows assignment := by
  exact program.source_satisfies_of_selector_one assignment satisfied
    (selected receipt) member satisfied.1

private theorem satisfies_member
    (rows : List OwnedRow)
    (assignment : ColumnId → Field)
    (satisfied : Goldilocks.Satisfies rows assignment)
    (row : OwnedRow)
    (member : row ∈ rows) :
    row.row.Holds assignment := by
  induction rows with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      rcases List.mem_cons.mp member with rfl | tailMember
      · exact satisfied.1
      · exact inductionHypothesis satisfied.2 tailMember

theorem moduleBytes_bound
    (assignment : ColumnId → Field)
    (satisfied : program.Satisfies assignment)
    (index : Fin moduleByteCount) :
    assignment (moduleByteColumn index) = moduleByteValue index := by
  have allRows := sourceSatisfies moduleByteReceipt (by simp [program])
    assignment satisfied
  have rowHolds := satisfies_member moduleByteRows assignment allRows
    (moduleByteRow index) (List.mem_ofFn.mpr ⟨index, rfl⟩)
  have oneExact : assignment oneColumn = 1 := by
    simpa [program] using satisfied.1
  simpa [moduleByteRow, Row.Holds, LinearCombination.eval,
    Nightstream.Implementation.Lowering.Goldilocks.singleton,
    moduleByteValue, oneExact, Fin.one_mul, Fin.mul_one] using rowHolds

private theorem field_42_mul_6 : (42 : Field) * 6 = 252 := by
  rfl

theorem product_bound
    (assignment : ColumnId → Field)
    (satisfied : program.Satisfies assignment) :
    assignment productColumn = 252 := by
  have rows := sourceSatisfies productReceipt (by simp [program])
    assignment satisfied
  have rowHolds := rows.1
  have oneExact : assignment oneColumn = 1 := by
    simpa [program] using satisfied.1
  simpa [productReceipt, productRow, Row.Holds, LinearCombination.eval,
    Nightstream.Implementation.Lowering.Goldilocks.singleton,
    oneExact, field_42_mul_6, Fin.one_mul, Fin.mul_one] using rowHolds.symm

theorem output_bound
    (assignment : ColumnId → Field)
    (satisfied : program.Satisfies assignment) :
    assignment outputColumn = 252 := by
  have rows := sourceSatisfies outputReceipt (by simp [program])
    assignment satisfied
  have rowHolds := rows.1
  have productExact := product_bound assignment satisfied
  have oneExact : assignment oneColumn = 1 := by
    simpa [program] using satisfied.1
  simpa [outputReceipt, outputRow, Row.Holds, LinearCombination.eval,
    Nightstream.Implementation.Lowering.Goldilocks.singleton,
    oneExact, productExact, Fin.one_mul, Fin.mul_one] using rowHolds.symm

/-- Any satisfying assignment binds the exact module bytes and accepts only
the result computed by those bytes. -/
theorem soundness
    (assignment : ColumnId → Field)
    (satisfied : program.Satisfies assignment) :
    (∀ index, assignment (moduleByteColumn index) = moduleByteValue index) ∧
      assignment outputColumn = 252 ∧
      module.run = some (assignment outputColumn).val := by
  refine ⟨moduleBytes_bound assignment satisfied,
    output_bound assignment satisfied, ?_⟩
  rw [output_bound assignment satisfied]
  exact module_computes_252

def honestAssignment (column : ColumnId) : Field :=
  if column = oneColumn then 1
  else if column = productColumn then 252
  else if column = outputColumn then 252
  else if byteColumn :
      column.owner = byteOwner ∧
        column.bundleIndex = 0 ∧
        column.coordinateIndex < moduleByteCount then
    moduleByteValue ⟨column.coordinateIndex, byteColumn.2.2⟩
  else 0

@[simp] theorem honestAssignment_one : honestAssignment oneColumn = 1 := by
  simp [honestAssignment]

@[simp] theorem honestAssignment_product :
    honestAssignment productColumn = 252 := by
  simp [honestAssignment, productColumn, productOwner, oneColumn]

@[simp] theorem honestAssignment_output :
    honestAssignment outputColumn = 252 := by
  simp [honestAssignment, outputColumn, outputOwner, productColumn,
    productOwner, oneColumn]

@[simp] theorem honestAssignment_moduleByte (index : Fin moduleByteCount) :
    honestAssignment (moduleByteColumn index) = moduleByteValue index := by
  simp [honestAssignment, moduleByteColumn, byteOwner, oneColumn,
    productColumn, productOwner, outputColumn, outputOwner]

private theorem satisfies_of_all
    (rows : List OwnedRow)
    (assignment : ColumnId → Field)
    (all : ∀ row, row ∈ rows → row.row.Holds assignment) :
    Goldilocks.Satisfies rows assignment := by
  induction rows with
  | nil => trivial
  | cons head tail inductionHypothesis =>
      exact ⟨all head List.mem_cons_self,
        inductionHypothesis (fun row member =>
          all row (List.mem_cons_of_mem head member))⟩

private theorem honest_moduleByteRows :
    Goldilocks.Satisfies moduleByteRows honestAssignment := by
  apply satisfies_of_all
  intro row member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  simp [moduleByteRow, Row.Holds, LinearCombination.eval,
    Nightstream.Implementation.Lowering.Goldilocks.singleton,
    Fin.one_mul, Fin.mul_one]

private theorem honest_productRows :
    Goldilocks.Satisfies productReceipt.rows honestAssignment := by
  simp only [productReceipt, productRow, Goldilocks.satisfies_cons,
    Goldilocks.satisfies_nil, Row.Holds, LinearCombination.eval,
    Nightstream.Implementation.Lowering.Goldilocks.singleton,
    honestAssignment_one, honestAssignment_product, Fin.one_mul, Fin.mul_one,
    Fin.add_zero, field_42_mul_6, and_self]

private theorem honest_outputRows :
    Goldilocks.Satisfies outputReceipt.rows honestAssignment := by
  simp only [outputReceipt, outputRow, Goldilocks.satisfies_cons,
    Goldilocks.satisfies_nil, Row.Holds, LinearCombination.eval,
    Nightstream.Implementation.Lowering.Goldilocks.singleton,
    honestAssignment_one, honestAssignment_product, honestAssignment_output,
    Fin.one_mul, Fin.mul_one, Fin.add_zero, and_self]

theorem honest_satisfies : program.Satisfies honestAssignment := by
  refine ⟨honestAssignment_one, ?_⟩
  apply (NativeCcsProgram.Program.satisfies_flattened_receipts_iff
    program.receipts honestAssignment).2
  intro receipt member
  simp [program] at member
  rcases member with rfl | rfl | rfl | rfl
  · exact NativeCcsSelector.complete oneColumn [] honestAssignment trivial
  · exact NativeCcsSelector.complete oneColumn moduleByteRows honestAssignment
      honest_moduleByteRows
  · exact NativeCcsSelector.complete oneColumn productReceipt.rows
      honestAssignment honest_productRows
  · exact NativeCcsSelector.complete oneColumn outputReceipt.rows
      honestAssignment honest_outputRows

theorem honest_output : honestAssignment outputColumn = 252 := by
  exact honestAssignment_output

/-- Static facts needed by the finite four-matrix compiler. -/
def valid : NativeCcsCompiler.Valid program where
  oneAllocated := by simp [program, selected, NativeCcsProgram.Program.columnIds,
    NativeCcsProgram.Program.allocations, SelectedReceipt.allocations,
    InstructionReceipt.prelude, preludeColumns]
  columnIdsNodup := program_columnIds_nodup
  rowsSupported := by
    intro row rowMember column columnMember
    simp [program, selected, NativeCcsProgram.Program.rows,
      SelectedReceipt.rows, NativeCcsSelector.select,
      InstructionReceipt.prelude, moduleByteReceipt, moduleByteRows,
      moduleByteAllocations, productReceipt, outputReceipt] at rowMember
    rcases rowMember with ⟨index, rfl⟩ | rfl | rfl
    · simp [SelectedRow.columnIds, moduleByteRow, OwnedRow.columnIds,
        Row.columnIds,
        Nightstream.Implementation.Lowering.Goldilocks.singleton] at columnMember
      rcases columnMember with rfl | remaining
      · exact oneColumn_mem
      · rcases remaining with rfl | rfl
        · exact moduleByteColumn_mem index
        · exact oneColumn_mem
    · simp [SelectedRow.columnIds, productRow, OwnedRow.columnIds,
        Row.columnIds,
        Nightstream.Implementation.Lowering.Goldilocks.singleton] at columnMember
      rcases columnMember with rfl | rfl
      · exact oneColumn_mem
      · exact productColumn_mem
    · simp [SelectedRow.columnIds, outputRow, OwnedRow.columnIds,
        Row.columnIds,
        Nightstream.Implementation.Lowering.Goldilocks.singleton] at columnMember
      rcases columnMember with rfl | remaining
      · exact oneColumn_mem
      · rcases remaining with rfl | remaining
        · exact productColumn_mem
        · rcases remaining with rfl | rfl
          · exact oneColumn_mem
          · exact outputColumn_mem

def rowDomain : NativeCcsCompiler.RowDomain program where
  rowVariables := 6
  rowsCovered := by rw [program_rows_length]; decide

theorem finite_accepts_honest :
    NativeCcsCompiler.IndexedAccepts program valid rowDomain
      (NativeCcsCompiler.indexedAssignment program honestAssignment) := by
  exact (NativeCcsCompiler.indexedAssignment_accepts_iff
    program valid rowDomain honestAssignment).2 honest_satisfies

theorem matrixCount_exact :
    (NativeCcsCompiler.system program valid rowDomain).constraintPolynomial =
      NativeCcsSelector.constraintPolynomial :=
  NativeCcsCompiler.matrix_count_exact program valid rowDomain

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.ModuleProofProgram
