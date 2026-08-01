import Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram

/-!
Contract: proof-free manifest for the native four-matrix CCS selector.

Assurance tier: model-level.

Owns:
- the exact four-matrix, degree-three selector polynomial description;
- normalized A/B/C rows and one selector column per physical receipt;
- exact receipt order, allocation ownership, row identity, cost, and
  satisfaction round trips.

Does not own: protocol metadata, JSON, Rust parsing, or a deployment value.

Emits constraints: none. It serializes an existing native CCS program.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.NativeCcsManifest

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector

private abbrev Field := Nightstream.SuperNeo.Concrete.F

/-- The only coefficients in the selected-CCS polynomial. -/
inductive Sign where
  | positive
  | negative
deriving DecidableEq, Repr

/-- One sparse multivariate monomial. Exponents are in matrix order
`[A, B, C, S]`. -/
structure PolynomialTerm where
  sign : Sign
  exponents : List Nat
deriving DecidableEq, Repr

/-- The exact polynomial `A * B * S - C * S`. -/
def selectorPolynomial : List PolynomialTerm :=
  [
    { sign := .positive, exponents := [1, 1, 0, 1] },
    { sign := .negative, exponents := [0, 0, 1, 1] }
  ]

theorem selectorPolynomial_exact :
    selectorPolynomial =
      [
        { sign := .positive, exponents := [1, 1, 0, 1] },
        { sign := .negative, exponents := [0, 0, 1, 1] }
      ] :=
  rfl

/-- Proof-erased selected receipt recovered by a manifest consumer. -/
structure SelectedReceiptImage where
  owner : PhysicalOwner
  kind : InstructionKind
  allocations : List OwnedColumn
  selector : ColumnId
  rows : List SelectedRow
deriving DecidableEq, Repr

/-- Proof-free selected receipt in exact emission order. -/
structure ManifestReceipt where
  owner : PhysicalOwner
  kind : InstructionKind
  allocations : List OwnedColumn
  selector : ColumnId
  rows : List CanonicalManifest.ManifestRow
deriving DecidableEq, Repr

namespace ManifestReceipt

def ofSelectedReceipt
    (receipt : SelectedReceipt) : ManifestReceipt where
  owner := receipt.receipt.owner
  kind := receipt.receipt.kind
  allocations := receipt.receipt.allocations
  selector := receipt.selector
  rows :=
    receipt.receipt.rows.map CanonicalManifest.ManifestRow.ofOwnedRow

def decode (receipt : ManifestReceipt) : SelectedReceiptImage where
  owner := receipt.owner
  kind := receipt.kind
  allocations := receipt.allocations
  selector := receipt.selector
  rows :=
    receipt.rows.map fun row =>
      { source := row.decode, selector := receipt.selector }

def imageOf (receipt : SelectedReceipt) : SelectedReceiptImage where
  owner := receipt.receipt.owner
  kind := receipt.receipt.kind
  allocations := receipt.receipt.allocations
  selector := receipt.selector
  rows :=
    receipt.receipt.rows.map fun row =>
      { source := normalizeOwnedRow row, selector := receipt.selector }

@[simp] theorem decode_ofSelectedReceipt
    (receipt : SelectedReceipt) :
    (ofSelectedReceipt receipt).decode = imageOf receipt := by
  cases receipt with
  | mk source selector =>
      simp [ofSelectedReceipt, decode, imageOf, List.map_map,
        Function.comp_def]

def asCanonical (receipt : ManifestReceipt) :
    CanonicalManifest.ManifestReceipt where
  owner := receipt.owner
  kind := receipt.kind
  allocations := receipt.allocations
  rows := receipt.rows

def cost (receipt : ManifestReceipt) : Cost :=
  receipt.asCanonical.cost

@[simp] theorem cost_ofSelectedReceipt
    (receipt : SelectedReceipt) :
    (ofSelectedReceipt receipt).cost = receipt.cost := by
  change
    (CanonicalManifest.ManifestReceipt.ofReceipt
      receipt.receipt).cost = receipt.receipt.cost
  exact
    CanonicalManifest.ManifestReceipt.cost_ofReceipt receipt.receipt

@[simp] theorem cost_recurringRows (receipt : ManifestReceipt) :
    receipt.cost.recurringRows = receipt.rows.length := by
  exact CanonicalManifest.ManifestReceipt.cost_recurringRows
    receipt.asCanonical

end ManifestReceipt

/-- Proof-erased normalized program recovered by a consumer. -/
structure ProgramImage where
  one : ColumnId
  matrixCount : Nat
  polynomialDegree : Nat
  polynomial : List PolynomialTerm
  receipts : List SelectedReceiptImage
deriving DecidableEq, Repr

namespace ProgramImage

def rows (program : ProgramImage) : List SelectedRow :=
  program.receipts.flatMap SelectedReceiptImage.rows

def columns (program : ProgramImage) : List OwnedColumn :=
  program.receipts.flatMap SelectedReceiptImage.allocations

def Satisfies
    (program : ProgramImage)
    (assignment : ColumnId → Field) : Prop :=
  assignment program.one = 1 ∧
    NativeCcsSelector.Satisfies program.rows assignment

end ProgramImage

/-- Proof-free native CCS program. The polynomial metadata is part of the
serialized authority and is checked by the consumer. -/
structure Program where
  one : ColumnId
  matrixCount : Nat
  polynomialDegree : Nat
  polynomial : List PolynomialTerm
  receipts : List ManifestReceipt
deriving DecidableEq, Repr

namespace Program

def ofProgram (program : NativeCcsProgram.Program) : Program where
  one := program.one
  matrixCount := NativeCcsSelector.matrixCount
  polynomialDegree := NativeCcsSelector.polynomialDegree
  polynomial := selectorPolynomial
  receipts := program.receipts.map ManifestReceipt.ofSelectedReceipt

def decode (program : Program) : ProgramImage where
  one := program.one
  matrixCount := program.matrixCount
  polynomialDegree := program.polynomialDegree
  polynomial := program.polynomial
  receipts := program.receipts.map ManifestReceipt.decode

def imageOf (program : NativeCcsProgram.Program) : ProgramImage where
  one := program.one
  matrixCount := NativeCcsSelector.matrixCount
  polynomialDegree := NativeCcsSelector.polynomialDegree
  polynomial := selectorPolynomial
  receipts := program.receipts.map ManifestReceipt.imageOf

/-- Static fail-closed shape check used by proof-free consumers. -/
def Valid (program : Program) : Prop :=
  program.matrixCount = NativeCcsSelector.matrixCount ∧
    program.polynomialDegree = NativeCcsSelector.polynomialDegree ∧
    program.polynomial = selectorPolynomial

@[simp] theorem valid_ofProgram
    (program : NativeCcsProgram.Program) :
    (ofProgram program).Valid :=
  ⟨rfl, rfl, rfl⟩

@[simp] theorem matrixCount_ofProgram
    (program : NativeCcsProgram.Program) :
    (ofProgram program).matrixCount = 4 :=
  rfl

@[simp] theorem polynomialDegree_ofProgram
    (program : NativeCcsProgram.Program) :
    (ofProgram program).polynomialDegree = 3 :=
  rfl

@[simp] theorem polynomial_ofProgram
    (program : NativeCcsProgram.Program) :
    (ofProgram program).polynomial = selectorPolynomial :=
  rfl

/-- Exact proof-erasure round trip. -/
theorem decode_ofProgram
    (program : NativeCcsProgram.Program) :
    (ofProgram program).decode = imageOf program := by
  simp [ofProgram, decode, imageOf, List.map_map, Function.comp_def]

private theorem flatMap_image_rows
    (receipts : List SelectedReceipt) :
    (receipts.map ManifestReceipt.imageOf).flatMap
        SelectedReceiptImage.rows =
      (receipts.flatMap SelectedReceipt.rows).map fun row =>
        { source := normalizeOwnedRow row.source
          selector := row.selector } := by
  induction receipts with
  | nil =>
      rfl
  | cons receipt rest inductionHypothesis =>
      simp [ManifestReceipt.imageOf, SelectedReceipt.rows,
        NativeCcsSelector.select, List.map_append, inductionHypothesis]

theorem decoded_rows_eq_normalized
    (program : NativeCcsProgram.Program) :
    (ofProgram program).decode.rows =
      program.rows.map fun row =>
        { source := normalizeOwnedRow row.source
          selector := row.selector } := by
  rw [decode_ofProgram]
  simpa [imageOf, ProgramImage.rows, NativeCcsProgram.Program.rows] using
    flatMap_image_rows program.receipts

private theorem selected_holds_normalized_iff
    (row : SelectedRow)
    (assignment : ColumnId → Field) :
    ({ source := normalizeOwnedRow row.source
       selector := row.selector } : SelectedRow).Holds assignment ↔
      row.Holds assignment := by
  unfold SelectedRow.Holds NativeCcsSelector.polynomial
  simp only [normalizeOwnedRow, normalizeRow]
  rw [eval_normalizeCombination, eval_normalizeCombination,
    eval_normalizeCombination]

private theorem satisfies_map_normalized
    (rows : List SelectedRow)
    (assignment : ColumnId → Field) :
    NativeCcsSelector.Satisfies
        (rows.map fun row =>
          { source := normalizeOwnedRow row.source
            selector := row.selector }) assignment ↔
      NativeCcsSelector.Satisfies rows assignment := by
  induction rows with
  | nil =>
      rfl
  | cons row rest inductionHypothesis =>
      simp only [List.map_cons, NativeCcsSelector.satisfies_cons]
      rw [selected_holds_normalized_iff, inductionHypothesis]

/-- Decoding and normalization preserve the exact native CCS relation. -/
theorem decoded_satisfies_iff
    (program : NativeCcsProgram.Program)
    (assignment : ColumnId → Field) :
    NativeCcsSelector.Satisfies
        (ofProgram program).decode.rows assignment ↔
      NativeCcsSelector.Satisfies program.rows assignment := by
  rw [decoded_rows_eq_normalized]
  exact satisfies_map_normalized program.rows assignment

/-- The complete decoded manifest and its source program accept the same
assignments, including the constant-one boundary. -/
theorem decoded_program_satisfies_iff
    (program : NativeCcsProgram.Program)
    (assignment : ColumnId → Field) :
    (ofProgram program).decode.Satisfies assignment ↔
      program.Satisfies assignment := by
  unfold ProgramImage.Satisfies NativeCcsProgram.Program.Satisfies
  change
    (assignment program.one = 1 ∧
        NativeCcsSelector.Satisfies
          (ofProgram program).decode.rows assignment) ↔
      assignment program.one = 1 ∧
        NativeCcsSelector.Satisfies program.rows assignment
  rw [decoded_satisfies_iff]

def rows (program : Program) : List CanonicalManifest.ManifestRow :=
  program.receipts.flatMap ManifestReceipt.rows

def columns (program : Program) : List OwnedColumn :=
  program.receipts.flatMap ManifestReceipt.allocations

def selectors (program : Program) : List ColumnId :=
  program.receipts.map ManifestReceipt.selector

def cost (program : Program) : Cost :=
  Cost.sum (program.receipts.map ManifestReceipt.cost)

theorem cost_ofProgram
    (program : NativeCcsProgram.Program) :
    (ofProgram program).cost = program.cost := by
  simp [cost, ofProgram, NativeCcsProgram.Program.cost,
    List.map_map, Function.comp_def]

theorem rows_length_ofProgram
    (program : NativeCcsProgram.Program) :
    (ofProgram program).rows.length = program.rows.length := by
  simp [rows, ofProgram, ManifestReceipt.ofSelectedReceipt,
    NativeCcsProgram.Program.rows, SelectedReceipt.rows,
    NativeCcsSelector.select, List.flatMap_map]

theorem columns_ofProgram
    (program : NativeCcsProgram.Program) :
    (ofProgram program).columns = program.allocations := by
  unfold columns ofProgram NativeCcsProgram.Program.allocations
    SelectedReceipt.allocations
  simp [ManifestReceipt.ofSelectedReceipt, List.flatMap_map]

theorem selectors_ofProgram
    (program : NativeCcsProgram.Program) :
    (ofProgram program).selectors =
      program.receipts.map SelectedReceipt.selector := by
  simp [selectors, ofProgram, ManifestReceipt.ofSelectedReceipt,
    Function.comp_def]

theorem cost_recurringRows (program : Program) :
    program.cost.recurringRows = program.rows.length := by
  unfold cost rows
  induction program.receipts with
  | nil =>
      rfl
  | cons receipt rest inductionHypothesis =>
      simp [Cost.sum, ManifestReceipt.cost_recurringRows,
        inductionHypothesis]

end Program

end Nightstream.Implementation.Lowering.Goldilocks.NativeCcsManifest
