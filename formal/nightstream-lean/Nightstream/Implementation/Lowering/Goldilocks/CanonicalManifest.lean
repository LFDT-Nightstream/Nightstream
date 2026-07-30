import Nightstream.Implementation.Lowering.Goldilocks.Compiler

/-!
Proof-free canonical serialization of a receipt-conserved Goldilocks program.

The manifest stores structural column and row identifiers, allocation classes,
receipt order, and field-canonical sparse coefficients as natural numbers.
Linear combinations are normalized before serialization: duplicate columns
are merged in the field and zero coefficients are removed.

Owns: normalization and its evaluation theorem; proof-erased receipt/program
images; exact encode/decode round trip; exact A/B/C nonzero counts and maximum
row support.

Does not own: a protocol profile, codec segment names, Rust, file I/O, source
hashes, or semantic refinement of a particular program.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest

open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo.Concrete

/-! ## Field-canonical normalization -/

private theorem fadd_assoc (left middle right : F) :
    (left + middle) + right = left + (middle + right) := by
  apply Fin.ext
  simp only [Fin.val_add]
  rw [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

private theorem fadd_comm (left right : F) :
    left + right = right + left := by
  apply Fin.ext
  simp only [Fin.val_add, Nat.add_comm]

private theorem fadd_mul (left middle right : F) :
    (left + middle) * right = left * right + middle * right := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul]
  rw [Nat.mod_mul_mod, Nat.add_mul, ← Nat.add_mod]

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨fadd_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨fadd_comm⟩

/-- Insert one term, merging it into the existing entry for the same
structural column. -/
def insertTerm (term : Term) : LinearCombination → LinearCombination
  | [] => [term]
  | entry :: rest =>
      if entry.column = term.column then
        { column := entry.column
          coefficient := entry.coefficient + term.coefficient } :: rest
      else
        entry :: insertTerm term rest

theorem eval_insertTerm
    (assignment : ColumnId → F)
    (term : Term)
    (combination : LinearCombination) :
    LinearCombination.eval assignment (insertTerm term combination) =
      term.coefficient * assignment term.column +
        LinearCombination.eval assignment combination := by
  induction combination with
  | nil =>
      simp [insertTerm, LinearCombination.eval]
  | cons entry rest inductionHypothesis =>
      unfold insertTerm
      by_cases same : entry.column = term.column
      · rw [if_pos same]
        simp only [LinearCombination.eval]
        rw [same]
        rw [fadd_mul]
        ac_rfl
      · rw [if_neg same]
        simp only [LinearCombination.eval, inductionHypothesis]
        ac_rfl

/-- Structural support of a sparse combination. -/
def Mentions (combination : LinearCombination) (column : ColumnId) : Prop :=
  column ∈ combination.map Term.column

theorem mentions_insertTerm
    (term : Term)
    (combination : LinearCombination)
    (column : ColumnId) :
    Mentions (insertTerm term combination) column ↔
      Mentions combination column ∨ column = term.column := by
  induction combination with
  | nil =>
      simp [insertTerm, Mentions]
  | cons entry rest inductionHypothesis =>
      unfold insertTerm
      by_cases same : entry.column = term.column
      · rw [if_pos same]
        simp only [Mentions, List.map_cons, List.mem_cons]
        constructor
        · intro mentioned
          exact Or.inl mentioned
        · rintro (mentioned | rfl)
          · exact mentioned
          · exact Or.inl same.symm
      · rw [if_neg same]
        simp only [Mentions, List.map_cons, List.mem_cons] at inductionHypothesis ⊢
        constructor
        · rintro (atHead | inTail)
          · exact Or.inl (Or.inl atHead)
          · rcases inductionHypothesis.1 inTail with old | inserted
            · exact Or.inl (Or.inr old)
            · exact Or.inr inserted
        · rintro ((atHead | old) | inserted)
          · exact Or.inl atHead
          · exact Or.inr (inductionHypothesis.2 (Or.inl old))
          · exact Or.inr (inductionHypothesis.2 (Or.inr inserted))

theorem insertTerm_nodup
    (term : Term)
    (combination : LinearCombination)
    (nodup : (combination.map Term.column).Nodup) :
    ((insertTerm term combination).map Term.column).Nodup := by
  induction combination with
  | nil =>
      simp [insertTerm]
  | cons entry rest inductionHypothesis =>
      rw [List.map_cons, List.nodup_cons] at nodup
      unfold insertTerm
      by_cases same : entry.column = term.column
      · rw [if_pos same]
        simp only [List.map_cons, List.nodup_cons]
        exact nodup
      · rw [if_neg same]
        simp only [List.map_cons, List.nodup_cons]
        refine ⟨?_, inductionHypothesis nodup.2⟩
        intro mentioned
        rcases (mentions_insertTerm term rest entry.column).1 mentioned with
          old | inserted
        · exact nodup.1 old
        · exact same inserted

/-- Aggregate duplicate structural columns in stable fold order. -/
def aggregate (combination : LinearCombination) : LinearCombination :=
  combination.foldr insertTerm []

theorem eval_aggregate
    (assignment : ColumnId → F)
    (combination : LinearCombination) :
    LinearCombination.eval assignment (aggregate combination) =
      LinearCombination.eval assignment combination := by
  unfold aggregate
  induction combination with
  | nil =>
      rfl
  | cons term rest inductionHypothesis =>
      simp only [List.foldr_cons]
      rw [eval_insertTerm, inductionHypothesis]
      rfl

theorem aggregate_nodup
    (combination : LinearCombination) :
    ((aggregate combination).map Term.column).Nodup := by
  unfold aggregate
  induction combination with
  | nil =>
      simp
  | cons term rest inductionHypothesis =>
      simp only [List.foldr_cons]
      exact insertTerm_nodup term _ inductionHypothesis

/-- Remove entries whose merged field coefficient is zero. -/
def dropZeros : LinearCombination → LinearCombination
  | [] => []
  | term :: rest =>
      if term.coefficient = 0 then
        dropZeros rest
      else
        term :: dropZeros rest

theorem eval_dropZeros
    (assignment : ColumnId → F)
    (combination : LinearCombination) :
    LinearCombination.eval assignment (dropZeros combination) =
      LinearCombination.eval assignment combination := by
  induction combination with
  | nil =>
      rfl
  | cons term rest inductionHypothesis =>
      unfold dropZeros
      by_cases zero : term.coefficient = 0
      · rw [if_pos zero, inductionHypothesis]
        simp only [LinearCombination.eval, zero, Fin.zero_mul,
          Fin.zero_add]
      · rw [if_neg zero]
        simp only [LinearCombination.eval, inductionHypothesis]

theorem dropZeros_nonzero
    (combination : LinearCombination) :
    ∀ term ∈ dropZeros combination, term.coefficient ≠ 0 := by
  induction combination with
  | nil =>
      simp [dropZeros]
  | cons term rest inductionHypothesis =>
      unfold dropZeros
      by_cases zero : term.coefficient = 0
      · rw [if_pos zero]
        exact inductionHypothesis
      · rw [if_neg zero]
        intro candidate member
        rcases List.mem_cons.1 member with rfl | inRest
        · exact zero
        · exact inductionHypothesis candidate inRest

theorem mentions_dropZeros_subset
    (combination : LinearCombination)
    (column : ColumnId)
    (mentioned : Mentions (dropZeros combination) column) :
    Mentions combination column := by
  induction combination with
  | nil =>
      simp [dropZeros, Mentions] at mentioned
  | cons term rest inductionHypothesis =>
      unfold dropZeros at mentioned
      simp only [Mentions, List.map_cons, List.mem_cons]
      by_cases zero : term.coefficient = 0
      · rw [if_pos zero] at mentioned
        exact Or.inr (inductionHypothesis mentioned)
      · rw [if_neg zero] at mentioned
        rcases
          (show
            column = term.column ∨
              column ∈ (dropZeros rest).map Term.column
            from by
              simpa only [Mentions, List.map_cons, List.mem_cons] using
                mentioned) with atHead | inRest
        · exact Or.inl atHead
        · exact Or.inr (inductionHypothesis inRest)

theorem dropZeros_nodup
    (combination : LinearCombination)
    (nodup : (combination.map Term.column).Nodup) :
    ((dropZeros combination).map Term.column).Nodup := by
  induction combination with
  | nil =>
      simp [dropZeros]
  | cons term rest inductionHypothesis =>
      rw [List.map_cons, List.nodup_cons] at nodup
      unfold dropZeros
      by_cases zero : term.coefficient = 0
      · rw [if_pos zero]
        exact inductionHypothesis nodup.2
      · rw [if_neg zero]
        simp only [List.map_cons, List.nodup_cons]
        refine ⟨?_, inductionHypothesis nodup.2⟩
        intro member
        exact nodup.1
          (mentions_dropZeros_subset rest term.column member)

/-- Stable, field-canonical sparse normal form. -/
def normalizeCombination
    (combination : LinearCombination) : LinearCombination :=
  dropZeros (aggregate combination)

theorem eval_normalizeCombination
    (assignment : ColumnId → F)
    (combination : LinearCombination) :
    LinearCombination.eval assignment
        (normalizeCombination combination) =
      LinearCombination.eval assignment combination := by
  rw [normalizeCombination, eval_dropZeros, eval_aggregate]

theorem normalizeCombination_nonzero
    (combination : LinearCombination) :
    ∀ term ∈ normalizeCombination combination,
      term.coefficient ≠ 0 :=
  dropZeros_nonzero (aggregate combination)

theorem normalizeCombination_nodup
    (combination : LinearCombination) :
    ((normalizeCombination combination).map Term.column).Nodup := by
  exact dropZeros_nodup _ (aggregate_nodup combination)

/-- Normalize all three sparse sides without changing the row identity. -/
def normalizeRow (row : Row) : Row where
  a := normalizeCombination row.a
  b := normalizeCombination row.b
  c := normalizeCombination row.c

def normalizeOwnedRow (row : OwnedRow) : OwnedRow where
  id := row.id
  row := normalizeRow row.row

theorem normalizeRow_holds_iff
    (assignment : ColumnId → F)
    (row : Row) :
    (normalizeRow row).Holds assignment ↔ row.Holds assignment := by
  unfold Row.Holds normalizeRow
  rw [eval_normalizeCombination, eval_normalizeCombination,
    eval_normalizeCombination]

theorem satisfies_map_normalizeOwnedRow
    (rows : List OwnedRow)
    (assignment : ColumnId → F) :
    Satisfies (rows.map normalizeOwnedRow) assignment ↔
      Satisfies rows assignment := by
  induction rows with
  | nil =>
      rfl
  | cons row rest inductionHypothesis =>
      simp only [List.map_cons, satisfies_cons]
      change
        (normalizeRow row.row).Holds assignment ∧
            Satisfies (rest.map normalizeOwnedRow) assignment ↔
          row.row.Holds assignment ∧ Satisfies rest assignment
      rw [normalizeRow_holds_iff]
      exact and_congr_right (fun _ => inductionHypothesis)

/-! ## Proof-free row and receipt records -/

/-- Serialized sparse term. `coefficient` is the canonical Goldilocks
residue, represented without a dependent proof. -/
structure ManifestTerm where
  column : ColumnId
  coefficient : Nat
deriving DecidableEq, Repr

namespace ManifestTerm

def ofTerm (term : Term) : ManifestTerm where
  column := term.column
  coefficient := term.coefficient.val

def decode (term : ManifestTerm) : Term where
  column := term.column
  coefficient :=
    ⟨term.coefficient % goldilocksModulus,
      Nat.mod_lt _ (by decide)⟩

@[simp] theorem decode_ofTerm (term : Term) :
    (ofTerm term).decode = term := by
  cases term with
  | mk column coefficient =>
      simp [ofTerm, decode, Nat.mod_eq_of_lt coefficient.isLt]

end ManifestTerm

abbrev ManifestCombination := List ManifestTerm

def encodeCombination
    (combination : LinearCombination) : ManifestCombination :=
  (normalizeCombination combination).map ManifestTerm.ofTerm

def decodeCombination
    (combination : ManifestCombination) : LinearCombination :=
  combination.map ManifestTerm.decode

@[simp] theorem decode_encodeCombination
    (combination : LinearCombination) :
    decodeCombination (encodeCombination combination) =
      normalizeCombination combination := by
  simp [decodeCombination, encodeCombination, List.map_map,
    Function.comp_def]

theorem encodeCombination_nonzero
    (combination : LinearCombination) :
    ∀ term ∈ encodeCombination combination, term.coefficient ≠ 0 := by
  intro encoded member
  rcases List.mem_map.1 member with ⟨term, termMember, rfl⟩
  exact
    fun zero =>
      normalizeCombination_nonzero combination term termMember
        (Fin.ext zero)

theorem encodeCombination_canonical
    (combination : LinearCombination) :
    ∀ term ∈ encodeCombination combination,
      term.coefficient < goldilocksModulus := by
  intro encoded member
  rcases List.mem_map.1 member with ⟨term, _, rfl⟩
  exact term.coefficient.isLt

theorem encodeCombination_columns_nodup
    (combination : LinearCombination) :
    ((encodeCombination combination).map ManifestTerm.column).Nodup := by
  simpa [encodeCombination, List.map_map, Function.comp_def,
    ManifestTerm.ofTerm] using normalizeCombination_nodup combination

structure ManifestRow where
  id : RowId
  a : ManifestCombination
  b : ManifestCombination
  c : ManifestCombination
deriving DecidableEq, Repr

namespace ManifestRow

def ofOwnedRow (row : OwnedRow) : ManifestRow where
  id := row.id
  a := encodeCombination row.row.a
  b := encodeCombination row.row.b
  c := encodeCombination row.row.c

def decode (row : ManifestRow) : OwnedRow where
  id := row.id
  row := {
    a := decodeCombination row.a
    b := decodeCombination row.b
    c := decodeCombination row.c
  }

@[simp] theorem decode_ofOwnedRow (row : OwnedRow) :
    (ofOwnedRow row).decode = normalizeOwnedRow row := by
  cases row
  simp [ofOwnedRow, decode, normalizeOwnedRow, normalizeRow]

def aNonzeros (row : ManifestRow) : Nat := row.a.length
def bNonzeros (row : ManifestRow) : Nat := row.b.length
def cNonzeros (row : ManifestRow) : Nat := row.c.length

def support (row : ManifestRow) : Nat :=
  row.aNonzeros + row.bNonzeros + row.cNonzeros

end ManifestRow

/-- Proof-erased receipt image after decoding a manifest. -/
structure ReceiptImage where
  owner : PhysicalOwner
  kind : InstructionKind
  allocations : List OwnedColumn
  rows : List OwnedRow
deriving DecidableEq, Repr

/-- Proof-free receipt record in exact emission order. -/
structure ManifestReceipt where
  owner : PhysicalOwner
  kind : InstructionKind
  allocations : List OwnedColumn
  rows : List ManifestRow
deriving DecidableEq, Repr

namespace ManifestReceipt

def ofReceipt (receipt : InstructionReceipt) : ManifestReceipt where
  owner := receipt.owner
  kind := receipt.kind
  allocations := receipt.allocations
  rows := receipt.rows.map ManifestRow.ofOwnedRow

def decode (receipt : ManifestReceipt) : ReceiptImage where
  owner := receipt.owner
  kind := receipt.kind
  allocations := receipt.allocations
  rows := receipt.rows.map ManifestRow.decode

def imageOf (receipt : InstructionReceipt) : ReceiptImage where
  owner := receipt.owner
  kind := receipt.kind
  allocations := receipt.allocations
  rows := receipt.rows.map normalizeOwnedRow

@[simp] theorem decode_ofReceipt (receipt : InstructionReceipt) :
    (ofReceipt receipt).decode = imageOf receipt := by
  simp [ofReceipt, decode, imageOf, List.map_map, Function.comp_def]

/-- Exact four-way physical cost of one proof-free receipt. -/
def cost (receipt : ManifestReceipt) : Cost :=
  columnCost receipt.allocations +
    Cost.sum (receipt.rows.map fun _ => Cost.oneRow)

private theorem columnCost_recurringRows
    (columns : List OwnedColumn) :
    (columnCost columns).recurringRows = 0 := by
  induction columns with
  | nil =>
      rfl
  | cons column rest inductionHypothesis =>
      unfold columnCost at inductionHypothesis ⊢
      simp only [List.map_cons, Cost.sum, Cost.add_recurringRows]
      rw [inductionHypothesis]
      cases column.ownership <;> rfl

private theorem rowCost_recurringRows
    (rows : List ManifestRow) :
    (Cost.sum (rows.map fun _ => Cost.oneRow)).recurringRows =
      rows.length := by
  induction rows with
  | nil =>
      rfl
  | cons row rest inductionHypothesis =>
      simp only [List.map_cons, Cost.sum, Cost.add_recurringRows,
        List.length_cons]
      rw [inductionHypothesis]
      simp [Cost.oneRow, Nat.add_comm]

@[simp] theorem cost_ofReceipt (receipt : InstructionReceipt) :
    cost (ofReceipt receipt) = receipt.cost := by
  simp [cost, ofReceipt, InstructionReceipt.cost, physicalCost, rowCost,
    List.map_map, Function.comp_def]

@[simp] theorem cost_recurringRows (receipt : ManifestReceipt) :
    receipt.cost.recurringRows = receipt.rows.length := by
  simp [cost, columnCost_recurringRows, rowCost_recurringRows]

end ManifestReceipt

/-- Proof-erased normalized program data recovered by a consumer. -/
structure ProgramImage where
  one : ColumnId
  receipts : List ReceiptImage
deriving DecidableEq, Repr

namespace ProgramImage

def columns (program : ProgramImage) : List OwnedColumn :=
  program.receipts.flatMap ReceiptImage.allocations

def rows (program : ProgramImage) : List OwnedRow :=
  program.receipts.flatMap ReceiptImage.rows

end ProgramImage

/-- Minimal proof-free canonical program manifest. Protocol metadata is
layered on top without changing this round-trip boundary. -/
structure Program where
  one : ColumnId
  receipts : List ManifestReceipt
deriving DecidableEq, Repr

namespace Program

private theorem flatMap_image_rows
    (receipts : List InstructionReceipt) :
    (receipts.map ManifestReceipt.imageOf).flatMap ReceiptImage.rows =
      (receipts.flatMap (fun receipt => receipt.rows)).map
        normalizeOwnedRow := by
  induction receipts with
  | nil =>
      rfl
  | cons receipt rest inductionHypothesis =>
      simp [ManifestReceipt.imageOf, List.map_append,
        inductionHypothesis]

def ofEncoding
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source) : Program where
  one := encoding.one
  receipts := encoding.receipts.map ManifestReceipt.ofReceipt

def decode (program : Program) : ProgramImage where
  one := program.one
  receipts := program.receipts.map ManifestReceipt.decode

def imageOfEncoding
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source) : ProgramImage where
  one := encoding.one
  receipts := encoding.receipts.map ManifestReceipt.imageOf

/-- Exact proof-erasure round trip. No digest or source hash participates. -/
theorem decode_ofEncoding
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source) :
    (ofEncoding encoding).decode = imageOfEncoding encoding := by
  simp [ofEncoding, decode, imageOfEncoding, List.map_map,
    Function.comp_def]

theorem decoded_columns_eq
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source) :
    (ofEncoding encoding).decode.columns = encoding.columns := by
  simp [ofEncoding, decode, ProgramImage.columns,
    ManifestReceipt.decode, ManifestReceipt.ofReceipt,
    Encoding.columns, List.flatMap_map]

theorem decoded_rows_eq_normalized
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source) :
    (ofEncoding encoding).decode.rows =
      encoding.rows.map normalizeOwnedRow := by
  rw [decode_ofEncoding]
  simpa [imageOfEncoding, ProgramImage.rows, Encoding.rows] using
    flatMap_image_rows encoding.receipts

/-- The proof-free manifest and the canonical program accept exactly the same
assignments. -/
theorem decoded_satisfies_iff
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source)
    (assignment : ColumnId → F) :
    Satisfies (ofEncoding encoding).decode.rows assignment ↔
      Satisfies encoding.rows assignment := by
  rw [decoded_rows_eq_normalized]
  exact satisfies_map_normalizeOwnedRow encoding.rows assignment

end Program

/-! ## Exact coefficient statistics -/

structure Statistics where
  aNonzeros : Nat
  bNonzeros : Nat
  cNonzeros : Nat
  maxRowSupport : Nat
deriving DecidableEq, Repr

namespace Statistics

def zero : Statistics := ⟨0, 0, 0, 0⟩

def addRow (row : ManifestRow) (rest : Statistics) : Statistics where
  aNonzeros := row.aNonzeros + rest.aNonzeros
  bNonzeros := row.bNonzeros + rest.bNonzeros
  cNonzeros := row.cNonzeros + rest.cNonzeros
  maxRowSupport := max row.support rest.maxRowSupport

def ofRows : List ManifestRow → Statistics
  | [] => zero
  | row :: rest => addRow row (ofRows rest)

def totalNonzeros (statistics : Statistics) : Nat :=
  statistics.aNonzeros + statistics.bNonzeros + statistics.cNonzeros

theorem aNonzeros_exact (rows : List ManifestRow) :
    (ofRows rows).aNonzeros =
      (rows.map ManifestRow.aNonzeros).sum := by
  induction rows with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      simp [ofRows, addRow, inductionHypothesis]

theorem bNonzeros_exact (rows : List ManifestRow) :
    (ofRows rows).bNonzeros =
      (rows.map ManifestRow.bNonzeros).sum := by
  induction rows with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      simp [ofRows, addRow, inductionHypothesis]

theorem cNonzeros_exact (rows : List ManifestRow) :
    (ofRows rows).cNonzeros =
      (rows.map ManifestRow.cNonzeros).sum := by
  induction rows with
  | nil => rfl
  | cons row rest inductionHypothesis =>
      simp [ofRows, addRow, inductionHypothesis]

theorem support_le_max
    (rows : List ManifestRow)
    (row : ManifestRow)
    (member : row ∈ rows) :
    row.support ≤ (ofRows rows).maxRowSupport := by
  induction rows with
  | nil =>
      simp at member
  | cons head rest inductionHypothesis =>
      rcases List.mem_cons.1 member with rfl | inRest
      · exact Nat.le_max_left _ _
      · exact Nat.le_trans
          (inductionHypothesis inRest)
          (Nat.le_max_right _ _)

end Statistics

def Program.rows (program : Program) : List ManifestRow :=
  program.receipts.flatMap ManifestReceipt.rows

def Program.columns (program : Program) : List OwnedColumn :=
  program.receipts.flatMap ManifestReceipt.allocations

def Program.statistics (program : Program) : Statistics :=
  Statistics.ofRows program.rows

/-- Exact four-way cost recovered from proof-free receipt data. -/
def Program.cost (program : Program) : Cost :=
  Cost.sum (program.receipts.map ManifestReceipt.cost)

/-- The proof-free manifest cost is definitionally attached to the same
receipt fold as the proof-carrying encoding. -/
theorem Program.cost_ofEncoding
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source) :
    (Program.ofEncoding encoding).cost = encoding.cost := by
  rw [Encoding.cost_eq_receipt_cost]
  simp [Program.cost, Program.ofEncoding, Encoding.receiptCost,
    List.map_map, Function.comp_def]

theorem Program.cost_recurringRows (program : Program) :
    program.cost.recurringRows = program.rows.length := by
  unfold Program.cost Program.rows
  induction program.receipts with
  | nil =>
      rfl
  | cons receipt rest inductionHypothesis =>
      simp [Cost.sum, inductionHypothesis]

/-- Serialization preserves the exact allocation stream, including ownership
classes and receipt order. -/
theorem Program.columns_ofEncoding
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source) :
    (Program.ofEncoding encoding).columns = encoding.columns := by
  simp [Program.columns, Program.ofEncoding, ManifestReceipt.ofReceipt,
    Encoding.columns, List.flatMap_map]

/-- Normalization changes sparse coefficients, never the emitted row count. -/
theorem Program.rows_length_ofEncoding
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source) :
    (Program.ofEncoding encoding).rows.length = encoding.rows.length := by
  simp [Program.rows, Program.ofEncoding, ManifestReceipt.ofReceipt,
    Encoding.rows, List.flatMap_map]

theorem Program.all_coefficients_nonzero
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source) :
    ∀ row ∈ (Program.ofEncoding encoding).rows,
      (∀ term ∈ row.a, term.coefficient ≠ 0) ∧
      (∀ term ∈ row.b, term.coefficient ≠ 0) ∧
      (∀ term ∈ row.c, term.coefficient ≠ 0) := by
  intro row rowMember
  rcases List.mem_flatMap.1 rowMember with
    ⟨receipt, receiptMember, rowMember⟩
  rcases List.mem_map.1 receiptMember with
    ⟨sourceReceipt, _, rfl⟩
  rcases List.mem_map.1 rowMember with
    ⟨sourceRow, _, rfl⟩
  exact ⟨
    encodeCombination_nonzero sourceRow.row.a,
    encodeCombination_nonzero sourceRow.row.b,
    encodeCombination_nonzero sourceRow.row.c
  ⟩

theorem Program.all_coefficients_canonical
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source) :
    ∀ row ∈ (Program.ofEncoding encoding).rows,
      (∀ term ∈ row.a, term.coefficient < goldilocksModulus) ∧
      (∀ term ∈ row.b, term.coefficient < goldilocksModulus) ∧
      (∀ term ∈ row.c, term.coefficient < goldilocksModulus) := by
  intro row rowMember
  rcases List.mem_flatMap.1 rowMember with
    ⟨receipt, receiptMember, rowMember⟩
  rcases List.mem_map.1 receiptMember with
    ⟨sourceReceipt, _, rfl⟩
  rcases List.mem_map.1 rowMember with
    ⟨sourceRow, _, rfl⟩
  exact ⟨
    encodeCombination_canonical sourceRow.row.a,
    encodeCombination_canonical sourceRow.row.b,
    encodeCombination_canonical sourceRow.row.c
  ⟩

theorem Program.all_combination_columns_nodup
    {signature : Signature}
    {input output : Schema signature.types}
    {source : Typed.Program signature input output}
    (encoding : Encoding source) :
    ∀ row ∈ (Program.ofEncoding encoding).rows,
      (row.a.map ManifestTerm.column).Nodup ∧
      (row.b.map ManifestTerm.column).Nodup ∧
      (row.c.map ManifestTerm.column).Nodup := by
  intro row rowMember
  rcases List.mem_flatMap.1 rowMember with
    ⟨receipt, receiptMember, rowMember⟩
  rcases List.mem_map.1 receiptMember with
    ⟨sourceReceipt, _, rfl⟩
  rcases List.mem_map.1 rowMember with
    ⟨sourceRow, _, rfl⟩
  exact ⟨
    encodeCombination_columns_nodup sourceRow.row.a,
    encodeCombination_columns_nodup sourceRow.row.b,
    encodeCombination_columns_nodup sourceRow.row.c
  ⟩

end Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
