import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.LeanCompiler.StableRows

/-!
Contract: instantiate the Lean-owned selective compiler from one complete
receipt-conserved `Goldilocks.Encoding`.

Assurance tier: model-level.

Owns: the canonical finite index of allocated structural columns, proof that
every emitted row reference is allocated by some receipt, exact reconstruction
of stable assignments from that index, and soundness plus honest completeness
of the compiled selective relation against `Encoding.PhysicalSatisfies`.

Does not own: a concrete F-prime application, selection of Step or Terminal,
the minimal Boolean row-domain witness, low-norm encodings, a fixed-point
shape theorem, Rust, generated artifacts, or protocol security events.

Emits constraints: no new R1CS rows. The compiled relation has exactly one
finite selective row for each row in `encoding.rows`.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.EncodingRows

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.R1CS.SelectiveCcs
open Nightstream.Implementation.R1CS.SelectiveCcs.LeanCompiler

universe u

/-- A proof-carrying position of one structural column in an ordered
allocation list. -/
structure Location (columns : List ColumnId) (column : ColumnId) where
  index : Fin columns.length
  atIndex : columns.get index = column

/-- Locate a member by the list's first exact structural occurrence. -/
def locate
    (columns : List ColumnId)
    (column : ColumnId)
    (member : column ∈ columns) :
    Location columns column := by
  match found : columns.idxOf? column with
  | none =>
      exact False.elim ((List.idxOf?_eq_none_iff.mp found) member)
  | some index =>
      have witness := List.idxOf?_eq_some_iff.mp found
      have indexLt : index < columns.length :=
        Exists.elim witness fun bound _ => bound
      have atIndex : columns[index] = column :=
        Exists.elim witness fun _ exactAndFirst => exactAndFirst.1
      exact ⟨⟨index, indexLt⟩, atIndex⟩

/-- Total column index for one encoding. Unallocated values map to the
already-allocated constant-one position. This fallback is never used by an
emitted row because `encoding_rows_supported` proves exact coverage. -/
def columnIndex
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (column : ColumnId) :
    Fin encoding.columnIds.length :=
  if member : column ∈ encoding.columnIds then
    (locate encoding.columnIds column member).index
  else
    (locate encoding.columnIds encoding.one encoding.oneAllocated).index

/-- Decode one finite matrix column back to its structural allocation. -/
def columnAt
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (index : Fin encoding.columnIds.length) :
    ColumnId :=
  encoding.columnIds.get index

theorem columnAt_columnIndex
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (column : ColumnId)
    (member : column ∈ encoding.columnIds) :
    columnAt encoding (columnIndex encoding column) = column := by
  unfold columnIndex
  rw [dif_pos member]
  exact (locate encoding.columnIds column member).atIndex

private theorem referenced_in_receipts
    (available : List ColumnId) :
    ∀ (receipts : List InstructionReceipt),
      ReceiptsWellScoped available receipts →
      ∀ receipt,
        receipt ∈ receipts →
        ∀ column,
          column ∈ receipt.referencedColumns →
          column ∈ available ∨
            column ∈ receipts.flatMap InstructionReceipt.columnIds
  | [], _, receipt, member, _, _ => by
      simp at member
  | head :: tail, wellScoped, receipt, member, column, referenced => by
      change
        head.WellScopedAfter available ∧
          ReceiptsWellScoped (available ++ head.columnIds) tail
        at wellScoped
      rcases List.mem_cons.mp member with rfl | inTail
      · rcases wellScoped.1 column referenced with old | current
        · exact Or.inl old
        · exact Or.inr
            (by
              simp only [List.flatMap_cons, List.mem_append]
              exact Or.inl current)
      · have inside :=
          referenced_in_receipts
            (available ++ head.columnIds) tail wellScoped.2
            receipt inTail column referenced
        rcases inside with inAvailable | inLater
        · rcases List.mem_append.mp inAvailable with old | inHead
          · exact Or.inl old
          · exact Or.inr
              (by
                simp only [List.flatMap_cons, List.mem_append]
                exact Or.inl inHead)
        · exact Or.inr
            (by
              simp only [List.flatMap_cons, List.mem_append]
              exact Or.inr inLater)

/-- Every structural column mentioned by an emitted row is present in the
encoding's receipt-derived allocation list. -/
theorem encoding_row_column_allocated
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (owned : OwnedRow)
    (rowMember : owned ∈ encoding.rows)
    (column : ColumnId)
    (mentioned : column ∈ owned.row.columnIds) :
    column ∈ encoding.columnIds := by
  rcases List.mem_flatMap.mp rowMember with
    ⟨receipt, receiptMember, ownedMember⟩
  have referenced : column ∈ receipt.referencedColumns := by
    unfold InstructionReceipt.referencedColumns
    apply List.mem_flatMap.mpr
    refine ⟨owned, ownedMember, ?_⟩
    simpa [InstructionReceipt.rowColumns, Row.columnIds] using mentioned
  have all :=
    referenced_in_receipts [] encoding.receipts encoding.wellScoped
      receipt receiptMember column referenced
  rcases all with impossible | allocated
  · simp at impossible
  · rw [encoding.columnIds_eq_receipt_columnIds]
    exact allocated

theorem encoding_rows_supported
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    ∀ owned ∈ encoding.rows,
      ∀ column ∈ owned.row.columnIds,
        column ∈ encoding.columnIds :=
  encoding_row_column_allocated encoding

/-- Rebuild an indexed assignment from one stable structural assignment. -/
def indexedAssignment
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (assignment : ColumnId → F) :
    Fin encoding.columnIds.length → F :=
  fun index => assignment (columnAt encoding index)

theorem pulled_indexed_at_allocated
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (assignment : ColumnId → F)
    (column : ColumnId)
    (member : column ∈ encoding.columnIds) :
    StableRows.pulledAssignment (columnIndex encoding)
        (indexedAssignment encoding assignment) column =
      assignment column := by
  unfold StableRows.pulledAssignment indexedAssignment
  rw [columnAt_columnIndex encoding column member]

private theorem combination_eval_congr
    (combination : LinearCombination)
    (left right : ColumnId → F)
    (agree :
      ∀ term ∈ combination,
        left term.column = right term.column) :
    combination.eval left = combination.eval right := by
  induction combination with
  | nil =>
      rfl
  | cons term tail inductionHypothesis =>
      simp only [LinearCombination.eval]
      rw [agree term (by simp)]
      rw [inductionHypothesis (fun candidate member =>
        agree candidate (by simp [member]))]

private theorem term_column_in_row_of_a
    (row : Row) (term : Term) (member : term ∈ row.a) :
    term.column ∈ row.columnIds := by
  unfold Row.columnIds
  apply List.mem_map.mpr
  exact ⟨term, by simp [member], rfl⟩

private theorem term_column_in_row_of_b
    (row : Row) (term : Term) (member : term ∈ row.b) :
    term.column ∈ row.columnIds := by
  unfold Row.columnIds
  apply List.mem_map.mpr
  exact ⟨term, by simp [member], rfl⟩

private theorem term_column_in_row_of_c
    (row : Row) (term : Term) (member : term ∈ row.c) :
    term.column ∈ row.columnIds := by
  unfold Row.columnIds
  apply List.mem_map.mpr
  exact ⟨term, by simp [member], rfl⟩

private theorem row_holds_pulled_indexed_iff
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (assignment : ColumnId → F)
    (owned : OwnedRow)
    (member : owned ∈ encoding.rows) :
    owned.row.Holds
        (StableRows.pulledAssignment (columnIndex encoding)
          (indexedAssignment encoding assignment)) ↔
      owned.row.Holds assignment := by
  unfold Row.Holds
  have aEqual := combination_eval_congr owned.row.a
    (StableRows.pulledAssignment (columnIndex encoding)
      (indexedAssignment encoding assignment))
    assignment
    (fun term termMember =>
      pulled_indexed_at_allocated encoding assignment term.column
        (encoding_row_column_allocated encoding owned member term.column
          (term_column_in_row_of_a owned.row term termMember)))
  have bEqual := combination_eval_congr owned.row.b
    (StableRows.pulledAssignment (columnIndex encoding)
      (indexedAssignment encoding assignment))
    assignment
    (fun term termMember =>
      pulled_indexed_at_allocated encoding assignment term.column
        (encoding_row_column_allocated encoding owned member term.column
          (term_column_in_row_of_b owned.row term termMember)))
  have cEqual := combination_eval_congr owned.row.c
    (StableRows.pulledAssignment (columnIndex encoding)
      (indexedAssignment encoding assignment))
    assignment
    (fun term termMember =>
      pulled_indexed_at_allocated encoding assignment term.column
        (encoding_row_column_allocated encoding owned member term.column
          (term_column_in_row_of_c owned.row term termMember)))
  rw [aEqual, bEqual, cEqual]

private theorem satisfies_iff_forall_member
    (rows : List OwnedRow)
    (assignment : ColumnId → F) :
    Satisfies rows assignment ↔
      ∀ row ∈ rows, row.row.Holds assignment := by
  induction rows with
  | nil =>
      simp
  | cons head tail inductionHypothesis =>
      rw [satisfies_cons, inductionHypothesis]
      constructor
      · rintro ⟨headHolds, tailHolds⟩ row member
        rcases List.mem_cons.mp member with rfl | inTail
        · exact headHolds
        · exact tailHolds row inTail
      · intro all
        exact ⟨
          all head (by simp),
          fun row member => all row (by simp [member])
        ⟩

theorem satisfies_pulled_indexed_iff
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (assignment : ColumnId → F) :
    Satisfies encoding.rows
        (StableRows.pulledAssignment (columnIndex encoding)
          (indexedAssignment encoding assignment)) ↔
      Satisfies encoding.rows assignment := by
  rw [satisfies_iff_forall_member, satisfies_iff_forall_member]
  constructor
  · intro all row member
    exact (row_holds_pulled_indexed_iff encoding assignment row member).mp
      (all row member)
  · intro all row member
    exact (row_holds_pulled_indexed_iff encoding assignment row member).mpr
      (all row member)

/-- Source rows compiled for this exact receipt-conserved encoding. -/
def program
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    List (DirectRows.SourceRow encoding.columnIds.length) :=
  StableRows.program (columnIndex encoding) encoding.rows

@[simp] theorem program_length
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    (program encoding).length = encoding.rows.length := by
  simp [program, StableRows.program]

/-- The complete indexed acceptance condition. The constant-one verifier
boundary is part of the condition, not an external premise. -/
def IndexedAccepts
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (profile :
      RelationProfile.Profile (program encoding).length
        encoding.columnIds.length)
    (assignment : Fin encoding.columnIds.length → F) :
    Prop :=
  assignment (columnIndex encoding encoding.one) = 1 ∧
    ConstraintSatisfied baseOps
      (DirectRows.paperSystem
        (DirectRows.relation
          (columnIndex encoding encoding.one)
          (program encoding))
        profile)
      assignment

/-- Current-program CIR-SOUND and indexed completeness: the compiled
selective relation accepts exactly the receipt-derived physical program on
the pulled stable assignment. -/
theorem indexedAccepts_iff_physicalSatisfies
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (profile :
      RelationProfile.Profile (program encoding).length
        encoding.columnIds.length)
    (assignment : Fin encoding.columnIds.length → F) :
    IndexedAccepts encoding profile assignment ↔
      encoding.PhysicalSatisfies
        (StableRows.pulledAssignment (columnIndex encoding) assignment) := by
  constructor
  · rintro ⟨constantOne, satisfied⟩
    refine ⟨constantOne, ?_⟩
    exact
      (StableRows.constraintSatisfied_iff
        (columnIndex encoding) encoding.one encoding.rows profile assignment
        constantOne).mp satisfied
  · rintro ⟨constantOne, satisfied⟩
    refine ⟨constantOne, ?_⟩
    exact
      (StableRows.constraintSatisfied_iff
        (columnIndex encoding) encoding.one encoding.rows profile assignment
        constantOne).mpr satisfied

/-- Current-program CIR-COMPLETE: every satisfying stable physical assignment
reassembles into the canonical indexed assignment and satisfies the same
compiled selective relation. The reverse direction is also exact. -/
theorem indexedAssignment_accepts_iff
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (profile :
      RelationProfile.Profile (program encoding).length
        encoding.columnIds.length)
    (assignment : ColumnId → F) :
    IndexedAccepts encoding profile
        (indexedAssignment encoding assignment) ↔
      encoding.PhysicalSatisfies assignment := by
  rw [indexedAccepts_iff_physicalSatisfies]
  unfold Encoding.PhysicalSatisfies
  have oneEqual :=
    pulled_indexed_at_allocated encoding assignment encoding.one
      encoding.oneAllocated
  rw [oneEqual, satisfies_pulled_indexed_iff]

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.EncodingRows
