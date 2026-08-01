import Nightstream.Implementation.Lowering.Goldilocks.CallRecipe
import Nightstream.Implementation.Lowering.Goldilocks.ColumnPlan

/-!
Contract: receipt-conserving physical programs for the selected Goldilocks
lowering vocabulary.

Owns:
- the only physical instruction receipt consumed by the row compiler;
- exact flattening of receipt allocations and rows;
- intrinsic physical occurrences and their unique instruction owner;
- collision-free physical identities, row scoping, and definitional cost.

Does not own: a Rust emitter, generated artifacts, protocol-call semantics,
normal-form selection, or whole-program semantic refinement.

Every physical row and allocated column is obtained by flattening the
nonoptional instruction receipts.  There is no extra-row or glue-row field.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

/-- Selected physical instruction classes.  Prelude and branch entries are
real lowering instructions rather than out-of-band compiler glue. -/
inductive InstructionKind where
  | prelude
  | input
  | literal
  | affine
  | product
  | bit
  | call
  | assertion
  | branchControl
  | branchJoin
deriving DecidableEq, Repr

/-- One complete nonoptional emission receipt.

The ownership equalities forbid a receipt from claiming occurrences emitted
under another structural owner.  Dependencies may refer to earlier receipts;
that ordering condition is checked by `ReceiptsWellScoped`. -/
structure InstructionReceipt where
  owner : PhysicalOwner
  kind : InstructionKind
  allocations : List OwnedColumn
  rows : List OwnedRow
  allocationsOwned :
    ∀ column, column ∈ allocations -> column.id.owner = owner
  rowsOwned :
    ∀ row, row ∈ rows -> row.id.owner = owner

namespace InstructionReceipt

def columnIds (receipt : InstructionReceipt) : List ColumnId :=
  receipt.allocations.map (fun column => column.id)

def rowIds (receipt : InstructionReceipt) : List RowId :=
  receipt.rows.map (fun row => row.id)

def cost (receipt : InstructionReceipt) : Cost :=
  physicalCost receipt.allocations receipt.rows

def rowColumns (row : OwnedRow) : List ColumnId :=
  (row.row.a ++ row.row.b ++ row.row.c).map (fun term => term.column)

def referencedColumns (receipt : InstructionReceipt) : List ColumnId :=
  receipt.rows.flatMap rowColumns

/-- Every row dependency is either already allocated by an earlier receipt or
is allocated by this receipt. -/
def WellScopedAfter
    (available : List ColumnId)
    (receipt : InstructionReceipt) : Prop :=
  ∀ column, column ∈ receipt.referencedColumns ->
    column ∈ available ∨ column ∈ receipt.columnIds

theorem allocation_has_receipt_owner
    (receipt : InstructionReceipt)
    (column : OwnedColumn)
    (member : column ∈ receipt.allocations) :
    column.id.owner = receipt.owner :=
  receipt.allocationsOwned column member

theorem row_has_receipt_owner
    (receipt : InstructionReceipt)
    (row : OwnedRow)
    (member : row ∈ receipt.rows) :
    row.id.owner = receipt.owner :=
  receipt.rowsOwned row member

end InstructionReceipt

/-- Receipt order is execution/allocation order. -/
def ReceiptsWellScoped :
    List ColumnId -> List InstructionReceipt -> Prop
  | _, [] => True
  | available, receipt :: tail =>
      receipt.WellScopedAfter available ∧
        ReceiptsWellScoped
          (available ++ receipt.columnIds) tail

/-- The selected physical encoding of one typed program.

The semantic program is retained as an index.  Obligation 11 proves that its
rows refine that program; this module fixes the smaller obligation-10 surface:
all physical data are receipt-derived, collision-free, and exactly costed. -/
structure Encoding
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (source : Program signature input output) where
  receipts : List InstructionReceipt
  one : ColumnId
  /-- The physical verifier boundary fixes the selected canonical public
  constant-one coordinate; an encoding cannot substitute another allocated
  wire and call it one. -/
  oneExact : one = oneColumn
  /-- The canonical one coordinate is not merely present under the right ID:
  it is the verifier-owned public prelude allocation. -/
  oneAllocationExact :
    ({ id := oneColumn, ownership := .publicColumn } : OwnedColumn) ∈
      receipts.flatMap (fun receipt => receipt.allocations)
  columnIdsNodup :
    (receipts.flatMap InstructionReceipt.columnIds).Nodup
  rowIdsNodup :
    (receipts.flatMap InstructionReceipt.rowIds).Nodup
  wellScoped : ReceiptsWellScoped [] receipts

namespace Encoding

def columns
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) : List OwnedColumn :=
  encoding.receipts.flatMap (fun receipt => receipt.allocations)

def rows
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) : List OwnedRow :=
  encoding.receipts.flatMap (fun receipt => receipt.rows)

def columnIds
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) : List ColumnId :=
  encoding.columns.map (fun column => column.id)

/-- The exact public-prelude ownership fact implies ordinary ID membership;
callers never need a second, independently supplied membership premise. -/
theorem oneAllocated
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    encoding.one ∈ encoding.columnIds := by
  rw [encoding.oneExact]
  exact List.mem_map.mpr ⟨
    ({ id := oneColumn, ownership := .publicColumn } : OwnedColumn),
    encoding.oneAllocationExact,
    rfl
  ⟩

/-- Any scoped receipt prefix supports all dependencies of all rows that it
contains. The result keeps the initial available prefix explicit. -/
theorem receipts_rows_supported
    (available : List ColumnId)
    (receipts : List InstructionReceipt)
    (scoping : ReceiptsWellScoped available receipts)
    (receipt : InstructionReceipt)
    (receiptMember : receipt ∈ receipts)
    (row : OwnedRow)
    (rowMember : row ∈ receipt.rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈
      available ++ receipts.flatMap InstructionReceipt.columnIds := by
  induction receipts generalizing available with
  | nil =>
      simp at receiptMember
  | cons head tail inductionHypothesis =>
      rcases scoping with ⟨headScoped, tailScoped⟩
      rcases List.mem_cons.1 receiptMember with rfl | tailMember
      · have referenced :
          column ∈ InstructionReceipt.referencedColumns receipt := by
          apply List.mem_flatMap.mpr
          refine ⟨row, rowMember, ?_⟩
          simpa [InstructionReceipt.rowColumns, OwnedRow.columnIds,
            Row.columnIds] using columnMember
        rcases headScoped column referenced with prior | current
        · exact List.mem_append_left _ prior
        · exact List.mem_append_right available
            (List.mem_append_left _ current)
      · have supported :=
          inductionHypothesis
            (available ++ head.columnIds) tailScoped tailMember
        simpa [List.append_assoc] using supported

def rowIds
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) : List RowId :=
  encoding.rows.map (fun row => row.id)

/-- Physical cost is computed from the exact emitted lists. -/
def cost
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) : Cost :=
  physicalCost encoding.columns encoding.rows

/-- Receipt cost is the structural fold used before any artifact exists. -/
def receiptCost
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) : Cost :=
  Cost.sum (encoding.receipts.map InstructionReceipt.cost)

/-- A physical column occurrence is intrinsically located inside exactly one
instruction receipt. -/
abbrev ColumnOccurrence
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :=
  (receipt : Fin encoding.receipts.length) ×
    Fin (encoding.receipts.get receipt).allocations.length

/-- A physical row occurrence is intrinsically located inside exactly one
instruction receipt. -/
abbrev RowOccurrence
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :=
  (receipt : Fin encoding.receipts.length) ×
    Fin (encoding.receipts.get receipt).rows.length

def columnAt
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    {encoding : Encoding source}
    (occurrence : encoding.ColumnOccurrence) : OwnedColumn :=
  (encoding.receipts.get occurrence.1).allocations.get occurrence.2

def rowAt
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    {encoding : Encoding source}
    (occurrence : encoding.RowOccurrence) : OwnedRow :=
  (encoding.receipts.get occurrence.1).rows.get occurrence.2

def columnInstruction
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    {encoding : Encoding source}
    (occurrence : encoding.ColumnOccurrence) :
    Fin encoding.receipts.length :=
  occurrence.1

def rowInstruction
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    {encoding : Encoding source}
    (occurrence : encoding.RowOccurrence) :
    Fin encoding.receipts.length :=
  occurrence.1

private theorem member_has_unique_flatMap_owner
    {Owner Value : Type}
    (owners : List Owner)
    (values : Owner -> List Value)
    {value : Value}
    (nodup : (owners.flatMap values).Nodup)
    (member : value ∈ owners.flatMap values) :
    ∃ owner,
      owner ∈ owners ∧ value ∈ values owner ∧
        ∀ candidate,
          candidate ∈ owners -> value ∈ values candidate ->
            candidate = owner := by
  induction owners with
  | nil =>
      simp at member
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons] at nodup member
      have split := List.nodup_append.mp nodup
      rcases List.mem_append.mp member with inHead | inTail
      · refine ⟨head, List.mem_cons_self, inHead, ?_⟩
        intro candidate candidateMember candidateValue
        rcases List.mem_cons.mp candidateMember with equal | candidateTail
        · exact equal
        · have valueInTail : value ∈ tail.flatMap values :=
            List.mem_flatMap.mpr
              ⟨candidate, candidateTail, candidateValue⟩
          exact False.elim
            (split.2.2 value inHead value valueInTail rfl)
      · rcases inductionHypothesis split.2.1 inTail with
          ⟨owner, ownerMember, ownerValue, unique⟩
        refine ⟨owner, List.mem_cons_of_mem head ownerMember, ownerValue, ?_⟩
        intro candidate candidateMember candidateValue
        rcases List.mem_cons.mp candidateMember with equal | candidateTail
        · subst candidate
          exact False.elim
            (split.2.2 value candidateValue value inTail rfl)
        · exact unique candidate candidateTail candidateValue

/-- Physical column identities are exactly the receipt-local identity lists.
This equality is occurrence-preserving; it is not a set projection. -/
theorem columnIds_eq_receipt_columnIds
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    encoding.columnIds =
      encoding.receipts.flatMap InstructionReceipt.columnIds := by
  unfold columnIds columns
  induction encoding.receipts with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.map_append,
        InstructionReceipt.columnIds, inductionHypothesis]

/-- Every dependency of every emitted row occurs in the exact flattened
allocation list. This is a consequence of receipt order and does not use a
separate column census. -/
theorem rows_supported
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (receipt : InstructionReceipt)
    (receiptMember : receipt ∈ encoding.receipts)
    (row : OwnedRow)
    (rowMember : row ∈ receipt.rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ encoding.columnIds := by
  rw [encoding.columnIds_eq_receipt_columnIds]
  simpa using
    receipts_rows_supported [] encoding.receipts encoding.wellScoped
      receipt receiptMember row rowMember column columnMember

/-- Physical row identities are exactly the receipt-local identity lists. -/
theorem rowIds_eq_receipt_rowIds
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    encoding.rowIds =
      encoding.receipts.flatMap InstructionReceipt.rowIds := by
  unfold rowIds rows
  induction encoding.receipts with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.map_append,
        InstructionReceipt.rowIds, inductionHypothesis]

/-- Every emitted physical column identity occurs in exactly one instruction
receipt.  Unlike `column_has_exactly_one_instruction`, this theorem starts
from the flattened physical identity rather than an occurrence that already
contains its owner index. -/
theorem column_identity_has_exactly_one_instruction
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (column : ColumnId)
    (member : column ∈ encoding.columnIds) :
    ∃ receipt,
      receipt ∈ encoding.receipts ∧
        column ∈ receipt.columnIds ∧
          ∀ candidate,
            candidate ∈ encoding.receipts ->
            column ∈ candidate.columnIds ->
            candidate = receipt := by
  rw [encoding.columnIds_eq_receipt_columnIds] at member
  exact member_has_unique_flatMap_owner
    encoding.receipts InstructionReceipt.columnIds
    encoding.columnIdsNodup member

/-- Every emitted physical row identity occurs in exactly one instruction
receipt. -/
theorem row_identity_has_exactly_one_instruction
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (row : RowId)
    (member : row ∈ encoding.rowIds) :
    ∃ receipt,
      receipt ∈ encoding.receipts ∧
        row ∈ receipt.rowIds ∧
          ∀ candidate,
            candidate ∈ encoding.receipts ->
            row ∈ candidate.rowIds ->
            candidate = receipt := by
  rw [encoding.rowIds_eq_receipt_rowIds] at member
  exact member_has_unique_flatMap_owner
    encoding.receipts InstructionReceipt.rowIds
    encoding.rowIdsNodup member

theorem column_has_exactly_one_instruction
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    {encoding : Encoding source}
    (occurrence : encoding.ColumnOccurrence) :
    ∃ instruction : Fin encoding.receipts.length,
      instruction = encoding.columnInstruction occurrence ∧
        ∀ candidate,
          candidate = encoding.columnInstruction occurrence ->
            candidate = instruction := by
  refine ⟨encoding.columnInstruction occurrence, rfl, ?_⟩
  intro candidate equal
  exact equal

theorem row_has_exactly_one_instruction
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    {encoding : Encoding source}
    (occurrence : encoding.RowOccurrence) :
    ∃ instruction : Fin encoding.receipts.length,
      instruction = encoding.rowInstruction occurrence ∧
        ∀ candidate,
          candidate = encoding.rowInstruction occurrence ->
            candidate = instruction := by
  refine ⟨encoding.rowInstruction occurrence, rfl, ?_⟩
  intro candidate equal
  exact equal

/-- There are no physical columns outside instruction receipts. -/
theorem columns_conserved
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    encoding.columns =
      encoding.receipts.flatMap (fun receipt => receipt.allocations) :=
  rfl

/-- There are no physical rows outside instruction receipts. -/
theorem rows_conserved
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    encoding.rows =
      encoding.receipts.flatMap (fun receipt => receipt.rows) :=
  rfl

theorem column_identities_nodup
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    encoding.columnIds.Nodup := by
  have mapFlatten :
      (encoding.receipts.flatMap fun receipt => receipt.allocations).map
          (fun column => column.id) =
        encoding.receipts.flatMap InstructionReceipt.columnIds := by
    induction encoding.receipts with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.flatMap_cons, List.map_append,
          InstructionReceipt.columnIds, inductionHypothesis]
  rw [columnIds, columns, mapFlatten]
  exact encoding.columnIdsNodup

theorem row_identities_nodup
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    encoding.rowIds.Nodup := by
  have mapFlatten :
      (encoding.receipts.flatMap fun receipt => receipt.rows).map
          (fun row => row.id) =
        encoding.receipts.flatMap InstructionReceipt.rowIds := by
    induction encoding.receipts with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.flatMap_cons, List.map_append,
          InstructionReceipt.rowIds, inductionHypothesis]
  rw [rowIds, rows, mapFlatten]
  exact encoding.rowIdsNodup

private theorem physicalCost_flatten
    (receipts : List InstructionReceipt) :
    physicalCost
        (receipts.flatMap fun receipt => receipt.allocations)
        (receipts.flatMap fun receipt => receipt.rows) =
      Cost.sum (receipts.map InstructionReceipt.cost) := by
  induction receipts with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.map_cons, Cost.sum]
      rw [physicalCost_append, inductionHypothesis]
      rfl

/-- Exact four-way accounting follows from the receipt fold, not a census. -/
theorem cost_eq_receipt_cost
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    encoding.cost = encoding.receiptCost := by
  exact physicalCost_flatten encoding.receipts

/-- Physical satisfaction includes the verifier-fixed constant-one boundary.
It does not infer authority for that coordinate from a circular R1CS row. -/
def PhysicalSatisfies
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (assignment : ColumnId -> Field) : Prop :=
  assignment encoding.one = 1 ∧ Satisfies encoding.rows assignment

end Encoding

end Nightstream.Implementation.Lowering.Goldilocks
