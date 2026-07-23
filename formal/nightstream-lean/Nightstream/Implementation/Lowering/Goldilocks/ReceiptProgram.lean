import Nightstream.Implementation.Lowering.Goldilocks.InstructionReceipts

/-!
Contract: construct a conserved Goldilocks `Encoding` from one exact ordered
list of physical instruction receipts.

Owns:
- the receipt-list evidence needed for globally unique physical identities;
- derivation of flattened column/row uniqueness from receipt ownership and
  receipt-local uniqueness;
- construction of `Encoding` with the actual public prelude receipt.

Does not own: source-program refinement, instruction selection, arbitrary glue
rows, Rust emission, generated artifacts, or protocol semantics.

Every physical occurrence in the resulting encoding comes from the supplied
receipt list.  The constructor takes no independent one-column membership
premise: membership of `InstructionReceipt.prelude` supplies the exact public
allocation.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

/-- Structural evidence that one exact ordered receipt list is a conserved
physical program. -/
structure ReceiptProgram
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (source : Program signature input output) where
  receipts : List InstructionReceipt
  preludeMember : InstructionReceipt.prelude ∈ receipts
  ownersNodup :
    (receipts.map fun receipt => receipt.owner).Nodup
  localColumnIdsNodup :
    ∀ receipt, receipt ∈ receipts -> receipt.columnIds.Nodup
  localRowIdsNodup :
    ∀ receipt, receipt ∈ receipts -> receipt.rowIds.Nodup
  wellScoped : ReceiptsWellScoped [] receipts

namespace ReceiptProgram

private theorem member_eq_of_owner_eq
    {Receipt Owner : Type}
    (ownerOf : Receipt -> Owner)
    (receipts : List Receipt)
    (ownersNodup : (receipts.map ownerOf).Nodup)
    {first second : Receipt}
    (firstMember : first ∈ receipts)
    (secondMember : second ∈ receipts)
    (ownersEqual : ownerOf first = ownerOf second) :
    first = second := by
  induction receipts with
  | nil =>
      simp at firstMember
  | cons head tail inductionHypothesis =>
      have ownerSplit :
          ownerOf head ∉ tail.map ownerOf ∧
            (tail.map ownerOf).Nodup := by
        simpa only [List.map_cons, List.nodup_cons] using ownersNodup
      rcases List.mem_cons.mp firstMember with firstEqual | firstTail
      · subst first
        rcases List.mem_cons.mp secondMember with secondEqual | secondTail
        · exact secondEqual.symm
        · exact False.elim (ownerSplit.1
            (List.mem_map.mpr
              ⟨second, secondTail, ownersEqual.symm⟩))
      · rcases List.mem_cons.mp secondMember with secondEqual | secondTail
        · subst second
          exact False.elim (ownerSplit.1
            (List.mem_map.mpr
              ⟨first, firstTail, ownersEqual⟩))
        · exact inductionHypothesis ownerSplit.2
            firstTail secondTail

/-- Distinct positions in a receipt program cannot reuse a structural owner. -/
theorem receipt_eq_of_owner_eq
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : ReceiptProgram source)
    {first second : InstructionReceipt}
    (firstMember : first ∈ program.receipts)
    (secondMember : second ∈ program.receipts)
    (ownersEqual : first.owner = second.owner) :
    first = second :=
  member_eq_of_owner_eq
    (fun receipt : InstructionReceipt => receipt.owner)
    program.receipts program.ownersNodup
    firstMember secondMember ownersEqual

private theorem flatMap_ids_nodup
    {Receipt Owner Id : Type}
    (ownerOf : Receipt -> Owner)
    (idOwner : Id -> Owner)
    (ids : Receipt -> List Id)
    (receipts : List Receipt)
    (ownersNodup : (receipts.map ownerOf).Nodup)
    (localNodup :
      ∀ receipt, receipt ∈ receipts -> (ids receipt).Nodup)
    (idsOwned :
      ∀ receipt id, id ∈ ids receipt ->
        idOwner id = ownerOf receipt) :
    (receipts.flatMap ids).Nodup := by
  induction receipts with
  | nil =>
      exact List.nodup_nil
  | cons head tail inductionHypothesis =>
      have ownerSplit :
          ownerOf head ∉ tail.map ownerOf ∧
            (tail.map ownerOf).Nodup := by
        simpa only [List.map_cons, List.nodup_cons] using ownersNodup
      rw [List.flatMap_cons, List.nodup_append]
      refine ⟨
        localNodup head List.mem_cons_self,
        inductionHypothesis ownerSplit.2
          (fun receipt member =>
            localNodup receipt (List.mem_cons_of_mem head member)),
        ?_
      ⟩
      intro headId headMember tailId tailMember idsEqual
      rcases List.mem_flatMap.mp tailMember with
        ⟨tailReceipt, tailReceiptMember, tailIdMember⟩
      have ownersEqual : ownerOf head = ownerOf tailReceipt := by
        calc
          ownerOf head = idOwner headId :=
            (idsOwned head headId headMember).symm
          _ = idOwner tailId := congrArg idOwner idsEqual
          _ = ownerOf tailReceipt :=
            idsOwned tailReceipt tailId tailIdMember
      exact False.elim (ownerSplit.1
        (List.mem_map.mpr
          ⟨tailReceipt, tailReceiptMember, ownersEqual.symm⟩))

private theorem column_id_has_receipt_owner
    (receipt : InstructionReceipt)
    (id : ColumnId)
    (member : id ∈ receipt.columnIds) :
    id.owner = receipt.owner := by
  rcases List.mem_map.mp member with ⟨column, columnMember, rfl⟩
  exact receipt.allocationsOwned column columnMember

private theorem row_id_has_receipt_owner
    (receipt : InstructionReceipt)
    (id : RowId)
    (member : id ∈ receipt.rowIds) :
    id.owner = receipt.owner := by
  rcases List.mem_map.mp member with ⟨row, rowMember, rfl⟩
  exact receipt.rowsOwned row rowMember

/-- Receipt-owner uniqueness and receipt-local column uniqueness imply global
uniqueness of the exact flattened physical column identities. -/
theorem flattenedColumnIdsNodup
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : ReceiptProgram source) :
    (program.receipts.flatMap InstructionReceipt.columnIds).Nodup :=
  flatMap_ids_nodup
    (fun receipt : InstructionReceipt => receipt.owner)
    (fun id : ColumnId => id.owner)
    InstructionReceipt.columnIds
    program.receipts program.ownersNodup
    program.localColumnIdsNodup
    column_id_has_receipt_owner

/-- Receipt-owner uniqueness and receipt-local row uniqueness imply global
uniqueness of the exact flattened physical row identities. -/
theorem flattenedRowIdsNodup
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : ReceiptProgram source) :
    (program.receipts.flatMap InstructionReceipt.rowIds).Nodup :=
  flatMap_ids_nodup
    (fun receipt : InstructionReceipt => receipt.owner)
    (fun id : RowId => id.owner)
    InstructionReceipt.rowIds
    program.receipts program.ownersNodup
    program.localRowIdsNodup
    row_id_has_receipt_owner

/-- Construct the conserved compiler encoding.  There are no physical rows or
columns outside `program.receipts`. -/
def toEncoding
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : ReceiptProgram source) :
    Encoding source where
  receipts := program.receipts
  one := oneColumn
  oneExact := rfl
  oneAllocationExact := by
    refine List.mem_flatMap.mpr
      ⟨InstructionReceipt.prelude, program.preludeMember, ?_⟩
    change
      ({ id := oneColumn, ownership := .publicColumn } : OwnedColumn) ∈
        preludeColumns
    exact List.mem_singleton.mpr rfl
  columnIdsNodup := program.flattenedColumnIdsNodup
  rowIdsNodup := program.flattenedRowIdsNodup
  wellScoped := program.wellScoped

@[simp] theorem toEncoding_receipts
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : ReceiptProgram source) :
    program.toEncoding.receipts = program.receipts :=
  rfl

@[simp] theorem toEncoding_one
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : ReceiptProgram source) :
    program.toEncoding.one = oneColumn :=
  rfl

end ReceiptProgram

end Nightstream.Implementation.Lowering.Goldilocks
