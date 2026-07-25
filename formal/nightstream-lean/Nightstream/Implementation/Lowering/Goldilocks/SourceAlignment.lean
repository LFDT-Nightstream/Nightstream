import Nightstream.Implementation.Lowering.Goldilocks.InputReceipts
import Nightstream.Implementation.Lowering.Goldilocks.ReceiptProgram

/-!
Contract: exact structural alignment between a typed source program and a
receipt-conserved Goldilocks physical program.

Owns:
- the complete expected physical-owner skeleton of every typed block;
- explicit prelude and one input owner per source-schema port;
- one nonoptional physical receipt owner for every primitive, both branch
  activation equations, every branch join (including an empty join), and
  every continuation instruction;
- bidirectional exactness and unique-receipt theorems for source owners.

Does not own: concrete primitive or branch recipes, row semantics, normal-form
selection, Rust emission, generated artifacts, or source-to-R1CS refinement.

The owner equality is occurrence-preserving and ordered.  It rules out both a
missing source instruction receipt and an extra receipt not named by the
typed source.  A receipt may own zero rows or columns—for example an empty
yield or empty branch join—but the receipt itself remains nonoptional.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.Goldilocks

open Nightstream.Implementation.Lowering.Typed

universe u

namespace SourceAlignment

/-- Canonical owner order for one typed block.

Branch order is: true activation, false activation, true arm, false arm,
joined-output receipt, continuation.  This fixes a deterministic lowering
order independently of any Rust artifact. -/
def blockOwners
    {signature : Signature.{u}} :
    {input output : Schema signature.types} ->
      OwnerPath ->
      Block signature input output ->
      List PhysicalOwner
  | _, _, _, .yield _ => []
  | _, _, path, .step _ rest =>
      .typed (.instruction path) ::
        blockOwners (.rest path) rest
  | _, _, path, .branch _ onTrue onFalse continuation =>
      [.branchActivation path true,
        .branchActivation path false] ++
        blockOwners (.trueArm path) onTrue ++
        blockOwners (.falseArm path) onFalse ++
        [.typed (.branch path)] ++
        blockOwners (.continuation path) continuation

/-- One owner per runtime source-schema port in exact schema order. -/
def inputOwners
    {types : TypeSystem.{u}}
    (input : Schema types) : List PhysicalOwner :=
  (List.range input.length).map
    (fun slot => .typed (.input slot))

/-- Complete physical owner skeleton of a typed program. -/
def programOwners
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (source : Program signature input output) :
    List PhysicalOwner :=
  .prelude :: inputOwners input ++
    blockOwners .root source.body

/-- A conserved physical receipt program whose ordered receipt owners are
exactly the structural owners of its typed source. -/
structure AlignedReceiptProgram
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    (source : Program signature input output) where
  physical : ReceiptProgram source
  ownersExact :
    physical.receipts.map (fun receipt => receipt.owner) =
      programOwners source

namespace AlignedReceiptProgram

/-- Forget only the source-owner equality; all physical conservation evidence
remains in the resulting encoding. -/
def toEncoding
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source) :
    Encoding source :=
  program.physical.toEncoding

@[simp] theorem toEncoding_receipts
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source) :
    program.toEncoding.receipts = program.physical.receipts :=
  rfl

/-- Every physical receipt is named by the typed source skeleton. -/
theorem receipt_owner_expected
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source)
    (receipt : InstructionReceipt)
    (member : receipt ∈ program.physical.receipts) :
    receipt.owner ∈ programOwners source := by
  rw [← program.ownersExact]
  exact List.mem_map.mpr ⟨receipt, member, rfl⟩

/-- Every typed source owner has one and only one physical receipt.

Uniqueness is structural receipt uniqueness, not merely equality of the
emitted equations. -/
theorem expected_owner_has_exactly_one_receipt
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source)
    (owner : PhysicalOwner)
    (expected : owner ∈ programOwners source) :
    ∃ receipt,
      receipt ∈ program.physical.receipts ∧
        receipt.owner = owner ∧
          ∀ candidate,
            candidate ∈ program.physical.receipts ->
            candidate.owner = owner ->
            candidate = receipt := by
  have mapped :
      owner ∈
        program.physical.receipts.map
          (fun receipt => receipt.owner) := by
    rw [program.ownersExact]
    exact expected
  rcases List.mem_map.mp mapped with
    ⟨receipt, receiptMember, receiptOwner⟩
  refine ⟨receipt, receiptMember, receiptOwner, ?_⟩
  intro candidate candidateMember candidateOwner
  exact program.physical.receipt_eq_of_owner_eq
    candidateMember receiptMember
    (candidateOwner.trans receiptOwner.symm)

/-- Ordered owner equality also makes the expected source skeleton
collision-free. -/
theorem expected_owners_nodup
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source) :
    (programOwners source).Nodup := by
  rw [← program.ownersExact]
  exact program.physical.ownersNodup

/-- Every physical column identity has exactly one receipt and therefore one
expected source owner. -/
theorem column_identity_has_exactly_one_source_owner
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source)
    (column : ColumnId)
    (member : column ∈ program.toEncoding.columnIds) :
    ∃ receipt,
      receipt ∈ program.physical.receipts ∧
        receipt.owner ∈ programOwners source ∧
        column ∈ receipt.columnIds ∧
          ∀ candidate,
            candidate ∈ program.physical.receipts ->
            column ∈ candidate.columnIds ->
            candidate = receipt := by
  rcases program.toEncoding.column_identity_has_exactly_one_instruction
      column member with
    ⟨receipt, receiptMember, columnMember, unique⟩
  exact ⟨receipt, receiptMember,
    program.receipt_owner_expected receipt receiptMember,
    columnMember, unique⟩

/-- Every physical row identity has exactly one receipt and therefore one
expected source owner. -/
theorem row_identity_has_exactly_one_source_owner
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source)
    (row : RowId)
    (member : row ∈ program.toEncoding.rowIds) :
    ∃ receipt,
      receipt ∈ program.physical.receipts ∧
        receipt.owner ∈ programOwners source ∧
        row ∈ receipt.rowIds ∧
          ∀ candidate,
            candidate ∈ program.physical.receipts ->
            row ∈ candidate.rowIds ->
            candidate = receipt := by
  rcases program.toEncoding.row_identity_has_exactly_one_instruction
      row member with
    ⟨receipt, receiptMember, rowMember, unique⟩
  exact ⟨receipt, receiptMember,
    program.receipt_owner_expected receipt receiptMember,
    rowMember, unique⟩

/-- No physical columns exist outside source-aligned receipts. -/
theorem columns_conserved
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source) :
    program.toEncoding.columns =
      program.physical.receipts.flatMap
        (fun receipt => receipt.allocations) :=
  rfl

/-- No physical rows exist outside source-aligned receipts. -/
theorem rows_conserved
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source) :
    program.toEncoding.rows =
      program.physical.receipts.flatMap
        (fun receipt => receipt.rows) :=
  rfl

/-- Exact four-way physical cost is computed by folding the source-aligned
receipt program. -/
theorem cost_eq_receipt_cost
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (program : AlignedReceiptProgram source) :
    program.toEncoding.cost =
      Cost.sum
        (program.physical.receipts.map
          InstructionReceipt.cost) := by
  exact program.toEncoding.cost_eq_receipt_cost

end AlignedReceiptProgram

end SourceAlignment

end Nightstream.Implementation.Lowering.Goldilocks
