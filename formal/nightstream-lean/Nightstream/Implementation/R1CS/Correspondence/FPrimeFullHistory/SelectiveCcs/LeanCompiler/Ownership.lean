import Nightstream.Implementation.Lowering.Goldilocks.SourceAlignment
import Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.EncodingRows

/-!
Contract: carry the exact receipt and typed-source ownership of one
receipt-conserved encoding through the Lean-owned selective-CCS compiler.

Assurance tier: model-level.

Owns: occurrence-preserving source/compiled row positions, exact row count,
one unique receipt for every compiled row, no receipt row outside the compiled
program, and the source-owner refinement for aligned receipt programs.

Does not own: row semantics, an application selection, a concrete fixed-point
profile, Rust, generated artifacts, or protocol security events.

Emits constraints: no new rows.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.Ownership

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks

universe u

/-- Transport a compiled row position to the source physical row at the same
ordinal. The compiler is occurrence-preserving, so this changes only the
length proof. -/
def sourcePosition
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (position : Fin (EncodingRows.program encoding).length) :
    Fin encoding.rows.length :=
  Fin.cast (EncodingRows.program_length encoding) position

/-- Transport a source physical row position to the compiled row at the same
ordinal. -/
def compiledPosition
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (position : Fin encoding.rows.length) :
    Fin (EncodingRows.program encoding).length :=
  Fin.cast (EncodingRows.program_length encoding).symm position

@[simp] theorem sourcePosition_value
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (position : Fin (EncodingRows.program encoding).length) :
    (sourcePosition encoding position).val = position.val :=
  rfl

@[simp] theorem compiledPosition_value
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (position : Fin encoding.rows.length) :
    (compiledPosition encoding position).val = position.val :=
  rfl

/-- The physical row occurrence compiled at one selective row position. -/
def sourceAt
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (position : Fin (EncodingRows.program encoding).length) :
    OwnedRow :=
  encoding.rows.get (sourcePosition encoding position)

/-- The compiler changes the coefficient representation but preserves the
row occurrence ordinal exactly. -/
theorem compiledAt_eq_sourceAt
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (position : Fin (EncodingRows.program encoding).length) :
    (EncodingRows.program encoding).get position =
      StableRows.row (EncodingRows.columnIndex encoding)
        (sourceAt encoding position).row := by
  unfold EncodingRows.program StableRows.program sourceAt sourcePosition
  simp

theorem sourceAt_mem
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (position : Fin (EncodingRows.program encoding).length) :
    sourceAt encoding position ∈ encoding.rows :=
  List.get_mem encoding.rows (sourcePosition encoding position)

theorem sourceAt_id_mem
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (position : Fin (EncodingRows.program encoding).length) :
    (sourceAt encoding position).id ∈ encoding.rowIds := by
  unfold Encoding.rowIds
  exact List.mem_map.mpr
    ⟨sourceAt encoding position, sourceAt_mem encoding position, rfl⟩

/-- Every compiled row has one unique deepest instruction receipt. The
receipt is selected by the source `RowId`, never by row-value equality. -/
theorem compiledRow_has_exactly_one_instruction
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (position : Fin (EncodingRows.program encoding).length) :
    ∃ receipt,
      receipt ∈ encoding.receipts ∧
        (sourceAt encoding position).id ∈ receipt.rowIds ∧
          ∀ candidate,
            candidate ∈ encoding.receipts →
            (sourceAt encoding position).id ∈ candidate.rowIds →
            candidate = receipt :=
  encoding.row_identity_has_exactly_one_instruction
    (sourceAt encoding position).id
    (sourceAt_id_mem encoding position)

/-- Every receipt-owned physical row appears at one compiled row position.
This is the no-row-outside-the-program direction. -/
theorem receiptRow_has_compiled_position
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source)
    (receipt : InstructionReceipt)
    (receiptMember : receipt ∈ encoding.receipts)
    (owned : OwnedRow)
    (ownedMember : owned ∈ receipt.rows) :
    ∃ position : Fin (EncodingRows.program encoding).length,
      sourceAt encoding position = owned := by
  have inRows : owned ∈ encoding.rows := by
    unfold Encoding.rows
    exact List.mem_flatMap.mpr
      ⟨receipt, receiptMember, ownedMember⟩
  rcases List.mem_iff_get.mp inRows with ⟨sourceIndex, sourceEqual⟩
  refine ⟨compiledPosition encoding sourceIndex, ?_⟩
  unfold sourceAt sourcePosition compiledPosition
  simpa using sourceEqual

/-- The selective source-row count is the exact flattened receipt-row count.
There is no formula-only row estimate in this equality. -/
theorem compiledRowCount_eq_receiptRows
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    (EncodingRows.program encoding).length =
      (encoding.receipts.map fun receipt => receipt.rows.length).sum := by
  rw [EncodingRows.program_length]
  unfold Encoding.rows
  induction encoding.receipts with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.length_append, List.map_cons,
        List.sum_cons, inductionHypothesis]

/-- The compiled-row count is the recurring-row component of the exact
receipt-derived physical cost. -/
theorem compiledRowCount_eq_cost
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (encoding : Encoding source) :
    (EncodingRows.program encoding).length =
      encoding.cost.recurringRows := by
  calc
    (EncodingRows.program encoding).length =
        encoding.rows.length :=
      EncodingRows.program_length encoding
    _ =
        (CanonicalManifest.Program.ofEncoding encoding).rows.length :=
      (CanonicalManifest.Program.rows_length_ofEncoding encoding).symm
    _ =
        (CanonicalManifest.Program.ofEncoding encoding).cost.recurringRows :=
      (CanonicalManifest.Program.cost_recurringRows
        (CanonicalManifest.Program.ofEncoding encoding)).symm
    _ = encoding.cost.recurringRows :=
      congrArg Cost.recurringRows
        (CanonicalManifest.Program.cost_ofEncoding encoding)

/-- Source-aligned current-program obligation tree: every compiled row has
one unique receipt whose owner occurs in the typed source skeleton. -/
theorem compiledRow_has_exactly_one_source_owner
    {signature : Signature.{u}}
    {input output : Schema signature.types}
    {source : Program signature input output}
    (aligned : SourceAlignment.AlignedReceiptProgram source)
    (position : Fin
      (EncodingRows.program aligned.toEncoding).length) :
    ∃ receipt,
      receipt ∈ aligned.physical.receipts ∧
        receipt.owner ∈ SourceAlignment.programOwners source ∧
        (sourceAt aligned.toEncoding position).id ∈ receipt.rowIds ∧
          ∀ candidate,
            candidate ∈ aligned.physical.receipts →
            (sourceAt aligned.toEncoding position).id ∈
              candidate.rowIds →
            candidate = receipt := by
  rcases aligned.row_identity_has_exactly_one_source_owner
      (sourceAt aligned.toEncoding position).id
      (sourceAt_id_mem aligned.toEncoding position) with
    ⟨receipt, receiptMember, expected, rowMember, unique⟩
  exact ⟨receipt, receiptMember, expected, rowMember, unique⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.Ownership
