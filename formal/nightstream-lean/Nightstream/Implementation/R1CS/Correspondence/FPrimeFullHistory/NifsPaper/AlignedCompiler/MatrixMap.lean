import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCompiler.AssignmentMap
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.AlignedCcsRelation

/-!
Matrix-row refinement for the aligned F' compiler.

Owns: an implementation-independent per-coordinate row compiler and proof that
the semantic aligned CCS row equals that compiler at every in-range column.

Does not own: Rust `CcsMatrix` storage, entry ordering, duplicate-entry
normalization, generated artifacts, Ajtai setup, or production conformance.

Emits constraints: no.

Authority boundary: an aligned row coordinate either decodes to one old
verifier-owned coefficient or is one of the thirteen fixed zeros. There is no
third prover-controlled source.

| Protocol | Phase | Constraint family | Mathematical obligation | Result |
|---|---|---|---|---|
| F' / CCS | matrix lowering | old coefficient | decoded old columns read exactly the old row value | `compiledRowValue_old` |
| F' / CCS | matrix lowering | padding coefficient | an undecodable aligned column reads zero | `compiledRowValue_padding` |
| F' / CCS | matrix lowering | complete row | every in-range aligned row value equals the independent decoder | `alignRow_getD_eq_compiledRowValue` |
| F' / CCS | matrix lowering | source exclusivity | every aligned column is old-owned or fixed padding | `compiledRowValue_cases` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedPublicInput
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCcsRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.ColumnMap
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.AssignmentMap

/-- Independent scalar-row compiler: decode an aligned coordinate to its old
owner, or emit the fixed additive identity for a padding coordinate. -/
def compiledRowValue (row : List F) (alignedColumn : Nat) : F :=
  match unalignIndex? alignedColumn with
  | some oldColumn => row.getD oldColumn 0
  | none => 0

theorem compiledRowValue_old (row : List F) (oldColumn : Nat) :
    compiledRowValue row (alignedIndex oldColumn) =
      row.getD oldColumn 0 := by
  simp [compiledRowValue, unalignIndex?_alignedIndex]

theorem compiledRowValue_padding (row : List F) (alignedColumn : Nat)
    (isPadding : logicalPublicWidth ≤ alignedColumn ∧
      alignedColumn < alignedPublicWidth) :
    compiledRowValue row alignedColumn = 0 := by
  rw [compiledRowValue]
  rw [(unalignIndex?_eq_none_iff alignedColumn).2 isPadding]

/-- Every aligned row coordinate has exactly one of the two authorized
sources: an old coefficient or the fixed zero padding. -/
theorem compiledRowValue_cases (row : List F) (alignedColumn : Nat) :
    (∃ oldColumn,
        unalignIndex? alignedColumn = some oldColumn ∧
        compiledRowValue row alignedColumn = row.getD oldColumn 0) ∨
      (logicalPublicWidth ≤ alignedColumn ∧
        alignedColumn < alignedPublicWidth ∧
        compiledRowValue row alignedColumn = 0) := by
  cases decoded : unalignIndex? alignedColumn with
  | none =>
      right
      have padding := (unalignIndex?_eq_none_iff alignedColumn).1 decoded
      exact ⟨padding.1, padding.2,
        compiledRowValue_padding row alignedColumn padding⟩
  | some oldColumn =>
      left
      refine ⟨oldColumn, rfl, ?_⟩
      simp [compiledRowValue, decoded]

/-- Complete per-coordinate equivalence between the semantic list insertion
and the independent decoder-based row compiler. -/
theorem alignRow_getD_eq_compiledRowValue (row : List F)
    (hasPublic : logicalPublicWidth ≤ row.length)
    (alignedColumn : Fin (row.length + paddingWidth)) :
    (alignRow row).getD alignedColumn.val 0 =
      compiledRowValue row alignedColumn.val := by
  cases decoded : unalignIndex? alignedColumn.val with
  | none =>
      have padding := (unalignIndex?_eq_none_iff alignedColumn.val).1 decoded
      rw [compiledRowValue_padding row alignedColumn.val padding]
      exact getD_padding_zero row hasPublic alignedColumn.val padding
  | some oldColumn =>
      have mapped := alignedIndex_of_unalignIndex?_eq_some decoded
      rw [← mapped]
      change (insertPublicPadding row).getD (alignedIndex oldColumn) 0 =
        compiledRowValue row (alignedIndex oldColumn)
      rw [getD_alignedIndex row hasPublic oldColumn]
      exact (compiledRowValue_old row oldColumn).symm

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.AlignedCompiler.MatrixMap
