import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyPoseidonCallFamily
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.RowAction

/-!
Contract: row-indexed matrix-action bridge for one normalized production
PiRLC Poseidon2 call block.

Owns the implication from exact decoded final-row images and zero final-row
residuals to `EmittedBlock.Satisfied` on the same assignment.

Does not own a production matrix artifact, Rust provenance, selector
activation, replay execution, lifecycle semantics, or collision resistance.
The production matrix-slice certificate remains an explicit obligation.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonFinalRowBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallFamily
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonCallRowProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction

/-- The final relation row that owns one local row offset of a call block. -/
def finalRowIndex
    {rows : Nat} (block : EmittedBlock)
    (rowsFit : block.finalRowStart + block.rows.length <= rows)
    (offset : Fin block.rows.length) : Fin rows :=
  ⟨block.finalRowStart + offset.val, by
    have offsetLt := offset.isLt
    omega⟩

/-- Exact matrix-action identity for every row of one emitted block.
This is the trust-boundary premise that a production matrix certificate must
prove. Row metadata or a digest cannot construct this value. -/
structure FinalRowSliceExact
    {rows : Nat} (block : EmittedBlock)
    (relation : InterpretedRelation rows productionFinalColumns)
    (assignment : Fin productionFinalColumns -> F) : Prop where
  rowsFit : block.finalRowStart + block.rows.length <= rows
  pointExact : forall offset : Fin block.rows.length,
    rowPoint relation assignment (finalRowIndex block rowsFit offset) =
      absolutePoint block.site assignment (block.rows.get offset)

/-- Every decoded final row has zero residual on one assignment. -/
def AllRowsSatisfied
    {rows : Nat}
    (relation : InterpretedRelation rows productionFinalColumns)
    (assignment : Fin productionFinalColumns -> F) : Prop :=
  forall row, residualAt relation assignment row = 0

/-- Exact final matrix images plus final relation satisfaction imply all
canonical residuals in the indexed 86-row call block. -/
theorem final_rows_imply_emitted_block_satisfied
    {rows : Nat} {block : EmittedBlock}
    {relation : InterpretedRelation rows productionFinalColumns}
    {assignment : Fin productionFinalColumns -> F}
    (exact : FinalRowSliceExact block relation assignment)
    (satisfied : AllRowsSatisfied relation assignment) :
    block.Satisfied assignment := by
  intro row member
  rcases List.mem_iff_get.mp member with ⟨offset, rowExact⟩
  subst row
  have finalZero := satisfied (finalRowIndex block exact.rowsFit offset)
  rw [residualAt_eq_evaluate, exact.pointExact offset] at finalZero
  exact finalZero

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonFinalRowBridge
