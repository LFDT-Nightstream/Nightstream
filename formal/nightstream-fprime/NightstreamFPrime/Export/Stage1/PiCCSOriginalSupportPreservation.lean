import NightstreamFPrime.Export.Stage1.PiCCSOriginalSupport
import NightstreamFPrime.Export.Stage1.PiCCSOriginalReadsPreservation

/-!
A successful complete mask check gives the zero original assignment and zero
coefficient callbacks. The proof covers missing blocks and entries, carrier
tails and outside columns. No source index is removed or renumbered here.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalSupport

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

private theorem pair_eq_zero (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (zero : isZero masks source = true)
    (block : Nat) :
    ((masks[block]?.getD #[])[source.val]?.getD (0, 0)) = (0, 0) := by
  unfold isZero at zero
  by_cases live : block < masks.size
  · rw [Array.getElem?_eq_getElem live, Option.getD_some]
    exact beq_iff_eq.mp ((Array.all_eq_true.mp zero) block live)
  · simp only [Array.getElem?_eq_none (Nat.le_of_not_lt live), Option.getD_none,
      Array.getElem?_empty]

/-- A complete zero-mask check makes every original carrier column zero.
No capture-size or source-entry presence premise is needed. -/
theorem assignment_eq_zero (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (zero : isZero masks source = true) :
    PiCCSOriginalReads.assignment masks source = fun _ => 0 := by
  funext column
  unfold PiCCSOriginalReads.assignment SignedUnitSourceInput.scalar
  rw [pair_eq_zero masks source zero (column.val / ringDegree)]
  simp only [Nat.zero_testBit, Bool.false_eq_true, if_false]

/-- Every original coefficient callback is zero for arbitrary prepared forms
and arbitrary requested column widths, including the complete zero suffix. -/
theorem read_eq_zero {columns : Nat}
    (tables : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (masks : Array (Array (Nat × Nat))) (source : Fin productionShape.sourceCount)
    (zero : isZero masks source = true) :
    (PiCCSOriginalReads.read tables masks source : Fin ringDegree → Fin columns → F) =
      fun _ _ => 0 := by
  have emptyBlock (block : Nat) :
      PiCCSSourceImages.blockAt (fun _ => 0) block = ringFZero := by
    unfold PiCCSSourceImages.blockAt
    split <;> rfl
  funext output column
  rw [congrFun (PiCCSOriginalReads.read_eq_preparedRead tables masks source output) column,
    assignment_eq_zero masks source zero]
  unfold PiCCSSourceImages.preparedRead
  rw [emptyBlock]
  simp only [SparseForm.evalSparse, ringFZero, mul_zero, add_zero, List.foldl_fixed]

end NightstreamFPrime.Export.Stage1.PiCCSOriginalSupport
