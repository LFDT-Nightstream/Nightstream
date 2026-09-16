import NightstreamFPrime.Export.Stage1.PiCCSOriginalReads
import NightstreamFPrime.Export.Stage1.PiCCSSourceImagesPreservation

/-!
Proof-only equality of original-mask blocks and prepared coefficient callbacks
with the existing complete-assignment readers. No split, source-support or
witness-validity premise is used. Final message-family transport is separate.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalReads

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

/-- The block callback is the existing complete-assignment block at every index. -/
theorem blockAt_eq (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (block : Nat) :
    blockAt masks source block = PiCCSSourceImages.blockAt (assignment masks source) block := rfl

/-- Each lane of every in-range block is the original mask scalar, including
all lanes of the final carrier block. Missing source entries retain exact zero. -/
theorem blockAt_lane (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (block : Fin PiCCSSourceImages.blockCount)
    (lane : Fin ringDegree) :
    blockAt masks source block.val lane =
      SignedUnitSourceInput.scalar (masks[block.val]?.getD #[])
        ⟨source.val, source.isLt⟩ lane := by
  rw [blockAt, PiCCSSourceImages.blockAt, dif_pos block.isLt]
  change assignment masks source
    (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth) block lane) = _
  have decoded := Phi81CarrierLayout.decode_carrierColumn
    (logicalWidth := PiCCSSourceImages.logicalWidth) block lane
  have blockIndex :
      (Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth)
        block lane).val / ringDegree = block.val :=
    congrArg (fun value => value.1.val) decoded
  have laneIndex :
      (⟨(Phi81CarrierLayout.carrierColumn (logicalWidth := PiCCSSourceImages.logicalWidth)
        block lane).val % ringDegree, Nat.mod_lt _ (by decide)⟩ : Fin ringDegree) = lane :=
    congrArg Prod.snd decoded
  simp only [assignment, blockIndex, laneIndex]

/-- Direct signed accumulation equals the existing sparse evaluation for
arbitrary masks and coefficients, including positive/negative overlap. -/
theorem maskEval_eq_evalSparse (form : SparseForm ringDegree) (positive negative : Nat) :
    maskEval form positive negative =
      form.evalSparse (fun lane =>
        if positive.testBit lane.val then 1
        else if negative.testBit lane.val then -1 else 0) := by
  unfold maskEval SparseForm.evalSparse
  apply congrArg (fun step : F → SparseEntry ringDegree → F => form.entries.foldl step 0)
  funext total entry
  dsimp only
  split_ifs
  · exact congrArg (fun value : F => total + value) (Fin.mul_one entry.coefficient).symm
  · apply congrArg (fun value : F => total + value)
    calc
      -entry.coefficient = -(1 * entry.coefficient) :=
        congrArg (fun value : F => -value) (Fin.one_mul entry.coefficient).symm
      _ = (-1) * entry.coefficient := (Lean.Grind.Fin.neg_mul (1 : F) entry.coefficient).symm
      _ = entry.coefficient * (-1) := Fin.mul_comm _ _
  · calc
      total = total + (0 : F) := (Fin.add_zero total).symm
      _ = total + entry.coefficient * 0 :=
        congrArg (fun value : F => total + value) (Fin.mul_zero entry.coefficient).symm

/-- The mask reader is the old prepared reader for every supplied table.
The complete-carrier guard covers arbitrary column widths and zero suffixes. -/
theorem read_eq_preparedRead {columns : Nat}
    (tables : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (masks : Array (Array (Nat × Nat))) (source : Fin productionShape.sourceCount)
    (output : Fin ringDegree) :
    (read tables masks source output : Fin columns → F) =
      PiCCSSourceImages.preparedRead tables (assignment masks source) output := by
  funext column
  by_cases live : column.val / ringDegree < PiCCSSourceImages.blockCount
  · have blockValues :
        PiCCSSourceImages.blockAt (assignment masks source) (column.val / ringDegree) =
          fun lane => SignedUnitSourceInput.scalar
            (masks[column.val / ringDegree]?.getD #[]) ⟨source.val, source.isLt⟩ lane := by
      funext lane
      exact blockAt_lane masks source ⟨column.val / ringDegree, live⟩ lane
    simp only [read, if_pos live]
    rw [maskEval_eq_evalSparse, PiCCSSourceImages.preparedRead, blockValues]
    rfl
  · simp only [read, if_neg live, PiCCSSourceImages.preparedRead,
      PiCCSSourceImages.blockAt, dif_neg live, SparseForm.evalSparse,
      ringFZero, mul_zero, add_zero, List.foldl_fixed]

/-- Canonical prepared forms preserve the original kernel read at every
requested column and output coefficient of the same complete assignment. -/
theorem read_eq_kernelRead {columns : Nat} (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (output : Fin ringDegree) :
    (read (PiDECParentSparseRead.prepare ()) masks source output : Fin columns → F) =
      PiCCSSourceImages.kernelRead (assignment masks source) output := by
  rw [read_eq_preparedRead]
  exact PiCCSSourceImages.preparedRead_eq (assignment masks source) output

end NightstreamFPrime.Export.Stage1.PiCCSOriginalReads
