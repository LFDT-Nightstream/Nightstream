import NightstreamFPrime.Export.Stage1.PiCCSOriginalReads
import NightstreamFPrime.Export.Stage1.PiCCSSourceImagesPreservation

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalReads

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

theorem blockAt_eq (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (block : Nat) :
    blockAt masks source block = PiCCSSourceImages.blockAt (assignment masks source) block := rfl

theorem blockAt_lane (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (block : Fin PiCCSSourceImages.blockCount)
    (lane : Fin ringDegree) :
    blockAt masks source block.val lane =
      SignedUnitSourceInput.scalar (masks[block.val]?.getD #[])
        ⟨source.val, source.isLt⟩ lane := by
  rw [blockAt, PiCCSSourceImages.blockAt, dite_eq_left block.isLt]
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

theorem maskEval_eq_evalSparse (form : SparseForm ringDegree) (positive negative : Nat) :
    maskEval form positive negative =
      form.evalSparse (fun lane =>
        if positive.testBit lane.val then 1
        else if negative.testBit lane.val then -1 else 0) := by
  exact PiDECNativeSparseEvaluation.nativeEvalSparse_eq_spec form _

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
    simp only [read, ite_eq_left live]
    rw [maskEval_eq_evalSparse, PiCCSSourceImages.preparedRead, blockValues]
    rfl
  · simp only [read, ite_eq_right live, PiCCSSourceImages.preparedRead,
      PiCCSSourceImages.blockAt, dite_eq_right live, SparseForm.evalSparse,
      ringFZero, mul_zero, add_zero, List.foldl_fixed]

theorem read_eq_kernelRead {columns : Nat} (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (output : Fin ringDegree) :
    (read (PiDECParentSparseRead.prepare ()) masks source output : Fin columns → F) =
      PiCCSSourceImages.kernelRead (assignment masks source) output := by
  rw [read_eq_preparedRead]
  exact PiCCSSourceImages.preparedRead_eq (assignment masks source) output

end NightstreamFPrime.Export.Stage1.PiCCSOriginalReads
