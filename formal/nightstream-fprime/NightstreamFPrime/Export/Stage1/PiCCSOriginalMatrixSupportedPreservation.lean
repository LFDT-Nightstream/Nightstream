import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixSupported
import NightstreamFPrime.Export.Stage1.PiCCSOriginalSupportPreservation
import NightstreamFPrime.Export.Stage1.PiDECMatrixZeroRead

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixSupported

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle (productionShape)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (FixedArray)

private theorem get_ofFn {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = _
  rw [Vector.getElem_ofFn]

theorem weighted_source_port {columns arity count : Nat}
    (zeroSource : Fin productionShape.sourceCount → Bool)
    (firstRow : Nat) (point : CubePoint K arity)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount)
    (zeroRead : zeroSource source = true → read source = fun _ _ => 0) :
    ((weighted zeroSource firstRow point read forms).get (Fin.encodeProd (source, port))).toRing =
      ((PiCCSOriginalMatrixBatch.sum firstRow point read forms).get
        (Fin.encodeProd (source, port))).toRing := by
  rw [PiCCSOriginalMatrixBatch.sum_source_port]
  simp only [weighted, get_ofFn, Fin.decodeProd_encodeProd]
  by_cases zero : zeroSource source = true
  · rw [ite_eq_left zero, PiDECEvaluationBatch.zero_value, zeroRead zero,
      PiDECMatrixZeroRead.sparse_sum_zero]
  · simp only [ite_eq_right zero]
    exact PiDECMatrixWeightedRange.evaluate_prepare_toRing firstRow point (read source) forms port

theorem invocations_source_port {columns arity count : Nat}
    (zeroSource : Fin productionShape.sourceCount → Bool)
    (firstRow : Nat) (point : CubePoint K arity)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount)
    (zeroRead : zeroSource source = true → read source = fun _ _ => 0) :
    ((invocations zeroSource firstRow point read interfaces).get
        (Fin.encodeProd (source, port))).toRing =
      ((PiCCSOriginalMatrixBatch.sumInvocations firstRow point read interfaces).get
        (Fin.encodeProd (source, port))).toRing := by
  rw [PiCCSOriginalMatrixBatch.sumInvocations_source_port]
  by_cases allZero : ∀ source, zeroSource source = true
  · rw [invocations, ite_eq_left allZero, PiDECEvaluationBatch.zero_value, zeroRead (allZero source),
      PiDECMatrixZeroRead.invocation_sum_zero]
  · simp only [invocations, ite_eq_right allZero, get_ofFn, Fin.decodeProd_encodeProd]
    by_cases zero : zeroSource source = true
    · rw [ite_eq_left zero, PiDECEvaluationBatch.zero_value, zeroRead zero,
        PiDECMatrixZeroRead.invocation_sum_zero]
    · simp only [ite_eq_right zero]
      exact PiDECPoseidonColumnWeights.evaluate_prepare_toRing firstRow point (read source)
        interfaces port

theorem weighted_isZero_source_port {columns arity count : Nat}
    (tables : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (masks : Array (Array (Nat × Nat))) (firstRow : Nat) (point : CubePoint K arity)
    (forms : Vector (MatrixProgram.RowForms columns) count)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount) :
    ((weighted (PiCCSOriginalSupport.isZero masks) firstRow point
      (PiCCSOriginalReads.read tables masks) forms).get (Fin.encodeProd (source, port))).toRing =
      ((PiCCSOriginalMatrixBatch.sum firstRow point (PiCCSOriginalReads.read tables masks) forms).get
        (Fin.encodeProd (source, port))).toRing := by
  apply weighted_source_port
  exact PiCCSOriginalSupport.read_eq_zero tables masks source

theorem invocations_isZero_source_port {columns arity count : Nat}
    (tables : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (masks : Array (Array (Nat × Nat))) (firstRow : Nat) (point : CubePoint K arity)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount) :
    ((invocations (PiCCSOriginalSupport.isZero masks) firstRow point
      (PiCCSOriginalReads.read tables masks) interfaces).get (Fin.encodeProd (source, port))).toRing =
      ((PiCCSOriginalMatrixBatch.sumInvocations firstRow point
        (PiCCSOriginalReads.read tables masks) interfaces).get (Fin.encodeProd (source, port))).toRing := by
  apply invocations_source_port
  exact PiCCSOriginalSupport.read_eq_zero tables masks source

end NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixSupported
