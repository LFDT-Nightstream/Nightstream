import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixSupported
import NightstreamFPrime.Export.Stage1.PiCCSOriginalSupportPreservation
import NightstreamFPrime.Export.Stage1.PiDECMatrixZeroRead

/-!
Complete source/port preservation for guarded original matrix sums. The
arithmetic zero premise is derived from the full signed-mask check in the
specialized endpoints. No split, magnitude, or combined witness is used.
-/

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

/-- A sound zero flag preserves the complete sparse result for this source
and port, including both field coordinates of all 54 ring coefficients. -/
theorem sparse_source_port {columns arity count : Nat}
    (zeroSource : Fin productionShape.sourceCount → Bool)
    (firstRow : Nat) (point : CubePoint K arity)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount)
    (zeroRead : zeroSource source = true → read source = fun _ _ => 0) :
    ((sparse zeroSource firstRow point read forms).get (Fin.encodeProd (source, port))).toRing =
      ((PiCCSOriginalMatrixBatch.sum firstRow point read forms).get
        (Fin.encodeProd (source, port))).toRing := by
  rw [PiCCSOriginalMatrixBatch.sum_source_port]
  simp only [sparse, get_ofFn, Fin.decodeProd_encodeProd]
  by_cases zero : zeroSource source = true
  · rw [if_pos zero, PiDECEvaluationBatch.zero_value, zeroRead zero,
      PiDECMatrixZeroRead.sparse_sum_zero]
  · simp only [if_neg zero]

/-- The same flag preserves complete prepared numeric invocation ranges.
The existing zero-read theorem covers selectors, constants and output pins. -/
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
  simp only [invocations, get_ofFn, Fin.decodeProd_encodeProd]
  by_cases zero : zeroSource source = true
  · rw [if_pos zero, PiDECEvaluationBatch.zero_value, zeroRead zero,
      PiDECMatrixZeroRead.invocation_sum_zero]
  · simp only [if_neg zero]

/-- Flags computed from every original mask preserve the unguarded sparse
batch for arbitrary supplied basis tables, forms, offsets and source indices. -/
theorem sparse_isZero_source_port {columns arity count : Nat}
    (tables : FixedArray (FixedArray (SparseForm ringDegree) ringDegree) ringDegree)
    (masks : Array (Array (Nat × Nat))) (firstRow : Nat) (point : CubePoint K arity)
    (forms : Vector (MatrixProgram.RowForms columns) count)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount) :
    ((sparse (PiCCSOriginalSupport.isZero masks) firstRow point
      (PiCCSOriginalReads.read tables masks) forms).get (Fin.encodeProd (source, port))).toRing =
      ((PiCCSOriginalMatrixBatch.sum firstRow point (PiCCSOriginalReads.read tables masks) forms).get
        (Fin.encodeProd (source, port))).toRing := by
  apply sparse_source_port
  exact PiCCSOriginalSupport.read_eq_zero tables masks source

/-- Full-mask flags preserve every source of the numeric invocation batch.
Caching these flags changes no read or source identity. -/
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
