import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixBatch
import NightstreamFPrime.Export.Stage1.PiDECMatrixWeightedRange
import NightstreamFPrime.Export.Stage1.PiDECPoseidonColumnWeights

/-!
Skip a complete source sum when its supplied zero flag is true, then retain
all original source/port slots. The caller validates loaders and derives the
flags from the complete masks. Preservation is in the separate proof module.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixSupported

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle (productionShape)

/-- The zero branch precedes all source arithmetic. The column weights of the
rows are prepared once for every nonzero source. -/
@[specialize] def weighted {columns arity count : Nat}
    (zeroSource : Fin productionShape.sourceCount → Bool)
    (firstRow : Nat) (point : CubePoint K arity)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count) : PiCCSOriginalMatrixBatch.Batch :=
  let prepared := PiDECMatrixWeightedRange.prepare firstRow point forms
  let bySource := Vector.ofFn fun source : Fin productionShape.sourceCount =>
    if zeroSource source then PiDECEvaluationBatch.zero matrixCount
    else PiDECMatrixWeightedRange.evaluate prepared (read source)
  Vector.ofFn fun code =>
    let pair : Fin productionShape.sourceCount × Fin matrixCount := Fin.decodeProd code
    (bySource.get pair.1).get pair.2

/-- Skip preparation when every source is zero; otherwise prepare the invocation
weights once and retain every source/port slot. -/
@[specialize] def invocations {columns arity count : Nat}
    (zeroSource : Fin productionShape.sourceCount → Bool)
    (firstRow : Nat) (point : CubePoint K arity)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count) :
    PiCCSOriginalMatrixBatch.Batch :=
  if ∀ source, zeroSource source = true then
    PiDECEvaluationBatch.zero (productionShape.sourceCount * matrixCount)
  else
    let prepared := PiDECPoseidonColumnWeights.prepare firstRow point interfaces
    let bySource := Vector.ofFn fun source : Fin productionShape.sourceCount =>
      if zeroSource source then PiDECEvaluationBatch.zero matrixCount
      else PiDECPoseidonColumnWeights.evaluate prepared (read source)
    Vector.ofFn fun code =>
      let pair : Fin productionShape.sourceCount × Fin matrixCount := Fin.decodeProd code
      (bySource.get pair.1).get pair.2

end NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixSupported
