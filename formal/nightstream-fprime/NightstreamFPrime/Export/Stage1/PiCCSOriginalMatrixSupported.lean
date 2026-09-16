import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixBatch

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

/-- The zero branch precedes all sparse source arithmetic. Each nonzero source
uses the existing single-source range; flattening preserves every port slot. -/
def sparse {columns arity count : Nat}
    (zeroSource : Fin productionShape.sourceCount → Bool)
    (firstRow : Nat) (point : CubePoint K arity)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count) : PiCCSOriginalMatrixBatch.Batch :=
  let bySource := Vector.ofFn fun source : Fin productionShape.sourceCount =>
    if zeroSource source then PiDECEvaluationBatch.zero matrixCount
    else PiDECMatrixSparseRange.sum firstRow point (read source) forms
  Vector.ofFn fun code =>
    let pair : Fin productionShape.sourceCount × Fin matrixCount := Fin.decodeProd code
    (bySource.get pair.1).get pair.2

/-- The zero branch precedes numeric invocation preparation and weighting.
The stored numeric evaluator and complete output shape remain unchanged. -/
def invocations {columns arity count : Nat}
    (zeroSource : Fin productionShape.sourceCount → Bool)
    (firstRow : Nat) (point : CubePoint K arity)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count) : PiCCSOriginalMatrixBatch.Batch :=
  let bySource := Vector.ofFn fun source : Fin productionShape.sourceCount =>
    if zeroSource source then PiDECEvaluationBatch.zero matrixCount
    else PiDECMatrixInvocationRange.sum firstRow point (read source) interfaces
  Vector.ofFn fun code =>
    let pair : Fin productionShape.sourceCount × Fin matrixCount := Fin.decodeProd code
    (bySource.get pair.1).get pair.2

end NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixSupported
