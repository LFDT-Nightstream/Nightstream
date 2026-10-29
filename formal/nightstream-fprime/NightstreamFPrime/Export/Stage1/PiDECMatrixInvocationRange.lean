import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocation

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixInvocationRange

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

@[specialize] def sum {columns arity count : Nat} (firstRow : Nat)
    (point : CubePoint K arity) (read : Fin ringDegree → Fin columns → F)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count) :
    Vector MaterializedRingK matrixCount :=
  PiDECEvaluationBatch.sum count fun index =>
    if live : index < count then
      PiDECMatrixInvocation.sum (firstRow + 150 * index) point
        (PiDECMatrixInvocation.prepare read (interfaces.get ⟨index, live⟩))
    else PiDECEvaluationBatch.zero matrixCount

end NightstreamFPrime.Export.Stage1.PiDECMatrixInvocationRange
