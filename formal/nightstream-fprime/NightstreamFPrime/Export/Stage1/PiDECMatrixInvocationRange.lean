import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocation

/-!
Sum a contiguous ordered vector of loaded Poseidon interfaces. Each interface
contributes its existing 94-row invocation sum at the corresponding global
row offset. All partial and final arithmetic stays in the existing Lean
extension-ring batch operations.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixInvocationRange

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

/-- Compute each loaded interface once in invocation order for one child.
The Nat branch outside the vector is never visited by the counted sum. -/
def sum {columns arity count : Nat} (firstRow : Nat)
    (point : CubePoint K arity) (read : Fin ringDegree → Fin columns → F)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count) :
    Vector MaterializedRingK matrixCount :=
  PiDECEvaluationBatch.sum count fun index =>
    if live : index < count then
      PiDECMatrixInvocation.sum (firstRow + 94 * index) point
        (PiDECMatrixInvocation.prepare read (interfaces.get ⟨index, live⟩))
    else PiDECEvaluationBatch.zero matrixCount

end NightstreamFPrime.Export.Stage1.PiDECMatrixInvocationRange
