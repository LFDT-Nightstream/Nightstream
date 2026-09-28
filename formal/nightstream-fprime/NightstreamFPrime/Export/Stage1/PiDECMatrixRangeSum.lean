import NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch
import NightstreamFPrime.Spec.ProductionRelation

/-!
Add matrix range results in their existing child/matrix/ring order. The
arithmetic remains the existing PiDECEvaluationBatch addition. This module
owns no file format, point selection, or source-row correctness premise.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixRangeSum

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

abbrev Values := Vector (Vector MaterializedRingK matrixCount) productionGlobalParams.k

def zero : Values :=
  Vector.replicate productionGlobalParams.k (PiDECEvaluationBatch.zero matrixCount)

def add (left right : Values) : Values :=
  Vector.ofFn fun child => PiDECEvaluationBatch.add (left.get child) (right.get child)

private theorem get_ofFn {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = _
  rw [Vector.getElem_ofFn]

/-- The initial accumulator has zero in every complete matrix ring. -/
theorem zero_value (child : Fin productionGlobalParams.k) (port : Fin matrixCount) :
    (((zero).get child).get port).toRing = ringKZero := by
  change (((Vector.replicate productionGlobalParams.k
    (PiDECEvaluationBatch.zero matrixCount))[child.val]).get port).toRing = _
  rw [Vector.getElem_replicate, PiDECEvaluationBatch.zero_value]

/-- Every child/port projection is the existing complete extension-ring sum. -/
theorem add_value (left right : Values) (child : Fin productionGlobalParams.k)
    (port : Fin matrixCount) :
    (((add left right).get child).get port).toRing =
      ringKAdd ((left.get child).get port).toRing ((right.get child).get port).toRing := by
  rw [add, get_ofFn, PiDECEvaluationBatch.add_value]

/-- Adjacent batches at the same point compose to the exact combined row
range. This is a structural statement for arbitrary row producers. -/
theorem add_ranges {arity : Nat} (start leftCount rightCount : Nat)
    (point : CubePoint K arity)
    (rows : Fin productionGlobalParams.k → Nat → Vector StoredRing matrixCount)
    (child : Fin productionGlobalParams.k) (port : Fin matrixCount) :
    (((add
      (Vector.ofFn fun selected => PiDECEvaluationBatch.range start leftCount point (rows selected))
      (Vector.ofFn fun selected => PiDECEvaluationBatch.range (start + leftCount)
        rightCount point (rows selected))).get child).get port).toRing =
      ((PiDECEvaluationBatch.range start (leftCount + rightCount) point
        (rows child)).get port).toRing := by
  simp only [add, get_ofFn]
  exact (PiDECEvaluationBatch.range_append start leftCount rightCount point (rows child) port).symm

end NightstreamFPrime.Export.Stage1.PiDECMatrixRangeSum
