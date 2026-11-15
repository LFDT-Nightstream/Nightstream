import NightstreamFPrime.Spec.RingRelation.Types
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.RingEvaluation

/-! The matrix evaluations `y_j` of SuperNeo v1.2 Definition 21.
`Lifecycle.PaperAlgebra.padEvaluation` is the Pad evaluation `y`. -/

namespace NightstreamFPrime.Spec.RingRelation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def matrixEvaluation {shape : Shape}
    (system : Structure shape)
    (assignment : Assignment shape)
    (point : Point shape)
    (matrix : Fin shape.matrixCount) : Evaluation :=
  fun lane =>
    PaperJoint.RingEvaluation.evaluate system.matrixSource assignment point
      matrix lane

def evaluations {shape : Shape}
    (system : Structure shape)
    (assignment : Assignment shape)
    (point : Point shape) : Array Evaluation :=
  Array.ofFn fun matrix => matrixEvaluation system assignment point matrix

@[simp] theorem evaluations_size {shape : Shape}
    (system : Structure shape)
    (assignment : Assignment shape)
    (point : Point shape) :
    (evaluations system assignment point).size = shape.matrixCount := by
  simp [evaluations]

@[simp] theorem evaluations_get {shape : Shape}
    (system : Structure shape)
    (assignment : Assignment shape)
    (point : Point shape)
    (matrix : Fin shape.matrixCount) :
    (evaluations system assignment point)[matrix.val]'(by
      simpa only [evaluations, Array.size_ofFn] using matrix.isLt) =
      matrixEvaluation system assignment point matrix := by
  simp [evaluations]

end NightstreamFPrime.Spec.RingRelation
