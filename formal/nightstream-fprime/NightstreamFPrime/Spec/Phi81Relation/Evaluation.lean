import NightstreamFPrime.Spec.Phi81Relation.Types

/-!
Paper Phi81 CE evaluation packing over the batch-invariant relation carrier.

Protocol: SuperNeo v1.2 Definition 21.

Owns: the relation-level `RingK` value for one matrix, canonical array order,
and exact array size/index theorems.

Does not own: relation membership, commitments, public-input authority,
`yZcol`, transcript derivation, PiRLC/PiDEC homomorphism, Rust, R1CS, or
constraint counts.
-/

namespace NightstreamFPrime.Spec.Phi81Relation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def matrixEvaluation {shape : Shape}
    (system : Structure shape)
    (assignment : Assignment shape)
    (point : Point shape)
    (matrix : Fin shape.matrixCount) : Evaluation :=
  fun lane =>
    PaperJoint.Phi81Evaluation.evaluate system.matrixSource assignment point
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

end NightstreamFPrime.Spec.Phi81Relation
