import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanEvaluation
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81MatrixSource

/-!
Batch-invariant Phi81 carried-evaluation leaves.

Protocol: SuperNeo coefficient embedding and `Pi_CCS` output CE claims.
Phase: derived coefficient-matrix image at one verifier-owned row point.
Constraint family: semantic evaluation only; this file emits no rows.

Owns: the single mathematical formula that turns a canonical Phi81 matrix
source, one complete-carrier assignment, one matrix, and one coefficient lane
into a Boolean row table and its multilinear evaluation.

Does not own: source-batch ordering, fresh/running arity, claimed output
products, `yZcol`, public inputs, commitments, transcript derivation, Rust,
R1CS, or constraint counts.

Emits constraints: no.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81Evaluation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

def table
    {cubeVariables freshCount runningCount matrixCount columns blockCount : Nat}
    (matrixSource : MatrixCoefficientSource.MatrixSource F
      (Phi81MatrixSource.phi81Shape cubeVariables freshCount runningCount
        matrixCount)
      columns blockCount)
    (assignment : Assignment F columns)
    (matrix : Fin matrixCount)
    (lane : Fin ringDegree) : BooleanTable K cubeVariables :=
  BooleanTable.tabulate fun vertex =>
    K.embed <| matrixVectorAt ConcreteCarrier.baseOps
      (matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix lane)
      assignment vertex

def evaluate
    {cubeVariables freshCount runningCount matrixCount columns blockCount : Nat}
    (matrixSource : MatrixCoefficientSource.MatrixSource F
      (Phi81MatrixSource.phi81Shape cubeVariables freshCount runningCount
        matrixCount)
      columns blockCount)
    (assignment : Assignment F columns)
    (point : CubePoint K cubeVariables)
    (matrix : Fin matrixCount)
    (lane : Fin ringDegree) : K :=
  (table matrixSource assignment matrix lane).evaluate
    ConcreteCarrier.extensionOps point

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81Evaluation
