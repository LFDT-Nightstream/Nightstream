import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanEvaluation
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81MatrixSource

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
