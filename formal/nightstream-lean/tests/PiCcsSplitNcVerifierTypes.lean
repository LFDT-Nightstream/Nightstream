import Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-!
Regression for the Split-NC verifier-visible input boundary.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.verify.input` | independently chosen matrices and assignments erase to the same public input | hidden semantic sources become verifier authority |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Matrices and assignments are not fields of the verifier-visible input.

The two rich source records deliberately receive independent hidden source
families. Only the three public families are shared. If `PublicInput.ofSources`
ever begins carrying hidden semantic witnesses, this equality stops reducing
to `PublicInput.ofSources_eq`. -/
example
    {shape : SemanticShape}
    (leftMatrices rightMatrices : Fin shape.matrixCount ->
      BooleanMatrix F shape.rowVariables shape.logicalWidth)
    (constraintPolynomial :
      CCSResidualTable.ConstraintPolynomial F shape.matrixCount)
    (leftFresh rightFresh :
      Fin shape.freshCount -> Assignment F shape.logicalWidth)
    (leftRunning rightRunning :
      Fin shape.runningCount -> Assignment F shape.carrierWidth)
    (priorPoint : CubePoint K shape.rowVariables)
    (claimedCoefficient : CarriedCoordinate shape.paperShape -> K) :
    PublicInput.ofSources
        ({ matrices := leftMatrices
           constraintPolynomial := constraintPolynomial
           freshAssignments := leftFresh
           runningAssignments := leftRunning
           priorPoint := priorPoint
           claimedCoefficient := claimedCoefficient } : Data shape) =
      PublicInput.ofSources
        ({ matrices := rightMatrices
           constraintPolynomial := constraintPolynomial
           freshAssignments := rightFresh
           runningAssignments := rightRunning
           priorPoint := priorPoint
           claimedCoefficient := claimedCoefficient } : Data shape) := by
  apply PublicInput.ofSources_eq <;> rfl

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Tests
