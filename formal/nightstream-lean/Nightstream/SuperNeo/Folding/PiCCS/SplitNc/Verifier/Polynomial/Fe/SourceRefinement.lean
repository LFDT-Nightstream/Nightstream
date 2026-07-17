import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Fe
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe

/-!
Source refinement for the production-shaped Split-NC FE polynomial.

Protocol: SuperNeo `Pi_CCS`, FE branch.
Phase: authoritative source images to verifier-visible `yRing` values.
Constraint family: fresh constant-coefficient CCS images and running carried
coefficient evaluations only; this file emits no rows.

Owns: equality between the FE polynomial's source-derived `yRing` view and
the two independently stated FE source obligations: completed fresh CCS
matrix images and running carried-evaluation coefficients.

Does not own: the FE polynomial formula, transcript challenges, SumCheck,
Fiat--Shamir, output-message checking, Phi81 kernel correctness, Rust, R1CS,
row emission, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: this file introduces no evaluator or claim. Fresh values
are reduced to `Data.freshBatch` through the proved Phi81 constant-coefficient
matrix law. Running values are reduced to `Data.carriedData` through the
canonical image-table evaluation theorem. `Semantics.Fe.CarriedTruth` then
binds the public running claim family to that same source-derived view.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.source.fresh.constant.table` | constant-lane Phi81 table equals the completed CCS matrix-image table | derived | `freshConstantTable_eq_completedMatrixImageTable` |
| `nifs.pi_ccs.fe.source.fresh.constant.eval` | fresh constant-lane `sourceYRingAt` evaluates that CCS table | derived | `sourceYRingAt_fresh_constant_eq_completedMatrixImage` |
| `nifs.pi_ccs.fe.source.running.eval` | running `sourceYRingAt` at `priorPoint` equals `computedCoefficient` | derived | `sourceYRingAt_running_eq_computedCoefficient` |
| `nifs.pi_ccs.fe.input.running.binding` | carried FE truth binds every public prior claim to source-derived `yRing` | derived | `claimedYRing_eq_sourceYRingAt_of_carriedTruth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe

/-- The completed original-matrix image consumed by the fresh CCS relation,
lifted pointwise into the FE extension carrier. -/
def completedMatrixImageTable
    {shape : SemanticShape}
    (data : Data shape)
    (fresh : Fin shape.freshCount)
    (matrix : Fin shape.matrixCount) : BooleanTable K shape.rowVariables :=
  BooleanTable.tabulate fun vertex =>
    K.embed <| CCSResidualTable.matrixImagesAt ConcreteCarrier.baseOps
      data.freshBatch.system (data.freshBatch.assignments fresh) vertex matrix

/-- At table level, the source-derived constant Phi81 lane is exactly the
completed original CCS matrix image for the same fresh assignment. -/
theorem freshConstantTable_eq_completedMatrixImageTable
    {shape : SemanticShape}
    (data : Data shape)
    (fresh : Fin shape.freshCount)
    (matrix : Fin shape.matrixCount) :
    yRingTableForAssignment data
        (data.assignment (Data.freshIndex fresh)) matrix
        Phi81CoefficientKernel.constant =
      completedMatrixImageTable data fresh matrix := by
  rw [data.assignment_freshIndex]
  unfold yRingTableForAssignment yRingTableForMatrixSource
    Phi81Evaluation.table completedMatrixImageTable
  apply congrArg BooleanTable.tabulate
  funext vertex
  apply congrArg K.embed
  unfold CCSResidualTable.matrixImagesAt
  change
    PaperLinearAlgebra.matrixVectorAt ConcreteCarrier.baseOps
        (data.matrixSource.coefficientMatrix ConcreteCarrier.baseOps matrix
          Phi81CoefficientKernel.constant)
        (data.freshAssignment fresh) vertex =
      PaperLinearAlgebra.matrixVectorAt ConcreteCarrier.baseOps
        (data.matrixSource.matrices matrix)
        (data.freshAssignment fresh) vertex
  congr 1
  funext row column
  exact Phi81MatrixSource.coefficientMatrix_constant_apply
    shape.rowVariables shape.freshCount shape.runningCount shape.matrixCount
    shape.logicalWidth data.matrices data.constraintPolynomial matrix row column

/-- Evaluating the fresh constant-lane source view at any verifier-owned row
point is evaluating the completed CCS matrix-image table at that point. -/
theorem sourceYRingAt_fresh_constant_eq_completedMatrixImage
    {shape : SemanticShape}
    (data : Data shape)
    (row : CubePoint K shape.rowVariables)
    (fresh : Fin shape.freshCount)
    (matrix : Fin shape.matrixCount) :
    sourceYRingAt data row (Data.freshIndex fresh) matrix
        Phi81CoefficientKernel.constant =
      (completedMatrixImageTable data fresh matrix).evaluate
        ConcreteCarrier.extensionOps row := by
  unfold sourceYRingAt yRingForAssignment yRingForMatrixSource
    Phi81Evaluation.evaluate
  change
    (yRingTableForAssignment data
        (data.assignment (Data.freshIndex fresh)) matrix
        Phi81CoefficientKernel.constant).evaluate
        ConcreteCarrier.extensionOps row = _
  rw [freshConstantTable_eq_completedMatrixImageTable data fresh matrix]

/-- A running source-derived `yRing` value at the statement's prior point is
the independently defined explicit equality-weighted carried coefficient. -/
theorem sourceYRingAt_running_eq_computedCoefficient
    {shape : SemanticShape}
    (data : Data shape)
    (running : Fin shape.runningCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree) :
    sourceYRingAt data data.priorPoint (Data.runningIndex running) matrix lane =
      CarriedEvaluationResidual.computedCoefficient
        ConcreteCarrier.baseOps ConcreteCarrier.extensionOps K.embed
        data.carriedData
        { running := running, matrix := matrix, coefficient := lane } := by
  calc
    sourceYRingAt data data.priorPoint (Data.runningIndex running) matrix lane =
        (CarriedEvaluationResidual.imageTable ConcreteCarrier.baseOps K.embed
          data.carriedData
          { running := running, matrix := matrix, coefficient := lane }).evaluate
          ConcreteCarrier.extensionOps data.carriedData.priorPoint := by
      unfold sourceYRingAt yRingForAssignment yRingForMatrixSource
        Phi81Evaluation.evaluate Phi81Evaluation.table
        CarriedEvaluationResidual.imageTable
        CarriedEvaluationResidual.imageCoefficientAt
      rw [data.assignment_runningIndex]
      rfl
    _ = CarriedEvaluationResidual.computedCoefficient
          ConcreteCarrier.baseOps ConcreteCarrier.extensionOps K.embed
          data.carriedData
          { running := running, matrix := matrix, coefficient := lane } :=
      CarriedEvaluationResidual.imageTable_evaluate_eq_computedCoefficient
        ConcreteCarrier.baseOps ConcreteCarrier.extensionOps
        ConcreteCarrier.extensionLaws K.embed data.carriedData
        { running := running, matrix := matrix, coefficient := lane }

/-- Semantic carried truth binds the complete verifier-visible prior claim
family to the source-derived running `yRing` view at `data.priorPoint`. -/
theorem claimedYRing_eq_sourceYRingAt_of_carriedTruth
    {shape : SemanticShape}
    (data : Data shape)
    (truth : Semantics.Fe.CarriedTruth data) :
    (PublicInput.ofSources data).claimedYRing =
      fun running matrix lane =>
        sourceYRingAt data data.priorPoint (Data.runningIndex running)
          matrix lane := by
  funext running matrix lane
  change data.claimedCoefficient
      { running := running, matrix := matrix, coefficient := lane } = _
  calc
    data.claimedCoefficient
        { running := running, matrix := matrix, coefficient := lane } =
        CarriedEvaluationResidual.computedCoefficient
          ConcreteCarrier.baseOps ConcreteCarrier.extensionOps K.embed
          data.carriedData
          { running := running, matrix := matrix, coefficient := lane } :=
      truth { running := running, matrix := matrix, coefficient := lane }
    _ = sourceYRingAt data data.priorPoint (Data.runningIndex running)
          matrix lane :=
      (sourceYRingAt_running_eq_computedCoefficient
        data running matrix lane).symm

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.SourceRefinement
