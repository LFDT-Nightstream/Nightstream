import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe

/-!
Affine source leaves for the Split-NC FE degree proof.

Owns: row-coordinate affinity of source-derived `yRing`, lane-coordinate
affinity of padded numeric selectors, and the two affine projections of the
zero-padded 54-lane MLE.

Does not own: sparse CCS substitution, equality-gated branches, complete FE
degree bounds, SumCheck rounds, transcripts, Rust, R1CS, rows, or costs.

Emits constraints: no.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe.degree.source.row` | one source-derived `yRing` row slice is affine | derived | `sourceYRingAt_row_affine` |
| `nifs.pi_ccs.fe.degree.source.lane_weight` | one padded numeric lane weight is affine | derived | `laneWeight_affine` |
| `nifs.pi_ccs.fe.degree.source.padded.row` | fixed-lane mix of affine row values is affine | derived | `paddedLaneEvaluation_row_affine` |
| `nifs.pi_ccs.fe.degree.source.padded.lane` | fixed-row padded lane MLE is affine | derived | `paddedLaneEvaluation_lane_affine` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree.Source

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport

/-- One source-derived coefficient image is affine in every row coordinate. -/
theorem sourceYRingAt_row_affine
    {shape : SemanticShape}
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount)
    (lane : Fin ringDegree)
    (before after : List K)
    (length : before.length + 1 + after.length = shape.rowVariables) :
    Represents 1 fun point =>
      sourceYRingAt data (cubeSlice before after length point)
        source matrix lane := by
  unfold sourceYRingAt yRingForAssignment yRingForMatrixSource
    Phi81Evaluation.evaluate BooleanTable.evaluate
  exact evaluateCoordinates_affine
    (Phi81Evaluation.table data.matrixSource (data.assignment source)
      matrix lane)
    before after length

/-- One padded numeric lane selector is affine in every lane coordinate. -/
theorem laneWeight_affine
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (lane : Fin ringDegree)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    Represents 1 fun point =>
      NumericBooleanDomain.tensorWeight ops
        (liveLane covers lane)
        (cubeSlice before after length point) := by
  let vertex := NumericBooleanDomain.vertex domain.laneVariables
    (liveLane covers lane)
  have betaLength :
      (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex).length =
        domain.laneVariables :=
    SumCheckTruthPath.VertexEncoding.fieldCoordinates_length ops vertex
  rcases pointEqualityCoordinates_right_affine
    (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex)
    before after (by rw [betaLength]; exact length) with
    ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  calc
    polynomial.evaluate ops.toOps point =
        SumCheckTruthPath.pointEqualityCoordinates ops
          (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex)
          (before ++ point :: after) := represents point
    _ = SumCheckTruthPath.pointEquality ops
        (SumCheckTruthPath.VertexEncoding.toCubePoint ops vertex)
        (cubeSlice before after length point) := rfl
    _ = BooleanVertex.equalityWeight ops vertex
        (cubeSlice before after length point) :=
      SumCheckTruthPath.pointEquality_toCubePoint_eq_equalityWeight
        ops laws vertex (cubeSlice before after length point)
    _ = NumericBooleanDomain.tensorWeight ops
        (liveLane covers lane)
        (cubeSlice before after length point) := by
      exact (NumericBooleanDomain.tensorWeight_eq_equalityWeight ops
        (liveLane covers lane)
        (cubeSlice before after length point)).symm

/-- Fixing a lane point leaves the padded output evaluation affine in every
row coordinate. -/
theorem paddedLaneEvaluation_row_affine
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (data : Data shape)
    (source : Fin shape.sourceCount)
    (matrix : Fin shape.matrixCount)
    (lanePoint : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = shape.rowVariables) :
    Represents 1 fun point =>
      paddedLaneEvaluation covers
        (sourceYRingAt data (cubeSlice before after length point)
          source matrix)
        lanePoint := by
  rcases polynomial_sum_exists
    (canonicalFinIndices ringDegree)
    (fun lane => NumericBooleanDomain.tensorWeight ops
      (liveLane covers lane) lanePoint)
    (fun lane point =>
      sourceYRingAt data (cubeSlice before after length point)
        source matrix lane)
    (by
      intro lane _
      exact sourceYRingAt_row_affine data source matrix lane
        before after length) with
    ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  rfl

/-- Fixing a row leaves the zero-padded 54-lane MLE affine in every lane
coordinate. -/
theorem paddedLaneEvaluation_lane_affine
    {domain : FlatNcDomain}
    (covers : LaneCovers domain)
    (values : Fin ringDegree -> K)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    Represents 1 fun point =>
      paddedLaneEvaluation covers values
        (cubeSlice before after length point) := by
  rcases polynomial_sum_exists
    (canonicalFinIndices ringDegree)
    values
    (fun lane point => NumericBooleanDomain.tensorWeight ops
      (liveLane covers lane) (cubeSlice before after length point))
    (by
      intro lane _
      exact laneWeight_affine covers lane before after length) with
    ⟨polynomial, represents⟩
  refine ⟨polynomial, ?_⟩
  intro point
  rw [represents]
  calc
    FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane => ops.mul (values lane)
          (NumericBooleanDomain.tensorWeight ops (liveLane covers lane)
            (cubeSlice before after length point))) =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices ringDegree)
        (fun lane => ops.mul
          (NumericBooleanDomain.tensorWeight ops (liveLane covers lane)
            (cubeSlice before after length point))
          (values lane)) := by
        apply FiniteSumAlgebra.sumMap_congr
        intro lane _
        exact laws.mul_comm _ _
    _ = paddedLaneEvaluation covers values
        (cubeSlice before after length point) := rfl

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Fe.Degree.Source
