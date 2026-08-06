import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetConvention

/-!
Table-level residual construction for the paper-level `Pi_CCS` model.

Owns: canonically indexed CCS and norm Boolean tables, canonically indexed
carried scalar residuals, their serialization into the joint coefficient
model, independently defined leafwise obligations, and the resulting
`ResidualizationBoundary`.

Does not own: concrete CCS formulas, the norm range polynomial or field
no-wrap proof, carried target/evaluation formulas, the signed joint identity,
literal Lemma 7, SumCheck truth, transcript semantics, relation refinement,
Rust, R1CS, constraint removal, or production approval.

Emits constraints: no.

Authority boundary: callers supply explicit finite residual tables and one
scalar for each typed carried coordinate. They do not supply polynomial
evaluators, degree claims, bases, list lengths, or per-leaf equivalences. The
carried scalar is formula-agnostic here; a later concrete construction must
use the selected orientation `T_local - sum Eval_local` and separately prove
the signed joint identity.

| Residual family | Canonical source | Serialization order | Proven obligation |
|---|---|---|---|
| CCS | `Fin K -> BooleanTable ell` | fresh index increasing | every table leaf is zero |
| norm | `Fin (K+k) -> BooleanTable ell` | source index increasing | every table leaf is zero |
| carried evaluation | `CarriedCoordinate -> Field` | coefficient, matrix, running; running fastest | every scalar is zero |

For zero-based `(running, matrix, coefficient)`, the carried serialization is
coordinated with
`I = running + k * matrix + k * t * coefficient`.
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField uIndex uLeft uRight

private theorem map_flatMap_apply
    {Index : Type uIndex}
    {Value : Type uLeft}
    {Result : Type uRight}
    (mapValue : Value -> Result)
    (values : Index -> List Value) :
    forall indices : List Index,
      (indices.flatMap values).map mapValue =
        indices.flatMap fun index => (values index).map mapValue
  | [] => rfl
  | index :: indices => by
      simp [map_flatMap_apply mapValue values indices]

/-- Explicit table data before alpha/gamma compression. Indexed functions
make family sizes and carried-coordinate ownership verifier-shaped rather than
caller-supplied list metadata. -/
structure TableResidualData
    (Field : Type uField)
    (shape : Shape) where
  ccs : Fin shape.freshCount -> BooleanTable Field shape.cubeVariables
  norm : Fin shape.sourceCount -> BooleanTable Field shape.cubeVariables
  carriedEvaluation : CarriedCoordinate shape -> Field

namespace TableResidualData

/-- Carried scalars serialized in the exact coordinate order above. -/
def orderedCarriedEvaluation
    {Field : Type uField}
    {shape : Shape}
    (data : TableResidualData Field shape) : List Field :=
  (canonicalCarriedCoordinates shape).map data.carriedEvaluation

/-- Mechanical expansion of the carried serialization order. Together with
`CarriedCoordinate.localGammaExponent`, this fixes position order to
`I = running + k * matrix + k * t * coefficient`; no caller list can choose a
different permutation. -/
theorem orderedCarriedEvaluation_eq_formulaOrder
    {Field : Type uField}
    {shape : Shape}
    (data : TableResidualData Field shape) :
    data.orderedCarriedEvaluation =
      (canonicalFinIndices shape.coefficientCount).flatMap fun coefficient =>
        (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
          (canonicalFinIndices shape.runningCount).map fun running =>
            data.carriedEvaluation
              { running := running
                matrix := matrix
                coefficient := coefficient } := by
  simp [orderedCarriedEvaluation, canonicalCarriedCoordinates,
    map_flatMap_apply, Function.comp_def]

/-- Serialization cannot omit or insert a carried scalar. -/
theorem orderedCarriedEvaluation_length
    {Field : Type uField}
    {shape : Shape}
    (data : TableResidualData Field shape) :
    data.orderedCarriedEvaluation.length = shape.carriedEvaluationCount := by
  simp [orderedCarriedEvaluation, canonicalCarriedCoordinates_length]

/-- Canonical coefficient residuals derived from explicit table data. -/
def toResiduals
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : TableResidualData Field shape) :
    Residuals Field shape (canonicalAlphaBasis shape) where
  ccs := (canonicalFinIndices shape.freshCount).map fun index =>
    (data.ccs index).toAlphaPolynomial ops
  ccsCount := by simp [canonicalFinIndices]
  norm := (canonicalFinIndices shape.sourceCount).map fun index =>
    (data.norm index).toAlphaPolynomial ops
  normCount := by simp [canonicalFinIndices]
  carriedEvaluation := data.orderedCarriedEvaluation
  carriedEvaluationCount := data.orderedCarriedEvaluation_length

/-- Independently defined table obligations. These propositions inspect only
the input table leaves/scalars, never the derived polynomial coefficients. -/
def toTableObligations
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : TableResidualData Field shape) : Obligations shape where
  ccs := (canonicalFinIndices shape.freshCount).map fun index =>
    (data.ccs index).AllEntriesZero ops
  ccsCount := by simp [canonicalFinIndices]
  norm := (canonicalFinIndices shape.sourceCount).map fun index =>
    (data.norm index).AllEntriesZero ops
  normCount := by simp [canonicalFinIndices]
  carriedEvaluation := (canonicalCarriedCoordinates shape).map fun coordinate =>
    data.carriedEvaluation coordinate = ops.zero
  carriedEvaluationCount := by
    simp [canonicalCarriedCoordinates_length]

private theorem aligned_map_same_source
    {Index : Type uIndex}
    {Left : Type uLeft}
    {Right : Type uRight}
    (relation : Left -> Right -> Prop)
    (left : Index -> Left)
    (right : Index -> Right)
    (exact : forall index, relation (left index) (right index)) :
    forall indices : List Index,
      Aligned relation (indices.map left) (indices.map right)
  | [] => .nil
  | index :: indices =>
      .cons (left index) (right index) (indices.map left) (indices.map right)
        (exact index) (aligned_map_same_source relation left right exact indices)

/-- The arbitrary per-leaf iff boundary is closed at the explicit table
layer. This proves only table residualization; it does not construct the
tables from CCS, norm, or carried-evaluation semantics. -/
theorem residualizationBoundary
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationZeroLaws ops)
    (data : TableResidualData Field shape) :
    ResidualizationBoundary ops.toOps
      (data.toResiduals ops) (data.toTableObligations ops) := by
  constructor
  · exact aligned_map_same_source
      (fun residual obligation =>
        residual.CoefficientZero ops.toOps ↔ obligation)
      (fun index => (data.ccs index).toAlphaPolynomial ops)
      (fun index => (data.ccs index).AllEntriesZero ops)
      (fun index =>
        BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero
          ops laws (data.ccs index))
      (canonicalFinIndices shape.freshCount)
  · exact aligned_map_same_source
      (fun residual obligation =>
        residual.CoefficientZero ops.toOps ↔ obligation)
      (fun index => (data.norm index).toAlphaPolynomial ops)
      (fun index => (data.norm index).AllEntriesZero ops)
      (fun index =>
        BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero
          ops laws (data.norm index))
      (canonicalFinIndices shape.sourceCount)
  · exact aligned_map_same_source
      (fun residual obligation => residual = ops.zero ↔ obligation)
      data.carriedEvaluation
      (fun coordinate => data.carriedEvaluation coordinate = ops.zero)
      (fun _ => Iff.rfl)
      (canonicalCarriedCoordinates shape)

/-- Joint coefficient truth is equivalent to the explicit table obligations,
without any caller-supplied evaluator, basis, degree, or per-leaf iff. This is
still a table-level theorem, not concrete Lemma 7. -/
theorem coefficientTruth_iff_tableObligations
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (laws : InterpolationZeroLaws ops)
    (data : TableResidualData Field shape) :
    (data.toResiduals ops).CoefficientTruth ops.toOps ↔
      (data.toTableObligations ops).AllHold :=
  Residuals.coefficientTruth_iff_allObligations ops.toOps
    (data.toResiduals ops) (data.toTableObligations ops)
    (data.residualizationBoundary ops laws)

end TableResidualData

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
