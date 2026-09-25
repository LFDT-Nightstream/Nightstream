import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanTable
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/TableResiduals.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Table-level residual construction for the paper-level `Pi_CCS` model.

Owns: canonically indexed Pad, matrix, CCS, and norm residuals, their v1.1
serialization, independent leafwise obligations, and the resulting boundary.

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
| Pad evaluation | `PadCoordinate -> Field` | coefficient, running | every scalar is zero |
| matrix evaluation | `MatrixCoordinate -> Field` | coefficient, matrix, running | every scalar is zero |

For zero-based `(running, matrix, coefficient)`, the carried serialization is
coordinated with
`I = running + k * matrix + k * t * coefficient`.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

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
  padEvaluation : PadCoordinate shape -> Field
  matrixEvaluation : MatrixCoordinate shape -> Field

namespace TableResidualData

/-- Pad scalars serialized in exact `I_K` order. -/
def orderedPadEvaluation
    {Field : Type uField}
    {shape : Shape}
    (data : TableResidualData Field shape) : List Field :=
  (canonicalPadCoordinates shape).map data.padEvaluation

/-- Matrix scalars serialized in exact `I_A` order. -/
def orderedMatrixEvaluation
    {Field : Type uField}
    {shape : Shape}
    (data : TableResidualData Field shape) : List Field :=
  (canonicalMatrixCoordinates shape).map data.matrixEvaluation

theorem orderedPadEvaluation_length
    {Field : Type uField}
    {shape : Shape}
    (data : TableResidualData Field shape) :
    data.orderedPadEvaluation.length = shape.padEvaluationCount := by
  simp [orderedPadEvaluation, canonicalPadCoordinates_length]

/-- Mechanical expansion of the matrix serialization order. -/
theorem orderedMatrixEvaluation_eq_formulaOrder
    {Field : Type uField}
    {shape : Shape}
    (data : TableResidualData Field shape) :
    data.orderedMatrixEvaluation =
      (canonicalFinIndices shape.coefficientCount).flatMap fun coefficient =>
        (canonicalFinIndices shape.matrixCount).flatMap fun matrix =>
          (canonicalFinIndices shape.runningCount).map fun running =>
            data.matrixEvaluation
              { running := running
                matrix := matrix
                coefficient := coefficient } := by
  simp [orderedMatrixEvaluation, canonicalMatrixCoordinates,
    map_flatMap_apply, Function.comp_def]

/-- Matrix serialization cannot omit or insert a scalar. -/
theorem orderedMatrixEvaluation_length
    {Field : Type uField}
    {shape : Shape}
    (data : TableResidualData Field shape) :
    data.orderedMatrixEvaluation.length = shape.matrixEvaluationCount := by
  simp [orderedMatrixEvaluation, canonicalMatrixCoordinates_length]

/-- Canonical coefficient residuals derived from explicit table data. -/
def toResiduals
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : TableResidualData Field shape) :
    Residuals Field shape (canonicalAlphaBasis shape) where
  padEvaluation := data.orderedPadEvaluation
  padEvaluationCount := data.orderedPadEvaluation_length
  matrixEvaluation := data.orderedMatrixEvaluation
  matrixEvaluationCount := data.orderedMatrixEvaluation_length
  ccs := (canonicalFinIndices shape.freshCount).map fun index =>
    (data.ccs index).toAlphaPolynomial ops
  ccsCount := by simp [canonicalFinIndices]
  norm := (canonicalFinIndices shape.sourceCount).map fun index =>
    (data.norm index).toAlphaPolynomial ops
  normCount := by simp [canonicalFinIndices]

/-- Independently defined table obligations. These propositions inspect only
the input table leaves/scalars, never the derived polynomial coefficients. -/
def toTableObligations
    {Field : Type uField}
    {shape : Shape}
    (ops : InterpolationOps Field)
    (data : TableResidualData Field shape) : Obligations shape where
  padEvaluation := (canonicalPadCoordinates shape).map fun coordinate =>
    data.padEvaluation coordinate = ops.zero
  padEvaluationCount := by simp [canonicalPadCoordinates_length]
  matrixEvaluation := (canonicalMatrixCoordinates shape).map fun coordinate =>
    data.matrixEvaluation coordinate = ops.zero
  matrixEvaluationCount := by simp [canonicalMatrixCoordinates_length]
  ccs := (canonicalFinIndices shape.freshCount).map fun index =>
    (data.ccs index).AllEntriesZero ops
  ccsCount := by simp [canonicalFinIndices]
  norm := (canonicalFinIndices shape.sourceCount).map fun index =>
    (data.norm index).AllEntriesZero ops
  normCount := by simp [canonicalFinIndices]

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
      (fun residual obligation => residual = ops.zero ↔ obligation)
      data.padEvaluation
      (fun coordinate => data.padEvaluation coordinate = ops.zero)
      (fun _ => Iff.rfl)
      (canonicalPadCoordinates shape)
  · exact aligned_map_same_source
      (fun residual obligation => residual = ops.zero ↔ obligation)
      data.matrixEvaluation
      (fun coordinate => data.matrixEvaluation coordinate = ops.zero)
      (fun _ => Iff.rfl)
      (canonicalMatrixCoordinates shape)
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

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
