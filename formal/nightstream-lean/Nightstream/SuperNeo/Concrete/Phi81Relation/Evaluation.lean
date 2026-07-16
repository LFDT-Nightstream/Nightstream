import Nightstream.SuperNeo.Concrete.Phi81Relation.Types
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.Semantics

/-!
Paper Phi81 CE evaluation packing over the batch-invariant relation carrier.

Protocol: SuperNeo Definition 13 and `Pi_CCS` output claims.
Phase: one matrix evaluation and the complete matrix-indexed CE array.
Constraint family: semantic evaluation only; this file emits no rows.

Owns: the relation-level `RingK` value for one matrix; canonical array order;
exact array size/index theorems; and the adapter proving that every
fresh/running `OutputClaims.canonicalYRing` coordinate is the same
batch-invariant relation evaluation.

Does not own: relation membership, commitments, public-input authority,
`yZcol`, transcript derivation, PiRLC/PiDEC homomorphism, Rust, R1CS, or
constraint counts.

Emits constraints: no.

Authority boundary: every scalar leaf delegates to
`PaperJoint.Phi81Evaluation.evaluate`. `Structure.ofSourceData` drops only
batch counts; it preserves the sole matrices and constraint polynomial.
Neither array packing nor the adapter introduces another evaluation formula.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| carried CE | one matrix | 54 coefficient lanes | `matrixEvaluation` is the sole Phi81 leaf at every lane |
| carried CE | output product | matrix order | `evaluations` uses canonical `Fin matrixCount` order |
| assurance | array shape | size / indexed read | every declared matrix occurs exactly once |
| `Pi_CCS` handoff | source adapter | fresh and running counts | `canonicalYRing` equals the same relation leaf independently of batch arity |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-- One paper CE matrix evaluation. Its 54 coefficients are all derived from
the canonical matrix source and complete assignment. -/
def matrixEvaluation {shape : Shape}
    (system : Structure shape)
    (assignment : Assignment shape)
    (point : Point shape)
    (matrix : Fin shape.matrixCount) : Evaluation :=
  fun lane =>
    PaperJoint.Phi81Evaluation.evaluate system.matrixSource assignment point
      matrix lane

/-- Complete CE evaluation array in canonical matrix-index order. -/
def evaluations {shape : Shape}
    (system : Structure shape)
    (assignment : Assignment shape)
    (point : Point shape) : Array Evaluation :=
  Array.ofFn fun matrix => matrixEvaluation system assignment point matrix

/-- The CE array has exactly one value for every structure matrix. -/
@[simp] theorem evaluations_size {shape : Shape}
    (system : Structure shape)
    (assignment : Assignment shape)
    (point : Point shape) :
    (evaluations system assignment point).size = shape.matrixCount := by
  simp [evaluations]

/-- Reading the canonical array at a typed matrix index returns that matrix's
sole relation evaluation. -/
@[simp] theorem evaluations_get {shape : Shape}
    (system : Structure shape)
    (assignment : Assignment shape)
    (point : Point shape)
    (matrix : Fin shape.matrixCount) :
    (evaluations system assignment point)[matrix.val]'(by
      simpa only [evaluations, Array.size_ofFn] using matrix.isLt) =
      matrixEvaluation system assignment point matrix := by
  simp [evaluations]

/-! ## Batch-shaped `Pi_CCS` adapter -/

/-- Forget one `Pi_CCS` batch's fresh/running counts while retaining all
dimensions that define the persistent CCS/CE relation. -/
def Shape.ofSemantic
    (batchShape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= batchShape.carrierWidth) : Shape where
  rowVariables := batchShape.rowVariables
  logicalWidth := batchShape.logicalWidth
  matrixCount := batchShape.matrixCount
  publicRingColumns := publicRingColumns
  publicFits := publicFits

/-- Project the sole matrix and polynomial owners from one semantic source
batch into the batch-invariant relation structure. -/
def Structure.ofSourceData
    {batchShape : SemanticShape}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= batchShape.carrierWidth)
    (data : SplitNc.Sources.Data batchShape) :
    Structure (Shape.ofSemantic batchShape publicRingColumns publicFits) where
  matrices := data.matrices
  constraintPolynomial := data.constraintPolynomial

/-- Lane-level adapter: fresh/running source counts do not change the derived
Phi81 coefficient image. -/
theorem matrixEvaluation_apply_ofSourceData
    {batchShape : SemanticShape}
    {domain : FlatNcDomain}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= batchShape.carrierWidth)
    (data : SplitNc.Sources.Data batchShape)
    (points : VerifierPoints batchShape domain)
    (source : Fin batchShape.sourceCount)
    (matrix : Fin batchShape.matrixCount)
    (lane : Fin ringDegree) :
    matrixEvaluation
        (Structure.ofSourceData publicRingColumns publicFits data)
        (data.assignment source) points.rPrime matrix lane =
      canonicalYRing data points source matrix lane := by
  rfl

/-- Ring-level form of the same adapter. -/
theorem matrixEvaluation_ofSourceData
    {batchShape : SemanticShape}
    {domain : FlatNcDomain}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= batchShape.carrierWidth)
    (data : SplitNc.Sources.Data batchShape)
    (points : VerifierPoints batchShape domain)
    (source : Fin batchShape.sourceCount)
    (matrix : Fin batchShape.matrixCount) :
    matrixEvaluation
        (Structure.ofSourceData publicRingColumns publicFits data)
        (data.assignment source) points.rPrime matrix =
      fun lane => canonicalYRing data points source matrix lane := by
  funext lane
  exact matrixEvaluation_apply_ofSourceData publicRingColumns publicFits data
    points source matrix lane

/-- Array-indexed handoff theorem used by a later concrete `CE.Holds`
instantiation. -/
theorem evaluations_get_ofSourceData
    {batchShape : SemanticShape}
    {domain : FlatNcDomain}
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= batchShape.carrierWidth)
    (data : SplitNc.Sources.Data batchShape)
    (points : VerifierPoints batchShape domain)
    (source : Fin batchShape.sourceCount)
    (matrix : Fin batchShape.matrixCount) :
    (evaluations
      (Structure.ofSourceData publicRingColumns publicFits data)
      (data.assignment source) points.rPrime)[matrix.val]'(by
        simpa only [evaluations, Array.size_ofFn] using matrix.isLt) =
      fun lane => canonicalYRing data points source matrix lane := by
  rw [evaluations_get]
  exact matrixEvaluation_ofSourceData publicRingColumns publicFits data points
    source matrix

end Nightstream.SuperNeo.Concrete.Phi81Relation
