import NightstreamFPrime.Spec.Phi81Relation.Types

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/Evaluation.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Paper Phi81 CE evaluation packing over the batch-invariant relation carrier.

Protocol: SuperNeo Definition 13.
Phase: one matrix evaluation and the complete matrix-indexed CE array.
Constraint family: semantic evaluation only; this file emits no rows.

Owns: the relation-level `RingK` value for one matrix, canonical array order,
and exact array size/index theorems.

Does not own: relation membership, commitments, public-input authority,
`yZcol`, transcript derivation, PiRLC/PiDEC homomorphism, Rust, R1CS, or
constraint counts.

Emits constraints: no.

Authority boundary: every scalar leaf delegates to
`PaperJoint.Phi81Evaluation.evaluate`. Array packing does not introduce
another evaluation formula.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| carried CE | one matrix | 54 coefficient lanes | `matrixEvaluation` is the sole Phi81 leaf at every lane |
| carried CE | output product | matrix order | `evaluations` uses canonical `Fin matrixCount` order |
| assurance | array shape | size / indexed read | every declared matrix occurs exactly once |
-/

namespace NightstreamFPrime.Spec.Phi81Relation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

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

end NightstreamFPrime.Spec.Phi81Relation
