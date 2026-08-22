import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81Evaluation

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/Types.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Typed, batch-invariant carriers for the paper Phi81 CCS/CE relation.

Protocol: SuperNeo Definitions 11--13 specialized to the Phi81 carrier.
Phase: verifier-owned relation shape and sole matrix source.
Constraint family: none; this file emits no rows.

Owns: dimensions that remain fixed across bootstrap and active folds; exact
typed assignment, public-input, point, and evaluation carriers; one original
matrix/polynomial structure; canonical completion to the Phi81 carrier; and
the sole derived coefficient-matrix source.

Does not own: fresh/running batch counts, relation membership, commitments,
Ajtai keys, norm predicates, NIFS acceptance, transcript derivation, Rust,
R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: the public width is constructed as exactly
`54 * publicRingColumns`. A raw 257-field prefix is therefore not a value of
the intended fixed public carrier. The original matrices are supplied once at
`logicalWidth`; completed columns and every coefficient matrix are definitions.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| coefficient embedding | total carrier | logical / complete width | complete width is a whole number of 54-lane blocks |
| CCS/CE opening | public carrier | ring-aligned prefix | public width is exactly `54 * publicRingColumns` and fits the complete assignment |
| CCS/CE opening | private carrier | assignment | every complete coordinate has one typed owner |
| carried CE | point / evaluation | row cube / Phi81 ring | point arity and 54 output coefficients are intrinsic types |
| structure | coefficient source | original matrix / derived images | every coefficient matrix comes from `Phi81MatrixSource.source` |
-/

namespace NightstreamFPrime.Spec.Phi81Relation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

/-- Relation dimensions that do not depend on one fold's fresh/running arity.

`logicalWidth` is the sole caller-supplied matrix width. Assignments live at
its completed Phi81 carrier width so running CE openings retain every folded
coordinate. -/
structure Shape where
  rowVariables : Nat
  logicalWidth : Nat
  matrixCount : Nat
  publicRingColumns : Nat
  publicFits :
    ringDegree * publicRingColumns <=
      Phi81CarrierLayout.carrierWidth logicalWidth

namespace Shape

/-- Exact complete coefficient-carrier width. -/
def carrierWidth (shape : Shape) : Nat :=
  Phi81CarrierLayout.carrierWidth shape.logicalWidth

/-- Public field width, aligned by construction to complete Phi81 rings. -/
def publicWidth (shape : Shape) : Nat :=
  ringDegree * shape.publicRingColumns

/-- Batch-free paper shape used only by the canonical matrix source. -/
def sourceShape (shape : Shape) :
    NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Shape :=
  Phi81MatrixSource.phi81Shape shape.rowVariables 0 0 shape.matrixCount

/-- The typed public carrier always consists of whole Phi81 ring columns. -/
theorem publicWidth_aligned (shape : Shape) :
    ringDegree ∣ shape.publicWidth := by
  exact ⟨shape.publicRingColumns, rfl⟩

/-- The legacy 257-field prefix cannot be the typed paper public carrier:
every accepted width is an exact multiple of the 54-coefficient ring. -/
theorem publicWidth_ne_257 (shape : Shape) :
    shape.publicWidth ≠ 257 := by
  simp only [publicWidth, ringDegree]
  omega

/-- A public coordinate embeds into the complete assignment prefix without a
default or truncating read. -/
def publicColumn (shape : Shape)
    (column : Fin shape.publicWidth) : Fin shape.carrierWidth :=
  ⟨column.val, Nat.lt_of_lt_of_le column.isLt shape.publicFits⟩

@[simp] theorem publicColumn_val (shape : Shape)
    (column : Fin shape.publicWidth) :
    (shape.publicColumn column).val = column.val := by
  rfl

end Shape

/-- One complete Phi81 assignment. Its width cannot drift from the shape. -/
abbrev Assignment (shape : Shape) :=
  PaperLinearAlgebra.Assignment F shape.carrierWidth

/-- The paper input projection represented in exact coefficient order. The
width is definitionally a multiple of 54. -/
abbrev PublicInput (shape : Shape) :=
  PaperLinearAlgebra.Assignment F shape.publicWidth

/-- One dimension-checked CE row point. -/
abbrev Point (shape : Shape) := CubePoint K shape.rowVariables

/-- One paper CE matrix evaluation in `K[X]/Phi81`. -/
abbrev Evaluation := RingK

/-- Canonical typed projection of the aligned public prefix. -/
def projectPublicInput {shape : Shape}
    (assignment : Assignment shape) : PublicInput shape :=
  fun column => assignment (shape.publicColumn column)

/-- One paper structure owner. There is no caller field for a completed
matrix or coefficient-expanded matrix. -/
structure Structure (shape : Shape) where
  matrices : Fin shape.matrixCount ->
    BooleanMatrix F shape.rowVariables shape.logicalWidth
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial F shape.matrixCount

namespace Structure

/-- Complete and coefficient-expand the sole original matrix family using
the independently defined Phi81 kernel. Dummy batch counts are zero because
they are not relation dimensions and do not affect the matrix source. -/
def matrixSource {shape : Shape}
    (system : Structure shape) :
    MatrixCoefficientSource.MatrixSource F shape.sourceShape shape.carrierWidth
      (Phi81ColumnLayout.blockCount shape.carrierWidth) :=
  Phi81MatrixSource.source shape.rowVariables 0 0 shape.matrixCount
    shape.logicalWidth system.matrices system.constraintPolynomial

/-- The relation structure cannot substitute a caller-selected coefficient
kernel. -/
@[simp] theorem matrixSource_kernel_eq {shape : Shape}
    (system : Structure shape) :
    system.matrixSource.kernel = Phi81CoefficientKernel.phi81Kernel := by
  rfl

end Structure

end NightstreamFPrime.Spec.Phi81Relation
