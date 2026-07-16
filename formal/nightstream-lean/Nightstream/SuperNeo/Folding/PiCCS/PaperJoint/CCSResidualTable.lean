import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-!
Concrete CCS residual tables for the paper-level joint `Pi_CCS` model.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: construction of the uncompressed CCS block of `F(X, C)`.
Constraint family: one fresh-source CCS zero-set obligation per Boolean row.

Owns: an explicit sparse constraint polynomial, its evaluation over shared
finite matrix images, the CCS residual at each Boolean row, and construction
of the CCS family on the shared canonical Boolean domain.

Does not own: commitments, public-input projection, norm constraints, carried
evaluation constraints, the base-field-to-extension-field embedding,
`eq(X, A)`, gamma mixing, the signed `T - sum Q` identity, SumCheck,
Fiat--Shamir, production row order, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: matrices, assignments, and sparse monomials are explicit
paper-level data. Matrix-vector multiplication is imported from the sole
shared `PaperLinearAlgebra` owner and polynomial evaluation is derived here.
No caller supplies a CCS truth proposition, evaluator,
per-leaf equivalence, circuit witness, or implementation trace. Boolean-domain
order is imported from `BooleanDomain`; external implementation numbering is
still an explicit refinement boundary. This slice uses one algebraic carrier;
proving that base-field CCS residuals embed into the paper's extension-field
joint polynomial without changing zero is a separate, still-open refinement.

| Code owner | Paper object | Derived mathematical value | Proven result |
|---|---|---|---|
| imported `BooleanVertex` / `BooleanTable.tabulate` | `x in {0,1}^log(m)` | shared recursive low/high order | no CCS-local permutation |
| imported `PaperLinearAlgebra.matrixVectorAt` | `M_j`, `(M_j z)(x)` | shared finite dot product over assignment columns | no evaluator supplied |
| `ConstraintPolynomial` / `evaluatePolynomial` | sparse `f` in Definition 11 | finite sum of explicit monomials | arity is typed; degree bound is carried with terms |
| `residualAt` | `f((M_1z)(x), ..., (M_tz)(x))` | one CCS zero-set residual | exact table leaf |
| `residualTable` | one CCS alpha table | low branch then high branch | leaves equal canonical residual enumeration |
| `residualTable_allEntriesZero_iff_constraintSatisfied` | Item 1 of Lemma 7 for one fresh source | all Boolean residuals vanish | unconditional model-level equivalence |
| `residualPolynomial_coefficientZero_iff_constraintSatisfied` | alpha-coefficient form of the same item | canonical interpolation polynomial is zero | requires only interpolation zero laws |
| `allResidualTablesZero_iff_allConstraintsSatisfied` | all `K` fresh CCS sources | pointwise family truth | no caller-selected iff |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable

universe uField

open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

/-- One explicit sparse monomial in the paper's `t` matrix-image variables.
The dependent exponent function makes the variable arity intrinsic. -/
structure Monomial (Field : Type uField) (matrixCount : Nat) where
  coefficient : Field
  exponents : Fin matrixCount -> Nat

namespace Monomial

/-- Total degree derived from the explicit finite exponent vector. -/
def totalDegree
    {Field : Type uField}
    {matrixCount : Nat}
    (monomial : Monomial Field matrixCount) : Nat :=
  ((canonicalFinIndices matrixCount).map monomial.exponents).sum

end Monomial

/-- Definition 11's finite sparse polynomial `f in F^{<u}[X_1,...,X_t]`.
The degree condition is attached to explicit monomial data; there is no
function-valued polynomial evaluator or caller-declared degree oracle. -/
structure ConstraintPolynomial (Field : Type uField) (matrixCount : Nat) where
  degreeBound : Nat
  terms : List (Monomial Field matrixCount)
  termsBelowDegree :
    ∀ term, term ∈ terms -> term.totalDegree < degreeBound

/-- Paper-level Definition 11 data at the dimensions owned by `Shape`.
The common structure is shared by every fresh source in one batch. -/
structure Structure
    (Field : Type uField)
    (shape : Shape)
    (columns : Nat) where
  matrices : Fin shape.matrixCount ->
    BooleanMatrix Field shape.cubeVariables columns
  constraintPolynomial : ConstraintPolynomial Field shape.matrixCount

/-- Exponentiation derived from the same multiplication and unit used by all
other paper-level arithmetic in this slice. -/
def pow
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (value : Field) : Nat -> Field
  | 0 => ops.one
  | exponent + 1 => ops.mul (pow ops value exponent) value

/-- Evaluate one explicit monomial at the finite matrix-image vector. -/
def evaluateMonomial
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {matrixCount : Nat}
    (monomial : Monomial Field matrixCount)
    (point : Fin matrixCount -> Field) : Field :=
  (canonicalFinIndices matrixCount).foldl
    (fun accumulated index =>
      ops.mul accumulated
        (pow ops (point index) (monomial.exponents index)))
    monomial.coefficient

/-- Evaluate Definition 11's explicit sparse constraint polynomial. -/
def evaluatePolynomial
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {matrixCount : Nat}
    (polynomial : ConstraintPolynomial Field matrixCount)
    (point : Fin matrixCount -> Field) : Field :=
  polynomial.terms.foldl
    (fun accumulated monomial =>
      ops.add accumulated (evaluateMonomial ops monomial point))
    ops.zero

/-- The matrix-image vector `((M_1 z)(x), ..., (M_t z)(x))` at one Boolean
row. All arithmetic is derived from explicit structure and assignment data. -/
def matrixImagesAt
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns)
    (vertex : BooleanVertex shape.cubeVariables) :
    Fin shape.matrixCount -> Field :=
  fun matrix =>
    matrixVectorAt ops (system.matrices matrix) assignment vertex

/-- One paper CCS residual
`f((M_1 z)(x), ..., (M_t z)(x))` at a Boolean row. -/
def residualAt
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns)
    (vertex : BooleanVertex shape.cubeVariables) : Field :=
  evaluatePolynomial ops system.constraintPolynomial
    (matrixImagesAt ops system assignment vertex)

/-- Independent CCS zero-set obligation from Definition 12 / Lemma 7 Item 1.
It is defined directly over explicit matrices, assignment, and polynomial,
before any Boolean table or alpha interpolation is constructed. -/
def ConstraintSatisfied
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns) : Prop :=
  ∀ vertex, residualAt ops system assignment vertex = ops.zero

/-- Canonical residual table for one fresh CCS source. -/
def residualTable
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns) :
    BooleanTable Field shape.cubeVariables :=
  BooleanTable.tabulate (residualAt ops system assignment)

/-- The table exposes the exact CCS formula at every canonical Boolean leaf.
This theorem is the reviewable leaf-order bridge internal to the paper model. -/
theorem residualTable_entries_eq
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns) :
    (residualTable ops system assignment).entries =
      (BooleanVertex.all shape.cubeVariables).map
        (residualAt ops system assignment) := by
  exact BooleanTable.entries_tabulate _

/-- The constructed CCS table is leafwise zero exactly when the independent
paper CCS zero-set obligation holds. No equivalence is supplied by a caller. -/
theorem residualTable_allEntriesZero_iff_constraintSatisfied
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns) :
    (residualTable ops system assignment).AllEntriesZero ops ↔
      ConstraintSatisfied ops system assignment := by
  exact BooleanTable.tabulate_allEntriesZero_iff ops _

/-- The canonical alpha polynomial derived from the concrete CCS table is
coefficient-zero exactly when the independent CCS obligation holds. This
closes the CCS branch of table residualization, but not its placement in the
paper's signed joint `Q` identity. -/
theorem residualPolynomial_coefficientZero_iff_constraintSatisfied
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationZeroLaws ops)
    {shape : Shape}
    {columns : Nat}
    (system : Structure Field shape columns)
    (assignment : Assignment Field columns) :
    AlphaPolynomial.CoefficientZero ops.toOps
        ((residualTable ops system assignment).toAlphaPolynomial ops) ↔
      ConstraintSatisfied ops system assignment := by
  exact Iff.trans
    (Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero
        ops laws (residualTable ops system assignment))
    (residualTable_allEntriesZero_iff_constraintSatisfied
      ops system assignment)

/-- One common paper structure plus all `K` fresh assignments. Family arity
comes from `Shape.freshCount`; no caller-supplied list length can drift. -/
structure FreshBatch
    (Field : Type uField)
    (shape : Shape)
    (columns : Nat) where
  system : Structure Field shape columns
  assignments : Fin shape.freshCount -> Assignment Field columns

namespace FreshBatch

/-- The exact CCS table family expected by `TableResidualData.ccs`. -/
def residualTables
    {Field : Type uField}
    {shape : Shape}
    {columns : Nat}
    (ops : InterpolationOps Field)
    (batch : FreshBatch Field shape columns) :
    Fin shape.freshCount -> BooleanTable Field shape.cubeVariables :=
  fun source => residualTable ops batch.system (batch.assignments source)

/-- Every fresh source satisfies the independent paper CCS constraint family. -/
def AllConstraintsSatisfied
    {Field : Type uField}
    {shape : Shape}
    {columns : Nat}
    (ops : InterpolationOps Field)
    (batch : FreshBatch Field shape columns) : Prop :=
  ∀ source, ConstraintSatisfied ops batch.system (batch.assignments source)

/-- All constructed CCS tables are zero iff all `K` independently defined CCS
obligations hold. This is the protocol -> phase -> family composition theorem
for the CCS branch; norm and carried-evaluation branches remain separate. -/
theorem allResidualTablesZero_iff_allConstraintsSatisfied
    {Field : Type uField}
    {shape : Shape}
    {columns : Nat}
    (ops : InterpolationOps Field)
    (batch : FreshBatch Field shape columns) :
    (∀ source, (batch.residualTables ops source).AllEntriesZero ops) ↔
      batch.AllConstraintsSatisfied ops := by
  constructor
  · intro allZero source
    exact (residualTable_allEntriesZero_iff_constraintSatisfied
      ops batch.system (batch.assignments source)).mp (allZero source)
  · intro allSatisfied source
    exact (residualTable_allEntriesZero_iff_constraintSatisfied
      ops batch.system (batch.assignments source)).mpr
        (allSatisfied source)

end FreshBatch

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
