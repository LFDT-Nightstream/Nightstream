import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteJointData

/-! Provenance: adapted from `formal/nightstream-lean/Nightstream/SuperNeo/Folding/PiCCS/PaperJoint/UnifiedSources.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; split into the
SuperNeo v1.1 Pad and 14-matrix evaluation families. -/

/-!
One authoritative assignment family for every SuperNeo v1.1 `Pi_CCS`
obligation.

Protocol: SuperNeo v1.1 `Pi_CCS` (Section 7.3 / Appendix B.2).
Phase: source ownership before CCS, norm, and carried-evaluation residuals.
Constraint family: semantic witness connectivity only; this file emits no
rows.

Owns: the paper's zero-padding injection from assignment columns into its
Boolean row cube, including the exact consequence `columns <= 2^variables`;
one typed family of the exact `K+k` source assignments; canonical
fresh/running source injections; and derived CCS, norm, Pad-evaluation, and
matrix-evaluation views that provably read the same `z_i` values.

Does not own: commitment/public-input binding, proof that coefficient-expanded
matrix coefficient expansions come from Pad and the structure matrices,
extension-field lifting,
the nonlinear protocol polynomial, Fiat--Shamir, Rust, R1CS, or constraint
counts.

Emits constraints: no.

Authority boundary: callers provide assignments once. They cannot separately
choose the norm values or either evaluation assignment family. `ColumnLayout`
is injective from columns to rows and identifies padding rows explicitly, so
a norm check over Boolean vertices covers every authoritative assignment
column and checks zero on every padding row. The older
`ConcreteJointData.IndependentInputs` remains only as an internal derived view
for reusing already-proved family lemmas.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | source domain | Boolean row / assignment column | `ColumnLayout` is an injection with explicit padding and forces `columns <= 2^variables` |
| `Pi_CCS` | source ownership | all `K+k` vectors | `UnifiedInputs.assignments` is authoritative |
| `Pi_CCS` | CCS | first `K` sources | `freshBatch` reads `freshSourceIndex i` |
| `Pi_CCS` | norm | all `K+k` sources | `normBatch` reads the same assignment through `ColumnLayout` |
| `Pi_CCS` | Pad evaluation | final `k` sources | `padData` reads `runningSourceIndex i` |
| `Pi_CCS` | matrix evaluation | final `k` sources | `matrixData` reads the same `runningSourceIndex i` |
| assurance | connectivity | norm coverage | `normBatch_allStrictNormBounded_iff_allAssignmentsStrictNormBounded` |
| assurance | semantic closure | all three families | `toIndependentInputs_semanticTruth_iff` |
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

universe uExtension

/-- Exact layout for the paper requirement `n_F <= m = 2^ell`.

Every assignment column owns one Boolean row. A row either decodes to that
unique column or is a padding row. This is the typed form of
`M_1 = [I; 0]`; it prevents omission and duplication without requiring the
assignment width to equal the row count. -/
structure ColumnLayout (variables columns : Nat) where
  columns_le : columns <= 2 ^ variables
  toVertex : Fin columns -> BooleanVertex variables
  toColumn? : BooleanVertex variables -> Option (Fin columns)
  toColumn_toVertex : forall column,
    toColumn? (toVertex column) = some column
  toVertex_toColumn : forall vertex column,
    toColumn? vertex = some column -> toVertex column = vertex

namespace ColumnLayout

/-- Embed the canonical assignment-column order into the Boolean row cube. -/
def enumeratedVertices
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    List (BooleanVertex variables) :=
  (canonicalFinIndices columns).map layout.toVertex

/-- Two assignment columns cannot own the same Boolean row. -/
theorem toVertex_injective
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    Function.Injective layout.toVertex := by
  intro left right equal
  have decoded := congrArg layout.toColumn? equal
  rw [layout.toColumn_toVertex left,
    layout.toColumn_toVertex right] at decoded
  exact Option.some.inj decoded

/-- The embedded column rows contain no duplicate Boolean vertex. -/
theorem enumeratedVertices_nodup
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    layout.enumeratedVertices.Nodup := by
  exact (canonicalFinIndices_nodup columns).map layout.toVertex (by
    intro left right different equal
    exact different (layout.toVertex_injective equal))

/-- Every embedded assignment row occurs in the canonical Boolean cube. -/
theorem enumeratedVertices_subset_all
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    layout.enumeratedVertices ⊆ BooleanVertex.all variables := by
  intro vertex _
  exact BooleanVertex.mem_all vertex

/-- The paper layout proves the required dimension inequality
`n_F <= m = 2^ell`. -/
theorem columns_le_twoPow
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    columns <= 2 ^ variables :=
  layout.columns_le

/-- Read an assignment through `M_1 = [I; 0]`: live rows return their unique
assignment coordinate and padding rows return the additive zero. -/
def paddedValue
    {Value : Type}
    {variables columns : Nat}
    (layout : ColumnLayout variables columns)
    (zero : Value)
    (assignment : Fin columns -> Value)
    (vertex : BooleanVertex variables) : Value :=
  match layout.toColumn? vertex with
  | some column => assignment column
  | none => zero

/-- One entry of the paper's padded first matrix `M_1 = [I; 0]`.
Live rows contain one canonical unit entry. Padding rows are zero rows. -/
def paddedIdentityEntry
    {Value : Type}
    {variables columns : Nat}
    (layout : ColumnLayout variables columns)
    (zero one : Value)
    (vertex : BooleanVertex variables)
    (column : Fin columns) : Value :=
  match layout.toColumn? vertex with
  | some selected => if column = selected then one else zero
  | none => zero

/-- Every authoritative assignment coordinate survives the padding
injection exactly. -/
@[simp] theorem paddedValue_toVertex
    {Value : Type}
    {variables columns : Nat}
    (layout : ColumnLayout variables columns)
    (zero : Value)
    (assignment : Fin columns -> Value)
    (column : Fin columns) :
    layout.paddedValue zero assignment (layout.toVertex column) =
      assignment column := by
  simp [paddedValue, layout.toColumn_toVertex]

/-- The row owned by `column` has the expected unit entry. -/
@[simp] theorem paddedIdentityEntry_toVertex
    {Value : Type}
    {variables columns : Nat}
    (layout : ColumnLayout variables columns)
    (zero one : Value)
    (vertexColumn column : Fin columns) :
    layout.paddedIdentityEntry zero one (layout.toVertex vertexColumn) column =
      if column = vertexColumn then one else zero := by
  simp [paddedIdentityEntry, layout.toColumn_toVertex]

end ColumnLayout

/-- Canonical injection of one of the first `K` sources into `K+k`. -/
def freshSourceIndex
    {shape : Shape}
    (source : Fin shape.freshCount) : Fin shape.sourceCount :=
  ⟨source.val, by
    simp only [Shape.sourceCount]
    omega⟩

/-- Canonical injection of one of the final `k` sources into `K+k`. -/
def runningSourceIndex
    {shape : Shape}
    (source : Fin shape.runningCount) : Fin shape.sourceCount :=
  ⟨shape.freshCount + source.val, by
    simp only [Shape.sourceCount]
    omega⟩

/-- The fresh and running injections cannot alias. -/
theorem freshSourceIndex_ne_runningSourceIndex
    {shape : Shape}
    (fresh : Fin shape.freshCount)
    (running : Fin shape.runningCount) :
    freshSourceIndex fresh ≠ runningSourceIndex running := by
  intro equal
  have values := congrArg Fin.val equal
  simp only [freshSourceIndex, runningSourceIndex] at values
  omega

/-- Every source is owned by exactly one canonical side of the `K+k`
partition. -/
theorem source_eq_fresh_or_running
    {shape : Shape}
    (source : Fin shape.sourceCount) :
    (∃ fresh, source = freshSourceIndex fresh) ∨
      ∃ running, source = runningSourceIndex running := by
  by_cases isFresh : source.val < shape.freshCount
  · left
    let fresh : Fin shape.freshCount := ⟨source.val, isFresh⟩
    exact ⟨fresh, Fin.eq_of_val_eq rfl⟩
  · right
    have sourceBound : source.val < shape.freshCount + shape.runningCount := by
      simpa only [Shape.sourceCount] using source.isLt
    let running : Fin shape.runningCount :=
      ⟨source.val - shape.freshCount, by omega⟩
    refine ⟨running, Fin.eq_of_val_eq ?_⟩
    simp only [runningSourceIndex, running]
    omega

/-- The paper-level input bundle with a single owner for every source vector.
The coefficient-expanded matrices remain explicit because their derivation
from ring matrices is a separate refinement theorem. -/
structure UnifiedInputs
    (Extension : Type uExtension)
    (shape : Shape)
    (columns : Nat) where
  layout : ColumnLayout shape.cubeVariables columns
  system : CCSResidualTable.Structure F shape columns
  assignments : Fin shape.sourceCount -> Assignment F columns
  padCoefficientMatrices : Fin shape.coefficientCount ->
    BooleanMatrix F shape.cubeVariables columns
  matrixCoefficientMatrices :
    Fin shape.matrixCount -> Fin shape.coefficientCount ->
      BooleanMatrix F shape.cubeVariables columns
  priorPoint : CubePoint Extension shape.cubeVariables
  claimedPadCoefficient : PadCoordinate shape -> Extension
  claimedMatrixCoefficient : MatrixCoordinate shape -> Extension

namespace UnifiedInputs

/-- CCS view of the first `K` authoritative assignments. -/
def freshBatch
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) :
    CCSResidualTable.FreshBatch F shape columns where
  system := data.system
  assignments := fun source => data.assignments (freshSourceIndex source)

/-- Norm view of all `K+k` authoritative assignments through `M_1 = [I; 0]`.
Padding rows contain the canonical field zero. -/
def normBatch
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) :
    NormResidualTable.SourceBatch shape where
  assignments := fun source vertex =>
    data.layout.paddedValue 0 (data.assignments source) vertex

/-- Pad-evaluation view of the final `k` authoritative assignments. -/
def padData
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) :
    PadEvaluationResidual.EvaluationData F Extension shape columns where
  priorPoint := data.priorPoint
  assignments := fun source =>
    data.assignments (runningSourceIndex source)
  coefficientMatrices := data.padCoefficientMatrices
  claimedCoefficient := data.claimedPadCoefficient

/-- CCS-matrix evaluation view of the same final `k` assignments and prior
point. -/
def matrixData
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) :
    MatrixEvaluationResidual.EvaluationData F Extension shape columns where
  priorPoint := data.priorPoint
  assignments := fun source =>
    data.assignments (runningSourceIndex source)
  coefficientMatrices := data.matrixCoefficientMatrices
  claimedCoefficient := data.claimedMatrixCoefficient

/-- Internal projection used only to reuse the independently proved residual
family theorems. No caller can construct its three assignment views
independently through this API. -/
def toIndependentInputs
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) :
    ConcreteJointData.IndependentInputs Extension shape columns where
  ccs := data.freshBatch
  norm := data.normBatch
  priorPoint := data.priorPoint
  runningAssignments := fun source =>
    data.assignments (runningSourceIndex source)
  padCoefficientMatrices := data.padCoefficientMatrices
  matrixCoefficientMatrices := data.matrixCoefficientMatrices
  claimedPadCoefficient := data.claimedPadCoefficient
  claimedMatrixCoefficient := data.claimedMatrixCoefficient

/-- The CCS view reads the authoritative fresh assignment verbatim. -/
theorem freshBatch_assignment_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns)
    (source : Fin shape.freshCount)
    (column : Fin columns) :
    data.freshBatch.assignments source column =
      data.assignments (freshSourceIndex source) column := by
  rfl

/-- The norm view is exactly the assignment after the paper's zero-padding
injection. -/
theorem normBatch_assignment_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns)
    (source : Fin shape.sourceCount)
    (vertex : BooleanVertex shape.cubeVariables) :
    data.normBatch.assignments source vertex =
      data.layout.paddedValue 0 (data.assignments source) vertex := by
  rfl

/-- The Pad view reads the authoritative running assignment verbatim. -/
theorem padData_assignment_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns)
    (source : Fin shape.runningCount)
    (column : Fin columns) :
    data.padData.assignments source column =
      data.assignments (runningSourceIndex source) column := by
  rfl

/-- The matrix view reads the same authoritative running assignment
verbatim. -/
theorem matrixData_assignment_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns)
    (source : Fin shape.runningCount)
    (column : Fin columns) :
    data.matrixData.assignments source column =
      data.assignments (runningSourceIndex source) column := by
  rfl

/-- Looking up a norm value at the vertex corresponding to a column recovers
that exact authoritative column; no assignment coordinate can evade the norm
family. -/
theorem normBatch_at_toVertex_eq_assignment
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns)
    (source : Fin shape.sourceCount)
    (column : Fin columns) :
    data.normBatch.assignments source (data.layout.toVertex column) =
      data.assignments source column := by
  simp [normBatch]

/-- Authoritative strict norm over every actual source column. -/
def AllAssignmentsStrictNormBounded
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) : Prop :=
  ∀ source column, centeredMagnitude (data.assignments source column) < 2

/-- The Boolean norm family is exact for the authoritative assignment family.
Every live column has one row, and every padding row contains zero. -/
theorem normBatch_allStrictNormBounded_iff_allAssignmentsStrictNormBounded
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) :
    data.normBatch.AllStrictNormBounded ↔
      data.AllAssignmentsStrictNormBounded := by
  constructor
  · intro bounded source column
    simpa [normBatch] using
      bounded source (data.layout.toVertex column)
  · intro bounded source vertex
    cases decoded : data.layout.toColumn? vertex with
    | none =>
        change centeredMagnitude
          (data.layout.paddedValue 0 (data.assignments source) vertex) < 2
        simp [ColumnLayout.paddedValue, decoded, centeredMagnitude]
    | some column =>
        simpa [normBatch, ColumnLayout.paddedValue, decoded] using
          bounded source column

/-- Independent semantic truth over the one authoritative source family. -/
def SemanticTruth
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns)
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension) : Prop :=
  data.freshBatch.AllConstraintsSatisfied baseOps ∧
    data.AllAssignmentsStrictNormBounded ∧
    PadEvaluationResidual.AllClaimsHold baseOps extensionOps lift data.padData ∧
    MatrixEvaluationResidual.AllClaimsHold baseOps extensionOps lift
      data.matrixData

/-- Projecting to the old independent residual-family bundle preserves
exactly the stronger unified semantic truth; this theorem is the only intended
entry into those reusable family lemmas. -/
theorem toIndependentInputs_semanticTruth_iff
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns)
    (baseOps : InterpolationOps F)
    (extensionOps : InterpolationOps Extension)
    (lift : F -> Extension) :
    data.toIndependentInputs.SemanticTruth baseOps extensionOps lift ↔
      data.SemanticTruth baseOps extensionOps lift := by
  unfold ConcreteJointData.IndependentInputs.SemanticTruth SemanticTruth
  change
    data.freshBatch.AllConstraintsSatisfied baseOps ∧
        data.normBatch.AllStrictNormBounded ∧
          PadEvaluationResidual.AllClaimsHold baseOps extensionOps lift
            data.padData ∧
          MatrixEvaluationResidual.AllClaimsHold baseOps extensionOps lift
            data.matrixData ↔
      data.freshBatch.AllConstraintsSatisfied baseOps ∧
        data.AllAssignmentsStrictNormBounded ∧
          PadEvaluationResidual.AllClaimsHold baseOps extensionOps lift
            data.padData ∧
          MatrixEvaluationResidual.AllClaimsHold baseOps extensionOps lift
            data.matrixData
  rw [normBatch_allStrictNormBounded_iff_allAssignmentsStrictNormBounded]

end UnifiedInputs

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
