import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteJointData

/-!
One authoritative assignment family for every paper joint `Pi_CCS`
obligation.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: source ownership before CCS, norm, and carried-evaluation residuals.
Constraint family: semantic witness connectivity only; this file emits no
rows.

Owns: the paper's square-domain bijection between its Boolean cube and
assignment columns, including the exact consequence `columns = 2^variables`;
one typed family of the exact `K+k` source assignments; canonical
fresh/running source injections; and derived CCS, norm, and carried-evaluation
views that provably read the same `z_i` values.

Does not own: commitment/public-input binding, proof that coefficient-expanded
carried matrices come from the structure matrices, extension-field lifting,
the nonlinear protocol polynomial, Fiat--Shamir, Rust, R1CS, or constraint
counts.

Emits constraints: no.

Authority boundary: callers provide assignments once. They cannot separately
choose the norm values or the carried-evaluation assignments. `ColumnLayout`
is required to be bijective, so a norm check over Boolean vertices covers
every authoritative assignment column exactly once. The older
`ConcreteJointData.IndependentInputs` remains only as an internal derived view
for reusing already-proved family lemmas.

| Protocol | Phase | Family | Mathematical obligation |
|---|---|---|---|
| `Pi_CCS` | source domain | Boolean vertex / assignment column | `ColumnLayout` is a two-sided inverse and forces `columns = 2^variables` |
| `Pi_CCS` | source ownership | all `K+k` vectors | `UnifiedInputs.assignments` is authoritative |
| `Pi_CCS` | CCS | first `K` sources | `freshBatch` reads `freshSourceIndex i` |
| `Pi_CCS` | norm | all `K+k` sources | `normBatch` reads the same assignment through `ColumnLayout` |
| `Pi_CCS` | carried evaluation | final `k` sources | `carriedData` reads `runningSourceIndex i` |
| assurance | connectivity | norm coverage | `normBatch_allStrictNormBounded_iff_allAssignmentsStrictNormBounded` |
| assurance | semantic closure | all three families | `toIndependentInputs_semanticTruth_iff` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

universe uExtension

/-- Explicit two-sided correspondence for the paper assumption
`m = n_F = 2^ell`. Keeping both directions typed makes omission, duplication,
and independent row/column permutations visible refinement obligations. -/
structure ColumnLayout (variables columns : Nat) where
  toColumn : BooleanVertex variables -> Fin columns
  toVertex : Fin columns -> BooleanVertex variables
  toColumn_toVertex : forall column, toColumn (toVertex column) = column
  toVertex_toColumn : forall vertex, toVertex (toColumn vertex) = vertex

private theorem perm_of_nodup_and_same_members
    {Value : Type}
    [DecidableEq Value]
    {left right : List Value}
    (leftNodup : left.Nodup)
    (rightNodup : right.Nodup)
    (sameMembers : forall value, value ∈ left ↔ value ∈ right) :
    left.Perm right := by
  induction left generalizing right with
  | nil =>
      have rightNil : right = [] := by
        cases right with
        | nil => rfl
        | cons value tail =>
            have impossible : value ∈ ([] : List Value) :=
              (sameMembers value).mpr (by simp)
            simp at impossible
      subst right
      exact .refl []
  | cons head tail inductionHypothesis =>
      have leftParts := List.nodup_cons.mp leftNodup
      have headMemRight : head ∈ right :=
        (sameMembers head).mp (by simp)
      rcases List.mem_iff_append.mp headMemRight with
        ⟨before, after, rightEq⟩
      subst right
      have rightParts := List.nodup_append.mp rightNodup
      have headTailParts := List.nodup_cons.mp rightParts.2.1
      have headNotBefore : head ∉ before := by
        intro member
        exact rightParts.2.2 head member head (by simp) rfl
      have headNotRest : head ∉ before ++ after := by
        simp [headNotBefore, headTailParts.1]
      have restNodup : (before ++ after).Nodup := by
        apply List.nodup_append.mpr
        exact ⟨rightParts.1, headTailParts.2, fun left leftMem right rightMem =>
          rightParts.2.2 left leftMem right (by simp [rightMem])⟩
      have tailMembers : forall value,
          value ∈ tail ↔ value ∈ before ++ after := by
        intro value
        by_cases equal : value = head
        · subst value
          simp [leftParts.1, headNotRest]
        · simpa [equal] using sameMembers value
      have tailPerm : tail.Perm (before ++ after) :=
        inductionHypothesis leftParts.2 restNodup tailMembers
      exact (tailPerm.cons head).trans List.perm_middle.symm

namespace ColumnLayout

/-- Enumerate assignment columns by mapping the canonical Boolean-cube order
through the paper's square-domain layout. -/
def enumeratedColumns
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) : List (Fin columns) :=
  (BooleanVertex.all variables).map layout.toColumn

/-- The layout-derived column enumeration contains no duplicate column. -/
theorem enumeratedColumns_nodup
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    layout.enumeratedColumns.Nodup := by
  exact (BooleanVertex.all_nodup variables).map layout.toColumn (by
    intro left right different equal
    apply different
    calc
      left = layout.toVertex (layout.toColumn left) :=
        (layout.toVertex_toColumn left).symm
      _ = layout.toVertex (layout.toColumn right) := congrArg layout.toVertex equal
      _ = right := layout.toVertex_toColumn right)

/-- Every declared assignment column occurs in the layout-derived
enumeration. -/
theorem mem_enumeratedColumns
    {variables columns : Nat}
    (layout : ColumnLayout variables columns)
    (column : Fin columns) :
    column ∈ layout.enumeratedColumns := by
  apply List.mem_map.mpr
  exact ⟨layout.toVertex column, BooleanVertex.mem_all _,
    layout.toColumn_toVertex column⟩

/-- A `ColumnLayout` is not merely an ordering choice: its two-sided inverse
forces the assignment width to equal the cardinality of the Boolean row cube. -/
theorem columns_eq_twoPow
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    columns = 2 ^ variables := by
  have permutation :
      layout.enumeratedColumns.Perm (canonicalFinIndices columns) := by
    apply perm_of_nodup_and_same_members
    · exact layout.enumeratedColumns_nodup
    · exact canonicalFinIndices_nodup columns
    · intro column
      constructor
      · intro _
        simp [canonicalFinIndices]
      · intro _
        exact layout.mem_enumeratedColumns column
  have lengths := permutation.length_eq
  simp only [enumeratedColumns, List.length_map, BooleanVertex.all_length,
    canonicalFinIndices_length] at lengths
  exact lengths.symm

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
  coefficientMatrices :
    Fin shape.matrixCount -> Fin shape.coefficientCount ->
      BooleanMatrix F shape.cubeVariables columns
  priorPoint : CubePoint Extension shape.cubeVariables
  claimedCoefficient : CarriedCoordinate shape -> Extension

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

/-- Norm view of all `K+k` authoritative assignments through the sole
Boolean-vertex/column layout. -/
def normBatch
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) :
    NormResidualTable.SourceBatch shape where
  assignments := fun source vertex =>
    data.assignments source (data.layout.toColumn vertex)

/-- Carried-evaluation view of the final `k` authoritative assignments. -/
def carriedData
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) :
    CarriedEvaluationResidual.EvaluationData F Extension shape columns where
  priorPoint := data.priorPoint
  assignments := fun source =>
    data.assignments (runningSourceIndex source)
  coefficientMatrices := data.coefficientMatrices
  claimedCoefficient := data.claimedCoefficient

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
  carried := data.carriedData

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

/-- The norm view reads the authoritative source at the layout-selected
column. -/
theorem normBatch_assignment_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns)
    (source : Fin shape.sourceCount)
    (vertex : BooleanVertex shape.cubeVariables) :
    data.normBatch.assignments source vertex =
      data.assignments source (data.layout.toColumn vertex) := by
  rfl

/-- The carried view reads the authoritative running assignment verbatim. -/
theorem carriedData_assignment_eq
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns)
    (source : Fin shape.runningCount)
    (column : Fin columns) :
    data.carriedData.assignments source column =
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
  simp [normBatch, data.layout.toColumn_toVertex]

/-- Authoritative strict norm over every actual source column. -/
def AllAssignmentsStrictNormBounded
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) : Prop :=
  ∀ source column, centeredMagnitude (data.assignments source column) < 2

/-- The derived Boolean norm family is exact for the authoritative assignment
family because the layout is bijective. -/
theorem normBatch_allStrictNormBounded_iff_allAssignmentsStrictNormBounded
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) :
    data.normBatch.AllStrictNormBounded ↔
      data.AllAssignmentsStrictNormBounded := by
  constructor
  · intro bounded source column
    simpa [normBatch, data.layout.toColumn_toVertex] using
      bounded source (data.layout.toVertex column)
  · intro bounded source vertex
    exact bounded source (data.layout.toColumn vertex)

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
    CarriedEvaluationResidual.AllClaimsHold baseOps extensionOps lift
      data.carriedData

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
          CarriedEvaluationResidual.AllClaimsHold baseOps extensionOps lift
            data.carriedData ↔
      data.freshBatch.AllConstraintsSatisfied baseOps ∧
        data.AllAssignmentsStrictNormBounded ∧
          CarriedEvaluationResidual.AllClaimsHold baseOps extensionOps lift
            data.carriedData
  rw [normBatch_allStrictNormBounded_iff_allAssignmentsStrictNormBounded]

end UnifiedInputs

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources
