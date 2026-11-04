import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteJointData

/-! SuperNeo v1.2 Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PaperLinearAlgebra

universe uExtension

structure ColumnLayout (variables columns : Nat) where
  columns_le : columns <= 2 ^ variables
  toVertex : Fin columns -> BooleanVertex variables
  toColumn? : BooleanVertex variables -> Option (Fin columns)
  toColumn_toVertex : forall column,
    toColumn? (toVertex column) = some column
  toVertex_toColumn : forall vertex column,
    toColumn? vertex = some column -> toVertex column = vertex

namespace ColumnLayout

def enumeratedVertices
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    List (BooleanVertex variables) :=
  (canonicalFinIndices columns).map layout.toVertex

theorem toVertex_injective
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    Function.Injective layout.toVertex := by
  intro left right equal
  have decoded := congrArg layout.toColumn? equal
  rw [layout.toColumn_toVertex left,
    layout.toColumn_toVertex right] at decoded
  exact Option.some.inj decoded

theorem enumeratedVertices_nodup
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    layout.enumeratedVertices.Nodup := by
  exact (canonicalFinIndices_nodup columns).map layout.toVertex (by
    intro left right different equal
    exact different (layout.toVertex_injective equal))

theorem enumeratedVertices_subset_all
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    layout.enumeratedVertices ⊆ BooleanVertex.all variables := by
  intro vertex _
  exact BooleanVertex.mem_all vertex

theorem columns_le_twoPow
    {variables columns : Nat}
    (layout : ColumnLayout variables columns) :
    columns <= 2 ^ variables :=
  layout.columns_le

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

def freshSourceIndex
    {shape : Shape}
    (source : Fin shape.freshCount) : Fin shape.sourceCount :=
  ⟨source.val, by
    simp only [Shape.sourceCount]
    omega⟩

def runningSourceIndex
    {shape : Shape}
    (source : Fin shape.runningCount) : Fin shape.sourceCount :=
  ⟨shape.freshCount + source.val, by
    simp only [Shape.sourceCount]
    omega⟩

theorem freshSourceIndex_ne_runningSourceIndex
    {shape : Shape}
    (fresh : Fin shape.freshCount)
    (running : Fin shape.runningCount) :
    freshSourceIndex fresh ≠ runningSourceIndex running := by
  intro equal
  have values := congrArg Fin.val equal
  simp only [freshSourceIndex, runningSourceIndex] at values
  omega

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

def freshBatch
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) :
    CCSResidualTable.FreshBatch F shape columns where
  system := data.system
  assignments := fun source => data.assignments (freshSourceIndex source)

def normBatch
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) :
    NormResidualTable.SourceBatch shape where
  assignments := fun source vertex =>
    data.layout.paddedValue 0 (data.assignments source) vertex

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

def AllAssignmentsStrictNormBounded
    {Extension : Type uExtension}
    {shape : Shape}
    {columns : Nat}
    (data : UnifiedInputs Extension shape columns) : Prop :=
  ∀ source column, centeredMagnitude (data.assignments source column) < 2

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
