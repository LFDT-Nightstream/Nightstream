import NightstreamFPrime.Export.Stage1.PiDECEvaluationSparseBlock
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81MatrixSource

/-!
Derive a sparse Pad row from its authoritative column layout. Every carrier
column, including the completed tail, belongs to this identity matrix.
The stored block product uses the existing source's padded entry.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationPadBlock

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.UnifiedSources
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Layout.ProductionRelation

/-- Convert the layout's selected identity entry to the existing sparse form. -/
def form {arity columns : Nat} (layout : ColumnLayout arity columns)
    (vertex : BooleanVertex arity) : SparseForm columns :=
  match layout.toColumn? vertex with
  | some column => SparseForm.singleton column 1
  | none => SparseForm.empty

/-- The sparse row has exactly the authoritative padded-identity coefficient. -/
theorem form_coefficient {arity columns : Nat} (layout : ColumnLayout arity columns)
    (vertex : BooleanVertex arity) (column : Fin columns) :
    (form layout vertex).coefficient column =
      layout.paddedIdentityEntry (0 : F) 1 vertex column := by
  cases selected : layout.toColumn? vertex with
  | none =>
      simp only [form, selected, SparseForm.empty_coefficient, ColumnLayout.paddedIdentityEntry]
  | some found =>
      simp only [form, selected, SparseForm.singleton_coefficient,
        ColumnLayout.paddedIdentityEntry, eq_comm]

/-- Every returned coefficient is the complete-carrier Pad contribution in
the existing Phi81 source. Pad does not read the stored CCS matrices. -/
theorem rowBlock_paddedEntry
    (arity freshCount runningCount matrixCount logicalWidth : Nat)
    (matrices : Fin matrixCount → BooleanMatrix F arity logicalWidth)
    (polynomial : CCSResidualTable.ConstraintPolynomial F matrixCount)
    (layout : ColumnLayout arity (Phi81CarrierLayout.carrierWidth logicalWidth))
    (vertex : BooleanVertex arity)
    (block : Fin (Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth logicalWidth)))
    (children : Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (output : Fin ringDegree) :
    ((PiDECEvaluationBlock.rowBlock (form layout vertex) block.val children).get child).get output =
      sumRange baseOps ringDegree (fun index =>
        if live : index < ringDegree then
          (Phi81MatrixSource.source arity freshCount runningCount matrixCount
            logicalWidth matrices polynomial).paddedEntry baseOps
              (layout.paddedIdentityEntry (0 : F) 1) vertex block ⟨index, live⟩ *
            CarrierAction.kernelImage ⟨index, live⟩ (children.get child).get output
        else 0) := by
  rw [PiDECEvaluationSparseBlock.rowBlock_coefficient_value]
  apply sumRange_congr baseOps ringDegree
  intro index live
  simp only [dif_pos live]
  have within : block.val * ringDegree + index <
      Phi81CarrierLayout.carrierWidth logicalWidth :=
    Phi81CarrierLayout.flatIndex_lt_carrierWidth block ⟨index, live⟩
  rw [dif_pos within, form_coefficient]
  simp only [MatrixSource.paddedEntry, Phi81MatrixSource.source,
    Phi81CarrierLayout.layout_encode?_isSome, Phi81ColumnLayout.flatIndex]

end NightstreamFPrime.Export.Stage1.PiDECEvaluationPadBlock
