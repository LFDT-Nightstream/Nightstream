import NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch
import NightstreamFPrime.Export.Stage1.PiDECEvaluationHonestMessages

/-!
Execute child evaluations from complete stored block batches. The full
StoredAssignment conversion is a reference used only by the value theorems.
Each row calls the existing sparse block kernel once for all children.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECEvaluationFromBlocks

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment view)
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle

private abbrev selectedShape :=
  PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)

private abbrev selectedPlan := PerApplicationFixedPoint.structuralPlan
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits

private abbrev blockCount := Phi81ColumnLayout.blockCount selectedShape.carrierWidth

private theorem get_ofFn {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = _
  rw [Vector.getElem_ofFn]

/-- Reference conversion into the existing full assignment type. None of the
executable row or family definitions below calls this conversion. -/
def reference
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k) :
    Vector (StoredAssignment selectedShape.carrierWidth) productionGlobalParams.k :=
  Vector.ofFn fun child => Vector.ofFn fun column =>
    let packed := (Phi81CarrierLayout.layout selectedShape.logicalWidth).decode column
    ((blocks packed.1).get child).get packed.2

private theorem reference_value
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (column : Fin selectedShape.carrierWidth) :
    ((reference blocks).get child).get column =
      let packed := (Phi81CarrierLayout.layout selectedShape.logicalWidth).decode column
      ((blocks packed.1).get child).get packed.2 := by
  rw [reference, get_ofFn, get_ofFn]

/-- Reading a reference assignment block returns exactly the supplied stored
batch, at every child and lane. The sole indexing owner is the carrier layout. -/
theorem childBlocks_reference
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (block : Fin blockCount) :
    PiDECCommitmentFold.childBlocks (shape := selectedShape) (reference blocks) block =
      blocks block := by
  apply Vector.ext
  intro child childLt
  apply Vector.ext
  intro lane laneLt
  change ((PiDECCommitmentFold.childBlocks (shape := selectedShape)
      (reference blocks) block).get ⟨child, childLt⟩).get ⟨lane, laneLt⟩ =
    ((blocks block).get ⟨child, childLt⟩).get ⟨lane, laneLt⟩
  rw [PiDECCommitmentFold.childBlocks_value]
  change ((reference blocks).get ⟨child, childLt⟩).get
    (Phi81CarrierLayout.carrierColumn (logicalWidth := selectedShape.logicalWidth)
      block ⟨lane, laneLt⟩) = _
  rw [reference_value]
  change ((blocks (Phi81ColumnLayout.decode
      (Phi81CarrierLayout.carrierColumn (logicalWidth := selectedShape.logicalWidth)
        block ⟨lane, laneLt⟩)).1).get ⟨child, childLt⟩).get
      (Phi81ColumnLayout.decode
        (Phi81CarrierLayout.carrierColumn (logicalWidth := selectedShape.logicalWidth)
          block ⟨lane, laneLt⟩)).2 = _
  rw [Phi81CarrierLayout.decode_carrierColumn]

/-- The conversion also preserves every coordinate of an existing complete
stored assignment, including all carried tail coordinates. -/
theorem reference_childBlocks
    (assignments : Vector (StoredAssignment selectedShape.carrierWidth)
      productionGlobalParams.k) :
    reference (PiDECCommitmentFold.childBlocks (shape := selectedShape) assignments) =
      assignments := by
  apply Vector.ext
  intro child childLt
  apply Vector.ext
  intro column columnLt
  change ((reference (PiDECCommitmentFold.childBlocks (shape := selectedShape)
      assignments)).get ⟨child, childLt⟩).get ⟨column, columnLt⟩ =
    (assignments.get ⟨child, childLt⟩).get ⟨column, columnLt⟩
  rw [reference_value]
  change ((PiDECCommitmentFold.childBlocks (shape := selectedShape) assignments
      (Phi81ColumnLayout.decode (⟨column, columnLt⟩ : Fin selectedShape.carrierWidth)).1).get
      ⟨child, childLt⟩).get
      (Phi81ColumnLayout.decode (⟨column, columnLt⟩ : Fin selectedShape.carrierWidth)).2 = _
  rw [PiDECCommitmentFold.childBlocks_value]
  have roundtrip : Phi81CarrierLayout.carrierColumn
      (logicalWidth := selectedShape.logicalWidth)
      (Phi81ColumnLayout.decode (⟨column, columnLt⟩ : Fin selectedShape.carrierWidth)).1
      (Phi81ColumnLayout.decode (⟨column, columnLt⟩ : Fin selectedShape.carrierWidth)).2 =
        (⟨column, columnLt⟩ : Fin selectedShape.carrierWidth) := by
    apply Fin.ext
    exact Phi81ColumnLayout.flatIndex_decode ⟨column, columnLt⟩
  change (assignments.get ⟨child, childLt⟩).get
    (Phi81CarrierLayout.carrierColumn (logicalWidth := selectedShape.logicalWidth)
      (Phi81ColumnLayout.decode (⟨column, columnLt⟩ : Fin selectedShape.carrierWidth)).1
      (Phi81ColumnLayout.decode (⟨column, columnLt⟩ : Fin selectedShape.carrierWidth)).2) = _
  rw [roundtrip]

/-- Guarded access for the existing Nat-indexed sparse block kernel. -/
def blockAt (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (index : Nat) : Vector StoredRing productionGlobalParams.k :=
  if live : index < blockCount then blocks ⟨index, live⟩
  else Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero

private theorem blockAt_reference
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k) :
    blockAt blocks = fun index =>
      if live : index < blockCount then
        PiDECCommitmentFold.childBlocks (shape := selectedShape)
          (reference blocks) ⟨index, live⟩
      else Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero := by
  funext index
  simp only [blockAt, childBlocks_reference]

private def formAtPort {columns : Nat}
    (forms : Option (NightstreamFPrime.Layout.MatrixProgram.RowForms columns))
    (matrix : Fin Spec.ProductionRelation.matrixCount) : SparseForm columns :=
  match meaningfulPort? matrix with
  | none => SparseForm.empty
  | some port =>
      match forms with
      | none => SparseForm.empty
      | some row => row port

private theorem formAtPort_plan {columns : Nat} (plan : Plan columns)
    (row : Fin plan.rowCount) (matrix : Fin Spec.ProductionRelation.matrixCount) :
    formAtPort (some (plan.forms row)) matrix = plan.portForm row matrix := by
  unfold formAtPort Plan.portForm
  cases meaningfulPort? matrix <;> rfl

/-- Read the selected compact program, retaining the explicit zero port.
The canonical package owns ordinary source rows; no matrix row is supplied. -/
def programForm (row : Fin selectedPlan.rowCount)
    (matrix : Fin Spec.ProductionRelation.matrixCount) : SparseForm selectedShape.logicalWidth :=
  formAtPort ((PerApplicationMatrixProgram.matrixProgram
    Poseidon2HashChainV1Package.application).row? selectedShape.logicalWidth
    (PerApplicationCanonicalPackage.sourceRow
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits) row.val) matrix

/-- Active program rows are the exact selected structural-plan forms. -/
theorem programForm_value (row : Fin selectedPlan.rowCount)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    programForm row matrix = selectedPlan.portForm row matrix := by
  exact (congrArg (fun forms : Option (NightstreamFPrime.Layout.MatrixProgram.RowForms
      (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)) =>
        formAtPort forms matrix)
    (Poseidon2HashChainV1MatrixRows.compactProgram_row?_eq_structuralPlan_forms row)).trans
      (formAtPort_plan selectedPlan row matrix)

/-- One sparse matrix kernel call computes the complete child batch for this row. -/
def matrixRow (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (matrix : Fin Spec.ProductionRelation.matrixCount) (index : Nat) :
    Vector StoredRing productionGlobalParams.k :=
  if live : index < selectedPlan.rowCount then
    PiDECEvaluationBlockSupport.kernel (programForm ⟨index, live⟩ matrix)
      (blockAt blocks)
  else Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero

/-- Pad covers the full carrier and calls the same batched sparse kernel. -/
def padRow (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (index : Nat) : Vector StoredRing productionGlobalParams.k :=
  if live : index < selectedShape.carrierWidth then
    PiDECEvaluationBlockSupport.kernel
      (SparseForm.singleton (⟨index, live⟩ : Fin selectedShape.carrierWidth) 1)
      (blockAt blocks)
  else Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero

theorem matrixRow_child
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (matrix : Fin Spec.ProductionRelation.matrixCount) (index : Nat)
    (child : Fin productionGlobalParams.k) :
    (matrixRow blocks matrix index).get child =
      PiDECEvaluationSelectedPrefix.matrixRow (reference blocks) child matrix index := by
  by_cases live : index < selectedPlan.rowCount
  · simpa only [matrixRow, PiDECEvaluationSelectedPrefix.matrixRow, dif_pos live,
      programForm_value, PiDECEvaluationRows.row] using
      congrArg (fun source : Nat → Vector StoredRing productionGlobalParams.k =>
        (PiDECEvaluationBlockSupport.kernel
          (selectedPlan.portForm ⟨index, live⟩ matrix) source).get child)
        (blockAt_reference blocks)
  · simp only [matrixRow, PiDECEvaluationSelectedPrefix.matrixRow, dif_neg live]
    change (Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero)[child.val] = _
    rw [Vector.getElem_replicate]

theorem padRow_child
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (index : Nat) (child : Fin productionGlobalParams.k) :
    (padRow blocks index).get child =
      PiDECEvaluationSelectedPrefix.padRow (reference blocks) child index := by
  by_cases live : index < selectedShape.carrierWidth
  · simpa only [padRow, PiDECEvaluationSelectedPrefix.padRow, dif_pos live,
      PiDECEvaluationRows.row] using
      congrArg (fun source : Nat → Vector StoredRing productionGlobalParams.k =>
        (PiDECEvaluationBlockSupport.kernel
          (SparseForm.singleton (⟨index, live⟩ : Fin selectedShape.carrierWidth) 1)
          source).get child) (blockAt_reference blocks)
  · simp only [padRow, PiDECEvaluationSelectedPrefix.padRow, dif_neg live]
    change (Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero)[child.val] = _
    rw [Vector.getElem_replicate]

/-- Compute Pad and each matrix as one all-child batch before exposing family
views. This definition does not assemble or access a full reference assignment. -/
def familyFromBlocks
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (point : PaperAlgebra.Point) : Vector PaperAlgebra.Evaluation productionGlobalParams.k :=
  let pad := PiDECEvaluationBatch.accumulate selectedShape.carrierWidth point (padRow blocks)
  let matrices := PiRLCPartialTrace.FixedArray.ofFn
    (fun matrix : Fin productionShape.matrixCount =>
      PiDECEvaluationBatch.accumulate selectedPlan.rowCount point (matrixRow blocks matrix))
  Vector.ofFn fun child =>
    { pad := (pad.get child).toRing
      matrix := fun matrix => ((matrices.get matrix).get child).toRing }

/-- Every batched family equals the existing scalar family on the reference
assignment, with no expected-value or row-correctness premise. -/
theorem familyFromBlocks_eq_family
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (point : PaperAlgebra.Point) (child : Fin productionGlobalParams.k) :
    (familyFromBlocks blocks point).get child =
      PiDECEvaluationHonestMessages.family (reference blocks) point child := by
  simp only [familyFromBlocks, get_ofFn, PiDECEvaluationHonestMessages.family,
    PiRLCPartialTrace.FixedArray.get_ofFn]
  apply congrArg₂ (@StrongReduction.EvaluationFamily.mk K productionShape)
  · rw [PiDECEvaluationBatch.accumulate_child]
    apply congrArg (fun rows =>
      (PiDECEvaluationWeights.accumulate selectedShape.carrierWidth point rows).toRing)
    funext index
    exact padRow_child blocks index child
  · funext matrix
    rw [PiDECEvaluationBatch.accumulate_child]
    apply congrArg (fun rows =>
      (PiDECEvaluationWeights.accumulate selectedPlan.rowCount point rows).toRing)
    funext index
    exact matrixRow_child blocks matrix index child

/-- Specialize the block executor to the same successfully split stored
children. Only the checked split is assumed; all commitments and claimed
evaluations remain outside the construction inputs. -/
theorem familyFromBlocks_honestMessages
    (values : PiDECInputCheck.ParentValues)
    (parentWitness : StoredAssignment selectedShape.carrierWidth)
    (childWitnesses : Vector (StoredAssignment selectedShape.carrierWidth)
      productionGlobalParams.k)
    (success : StoredSplit.splitChecked parentWitness = some childWitnesses)
    (child : Fin productionGlobalParams.k) :
    #[(familyFromBlocks
      (PiDECCommitmentFold.childBlocks (shape := selectedShape) childWitnesses)
      values.point).get child] =
      (PiDEC.PaperVerifier.honestMessages
        (PaperAlgebra.piDecAlgebra Poseidon2HashChainV1Setup.productionAjtaiKey)
        (PiDECInputCheck.parent values) (view parentWitness) child).evaluations := by
  rw [familyFromBlocks_eq_family, reference_childBlocks]
  exact PiDECEvaluationHonestMessages.family_honestMessages
    values parentWitness childWitnesses success child

end NightstreamFPrime.Export.Stage1.PiDECEvaluationFromBlocks
