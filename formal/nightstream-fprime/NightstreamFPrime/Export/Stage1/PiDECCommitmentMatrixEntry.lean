import NightstreamFPrime.Export.Stage1.PiDECCommitmentMatrixWork
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1MatrixRows
import NightstreamFPrime.Export.MatrixProgram.CoefficientWork

/-!
Counted coefficient entries for the selected PiDEC commitment row packet.
All 14 matrix ports and 54 coefficient lanes reach the actual selected
matrix source. The packet starts at global logical row 6021547.
SuperNeo v1.1 Section 7.3 and Appendix B.2 own the coefficient-matrix check;
Section 7.5 and Appendix B.4 own these commitment recomposition rows.

The caller supplies a typed local row. This program constructs its forms
once, selects one stored port, and expands its 54 source lanes. Global row
dispatch and Boolean-vertex conversion are caller work. No packet or matrix
function is evaluated here. Work composes the existing declared callee clocks
and the named wrapper charges. A complete refinement of those clocks to source
execution or runtime is a separate obligation. Proof casts and clock
instrumentation are erased from the count.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECCommitmentMatrixEntry

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Layout
open _root_.NightstreamFPrime.Layout.ProductionRelation
open _root_.NightstreamFPrime.Lifecycle
open _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open _root_.NightstreamFPrime.Export.MatrixProgram
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

abbrev carrier : Phi81Relation.Shape :=
  PaperAlgebra.FullShape PiDECInputCheck.logicalWidth PiDECInputCheck.publicFits

abbrev selectedPlan : ProductionRelation.Plan PiDECInputCheck.logicalWidth :=
  PerApplicationFixedPoint.structuralPlan Poseidon2HashChainV1Package.application
    Poseidon2HashChainV1Package.fits

private theorem width_eq : PiDECInputCheck.logicalWidth = 254260583 :=
  Poseidon2HashChainV1Package.logicalWidth

attribute [local irreducible] PerApplicationFixedPoint.logicalWidth PiDECInputCheck.relation
  PiDECCommitmentMatrixWork.commitmentForms CoefficientWork.coefficient

section Embedding

variable {application : Lifecycle.Stage1.Application.Program} {columns relationWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationWidth}
  (relation : ProductionKey.LogicalRelation relationWidth publicFits)
  (fits : PerApplicationPackage.FitsTwoPow28 application)
  (geometry : ApplicationRetainedGeometry.Geometry application columns)

/-- A structural index through the existing append tree. This is used only
to prove the selected packet's global position and exact matrix images. -/
private def fullRow (row : Fin 1188) :
    Fin (DirectApplicationPrefixPlan.plan relation fits geometry).rowCount :=
  let sampler := DirectApplicationPrefixPlan.prefixGeometry geometry
  let dec := DirectApplicationPrefixPlan.piDecGeometry geometry
  let packet := ProductionRelation.Plan.rightIndex
    (PiDECDirectPlan.publicPlan relation dec).rowCount
    (PiDECDirectPlan.recompositionPlan relation dec).rowCount
    (ProductionRelation.Plan.leftIndex
      (PiDECDirectPlan.commitmentPlan relation dec).rowCount
      (PiDECDirectPlan.evaluationPlan relation dec).rowCount row)
  let afterDec := ProductionRelation.Plan.rightIndex
    (DirectPiRLCSamplerCompletePrefixPlan.piRlcCompletePlan relation sampler).rowCount
    (DirectPiRLCSamplerCompletePrefixPlan.piDecPlan relation sampler).rowCount packet
  let running := ProductionRelation.Plan.leftIndex
    (DirectPiRLCSamplerCompletePrefixPlan.piDecCompletePlan relation sampler).rowCount
    (DirectPiRLCSamplerCompletePrefixPlan.transitionPlan relation sampler).rowCount afterDec
  let app := ProductionRelation.Plan.leftIndex
    (DirectApplicationPrefixPlan.prefixPlan relation geometry).rowCount
    (DirectApplicationPrefixPlan.applicationPlan fits geometry).rowCount running
  let next := ProductionRelation.Plan.leftIndex
    (DirectApplicationPrefixPlan.prefixApplicationPlan relation fits geometry).rowCount
    (DirectApplicationPrefixPlan.nextPreimagePlan geometry).rowCount app
  ProductionRelation.Plan.leftIndex
    (DirectApplicationPrefixPlan.throughNextPreimagePlan relation fits geometry).rowCount
    (DirectApplicationPrefixPlan.publicOutputPlan geometry).rowCount next

private theorem fullRow_val (row : Fin 1188) :
    (fullRow relation fits geometry row).val = 6021547 + row.val := by
  simp only [fullRow, ProductionRelation.Plan.leftIndex_val, ProductionRelation.Plan.rightIndex_val,
    PiDECDirectPlan.publicPlan_rowCount,
    DirectPiRLCSamplerCompletePrefixPlan.piRlcCompletePlan_rowCount]
  omega

private theorem fullRow_port (row : Fin 1188) (matrix : Fin Spec.ProductionRelation.matrixCount) :
    (DirectApplicationPrefixPlan.plan relation fits geometry).portForm
        (fullRow relation fits geometry row) matrix =
      (PiDECMatrixProgram.commitmentDirectForms relation
        (DirectApplicationPrefixPlan.piDecGeometry geometry) row).portForm matrix := by
  simp only [fullRow, DirectApplicationPrefixPlan.plan,
    DirectApplicationPrefixPlan.throughNextPreimagePlan, DirectApplicationPrefixPlan.prefixApplicationPlan,
    DirectApplicationPrefixPlan.prefixPlan, DirectPiRLCSamplerCompletePrefixPlan.plan,
    DirectPiRLCSamplerCompletePrefixPlan.piDecCompletePlan,
    DirectPiRLCSamplerCompletePrefixPlan.piDecPlan, DirectPiDECPrefixPlan.piDecPlan,
    PiDECDirectPlan.plan, PiDECDirectPlan.recompositionPlan,
    ProductionRelation.Plan.append_portForm_left, ProductionRelation.Plan.append_portForm_right]
  simp only [ProductionRelation.Plan.portForm, OrdinaryRow.Forms.portForm,
    PiDECMatrixProgram.commitmentPlan_forms]
  rfl

end Embedding

private theorem plan_port_transport {columns : Nat} (left right : ProductionRelation.Plan columns)
    (equal : left = right) (leftRow : Fin left.rowCount) (rightRow : Fin right.rowCount)
    (same : leftRow.val = rightRow.val) (matrix : Fin Spec.ProductionRelation.matrixCount) :
    left.portForm leftRow matrix = right.portForm rightRow matrix := by
  cases equal
  have indices : leftRow = rightRow := Fin.ext same
  rw [indices]

/-- The constant offset is checked against the exact append tree below.
This coordinate is a semantic reference; entry does not execute this map. -/
def globalRow (row : Fin 1188) : Fin selectedPlan.rowCount :=
  ⟨6021547 + row.val, by
    have bound := row.isLt
    rw [Poseidon2HashChainV1Package.structuralRowCount]
    omega⟩

theorem globalRow_val (row : Fin 1188) : (globalRow row).val = 6021547 + row.val := rfl

private theorem globalRow_port (row : Fin 1188) (matrix : Fin Spec.ProductionRelation.matrixCount) :
    selectedPlan.portForm (globalRow row) matrix =
      (PiDECMatrixProgram.commitmentDirectForms PiDECInputCheck.relation
        (PerApplicationMatrixProgram.piDecGeometry Poseidon2HashChainV1Package.application) row).portForm matrix := by
  let geometry := PerApplicationFixedPoint.geometry Poseidon2HashChainV1Package.application
  let fits := Poseidon2HashChainV1Package.fits.package
  let whole := DirectApplicationPrefixPlan.plan PiDECInputCheck.relation fits geometry
  have equal : whole = selectedPlan := by
    dsimp only [whole]
    rw [PiDECInputCheck.relation_eq_selected]
    exact PerApplicationFixedPoint.plan_fixedPoint Poseidon2HashChainV1Package.application
      Poseidon2HashChainV1Package.fits
  have same : (fullRow PiDECInputCheck.relation fits geometry row).val = (globalRow row).val := by
    rw [fullRow_val, globalRow_val]
  have moved := plan_port_transport whole selectedPlan equal
    (fullRow PiDECInputCheck.relation fits geometry row) (globalRow row) same matrix
  exact moved.symm.trans (fullRow_port PiDECInputCheck.relation fits geometry row matrix)

/-- One matrix-index read; each test costs literal/comparison/branch (3).
A live field costs its record read and Result (2). Empty ports include
Unit/call/value/Result (4), plus the actual empty-form constructor work. -/
private def selectPort {columns : Nat} (forms : OrdinaryRow.Forms columns)
    (matrix : Fin Spec.ProductionRelation.matrixCount) : Result (SparseForm columns) :=
  let slot := matrix.val
  if slot = 1 then ⟨forms.selector, 6⟩
  else if slot = 2 then ⟨forms.a, 9⟩
  else if slot = 3 then ⟨forms.b, 12⟩
  else if slot = 4 then ⟨forms.c, 15⟩
  else
    let empty := SparseWork.empty ()
    ⟨empty.value, empty.work + 17⟩

private theorem selectPort_value {columns : Nat} (forms : OrdinaryRow.Forms columns)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    (selectPort forms matrix).value = forms.portForm matrix := by
  fin_cases matrix <;>
    simp [selectPort, OrdinaryRow.Forms.portForm, OrdinaryRow.Forms.meaningfulForm,
      Layout.ProductionRelation.meaningfulPort?, Spec.ProductionRelation.meaningfulPortCount,
      Spec.ProductionRelation.SelectivePolynomial.meaningfulPortCount,
      SparseWork.empty_value]

private theorem selectPort_work_le {columns : Nat} (forms : OrdinaryRow.Forms columns)
    (matrix : Fin Spec.ProductionRelation.matrixCount) :
    (selectPort forms matrix).work ≤ 20 := by
  dsimp only [selectPort]
  split_ifs <;> dsimp only [SparseWork.empty] <;> omega

private theorem selectPort_length_le {columns : Nat} (forms : OrdinaryRow.Forms columns)
    (matrix : Fin Spec.ProductionRelation.matrixCount)
    (lengths : forms.selector.entries.length = 1 ∧ forms.a.entries.length = 657 ∧
      forms.b.entries.length = 1 ∧ forms.c.entries.length = 42) :
    (selectPort forms matrix).value.entries.length ≤ 657 := by
  rcases lengths with ⟨selector, a, b, c⟩
  dsimp only [selectPort]
  split_ifs <;> simp only [SparseWork.empty, selector, a, b, c, List.length_nil] <;> omega

private theorem kernel_work_eq : StoredWitnessCheckEntries.kernelWork = 5208 := by rfl

private theorem coefficient_work_eq : CoefficientWork.workBound 657 = 637643 := by
  rw [CoefficientWork.workBound, kernel_work_eq]
  norm_num [ringDegree]

private theorem coefficient_work_mono {left right : Nat} (bound : left ≤ right) :
    CoefficientWork.workBound left ≤ CoefficientWork.workBound right := by
  unfold CoefficientWork.workBound
  exact Nat.add_le_add_right (Nat.add_le_add_right
    (Nat.mul_le_mul_left _ (Nat.add_le_add_right (Nat.add_le_add_right
      (Nat.add_le_add_right (Nat.add_le_add_right (Nat.mul_le_mul_left 10 bound) 8) _) 14) 8)) 3) 8

private theorem coefficient_cast {columns output : Nat} (equal : columns = output)
    (form : SparseForm columns) (coefficient : Fin ringDegree)
    (column : Fin (Phi81CarrierLayout.carrierWidth columns)) :
    CoefficientWork.coefficient (logicalWidth := output) (equal ▸ form) coefficient
        (congrArg Phi81CarrierLayout.carrierWidth equal ▸ column) =
      CoefficientWork.coefficient form coefficient column := by
  cases equal
  rfl

private theorem coefficient_for_relation {columns : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth columns}
    (relation : ProductionKey.LogicalRelation columns publicFits)
    (matrix : Fin productionShape.matrixCount) (vertex : BooleanVertex productionShape.cubeVariables)
    (form : SparseForm columns)
    (same : ∀ source, form.coefficient source = relation.matrices matrix vertex source)
    (coefficient : Fin productionShape.coefficientCount)
    (column : Fin (Phi81CarrierLayout.carrierWidth columns)) :
    (CoefficientWork.coefficient form coefficient column).value =
      (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation).matrixSource.coefficientMatrix
        baseOps matrix coefficient vertex column :=
  CoefficientWork.coefficient_value cubeVariables productionProfile.freshSources
    productionProfile.runningSources productionProfile.ccsMatrices columns relation.matrices
    Spec.ProductionRelation.polynomial matrix vertex form same coefficient column

private theorem selected_port_coefficient (row : Fin 1188)
    (matrix : Fin productionShape.matrixCount) (column : Fin PiDECInputCheck.logicalWidth) :
    ((selectPort (PiDECCommitmentMatrixWork.commitmentForms row).value matrix).value).coefficient column =
      PiDECInputCheck.relation.matrices matrix (selectedPlan.rowLayout.toVertex (globalRow row)) column := by
  rw [selectPort_value, PiDECCommitmentMatrixWork.commitmentForms_value, ← globalRow_port]
  have selected := Poseidon2HashChainV1MatrixRows.allPort_coefficient_eq_logicalRelation_matrix
    (globalRow row) matrix column
  rw [← PiDECInputCheck.relation_eq_selected] at selected
  exact selected

/-- One forms call, one port call and one coefficient call. Their three
value reads, the explicit selected-width literal and Result add eight.
Global dispatch and vertex construction are absent from this typed-row call. -/
def entry (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) (row : Fin 1188)
    (column : Fin carrier.carrierWidth) : Result F :=
  let forms := PiDECCommitmentMatrixWork.commitmentForms row
  let selected := selectPort forms.value matrix
  let result := CoefficientWork.coefficient (logicalWidth := 254260583)
    (width_eq ▸ selected.value) coefficient
    (congrArg Phi81CarrierLayout.carrierWidth width_eq ▸ column)
  ⟨result.value, forms.work + selected.work + result.work + 8⟩

/-- Bound for the composed declared clocks of this typed local-row call.
The selected kernel's declared bound 5208 is included in coefficient expansion. -/
def entryWork : Nat := 254729 + 20 + 637643 + 8

theorem entry_value (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) (row : Fin 1188)
    (column : Fin carrier.carrierWidth) :
    (entry matrix coefficient row column).value =
      (Lifecycle.PiRLC.v1_1.InputBinding.relationSource PiDECInputCheck.relation).matrixSource.coefficientMatrix
        baseOps matrix coefficient (selectedPlan.rowLayout.toVertex (globalRow row)) column := by
  change (CoefficientWork.coefficient (logicalWidth := 254260583)
    (width_eq ▸ (selectPort (PiDECCommitmentMatrixWork.commitmentForms row).value matrix).value)
    coefficient (congrArg Phi81CarrierLayout.carrierWidth width_eq ▸ column)).value = _
  rw [coefficient_cast]
  exact coefficient_for_relation PiDECInputCheck.relation matrix
    (selectedPlan.rowLayout.toVertex (globalRow row))
    (selectPort (PiDECCommitmentMatrixWork.commitmentForms row).value matrix).value
    (selected_port_coefficient row matrix) coefficient column

theorem entry_work_le (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) (row : Fin 1188)
    (column : Fin carrier.carrierWidth) :
    (entry matrix coefficient row column).work ≤ entryWork := by
  let forms := PiDECCommitmentMatrixWork.commitmentForms row
  let selected := selectPort forms.value matrix
  have formsWork : forms.work ≤ 254729 := PiDECCommitmentMatrixWork.commitmentForms_work_le row
  have portWork : selected.work ≤ 20 := selectPort_work_le forms.value matrix
  have length : selected.value.entries.length ≤ 657 := selectPort_length_le forms.value matrix
    (PiDECCommitmentMatrixWork.commitmentForms_lengths row)
  have coefficientWork := CoefficientWork.coefficient_work_le selected.value coefficient column
  have bound := coefficient_work_mono length
  rw [coefficient_work_eq] at bound
  dsimp only [entry]
  rw [coefficient_cast]
  change forms.work + selected.work + (CoefficientWork.coefficient selected.value coefficient column).work + 8 ≤ entryWork
  unfold entryWork
  omega

end NightstreamFPrime.Export.Stage1.PiDECCommitmentMatrixEntry
