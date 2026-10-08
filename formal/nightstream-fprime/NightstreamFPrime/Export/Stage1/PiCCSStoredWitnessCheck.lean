import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckEntries
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CheckedWitnessExtraction

/-!
Owns the selected PiCCS source statement for the actual application matrix
source and the frozen Ajtai setup. `statement` reads the existing typed
fresh/running public fields and equals the statement that `ProductionKey.key`
selects. Public-input reads and Pad entries return their values with their
declared work.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open NightstreamFPrime.Lifecycle
open CheckedWitnessExtraction
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

abbrev carrier : Phi81Relation.Shape :=
  PaperAlgebra.FullShape PiDECInputCheck.logicalWidth PiDECInputCheck.publicFits

/-- The sole selected commitment map, including the actual indexed key expansion. -/
def commit : Phi81Relation.Assignment carrier → PaperAlgebra.Commitment :=
  (PaperAlgebra.openingMaps Poseidon2HashChainV1Setup.productionAjtaiKey).commit

private theorem openingMaps_of_projection {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (key : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    openingMaps (carrier := PaperAlgebra.FullShape logicalWidth publicFits)
      (PaperAlgebra.openingMaps key).commit = PaperAlgebra.openingMaps key := by
  rfl

private theorem selected_openingMaps : openingMaps (carrier := carrier) commit =
    PaperAlgebra.openingMaps Poseidon2HashChainV1Setup.productionAjtaiKey :=
  openingMaps_of_projection (logicalWidth := PiDECInputCheck.logicalWidth)
    (publicFits := PiDECInputCheck.publicFits) Poseidon2HashChainV1Setup.productionAjtaiKey

/-- Computable projection of the selected key's statement. Matrix entries
remain behind the existing selected relation's access function. -/
def statement (input : PiCCSInputCheck.Input) :
    Statement K PaperAlgebra.Commitment (Phi81Relation.PublicInput carrier)
      productionShape carrier.carrierWidth
      (Phi81ColumnLayout.blockCount carrier.carrierWidth) baseOps where
  cubeLayout := (Lifecycle.PiRLC.v1_1.InputBinding.relationSource PiDECInputCheck.relation).cubeLayout
  matrixSource := (Lifecycle.PiRLC.v1_1.InputBinding.relationSource PiDECInputCheck.relation).matrixSource
  commitments := PiCCSInputCheck.outputCommitments input
  publicInputs := PiCCSInputCheck.outputPublicInputs input
  priorPoint := (PiCCSInputCheck.running input).point
  claimedPadCoefficient := (PiCCSInputCheck.verifierInput input).claimedPadCoefficient
  claimedMatrixCoefficient := (PiCCSInputCheck.verifierInput input).claimedMatrixCoefficient

/-- Read the actual fresh/running public arrays. Both paths charge source
index read, comparison, branch, data reads and return; the running path also
charges index subtraction. The costs are six and nine operations. -/
def publicInputRead (input : PiCCSInputCheck.Input) (source : Fin productionShape.sourceCount)
    (column : Fin carrier.publicWidth) : Result F :=
  if fresh : source.val < productionShape.freshCount then
    ⟨input.publicInput.get column, 1 + 1 + 1 + 1 + 1 + 1⟩
  else
    let running : Fin productionShape.runningCount :=
      ⟨source.val - productionShape.freshCount, by
        have sourceBound := source.isLt
        change source.val < productionShape.freshCount + productionShape.runningCount at sourceBound
        omega⟩
    ⟨(input.running.publicInputs.get running).get column, 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1⟩

theorem publicInputRead_value (input : PiCCSInputCheck.Input) (source : Fin productionShape.sourceCount)
    (column : Fin carrier.publicWidth) :
    (publicInputRead input source column).value = (statement input).publicInputs source column := by
  change (publicInputRead input source column).value =
    (Fin.addCases (motive := fun _ => Phi81Relation.PublicInput carrier)
      (PiCCSInputCheck.fresh input).publicInputs
      (PiCCSInputCheck.running input).publicInputs source) column
  refine Fin.addCases (fun fresh => ?_) (fun running => ?_) source
  · have isFresh : (Fin.castAdd productionShape.runningCount fresh).val < productionShape.freshCount := fresh.isLt
    rw [publicInputRead, dif_pos isFresh, Fin.addCases_left]
    rfl
  · have isRunning : ¬ (Fin.natAdd productionShape.freshCount running).val < productionShape.freshCount := by
      change ¬ productionShape.freshCount + running.val < productionShape.freshCount
      omega
    rw [publicInputRead, dif_neg isRunning, Fin.addCases_right]
    simp only [PiCCSInputCheck.running, PiCCSInputCheck.runningFromInput, Fin.natAdd, Nat.add_sub_cancel_left]

theorem publicInputRead_work (input : PiCCSInputCheck.Input) (source : Fin productionShape.sourceCount)
    (column : Fin carrier.publicWidth) :
    (publicInputRead input source column).work = if source.val < productionShape.freshCount then 6 else 9 := by
  unfold publicInputRead
  split <;> simp_all

theorem publicInputRead_work_le (input : PiCCSInputCheck.Input) (source : Fin productionShape.sourceCount)
    (column : Fin carrier.publicWidth) : (publicInputRead input source column).work ≤ 9 := by
  rw [publicInputRead_work]
  split <;> omega

private theorem padEntry_of_relation {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (coefficient : Fin productionShape.coefficientCount)
    (vertex : BooleanVertex productionShape.cubeVariables)
    (column : Fin (Phi81CarrierLayout.carrierWidth logicalWidth)) :
    (StoredWitnessCheckEntries.padEntry coefficient vertex column).value =
      (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation).matrixSource.coefficientMatrixOf baseOps
        (fun row column => (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation).cubeLayout.paddedIdentityEntry
          baseOps.zero baseOps.one row column) coefficient vertex column :=
  StoredWitnessCheckEntries.padEntry_value cubeVariables
    productionProfile.freshSources productionProfile.runningSources productionProfile.ccsMatrices
    logicalWidth relation.matrices Spec.ProductionRelation.polynomial relation.cubeFits coefficient vertex column

/-- The executed Pad entry is the selected statement's coefficientMatrixOf
entry at every coordinate, independent of the returned probe. -/
theorem padEntry_value (input : PiCCSInputCheck.Input)
    (coefficient : Fin productionShape.coefficientCount)
    (vertex : BooleanVertex productionShape.cubeVariables) (column : Fin carrier.carrierWidth) :
    (StoredWitnessCheckEntries.padEntry coefficient vertex column).value =
      (statement input).matrixSource.coefficientMatrixOf baseOps
        (fun row column => (statement input).cubeLayout.paddedIdentityEntry
          baseOps.zero baseOps.one row column) coefficient vertex column :=
  padEntry_of_relation (logicalWidth := PiDECInputCheck.logicalWidth)
    (publicFits := PiDECInputCheck.publicFits) PiDECInputCheck.relation coefficient vertex column

private theorem kernelConstant_of_relation {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation).matrixSource.kernel.constant =
      Phi81CoefficientKernel.constant := by
  rfl

end NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck
