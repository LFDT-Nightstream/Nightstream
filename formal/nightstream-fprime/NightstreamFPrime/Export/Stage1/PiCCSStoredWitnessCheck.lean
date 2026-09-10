import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheck
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckWork
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckPrimitives
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredOneRunExtraction

/-!
Selected Appendix B.2 witness check for the actual application matrix source
and frozen Ajtai setup. The statement reads the existing typed fresh/running
public fields. It equals the statement selected by `ProductionKey.key`.

The Boolean checker and its use by the stored source-return theorem are
proved here. A charged checker must still execute this Boolean computation
and count its key generation, matrix access, commitment, field operations,
public/probe reads, and representation work. This module supplies no clock.
The candidate coins remain explicit; no Fiat--Shamir distribution is claimed.
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

abbrev StoredWitness := StoredWitnessProjection.StoredWitness productionShape carrier
abbrev Candidate := StoredProbe productionShape × StoredWitness

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

/-- The executable statement is the literal selected NIFS statement. -/
theorem statement_eq_key (input : PiCCSInputCheck.Input) :
    statement input =
      (ProductionKey.key PiDECInputCheck.relation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
        (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input) := by
  rfl

/-- Check a returned probe and its stored witness at the selected statement. -/
def check (input : PiCCSInputCheck.Input) (candidate : Candidate) : Bool :=
  StoredWitnessCheck.check commit productionGlobalParams (statement input)
    (ProductionKey.degreeBound PiDECInputCheck.relation) (candidate.1.view, candidate.2)

/-- Exact selected public/ambient predicate; opening validity is checked. -/
theorem check_eq_true_iff (input : PiCCSInputCheck.Input)
    (probe : StoredProbe productionShape) (stored : StoredWitness) :
    check input (probe, stored) = true ↔
      probe.view.FixedWidthAccepted extensionOps K.embed
        ((ProductionKey.key PiDECInputCheck.relation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
          (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input))
        (ProductionKey.degreeBound PiDECInputCheck.relation) ∧
      AmbientOutputHolds extensionOps K.embed
        (PaperAlgebra.openingMaps Poseidon2HashChainV1Setup.productionAjtaiKey) productionGlobalParams
        ((ProductionKey.key PiDECInputCheck.relation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
          (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input))
        probe.view (StoredWitnessProjection.view stored) := by
  rw [← statement_eq_key input, ← selected_openingMaps]
  exact StoredWitnessCheck.check_eq_true_iff
    (shape := productionShape) (carrier := carrier)
    (blockCount := Phi81ColumnLayout.blockCount carrier.carrierWidth)
    commit productionGlobalParams (statement input)
    (ProductionKey.degreeBound PiDECInputCheck.relation) probe.view stored

/-- The remaining value-refinement obligation is equality to an implemented
Boolean check. The charged call's work is retained without a proposed bound. -/
theorem chargedCheck_correct (input : PiCCSInputCheck.Input)
    (charged : StoredOneRunExtraction.Check productionShape carrier)
    (computes : ∀ candidate, (charged candidate).value = check input candidate) :
    ∀ probe stored, (charged (probe, stored)).value = true ↔
      probe.view.FixedWidthAccepted extensionOps K.embed (statement input)
        (ProductionKey.degreeBound PiDECInputCheck.relation) ∧
      AmbientOutputHolds extensionOps K.embed (openingMaps commit) productionGlobalParams
        (statement input) probe.view (StoredWitnessProjection.view stored) := by
  intro probe stored
  rw [computes]
  exact StoredWitnessCheck.check_eq_true_iff commit productionGlobalParams (statement input)
    (ProductionKey.degreeBound PiDECInputCheck.relation) probe.view stored

/-- The selected source return consumes the proved checker correctness.
Only the charged implementation's value refinement remains as a premise. -/
theorem finish_source_iff (input : PiCCSInputCheck.Input)
    (charged : StoredOneRunExtraction.Check productionShape carrier)
    (computes : ∀ candidate, (charged candidate).value = check input candidate)
    (outcome : StoredOutcome productionShape carrier) :
    SourceReturned (shape := productionShape) (carrier := carrier)
      (blockCount := Phi81ColumnLayout.blockCount carrier.carrierWidth)
      commit productionGlobalParams (statement input) (finishStored charged outcome).value ↔
      StrongProbability.RelaxedSuccess
        (width := ProductionKey.degreeBound PiDECInputCheck.relation)
        (PaperAlgebra.openingMaps Poseidon2HashChainV1Setup.productionAjtaiKey) productionGlobalParams
        (statement input) (storedView outcome) ∧
      StrongProbability.SourceValid
        (PaperAlgebra.openingMaps Poseidon2HashChainV1Setup.productionAjtaiKey) productionGlobalParams
        (statement input) (storedView outcome) := by
  rw [← selected_openingMaps]
  exact finishStored_source_iff (shape := productionShape) (carrier := carrier)
    (blockCount := Phi81ColumnLayout.blockCount carrier.carrierWidth)
    (width := ProductionKey.degreeBound PiDECInputCheck.relation)
    charged commit productionGlobalParams (statement input)
    (chargedCheck_correct input charged computes) outcome

/-- The implemented charged driver supplies the selected checker refinement.
Its primitive implementation contracts remain explicit; the source-return
path has no separate assumed opening or whole-checker correctness premise. -/
theorem charged_finish_source_iff (input : PiCCSInputCheck.Input)
    (program : StoredWitnessCheckWork.Program productionShape carrier)
    (correct : StoredWitnessCheckWork.Correct (Commitment := PaperAlgebra.Commitment)
      (shape := productionShape) (carrier := carrier)
      (blockCount := Phi81ColumnLayout.blockCount carrier.carrierWidth)
      program commit productionGlobalParams (statement input)
      (ProductionKey.degreeBound PiDECInputCheck.relation))
    (outcome : StoredOutcome productionShape carrier) :
    SourceReturned (shape := productionShape) (carrier := carrier)
      (blockCount := Phi81ColumnLayout.blockCount carrier.carrierWidth)
      commit productionGlobalParams (statement input)
      (finishStored (StoredWitnessCheckWork.check program) outcome).value ↔
      StrongProbability.RelaxedSuccess
        (width := ProductionKey.degreeBound PiDECInputCheck.relation)
        (PaperAlgebra.openingMaps Poseidon2HashChainV1Setup.productionAjtaiKey) productionGlobalParams
        (statement input) (storedView outcome) ∧
      StrongProbability.SourceValid
        (PaperAlgebra.openingMaps Poseidon2HashChainV1Setup.productionAjtaiKey) productionGlobalParams
        (statement input) (storedView outcome) := by
  refine finish_source_iff input (StoredWitnessCheckWork.check program) ?_ outcome
  intro candidate
  exact StoredWitnessCheckWork.check_value (Commitment := PaperAlgebra.Commitment)
    (shape := productionShape) (carrier := carrier)
    (blockCount := Phi81ColumnLayout.blockCount carrier.carrierWidth)
    (width := ProductionKey.degreeBound PiDECInputCheck.relation)
    program commit productionGlobalParams (statement input) correct candidate

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

/-- Install the proved scalar operations and public/probe array readers.
The public gate and commitment/matrix entry calls stay explicit. -/
def scalarProgram (input : PiCCSInputCheck.Input)
    (program : StoredWitnessCheckWork.Program productionShape carrier) :
    StoredWitnessCheckWork.Program productionShape carrier :=
  StoredWitnessCheckPrimitives.withScalarChecks
    { program with
      publicInput := publicInputRead input
      padClaim := StoredProbe.padRead
      matrixClaim := StoredProbe.matrixRead }

def scalarBounds (bounds : StoredWitnessCheckWork.PrimitiveBounds) : StoredWitnessCheckWork.PrimitiveBounds :=
  StoredWitnessCheckPrimitives.withScalarBounds { bounds with publicInput := 9, padClaim := 4, matrixClaim := 5 }

/-- The selected stored return needs only the four remaining gate/access
refinements. Scalar operations and public/probe reads are proved. -/
theorem scalar_finish_source_iff (input : PiCCSInputCheck.Input)
    (program : StoredWitnessCheckWork.Program productionShape carrier)
    (publicCheck : ∀ probe, (program.publicCheck probe).value =
      ProtocolPolynomial.FixedWidth.check extensionOps (ProductionKey.degreeBound PiDECInputCheck.relation)
        ((statement input).verifierInput K.embed)
        probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
        ((statement input).projectOutput probe.view.response.fullOutput) probe.certificate)
    (commitmentCheck : ∀ stored source, (program.commitmentCheck stored source).value =
      decide (commit (stored.get source).get = (statement input).commitments source))
    (padEntry : ∀ coefficient vertex column, (program.padEntry coefficient vertex column).value =
      (statement input).matrixSource.coefficientMatrixOf baseOps
        (fun row column => (statement input).cubeLayout.paddedIdentityEntry
          baseOps.zero baseOps.one row column) coefficient vertex column)
    (matrixEntry : ∀ matrix coefficient vertex column,
      (program.matrixEntry matrix coefficient vertex column).value =
        (statement input).matrixSource.coefficientMatrix baseOps matrix coefficient vertex column)
    (outcome : StoredOutcome productionShape carrier) :
    SourceReturned (shape := productionShape) (carrier := carrier)
      (blockCount := Phi81ColumnLayout.blockCount carrier.carrierWidth)
      commit productionGlobalParams (statement input)
      (finishStored (StoredWitnessCheckWork.check (scalarProgram input program)) outcome).value ↔
      StrongProbability.RelaxedSuccess
        (width := ProductionKey.degreeBound PiDECInputCheck.relation)
        (PaperAlgebra.openingMaps Poseidon2HashChainV1Setup.productionAjtaiKey) productionGlobalParams
        (statement input) (storedView outcome) ∧
      StrongProbability.SourceValid
        (PaperAlgebra.openingMaps Poseidon2HashChainV1Setup.productionAjtaiKey) productionGlobalParams
        (statement input) (storedView outcome) := by
  apply charged_finish_source_iff input (scalarProgram input program) _ outcome
  exact StoredWitnessCheckPrimitives.withScalarChecks_correct
    (shape := productionShape) (carrier := carrier)
    (blockCount := Phi81ColumnLayout.blockCount carrier.carrierWidth)
    (width := ProductionKey.degreeBound PiDECInputCheck.relation)
    { program with
      publicInput := publicInputRead input
      padClaim := StoredProbe.padRead
      matrixClaim := StoredProbe.matrixRead } commit (statement input)
    publicCheck commitmentCheck (publicInputRead_value input) padEntry matrixEntry
    StoredProbe.padRead_value StoredProbe.matrixRead_value

/-- The selected work bound uses concrete scalar and public/probe read counts.
Only the four remaining gate/access calls need supplied work bounds. -/
theorem scalar_check_work_le (input : PiCCSInputCheck.Input)
    (program : StoredWitnessCheckWork.Program productionShape carrier)
    (bounds : StoredWitnessCheckWork.PrimitiveBounds)
    (publicCheck : ∀ probe, (program.publicCheck probe).work ≤ bounds.publicCheck)
    (commitmentCheck : ∀ stored source, (program.commitmentCheck stored source).work ≤ bounds.commitmentCheck)
    (padEntry : ∀ coefficient vertex column, (program.padEntry coefficient vertex column).work ≤ bounds.padEntry)
    (matrixEntry : ∀ matrix coefficient vertex column,
      (program.matrixEntry matrix coefficient vertex column).work ≤ bounds.matrixEntry)
    (candidate : Candidate) :
    (StoredWitnessCheckWork.check (scalarProgram input program) candidate).work ≤
      StoredWitnessCheckWork.workBound productionShape carrier (scalarBounds bounds) := by
  apply StoredWitnessCheckWork.check_work_le (scalarProgram input program) (scalarBounds bounds) _ candidate
  exact StoredWitnessCheckPrimitives.withScalarChecks_bounded
    { program with
      publicInput := publicInputRead input
      padClaim := StoredProbe.padRead
      matrixClaim := StoredProbe.matrixRead }
    { bounds with publicInput := 9, padClaim := 4, matrixClaim := 5 }
    publicCheck commitmentCheck (publicInputRead_work_le input) padEntry matrixEntry
    (fun probe source coefficient => le_of_eq (StoredProbe.padRead_work probe source coefficient))
    (fun probe source matrix coefficient => le_of_eq (StoredProbe.matrixRead_work probe source matrix coefficient))

end NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck
