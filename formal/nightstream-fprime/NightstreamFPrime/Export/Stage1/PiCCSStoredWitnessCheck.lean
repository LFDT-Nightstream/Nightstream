import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import NightstreamFPrime.Export.Stage1.SecurityInstance
import NightstreamFPrime.Export.Stage1.PiCCSStoredPublicCheck
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheck
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckWork
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckPrimitives
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckEntries
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredOneRunExtraction

/-!
Selected Appendix B.2 witness check for the actual application matrix source
and frozen Ajtai setup. The statement reads the existing typed fresh/running
public fields. It equals the statement selected by `ProductionKey.key`.

The selected charged checker has concrete public-gate, scalar, stored-read,
and Pad-entry counts. Commitment/key and CCS matrix-entry work remain
explicit primitive contracts. The candidate coins remain explicit; no
Fiat--Shamir distribution is claimed.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open NightstreamFPrime.Lifecycle
open CheckedWitnessExtraction
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

variable (inst : SecurityInstance)

abbrev carrier : Phi81Relation.Shape :=
  PaperAlgebra.FullShape inst.logicalWidth inst.publicFits

abbrev StoredWitness := StoredWitnessProjection.StoredWitness productionShape (carrier inst)
abbrev Candidate := StoredProbe productionShape × StoredWitness inst

/-- The sole selected commitment map, including the actual indexed key expansion. -/
def commit : Phi81Relation.Assignment (carrier inst) → PaperAlgebra.Commitment :=
  (PaperAlgebra.openingMaps inst.ajtai).commit

private theorem openingMaps_of_projection {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (key : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    openingMaps (carrier := PaperAlgebra.FullShape logicalWidth publicFits)
      (PaperAlgebra.openingMaps key).commit = PaperAlgebra.openingMaps key := by
  rfl

private theorem selected_openingMaps : openingMaps (carrier := carrier inst) (commit inst) =
    PaperAlgebra.openingMaps inst.ajtai :=
  openingMaps_of_projection (logicalWidth := inst.logicalWidth)
    (publicFits := inst.publicFits) inst.ajtai

/-- Computable projection of the selected key's statement. Matrix entries
remain behind the existing selected relation's access function. -/
def statement (input : PiCCSInputCheck.Input) :
    Statement K PaperAlgebra.Commitment (Phi81Relation.PublicInput (carrier inst))
      productionShape (carrier inst).carrierWidth
      (Phi81ColumnLayout.blockCount (carrier inst).carrierWidth) baseOps where
  cubeLayout := (Lifecycle.PiRLC.v1_1.InputBinding.relationSource inst.relation).cubeLayout
  matrixSource := (Lifecycle.PiRLC.v1_1.InputBinding.relationSource inst.relation).matrixSource
  commitments := PiCCSInputCheck.outputCommitments input
  publicInputs := PiCCSInputCheck.outputPublicInputs input
  priorPoint := (inst.running input).point
  claimedPadCoefficient := (PiCCSInputCheck.verifierInput input).claimedPadCoefficient
  claimedMatrixCoefficient := (PiCCSInputCheck.verifierInput input).claimedMatrixCoefficient

/-- The executable statement is the literal selected NIFS statement. -/
theorem statement_eq_key (input : PiCCSInputCheck.Input) :
    statement inst input =
      (ProductionKey.key inst.relation inst.ajtai).statement
        (inst.running input) (inst.fresh input) := by
  rfl

/-- Check a returned probe and its stored witness at the selected statement. -/
def check (input : PiCCSInputCheck.Input) (candidate : Candidate inst) : Bool :=
  StoredWitnessCheck.check (commit inst) productionGlobalParams (statement inst input)
    (ProductionKey.degreeBound inst.relation) (candidate.1.view, candidate.2)

/-- Exact selected public/ambient predicate; opening validity is checked. -/
theorem check_eq_true_iff (input : PiCCSInputCheck.Input)
    (probe : StoredProbe productionShape) (stored : StoredWitness inst) :
    check inst input (probe, stored) = true ↔
      probe.view.FixedWidthAccepted extensionOps K.embed
        ((ProductionKey.key inst.relation inst.ajtai).statement
          (inst.running input) (inst.fresh input))
        (ProductionKey.degreeBound inst.relation) ∧
      AmbientOutputHolds extensionOps K.embed
        (PaperAlgebra.openingMaps inst.ajtai) productionGlobalParams
        ((ProductionKey.key inst.relation inst.ajtai).statement
          (inst.running input) (inst.fresh input))
        probe.view (StoredWitnessProjection.view stored) := by
  rw [← statement_eq_key inst input, ← selected_openingMaps inst]
  exact StoredWitnessCheck.check_eq_true_iff
    (shape := productionShape) (carrier := carrier inst)
    (blockCount := Phi81ColumnLayout.blockCount (carrier inst).carrierWidth)
    (commit inst) productionGlobalParams (statement inst input)
    (ProductionKey.degreeBound inst.relation) probe.view stored

private def checkedSourceValue {shape : Shape} {carrier : Phi81Relation.Shape}
    (checked : (StoredProbe shape × StoredWitnessProjection.StoredWitness shape carrier) → Bool) :
    StoredOutcome shape carrier → Option (WitnessProjection.SourceWitness shape carrier)
  | none => none
  | some candidate =>
      if checked candidate then some (StoredWitnessProjection.project candidate.2).value
      else none

private theorem checkedSourceValue_source_iff
    {Commitment : Type*} {shape : Shape} {carrier : Phi81Relation.Shape}
    {blockCount width : Nat}
    (checked : (StoredProbe shape × StoredWitnessProjection.StoredWitness shape carrier) → Bool)
    (commit : Phi81Relation.Assignment carrier → Commitment) (params : GlobalParams)
    (statement : Statement K Commitment (Phi81Relation.PublicInput carrier)
      shape carrier.carrierWidth blockCount baseOps)
    (correct : ∀ probe stored, checked (probe, stored) = true ↔
      probe.view.FixedWidthAccepted extensionOps K.embed statement width ∧
        AmbientOutputHolds extensionOps K.embed (openingMaps commit) params statement probe.view
          (StoredWitnessProjection.view stored))
    (outcome : StoredOutcome shape carrier) :
    SourceReturned commit params statement (checkedSourceValue checked outcome) ↔
      StrongProbability.RelaxedSuccess (width := width) (openingMaps commit) params statement (storedView outcome) ∧
        StrongProbability.SourceValid (openingMaps commit) params statement (storedView outcome) := by
  cases outcome with
  | none =>
      simp [checkedSourceValue, SourceReturned, storedView,
        StrongProbability.RelaxedSuccess, StrongProbability.SourceValid]
  | some candidate =>
      rcases candidate with ⟨probe, stored⟩
      by_cases accepted : checked (probe, stored) = true
      · have valid := (correct probe stored).mp accepted
        have reconstructed := StoredWitnessProjection.reconstruct_project statement.publicInputs stored
          (fun source => (valid.2 (UnifiedSources.freshSourceIndex source)).1.2.1)
        simp [checkedSourceValue, accepted, SourceReturned, storedView,
          StrongProbability.RelaxedSuccess, StrongProbability.SourceValid,
          valid.1, valid.2, reconstructed]
      · have rejected : ¬ (probe.view.FixedWidthAccepted extensionOps K.embed statement width ∧
            AmbientOutputHolds extensionOps K.embed (openingMaps commit) params statement probe.view
              (StoredWitnessProjection.view stored)) := fun valid => accepted ((correct probe stored).mpr valid)
        simp [checkedSourceValue, accepted, SourceReturned, storedView,
          StrongProbability.RelaxedSuccess, StrongProbability.SourceValid, rejected]

/-- Execute the selected public/ambient Boolean check on the actual stored
candidate. Abort and rejection return none. Acceptance returns the existing
source projection. This value-only entrypoint assigns no checker clock. -/
def finishValue (input : PiCCSInputCheck.Input) (outcome : StoredOutcome productionShape (carrier inst)) :
    Option (WitnessProjection.SourceWitness productionShape (carrier inst)) :=
  checkedSourceValue (check inst input) outcome

/-- The selected checked return has exactly the existing B.2 source event,
with the actual Ajtai key and all matrix entries. SourceReturned uses the
original source relation and verifier-owned public prefixes. No checker or
primitive correctness premise remains; no work or EPT claim is made. -/
theorem finishValue_source_iff (input : PiCCSInputCheck.Input)
    (outcome : StoredOutcome productionShape (carrier inst)) :
    SourceReturned (shape := productionShape) (carrier := carrier inst)
      (blockCount := Phi81ColumnLayout.blockCount (carrier inst).carrierWidth)
      (commit inst) productionGlobalParams (statement inst input) (finishValue inst input outcome) ↔
      StrongProbability.RelaxedSuccess
        (width := ProductionKey.degreeBound inst.relation)
        (PaperAlgebra.openingMaps inst.ajtai) productionGlobalParams
        (statement inst input) (storedView outcome) ∧
      StrongProbability.SourceValid
        (PaperAlgebra.openingMaps inst.ajtai) productionGlobalParams
        (statement inst input) (storedView outcome) := by
  rw [← selected_openingMaps inst]
  exact checkedSourceValue_source_iff (shape := productionShape) (carrier := carrier inst)
    (blockCount := Phi81ColumnLayout.blockCount (carrier inst).carrierWidth)
    (width := ProductionKey.degreeBound inst.relation)
    (check inst input) (commit inst) productionGlobalParams (statement inst input)
    (fun probe stored => StoredWitnessCheck.check_eq_true_iff
      (shape := productionShape) (carrier := carrier inst)
      (blockCount := Phi81ColumnLayout.blockCount (carrier inst).carrierWidth)
      (commit inst) productionGlobalParams (statement inst input)
      (ProductionKey.degreeBound inst.relation) probe.view stored) outcome

/-- The remaining value-refinement obligation is equality to an implemented
Boolean check. The charged call's work is retained without a proposed bound. -/
theorem chargedCheck_correct (input : PiCCSInputCheck.Input)
    (charged : StoredOneRunExtraction.Check productionShape (carrier inst))
    (computes : ∀ candidate, (charged candidate).value = check inst input candidate) :
    ∀ probe stored, (charged (probe, stored)).value = true ↔
      probe.view.FixedWidthAccepted extensionOps K.embed (statement inst input)
        (ProductionKey.degreeBound inst.relation) ∧
      AmbientOutputHolds extensionOps K.embed (openingMaps (commit inst)) productionGlobalParams
        (statement inst input) probe.view (StoredWitnessProjection.view stored) := by
  intro probe stored
  rw [computes]
  exact StoredWitnessCheck.check_eq_true_iff (commit inst) productionGlobalParams (statement inst input)
    (ProductionKey.degreeBound inst.relation) probe.view stored

/-- The selected source return consumes the proved checker correctness.
Only the charged implementation's value refinement remains as a premise. -/
theorem finish_source_iff (input : PiCCSInputCheck.Input)
    (charged : StoredOneRunExtraction.Check productionShape (carrier inst))
    (computes : ∀ candidate, (charged candidate).value = check inst input candidate)
    (outcome : StoredOutcome productionShape (carrier inst)) :
    SourceReturned (shape := productionShape) (carrier := carrier inst)
      (blockCount := Phi81ColumnLayout.blockCount (carrier inst).carrierWidth)
      (commit inst) productionGlobalParams (statement inst input) (finishStored charged outcome).value ↔
      StrongProbability.RelaxedSuccess
        (width := ProductionKey.degreeBound inst.relation)
        (PaperAlgebra.openingMaps inst.ajtai) productionGlobalParams
        (statement inst input) (storedView outcome) ∧
      StrongProbability.SourceValid
        (PaperAlgebra.openingMaps inst.ajtai) productionGlobalParams
        (statement inst input) (storedView outcome) := by
  rw [← selected_openingMaps inst]
  exact finishStored_source_iff (shape := productionShape) (carrier := carrier inst)
    (blockCount := Phi81ColumnLayout.blockCount (carrier inst).carrierWidth)
    (width := ProductionKey.degreeBound inst.relation)
    charged (commit inst) productionGlobalParams (statement inst input)
    (chargedCheck_correct inst input charged computes) outcome

/-- The implemented charged driver supplies the selected checker refinement.
Its primitive implementation contracts remain explicit; the source-return
path has no separate assumed opening or whole-checker correctness premise. -/
theorem charged_finish_source_iff (input : PiCCSInputCheck.Input)
    (program : StoredWitnessCheckWork.Program productionShape (carrier inst))
    (correct : StoredWitnessCheckWork.Correct (Commitment := PaperAlgebra.Commitment)
      (shape := productionShape) (carrier := carrier inst)
      (blockCount := Phi81ColumnLayout.blockCount (carrier inst).carrierWidth)
      program (commit inst) productionGlobalParams (statement inst input)
      (ProductionKey.degreeBound inst.relation))
    (outcome : StoredOutcome productionShape (carrier inst)) :
    SourceReturned (shape := productionShape) (carrier := carrier inst)
      (blockCount := Phi81ColumnLayout.blockCount (carrier inst).carrierWidth)
      (commit inst) productionGlobalParams (statement inst input)
      (finishStored (StoredWitnessCheckWork.check program) outcome).value ↔
      StrongProbability.RelaxedSuccess
        (width := ProductionKey.degreeBound inst.relation)
        (PaperAlgebra.openingMaps inst.ajtai) productionGlobalParams
        (statement inst input) (storedView outcome) ∧
      StrongProbability.SourceValid
        (PaperAlgebra.openingMaps inst.ajtai) productionGlobalParams
        (statement inst input) (storedView outcome) := by
  refine finish_source_iff inst input (StoredWitnessCheckWork.check program) ?_ outcome
  intro candidate
  exact StoredWitnessCheckWork.check_value (Commitment := PaperAlgebra.Commitment)
    (shape := productionShape) (carrier := carrier inst)
    (blockCount := Phi81ColumnLayout.blockCount (carrier inst).carrierWidth)
    (width := ProductionKey.degreeBound inst.relation)
    program (commit inst) productionGlobalParams (statement inst input) correct candidate

/-- Read the actual fresh/running public arrays. Both paths charge source
index read, comparison, branch, data reads and return; the running path also
charges index subtraction. The costs are six and nine operations. -/
def publicInputRead (input : PiCCSInputCheck.Input) (source : Fin productionShape.sourceCount)
    (column : Fin (carrier inst).publicWidth) : Result F :=
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
    (column : Fin (carrier inst).publicWidth) :
    (publicInputRead inst input source column).value = (statement inst input).publicInputs source column := by
  change (publicInputRead inst input source column).value =
    (Fin.addCases (motive := fun _ => Phi81Relation.PublicInput (carrier inst))
      (inst.fresh input).publicInputs
      (inst.running input).publicInputs source) column
  refine Fin.addCases (fun fresh => ?_) (fun running => ?_) source
  · have isFresh : (Fin.castAdd productionShape.runningCount fresh).val < productionShape.freshCount := fresh.isLt
    rw [publicInputRead, dif_pos isFresh, Fin.addCases_left]
    rfl
  · have isRunning : ¬ (Fin.natAdd productionShape.freshCount running).val < productionShape.freshCount := by
      change ¬ productionShape.freshCount + running.val < productionShape.freshCount
      omega
    rw [publicInputRead, dif_neg isRunning, Fin.addCases_right]
    simp only
        [SecurityInstance.running, PiCCSInputCheck.runningAt, PiCCSInputCheck.runningFromInput,
        Fin.natAdd, Nat.add_sub_cancel_left]

theorem publicInputRead_work (input : PiCCSInputCheck.Input) (source : Fin productionShape.sourceCount)
    (column : Fin (carrier inst).publicWidth) :
    (publicInputRead inst input source column).work = if source.val < productionShape.freshCount
        then 6 else 9 := by
  unfold publicInputRead
  split <;> simp_all

theorem publicInputRead_work_le (input : PiCCSInputCheck.Input) (source : Fin productionShape.sourceCount)
    (column : Fin (carrier inst).publicWidth) : (publicInputRead inst input source column).work ≤ 9 := by
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
    (vertex : BooleanVertex productionShape.cubeVariables) (column : Fin (carrier inst).carrierWidth) :
    (StoredWitnessCheckEntries.padEntry coefficient vertex column).value =
      (statement inst input).matrixSource.coefficientMatrixOf baseOps
        (fun row column => (statement inst input).cubeLayout.paddedIdentityEntry
          baseOps.zero baseOps.one row column) coefficient vertex column :=
  padEntry_of_relation (logicalWidth := inst.logicalWidth)
    (publicFits := inst.publicFits) inst.relation coefficient vertex column

private theorem kernelConstant_of_relation {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation).matrixSource.kernel.constant =
      Phi81CoefficientKernel.constant := by
  rfl

private theorem outputMessage_of_constant {Commitment PublicInput : Type}
    {columns blockCount : Nat}
    (selected : Statement K Commitment PublicInput productionShape columns blockCount baseOps)
    (constant : selected.matrixSource.kernel.constant = Phi81CoefficientKernel.constant)
    (probe : StoredProbe productionShape) :
    PiCCSStoredPublicCheck.outputMessage probe = selected.projectOutput probe.view.response.fullOutput := by
  simp only [PiCCSStoredPublicCheck.outputMessage, Statement.projectOutput, constant]

/-- The complete counted public gate checks this selected statement for
every stored probe, retaining all raw-certificate rejection cases. -/
theorem publicCheck_value (input : PiCCSInputCheck.Input) (probe : StoredProbe productionShape) :
    (PiCCSStoredPublicCheck.check input probe).value =
      ProtocolPolynomial.FixedWidth.check extensionOps (ProductionKey.degreeBound inst.relation)
        ((statement inst input).verifierInput K.embed)
        probe.coins.alpha probe.coins.gamma probe.coins.roundPoint
        ((statement inst input).projectOutput probe.view.response.fullOutput) probe.certificate := by
  -- The checker's input does not depend on the width; this is definitional.
  have selectedInput : PiCCSInputCheck.verifierInput input = (statement inst input).verifierInput
      K.embed := rfl
  have kernel := kernelConstant_of_relation (logicalWidth := inst.logicalWidth)
    (publicFits := inst.publicFits) inst.relation
  have selectedOutput := outputMessage_of_constant (statement inst input) kernel probe
  rw [PiCCSStoredPublicCheck.check_value, ProductionKey.degreeBound_eq, selectedInput, selectedOutput]

/-- Install the public gate, scalar operations, stored readers, and Pad.
Only the commitment and CCS matrix-entry calls stay explicit. -/
def scalarProgram (input : PiCCSInputCheck.Input)
    (program : StoredWitnessCheckWork.Program productionShape (carrier inst)) :
    StoredWitnessCheckWork.Program productionShape (carrier inst) :=
  StoredWitnessCheckPrimitives.withScalarChecks
    { program with
      publicCheck := PiCCSStoredPublicCheck.check input
      publicInput := publicInputRead inst input
      padEntry := StoredWitnessCheckEntries.padEntry
      padClaim := StoredProbe.padRead
      matrixClaim := StoredProbe.matrixRead }

def scalarBounds (bounds : StoredWitnessCheckWork.PrimitiveBounds) : StoredWitnessCheckWork.PrimitiveBounds :=
  StoredWitnessCheckPrimitives.withScalarBounds
    { bounds with
      publicCheck := PiCCSStoredPublicCheck.workBound
      publicInput := 9
      padEntry := StoredWitnessCheckEntries.padWork productionShape.cubeVariables
      padClaim := 4
      matrixClaim := 5 }

/-- The selected stored return needs only commitment and matrix-entry
refinements. The public gate, scalar checks, stored reads, and Pad are proved. -/
theorem scalar_finish_source_iff (input : PiCCSInputCheck.Input)
    (program : StoredWitnessCheckWork.Program productionShape (carrier inst))
    (commitmentCheck : ∀ stored source, (program.commitmentCheck stored source).value =
      decide ((commit inst) (stored.get source).get = (statement inst input).commitments source))
    (matrixEntry : ∀ matrix coefficient vertex column,
      (program.matrixEntry matrix coefficient vertex column).value =
        (statement inst input).matrixSource.coefficientMatrix baseOps matrix coefficient vertex column)
    (outcome : StoredOutcome productionShape (carrier inst)) :
    SourceReturned (shape := productionShape) (carrier := carrier inst)
      (blockCount := Phi81ColumnLayout.blockCount (carrier inst).carrierWidth)
      (commit inst) productionGlobalParams (statement inst input)
      (finishStored (StoredWitnessCheckWork.check (scalarProgram inst input program)) outcome).value ↔
      StrongProbability.RelaxedSuccess
        (width := ProductionKey.degreeBound inst.relation)
        (PaperAlgebra.openingMaps inst.ajtai) productionGlobalParams
        (statement inst input) (storedView outcome) ∧
      StrongProbability.SourceValid
        (PaperAlgebra.openingMaps inst.ajtai) productionGlobalParams
        (statement inst input) (storedView outcome) := by
  apply charged_finish_source_iff inst input (scalarProgram inst input program) _ outcome
  exact StoredWitnessCheckPrimitives.withScalarChecks_correct
    (shape := productionShape) (carrier := carrier inst)
    (blockCount := Phi81ColumnLayout.blockCount (carrier inst).carrierWidth)
    (width := ProductionKey.degreeBound inst.relation)
    { program with
      publicCheck := PiCCSStoredPublicCheck.check input
      publicInput := publicInputRead inst input
      padEntry := StoredWitnessCheckEntries.padEntry
      padClaim := StoredProbe.padRead
      matrixClaim := StoredProbe.matrixRead } (commit inst) (statement inst input)
    (publicCheck_value inst input) commitmentCheck (publicInputRead_value inst input)
        (padEntry_value inst input) matrixEntry
    StoredProbe.padRead_value StoredProbe.matrixRead_value

/-- The selected work bound includes the full public gate and Pad execution.
Only commitment and CCS matrix-entry calls need supplied work bounds. -/
theorem scalar_check_work_le (input : PiCCSInputCheck.Input)
    (program : StoredWitnessCheckWork.Program productionShape (carrier inst))
    (bounds : StoredWitnessCheckWork.PrimitiveBounds)
    (commitmentCheck : ∀ stored source, (program.commitmentCheck stored source).work ≤ bounds.commitmentCheck)
    (matrixEntry : ∀ matrix coefficient vertex column,
      (program.matrixEntry matrix coefficient vertex column).work ≤ bounds.matrixEntry)
    (candidate : Candidate inst) :
    (StoredWitnessCheckWork.check (scalarProgram inst input program) candidate).work ≤
      StoredWitnessCheckWork.workBound productionShape (carrier inst) (scalarBounds bounds) := by
  apply StoredWitnessCheckWork.check_work_le (scalarProgram inst input program)
      (scalarBounds bounds) _ candidate
  exact StoredWitnessCheckPrimitives.withScalarChecks_bounded
    { program with
      publicCheck := PiCCSStoredPublicCheck.check input
      publicInput := publicInputRead inst input
      padEntry := StoredWitnessCheckEntries.padEntry
      padClaim := StoredProbe.padRead
      matrixClaim := StoredProbe.matrixRead }
    { bounds with
      publicCheck := PiCCSStoredPublicCheck.workBound
      publicInput := 9
      padEntry := StoredWitnessCheckEntries.padWork productionShape.cubeVariables
      padClaim := 4
      matrixClaim := 5 }
    (PiCCSStoredPublicCheck.check_work_le input) commitmentCheck (publicInputRead_work_le inst input)
    (fun coefficient vertex column => StoredWitnessCheckEntries.padEntry_work_le coefficient vertex column) matrixEntry
    (fun probe source coefficient => le_of_eq (StoredProbe.padRead_work probe source coefficient))
    (fun probe source matrix coefficient => le_of_eq (StoredProbe.matrixRead_work probe source matrix coefficient))

end NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck
