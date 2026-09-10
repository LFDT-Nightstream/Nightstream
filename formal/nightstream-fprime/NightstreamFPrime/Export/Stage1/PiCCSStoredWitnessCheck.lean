import NightstreamFPrime.Export.Stage1.PiDECInputCheck
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheck
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.StoredWitnessCheckWork
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

abbrev carrier : Phi81Relation.Shape :=
  PaperAlgebra.FullShape PiDECInputCheck.logicalWidth PiDECInputCheck.publicFits

abbrev StoredWitness := StoredWitnessProjection.StoredWitness productionShape carrier
abbrev Candidate := Probe K productionShape × StoredWitness

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
    (ProductionKey.degreeBound PiDECInputCheck.relation) candidate

/-- Exact selected public/ambient predicate; opening validity is checked. -/
theorem check_eq_true_iff (input : PiCCSInputCheck.Input)
    (probe : Probe K productionShape) (stored : StoredWitness) :
    check input (probe, stored) = true ↔
      probe.FixedWidthAccepted extensionOps K.embed
        ((ProductionKey.key PiDECInputCheck.relation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
          (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input))
        (ProductionKey.degreeBound PiDECInputCheck.relation) ∧
      AmbientOutputHolds extensionOps K.embed
        (PaperAlgebra.openingMaps Poseidon2HashChainV1Setup.productionAjtaiKey) productionGlobalParams
        ((ProductionKey.key PiDECInputCheck.relation Poseidon2HashChainV1Setup.productionAjtaiKey).statement
          (PiCCSInputCheck.running input) (PiCCSInputCheck.fresh input))
        probe (StoredWitnessProjection.view stored) := by
  rw [← statement_eq_key input, ← selected_openingMaps]
  exact StoredWitnessCheck.check_eq_true_iff
    (shape := productionShape) (carrier := carrier)
    (blockCount := Phi81ColumnLayout.blockCount carrier.carrierWidth)
    commit productionGlobalParams (statement input)
    (ProductionKey.degreeBound PiDECInputCheck.relation) probe stored

/-- The remaining value-refinement obligation is equality to an implemented
Boolean check. The charged call's work is retained without a proposed bound. -/
theorem chargedCheck_correct (input : PiCCSInputCheck.Input)
    (charged : StoredOneRunExtraction.Check productionShape carrier)
    (computes : ∀ candidate, (charged candidate).value = check input candidate) :
    ∀ probe stored, (charged (probe, stored)).value = true ↔
      probe.FixedWidthAccepted extensionOps K.embed (statement input)
        (ProductionKey.degreeBound PiDECInputCheck.relation) ∧
      AmbientOutputHolds extensionOps K.embed (openingMaps commit) productionGlobalParams
        (statement input) probe (StoredWitnessProjection.view stored) := by
  intro probe stored
  rw [computes]
  exact StoredWitnessCheck.check_eq_true_iff commit productionGlobalParams (statement input)
    (ProductionKey.degreeBound PiDECInputCheck.relation) probe stored

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

end NightstreamFPrime.Export.Stage1.PiCCSStoredWitnessCheck
