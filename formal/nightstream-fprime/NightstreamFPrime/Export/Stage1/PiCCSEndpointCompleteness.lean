import NightstreamFPrime.Export.Stage1.PiCCSCompletedReadout
import NightstreamFPrime.Export.Stage1.PermutationPlan

/-!
Owns the exact endpoint-column connection from the four C transcript packets
to their canonical retained forms. Actual permutation rows provide values;
the existing witness-start schedule provides addresses.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSEndpointCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open PiCCSTranscriptEndpointPlan
open PiCCSPoseidonPreservation
open PerApplicationAssignmentTransportExecution

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

private theorem physicalInvocation_data (index : InvocationIndex) :
    physicalInvocation index = (Data.permutationInvocations ()).get
      ⟨index.val, by
        rw [PoseidonRetainedBlock.data_permutationInvocations_length]
        exact (laterIndex index).isLt⟩ := by
  simp only [physicalInvocation, List.get_eq_getElem,
    PoseidonRetainedBlock.basePackage_permutationInvocations_eq]

/-- Every family endpoint is the output column of its actual last C
permutation. This includes the post-ordinary output-absorption family. -/
theorem physicalEndpoint_column (family : Fin familyCount) (lane : Fin laneCount) :
    (physicalInvocation (endpointInvocation family)).witnessStart + 584 + lane.val =
      Spartan.sourceToSpartan (endpointColumn family lane) := by
  let index := endpointInvocation family
  let sourceStart := if index.val < 718 then
      PiCCSInputs.phaseOffset + index.val * 592
    else PiCCSInvocations.outputWitnessStart + (index.val - 718) * 592
  have startEq : (physicalInvocation index).witnessStart = Spartan.sourceToSpartan sourceStart := by
    rw [physicalInvocation_data]
    by_cases beforeOutput : index.val < 718
    · rw [show sourceStart = PiCCSInputs.phaseOffset + index.val * 592 from if_pos beforeOutput]
      exact PermutationPlan.canonicalInvocation_witnessStart_of_transcript _ beforeOutput
    · rw [show sourceStart = PiCCSInvocations.outputWitnessStart + (index.val - 718) * 592 from if_neg beforeOutput]
      apply PermutationPlan.canonicalInvocation_witnessStart_of_output
      · omega
      · simpa only [PiCCSPoseidonPlan.invocationCount_eq] using
          (show index.val < PiCCSPoseidonPlan.invocationCount from index.isLt)
  have sourceLocal : Spartan.piCcsPhaseOffset ≤ sourceStart := by
    unfold sourceStart
    split
    · rw [PiCCSInputs.phaseOffset_eq]
      change 14751804 ≤ 14751804 + index.val * 592
      omega
    · rw [PiCCSInvocations.outputWitnessStart, PiCCSStarts.outputBindingWitnessStart_eq]
      change 14751804 ≤ 15256706 + (index.val - 718) * 592
      omega
  have endpointEq : sourceStart + 584 + lane.val = endpointColumn family lane := by
    fin_cases family <;>
      norm_num [sourceStart, index, endpointInvocation, endpointColumn, endpointStart,
        PiCCSTranscriptDirectSemantics.statementLast, PiCCSTranscriptDirectSemantics.challengeLast,
        PiCCSTranscriptDirectSemantics.roundLast, PiCCSTranscriptDirectSemantics.outputLast,
        PiCCSTranscriptDirectSemantics.roundCount, PiCCSInputs.phaseOffset_eq,
        PiCCSInvocations.challengeWitnessStart, PiCCSStarts.challengeWitnessStart_eq,
        PiCCSInvocations.roundWitnessStart, PiCCSStarts.roundTranscriptWitnessStart_eq,
        PiCCSInvocations.outputWitnessStart, PiCCSStarts.outputBindingWitnessStart_eq,
        PiCCSStarts.logicalFreshBase]
  calc
    (physicalInvocation (endpointInvocation family)).witnessStart + 584 + lane.val =
        Spartan.sourceToSpartan sourceStart + (584 + lane.val) := by rw [startEq]; omega
    _ = Spartan.sourceToSpartan (sourceStart + (584 + lane.val)) :=
      (Spartan.sourceToSpartan_add_of_piCcsLocal sourceStart (584 + lane.val) sourceLocal).symm
    _ = Spartan.sourceToSpartan (endpointColumn family lane) :=
      congrArg Spartan.sourceToSpartan (by omega)

/-- A canonical C endpoint form reads the same actual source column as the
lifecycle compiler. No endpoint-plan acceptance is assumed. -/
theorem endpointValue_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target))
    (family : Fin familyCount) (lane : Fin laneCount) :
    let raw := canonicalRawValues application (PerApplicationSourceAssignment.ofCompleted application target suffix)
    outputValue (PerApplicationCanonicalEncodes.poseidonGeometry application) raw.assignment
      (endpointInvocation family) lane = Spartan.pullback target (endpointColumn family lane) := by
  intro raw
  have output := congrFun
    (PiCCSCompletedReadout.outputValue_of_completed application relation target suffix physical
      (endpointInvocation family)) lane
  rw [physicalEndpoint_column] at output
  exact output

/-- The actual completed C phase makes all four endpoint pin families zero
on the canonical assignment, with exact source agreement derived from rows. -/
theorem rowsZero_of_completed
    (application : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (physical : NightstreamFPrime.Layout.PiCCS.v1_1.PhysicalHolds relation
      (PiCCSInputs.interface logicalWidth publicFits) PiCCSInputs.phaseOffset (Spartan.pullback target)) :
    let raw := canonicalRawValues application (PerApplicationSourceAssignment.ofCompleted application target suffix)
    (PiCCSTranscriptEndpointPlan.plan (PerApplicationCanonicalEncodes.poseidonGeometry application)
      (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry application)).RowsZero raw.assignment := by
  intro raw
  apply (PiCCSTranscriptEndpointPlan.rowsZero_iff _ _ raw.assignment
    (PerApplicationCanonicalAssignment.assignment_one raw)).mpr
  intro rowIndex
  let selected := descriptor rowIndex
  have direct := endpointValue_of_completed application relation target suffix physical selected.1 selected.2
  have source := sourceForm_eval (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry application)
    raw.assignment raw.base raw.groupValue raw.products
    (PerApplicationCanonicalEncodes.samplerPrefixEncodes raw).prior.pilotOrdinary.prior selected.1 selected.2
  have bound := endpointColumn_lt_source selected.1 selected.2
  have same := packageEnv_sourceAssignment application raw.base raw.groupValue raw.products _ bound
  have copied := PiCCSCompletedReadout.transitionEnv_of_completed application relation target suffix
    physical (endpointColumn selected.1 selected.2) bound
  have sourceValue : (sourceForm (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry application)
      selected.1 selected.2).eval raw.assignment =
      Spartan.pullback target (endpointColumn selected.1 selected.2) :=
    source.trans (same.trans copied)
  change (directForm (PerApplicationCanonicalEncodes.poseidonGeometry application)
    selected.1 selected.2).eval raw.assignment = _ at direct
  change (SparseForm.add (directForm (PerApplicationCanonicalEncodes.poseidonGeometry application)
      selected.1 selected.2)
    (SparseForm.scale (-1) (sourceForm (PerApplicationCanonicalEncodes.piCcsOrdinaryGeometry application)
      selected.1 selected.2))).eval raw.assignment = 0
  rw [SparseForm.add_eval, SparseForm.scale_eval, direct, sourceValue]
  simp

end NightstreamFPrime.Export.Stage1.PiCCSEndpointCompleteness
