import NightstreamFPrime.Export.Stage1.ApplicationWitnessCompleteness
import NightstreamFPrime.Export.Stage1.PilotHashRowsCompleteness
import NightstreamFPrime.Export.Stage1.PilotCompletedAssignment
import NightstreamFPrime.Export.Stage1.PiCCSCompletedAssignment
import NightstreamFPrime.Export.Stage1.PiRLCSamplerPoseidonCompleteness
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryCompleteness
import NightstreamFPrime.Export.Stage1.PiRLCRetainedCompleteness
import NightstreamFPrime.Export.Stage1.PiDECCompletedAssignment
import NightstreamFPrime.Export.Stage1.RunningTransitionCompletedAssignment
import NightstreamFPrime.Export.Stage1.NextPreimageCompleteness
import NightstreamFPrime.Export.Stage1.CanonicalPublicOutput

/-!
Owns construction of the complete selected plan from a semantic step and its
accepted NIFS advice. All phase rows, the strict carrier norm, public digest
and application advice refer to the same canonical completed assignment.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.SelectedAssignmentCompleteness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open PerApplicationCanonicalAssignment
open PerApplicationAssignmentTransportExecution
open Poseidon2HashChainV1Package (application fits)

private theorem prefix_rowsZero_of_completed
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (program : Lifecycle.Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (target : Env)
    (suffix : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (physical : R1CS.RowsHold target (Spartan.remappedRows relation)) :
    let raw := canonicalRawValues program
      (PerApplicationSourceAssignment.ofCompleted program target suffix)
    (DirectPiRLCSamplerCompletePrefixPlan.plan relation
      (PerApplicationCanonicalEncodes.samplerGeometry program)).RowsZero raw.assignment := by
  intro raw
  have pilot := PilotHashRowsCompleteness.rowsZero_of_spartanRows
    relation program target suffix physical
  have pilotRemaining := PilotCompletedAssignment.rowsZero_of_completed
    program relation target suffix physical
  have c := PiCCSCompletedAssignment.rowsZero_of_completed
    program relation target suffix physical
  have sampler := PiRLCSamplerPoseidonCompleteness.rowsZero_of_completed
    program relation ajtai target suffix physical
  have samplerOrdinary := PiRLCSamplerOrdinaryCompleteness.rowsZero_of_completed
    program relation ajtai target suffix physical
  have r := PiRLCRetainedCompleteness.rowsZero_of_completed
    program relation ajtai target suffix physical
  have d := PiDECCompletedAssignment.rowsZero_of_completed
    program relation target suffix physical
  have running := RunningTransitionCompletedAssignment.rowsZero_of_completed
    program relation target suffix physical
  have samplerPrefix :
      (DirectPiDECPrefixPlan.samplerPrefixPlan relation
        (PerApplicationCanonicalEncodes.piDecGeometry program)).RowsZero raw.assignment := by
    rw [DirectPiDECPrefixPlan.samplerPrefixPlan, Plan.append_rowsZero_iff]
    rw [DirectPiDECPrefixPlan.piCcsCompletePlan, Plan.append_rowsZero_iff]
    rw [DirectPiDECPrefixPlan.pilotBindingPrefixPlan, Plan.append_rowsZero_iff]
    rw [DirectPiDECPrefixPlan.pilotOrdinaryPrefixPlan, Plan.append_rowsZero_iff]
    rw [DirectPiDECPrefixPlan.piCcsCorePlan, Plan.append_rowsZero_iff]
    rw [DirectPiDECPrefixPlan.piCcsPoseidonPrefix, Plan.append_rowsZero_iff]
    exact ⟨⟨⟨⟨⟨⟨pilot, c.2.1⟩, c.1⟩, pilotRemaining.1⟩,
      pilotRemaining.2⟩, c.2.2⟩, sampler⟩
  apply (DirectPiRLCSamplerCompletePrefixPlan.rowsZero_iff relation
    (PerApplicationCanonicalEncodes.samplerGeometry program) raw.assignment).mpr
  exact ⟨samplerPrefix, samplerOrdinary, r, d, running⟩

/-- A selected semantic step constructs one complete canonical assignment.
Every structural row, the strict carrier norm, the public digest and the exact
application advice follow from the existing physical and retained builders.
No row validity, environment agreement or completeness callback is supplied. -/
theorem complete
    (ajtai : AjtaiKey
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (context : VerifierContext.Digest4)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application))
      (Lifecycle.Proof 9) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
        (publicFits := PerApplicationFixedPoint.publicFits application)) slotCount)
    (result : Running
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (step : StepHoldsFor (PerApplicationFixedPoint.relation application fits)
      ajtai context.toList application input output)
    (priorWellFormed : StateEncoding.WellFormed
      (priorHashPreimage (setup (PerApplicationFixedPoint.relation application fits)
        ajtai context.toList) input))
    (nextWellFormed : StateEncoding.WellFormed
      (nextHashPreimage (setup (PerApplicationFixedPoint.relation application fits)
        ajtai context.toList) input output))
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage
        (setup (PerApplicationFixedPoint.relation application fits) ajtai context.toList) input)))
    (accepted : Nifs.PaperNonInteractive.verify
      (ProductionKey.key (PerApplicationFixedPoint.relation application fits) ajtai)
      (input.running functionIndex) input.fresh input.nifsProof = some result)
    (recursiveResult : 0 < input.iteration → result = output.runningNext functionIndex)
    (witnessWidth : input.witness.length = Stage1.Poseidon2HashChainV1.messageWordCount) :
    ∃ raw : RawValues application,
      (PerApplicationFixedPoint.structuralPlan application fits).RowsZero raw.assignment ∧
      (∀ column, centeredMagnitude (raw.completeAssignment column) < 2) ∧
      Phi81Relation.projectPublicInput raw.completeAssignment = encHash output.x ∧
      Stage1.Application.witnessValue (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application) (SourceCompiler.sourceEnv raw.base) = input.witness ∧
      raw.outputDigest = output.x := by
  obtain ⟨digestFixed, target, suffix, _constant, physical, nextRows,
      _nifsOutput, _sources, _priorWords, _nextWords, _applicationSourceRows,
      actualWitness, outputDigest, publicInput, applicationRows⟩ :=
    ApplicationWitnessCompleteness.complete
      (PerApplicationFixedPoint.relation application fits) ajtai context input output result
      step priorWellFormed nextWellFormed freshPublic accepted recursiveResult witnessWidth
  let raw := canonicalRawValues application
    (PerApplicationSourceAssignment.ofCompleted application target suffix)
  have prefixRows := prefix_rowsZero_of_completed application
    (PerApplicationFixedPoint.relation application fits) ajtai target suffix physical
  have nextPlanRows := NextPreimageCompleteness.rowsZero_of_completed application target suffix nextRows
  have publicRows := CanonicalPublicOutput.rowsZero raw
  have allRows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero raw.assignment := by
    rw [← PerApplicationFixedPoint.plan_fixedPoint application fits]
    apply (DirectApplicationPrefixPlan.rowsZero_iff
      (PerApplicationFixedPoint.relation application fits) fits.package
      (PerApplicationFixedPoint.geometry application) raw.assignment).mpr
    exact ⟨⟨⟨prefixRows, applicationRows⟩, nextPlanRows⟩, publicRows⟩
  exact ⟨raw, allRows,
    PerApplicationSourceAssignment.completeAssignment_norm_of_physical
      application fits ajtai target suffix physical,
    publicInput, actualWitness, outputDigest⟩

end NightstreamFPrime.Export.Stage1.SelectedAssignmentCompleteness
