import NightstreamFPrime.Export.Stage1.Wide.PackageAuthority
import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCorrectness
import NightstreamFPrime.Export.Stage1.Wide.SelectedAssignmentCompleteness

/-! The prepared package's exact transport accepts the constructive physical
witness and returns the proved complete low-norm assignment. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PackageCompleteness

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint Spec.HyperNova.Construction2.Paper
open Layout.Stage1 ProductionRelation
open Poseidon2HashChainV1Package (application fits)

theorem complete (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (parts : AuthorityStream.Parts) (prepared : AuthorityStream.prepare compiled = .ok parts)
    (ajtai : AjtaiKey
      (logicalWidth := RetainedLayout.logicalWidth application)
      (publicFits := FixedPoint.publicFits application))
    (context : VerifierContext.Digest4)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := RetainedLayout.logicalWidth application)
        (publicFits := FixedPoint.publicFits application))
      (Fresh (logicalWidth := RetainedLayout.logicalWidth application)
        (publicFits := FixedPoint.publicFits application))
      (Lifecycle.Proof 9) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := RetainedLayout.logicalWidth application)
        (publicFits := FixedPoint.publicFits application)) slotCount)
    (result : Running
      (logicalWidth := RetainedLayout.logicalWidth application)
      (publicFits := FixedPoint.publicFits application))
    (step : Lifecycle.Stage1.Wide.Relation.StepHoldsFor (FixedPoint.relation application compiled fits)
      ajtai context.toList application input output)
    (priorWellFormed : StateEncoding.WellFormed
      (priorHashPreimage (Lifecycle.Stage1.Wide.Relation.setup (FixedPoint.relation application compiled fits)
        ajtai context.toList) input))
    (nextWellFormed : StateEncoding.WellFormed
      (nextHashPreimage (Lifecycle.Stage1.Wide.Relation.setup (FixedPoint.relation application compiled fits)
        ajtai context.toList) input output))
    (freshPublic : input.fresh.publicInputs ⟨0, by decide⟩ =
      encHash (stateHash (priorHashPreimage
        (Lifecycle.Stage1.Wide.Relation.setup (FixedPoint.relation application compiled fits) ajtai context.toList) input)))
    (accepted : Nifs.PaperNonInteractive.verify
      (PiRLC.Wide.Key.key (FixedPoint.relation application compiled fits) ajtai)
      (input.running functionIndex) input.fresh input.nifsProof = some result)
    (recursiveResult : 0 < input.iteration → result = output.runningNext functionIndex)
    (witnessWidth : input.witness.length = Stage1.Poseidon2HashChainV1.messageWordCount) :
    ∃ (env : Env) (message : Fin 4 → F),
      let suffix := ApplicationCompletedAssignment.suffix env message
      let assignment := SourceAssignment.assignment application env suffix
      let carrier := CarrierAssignment.values application env suffix
      AssignmentTransportExecution.execute application parts.transport parts.package.layout.totalColumnCount
        (AssignmentTransportSemantics.physicalValues application env suffix) = some assignment ∧
      (FixedPoint.structuralPlan application compiled fits).RowsZero assignment ∧
      (∀ column, centeredMagnitude (carrier column) < 2) ∧
      Phi81Relation.projectPublicInput carrier = encHash output.x ∧
      Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application)
        (SourceCompiler.sourceEnv (SourceAssignment.raw application env suffix).base) = input.witness ∧
      (SourceAssignment.raw application env suffix).outputDigest = output.x := by
  obtain ⟨env, message, rows, norm, publicOutput, advice, digest, ranges, physical⟩ :=
    SelectedAssignmentCompleteness.complete_with_values compiled ajtai context input output result
      step priorWellFormed nextWellFormed freshPublic accepted recursiveResult witnessWidth
  let relation := FixedPoint.relation application compiled fits
  let suffix := ApplicationCompletedAssignment.suffix env message
  obtain ⟨throughD, _⟩ :=
    (Layout.Stage1.Wide.PilotPiCCSPiRLCPiDECRunningTransition.physicalHolds_iff relation env).mp physical
  obtain ⟨throughR, _⟩ := (Layout.Stage1.Wide.PilotPiCCSPiRLCPiDEC.physicalHolds_iff relation env).mp throughD
  obtain ⟨throughC, rRows⟩ := (Layout.Stage1.Wide.PilotPiCCSPiRLC.physicalHolds_iff relation env).mp throughR
  have cRows := ((Layout.Stage1.PilotPiCCS.physicalHolds_iff relation env).mp throughC).2
  have executed := AssignmentTransportCorrectness.canonical_execute_eq_assignment
    application env suffix parts.transport (PackageAuthority.transport_emitted compiled parts prepared)
    relation ajtai input.nifsProof
    (Layout.PiCCS.v1_1.Assumptions.production relation
      (PiCCSInputs.interface _ _) PiCCSInputs.phaseOffset (PiCCSInputs.externalInputsLinear _ _) env)
    cRows (Layout.Stage1.Wide.PiRLCInputBounds.assumptions relation env) rRows ranges
  rw [← PackageAuthority.physical_width compiled parts prepared] at executed
  exact ⟨env, message, executed, rows, norm, publicOutput, advice, digest⟩

end NightstreamFPrime.Export.Stage1.Wide.PackageCompleteness
