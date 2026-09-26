import NightstreamFPrime.Export.Stage1.Wide.PackageCompleteness
import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportScratch

/-! The prepared package accepts the constructive physical source after all
ring-product scratch cells are erased. The retained assignment is unchanged. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PackageScratchCompleteness

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding Spec.Folding.PiCCS.PaperJoint Spec.HyperNova.Construction2.Paper
open Layout.Stage1 ProductionRelation
open Poseidon2HashChainV1Package (application fits)

theorem execute_erase (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (parts : AuthorityStream.Parts) (prepared : AuthorityStream.prepare compiled = .ok parts)
    (source : Env) :
    AssignmentTransportExecution.execute application parts.transport parts.package.layout.totalColumnCount
      (AssignmentTransportScratch.erase source) =
      AssignmentTransportExecution.execute application parts.transport parts.package.layout.totalColumnCount source := by
  rw [PackageAuthority.physical_width compiled parts prepared]
  exact AssignmentTransportScratch.execute_erase application _ parts.transport
    (PackageAuthority.transport_emitted compiled parts prepared) source

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
        (AssignmentTransportScratch.erase (AssignmentTransportSemantics.physicalValues application env suffix)) =
          some assignment ∧
      (FixedPoint.structuralPlan application compiled fits).RowsZero assignment ∧
      (∀ column, centeredMagnitude (carrier column) < 2) ∧
      Phi81Relation.projectPublicInput carrier = encHash output.x ∧
      Lifecycle.Stage1.Application.witnessValue (ApplicationInputs.interface application)
        (ApplicationInputs.localStart application)
        (SourceCompiler.sourceEnv (SourceAssignment.raw application env suffix).base) = input.witness ∧
      (SourceAssignment.raw application env suffix).outputDigest = output.x := by
  obtain ⟨env, message, executed, rows, norm, publicOutput, advice, digest⟩ :=
    PackageCompleteness.complete compiled parts prepared ajtai context input output result
      step priorWellFormed nextWellFormed freshPublic accepted recursiveResult witnessWidth
  refine ⟨env, message, ?_, rows, norm, publicOutput, advice, digest⟩
  rw [execute_erase compiled parts prepared]
  exact executed

end NightstreamFPrime.Export.Stage1.Wide.PackageScratchCompleteness
