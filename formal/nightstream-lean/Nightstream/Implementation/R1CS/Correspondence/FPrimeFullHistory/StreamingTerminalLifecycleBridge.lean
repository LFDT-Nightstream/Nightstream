import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLifecycleBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFinalSelectiveLifecycleBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutLifecycleBridge

/-!
Contract: compose the exact terminal XOut and delayed-finalizer row families
on one Goldilocks assignment.

The finalizer source boundary is one record equality. Its fields come from
the same typed Nebula value, trailing fresh batch, and opened fresh witness.
The theorem returns all exact XOut preimages and the checked final lane, or
one of the existing named Poseidon2 failures.

This module does not own the exact source encoding of the opened fresh
witness.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalLifecycleBridge

open Nightstream.Implementation.Nebula.StateOutputPoseidonBinding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLifecycleBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerSourceBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRelation
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionSound
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFinalSelectiveLifecycleBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutFinalPublicBinding
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutLifecycleBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonCallSequence
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonOutputCopyBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPublicBinding
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutSourceFinalBridge
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPublicHash
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen
  uRunningWitness uFreshWitness

private abbrev contextArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullXOutContext.rawArtifact

private abbrev phaseArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullPhaseSemantic.rawArtifact

private abbrev nebulaArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullNebulaStateDigest.rawArtifact

private abbrev finalizerArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact

private abbrev sourceArtifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding.rawArtifact

/-- Complete uncompressed source view consumed by the delayed finalizer. -/
structure FinalizerSources where
  lane : Lane
  openInput : Bool
  gammaInputs : List Nat
  dPre : Fin 3 ->
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRelation.Digest
  step : StepInput
  leafInputs : Fin 3 -> List Nat

def assignedFinalizerSources (assignment : Nat -> Nat) : FinalizerSources where
  lane := sourceLane assignment
  openInput := delayedOpenSource assignment
  gammaInputs :=
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaSourceBridge.inputValues
      assignment
  dPre := sourceDPre assignment
  step := sourceStep assignment
  leafInputs :=
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge.inputValues
      assignment

/-- One semantic source view derived from the same Nebula value and fresh
witness used by the terminal lifecycle relation. -/
def expectedFinalizerSources
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {Nebula : Type}
    {authority :
      TerminalAuthority Running Fresh RunningWitness FreshWitness Nebula}
    (compatible : Compatible authority)
    (nebula : Nebula) (latest : List Fresh) (witness : FreshWitness) :
    FinalizerSources where
  lane := compatible.lane nebula
  openInput := compatible.openInput latest witness
  gammaInputs := compatible.gammaInputs latest witness
  dPre := compatible.candidateDPre latest witness
  step := compatible.step latest witness
  leafInputs := compatible.leafInputs latest witness

/-- All exact terminal row families share one assignment. They recover the
typed terminal XOut preimages and the delayed finalizer result, or expose one
named Poseidon2 failure. -/
theorem rows_bind_terminal_or_failure
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {NifsProof : Type uNifsProof}
    {Nebula : Type}
    {NebulaOpen : Type uNebulaOpen}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {armCount : Nat}
    {configuration : Configuration Params StructureDigest Running Fresh
      NifsProof Nebula NebulaOpen armCount}
    {authority :
      TerminalAuthority Running Fresh RunningWitness FreshWitness Nebula}
    (terminal : Terminal configuration authority)
    (phaseEncoding : PhasePayloadEncoding Fresh)
    (nebulaEncoding : NebulaEncoding Nebula)
    (outerCompatible :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeXOutLifecycleBridge.Poseidon2Compatible
        configuration)
    (phaseCompatible : PhasePoseidon2Compatible configuration phaseEncoding)
    (nebulaCompatible : NebulaPoseidon2Compatible configuration nebulaEncoding)
    (finalizerCompatible : Compatible authority)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (xOutSourceSatisfied : sourceArtifact.Satisfied assignment)
    (contextSatisfied :
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullXOutContext.Satisfied
        assignment)
    (phaseSatisfied : phaseArtifact.Satisfied assignment)
    (nebulaSatisfied : nebulaArtifact.Satisfied assignment)
    {rows : Nat}
    {relation : InterpretedRelation rows callPlacement8.finalColumns}
    (exact0 : FinalRowSliceExact callPlacement0 callPlacement0_valid relation
      (projectedFinalAssignment assignment canonical))
    (exact1 : FinalRowSliceExact callPlacement1 callPlacement1_valid relation
      (projectedFinalAssignment assignment canonical))
    (exact2 : FinalRowSliceExact callPlacement2 callPlacement2_valid relation
      (projectedFinalAssignment assignment canonical))
    (exact3 : FinalRowSliceExact callPlacement3 callPlacement3_valid relation
      (projectedFinalAssignment assignment canonical))
    (exact4 : FinalRowSliceExact callPlacement4 callPlacement4_valid relation
      (projectedFinalAssignment assignment canonical))
    (exact5 : FinalRowSliceExact callPlacement5 callPlacement5_valid relation
      (projectedFinalAssignment assignment canonical))
    (exact6 : FinalRowSliceExact callPlacement6 callPlacement6_valid relation
      (projectedFinalAssignment assignment canonical))
    (exact7 : FinalRowSliceExact callPlacement7 callPlacement7_valid relation
      (projectedFinalAssignment assignment canonical))
    (exact8 : FinalRowSliceExact callPlacement8 callPlacement8_valid relation
      (projectedFinalAssignment assignment canonical))
    (copyExact : forall lane : Fin 4,
      OutputCopyRowExact (outputCopyAt lane) relation
        (projectedFinalAssignment assignment canonical))
    (linkExact : forall lane : Fin 4, forall bit : Fin 64,
      PublicLinkRowExact lane bit relation
        (projectedFinalAssignment assignment canonical))
    (finalSatisfied : AllRowsSatisfied relation
      (projectedFinalAssignment assignment canonical))
    (finalOne :
      absoluteValue (projectedFinalAssignment assignment canonical) 0 = 1)
    (finalSelectorOne :
      absoluteValue (projectedFinalAssignment assignment canonical)
        callPlacement8.selectorColumn = 1)
    (publicBinding : PublicAssignmentBinding (projectedFinalValues assignment)
      terminal.publicXOut)
    (openAlgebraSatisfied : finalizerArtifact.OpenAlgebraSatisfied assignment)
    (gammaTranscriptSatisfied :
      finalizerArtifact.GammaTranscriptSatisfied assignment)
    (gammaMuxSatisfied : finalizerArtifact.GammaMuxSatisfied assignment)
    (sourceBindingSatisfied : SourceSatisfied assignment)
    (decodeSatisfied : finalizerArtifact.DecodeSatisfied assignment)
    (advanceAlgebraSatisfied :
      finalizerArtifact.AdvanceAlgebraSatisfied assignment)
    (advanceChainSatisfied : forall lane,
      (Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer.advanceChainLink
        lane).Satisfied assignment)
    (opsLeafSatisfied :
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge.OpsSatisfied
        assignment)
    (isLeafSatisfied :
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge.IsSatisfied
        assignment)
    (fsLeafSatisfied :
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge.FsSatisfied
        assignment)
    (closeSatisfied : finalizerArtifact.CloseSatisfied assignment)
    (terminalClosed : finalizerArtifact.TerminalClosedSatisfied assignment)
    (sourceExact : assignedFinalizerSources assignment =
      expectedFinalizerSources finalizerCompatible terminal.nebula
        terminal.latest terminal.freshWitness) :
    (((assignmentFrame assignment =
          frame configuration terminal.state terminal.nebula) /\
        phaseInput assignment =
          phasePreimage phaseEncoding terminal.phaseState terminal.latest /\
        assignmentLaneBranch assignment =
          nebulaEncoding.branch terminal.nebula /\
        laneInput assignment = nebulaEncoding.preimage terminal.nebula) /\
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRowSound.finalLane
          assignment =
        finalizerCompatible.lane terminal.finalNebula /\
      Closed emittedHashSemantics
        (finalizerCompatible.lane terminal.finalNebula) /\
      authority.nebulaFinal terminal.finalNebula) \/ Failure := by
  have xOutResult := final_rows_bind_terminal_frame_or_collision terminal
    phaseEncoding nebulaEncoding outerCompatible phaseCompatible
    nebulaCompatible assignment canonical one xOutSourceSatisfied
    contextSatisfied phaseSatisfied nebulaSatisfied exact0 exact1 exact2 exact3
    exact4 exact5 exact6 exact7 exact8 copyExact linkExact finalSatisfied
    finalOne finalSelectorOne publicBinding
  rcases xOutResult with xOutExact | failure
  · have laneExact := congrArg FinalizerSources.lane sourceExact
    have openExact := congrArg FinalizerSources.openInput sourceExact
    have gammaExact := congrArg FinalizerSources.gammaInputs sourceExact
    have dPreExact := congrArg FinalizerSources.dPre sourceExact
    have stepExact := congrArg FinalizerSources.step sourceExact
    have leafExact := congrArg FinalizerSources.leafInputs sourceExact
    have finalizerResult := rows_bind_terminal_finalizer terminal
      finalizerCompatible assignment canonical one openAlgebraSatisfied
      gammaTranscriptSatisfied gammaMuxSatisfied sourceBindingSatisfied
      decodeSatisfied advanceAlgebraSatisfied advanceChainSatisfied
      opsLeafSatisfied isLeafSatisfied fsLeafSatisfied closeSatisfied
      terminalClosed laneExact openExact gammaExact dPreExact stepExact
      leafExact
    exact Or.inl ⟨xOutExact, finalizerResult⟩
  · exact Or.inr failure

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalLifecycleBridge
