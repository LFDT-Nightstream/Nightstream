import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingLifecycleRelation
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaSourceBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerSourceBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionSound

/-!
Contract: bind the exact Rust terminal finalizer transition to the abstract
terminal lifecycle authority.

The lifecycle authority owns the delayed fresh witness and the semantic
Nebula finalizer. Exact input equalities connect that authority to the
Rust-emitted lane, step, and leaf columns. The generated finalizer rows then
recover the same final Nebula lane and its closed predicate.

This module does not own the input equalities, finalizer row generation,
fresh-witness openings, or collision resistance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLifecycleBridge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRelation
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRelation
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRowSound
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenTransition
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaSourceBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerSourceBridge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionSound
open Nightstream.SuperNeo.Concrete

universe uParams uStructure uRunning uFresh uNifsProof uNebulaOpen
  uRunningWitness uFreshWitness

private abbrev artifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact

/-- Deterministic agreement between the abstract delayed Nebula finalizer and
the exact typed transition used by the Rust terminal relation. -/
structure Compatible
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {RunningWitness : Type uRunningWitness}
    {FreshWitness : Type uFreshWitness}
    {Nebula : Type}
    (authority :
      TerminalAuthority Running Fresh RunningWitness FreshWitness Nebula) where
  lane : Nebula -> Lane
  openInput : List Fresh -> FreshWitness -> Bool
  gammaInputs : List Fresh -> FreshWitness -> List Nat
  leafInputs : List Fresh -> FreshWitness -> Fin 3 -> List Nat
  candidateGamma : List Fresh -> FreshWitness -> Fin 2 -> K
  candidateDPre : List Fresh -> FreshWitness -> Fin 3 ->
    StreamingTerminalFullFinalizerTransitionRelation.Digest
  opened : Nebula -> List Fresh -> FreshWitness -> Lane
  step : List Fresh -> FreshWitness -> StepInput
  leaves : List Fresh -> FreshWitness -> LeafDigests
  openedExact : forall nebula latest witness,
    opened nebula latest witness =
      maybeOpenLane (lane nebula) (openInput latest witness)
        (candidateGamma latest witness) (candidateDPre latest witness)
  gammaTranscriptExact : forall
      (assignment : Nat -> Nat)
      (canonical : forall column, assignment column < goldilocksP)
      latest witness,
    inputValues assignment = gammaInputs latest witness ->
      computedGamma assignment canonical = candidateGamma latest witness
  leafComputationExact : forall assignment latest witness,
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge.inputValues
        assignment =
      leafInputs latest witness ->
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge.computedLeaves
        assignment =
      leaves latest witness
  finalizeExact : forall nebula latest witness finalNebula,
    authority.nebulaFinalize nebula latest witness = some finalNebula ->
      lane finalNebula =
        closeLane emittedHashSemantics
          (advanceLane emittedHashSemantics (opened nebula latest witness)
            (step latest witness) (leaves latest witness))

/-- Exact finalizer rows and authoritative input links recover the same final
Nebula value accepted by the terminal lifecycle relation. -/
private theorem rows_bind_terminal_finalizer_from_opened_exact
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
    (compatible : Compatible authority)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (decodeSatisfied : artifact.DecodeSatisfied assignment)
    (advanceAlgebraSatisfied : artifact.AdvanceAlgebraSatisfied assignment)
    (advanceChainSatisfied :
      forall lane, (advanceChainLink lane).Satisfied assignment)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment)
    (openedExact :
      openedLane assignment = compatible.opened terminal.nebula
        terminal.latest terminal.freshWitness)
    (stepExact :
      stepInput assignment =
        compatible.step terminal.latest terminal.freshWitness)
    (leavesExact :
      assignedLeaves assignment =
        compatible.leaves terminal.latest terminal.freshWitness) :
    finalLane assignment = compatible.lane terminal.finalNebula /\
      Closed emittedHashSemantics (compatible.lane terminal.finalNebula) /\
      authority.nebulaFinal terminal.finalNebula := by
  have transition := rows_imply_terminalTransition assignment canonical one
    decodeSatisfied advanceAlgebraSatisfied advanceChainSatisfied
    closeSatisfied terminalClosed
  have semanticOutput := compatible.finalizeExact terminal.nebula
    terminal.latest terminal.freshWitness terminal.finalNebula
    terminal.nebulaFinalized
  have finalExact :
      finalLane assignment = compatible.lane terminal.finalNebula := by
    calc
      finalLane assignment =
          closeLane emittedHashSemantics
            (advanceLane emittedHashSemantics (openedLane assignment)
              (stepInput assignment) (assignedLeaves assignment)) :=
        transition.outputExact
      _ = closeLane emittedHashSemantics
            (advanceLane emittedHashSemantics
              (compatible.opened terminal.nebula terminal.latest
                terminal.freshWitness)
              (compatible.step terminal.latest terminal.freshWitness)
              (compatible.leaves terminal.latest terminal.freshWitness)) := by
        rw [openedExact, stepExact, leavesExact]
      _ = compatible.lane terminal.finalNebula := semanticOutput.symm
  refine ⟨finalExact, ?_, terminal.nebulaFinal⟩
  rw [← finalExact]
  exact transition.after_closed

/-- Exact open, gamma-transcript, mux, advance, and close rows bind the Rust
finalizer to the terminal lifecycle authority. The remaining equalities name
the source-row obligations for the post-phase lane and delayed fresh input;
no digest is accepted as independent authority. -/
theorem rows_bind_terminal_finalizer
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
    (compatible : Compatible authority)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (openAlgebraSatisfied : artifact.OpenAlgebraSatisfied assignment)
    (gammaTranscriptSatisfied : artifact.GammaTranscriptSatisfied assignment)
    (gammaMuxSatisfied : artifact.GammaMuxSatisfied assignment)
    (sourceBindingSatisfied : SourceSatisfied assignment)
    (decodeSatisfied : artifact.DecodeSatisfied assignment)
    (advanceAlgebraSatisfied : artifact.AdvanceAlgebraSatisfied assignment)
    (advanceChainSatisfied :
      forall lane, (advanceChainLink lane).Satisfied assignment)
    (opsLeafSatisfied :
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge.OpsSatisfied
        assignment)
    (isLeafSatisfied :
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge.IsSatisfied
        assignment)
    (fsLeafSatisfied :
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge.FsSatisfied
        assignment)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment)
    (postPhaseSourceExact :
      sourceLane assignment = compatible.lane terminal.nebula)
    (openSourceExact :
      delayedOpenSource assignment =
        compatible.openInput terminal.latest terminal.freshWitness)
    (gammaInputExact :
      inputValues assignment =
        compatible.gammaInputs terminal.latest terminal.freshWitness)
    (dPreSourceExact :
      sourceDPre assignment =
        compatible.candidateDPre terminal.latest terminal.freshWitness)
    (stepSourceExact :
      sourceStep assignment =
        compatible.step terminal.latest terminal.freshWitness)
    (leafInputExact :
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge.inputValues
          assignment =
        compatible.leafInputs terminal.latest terminal.freshWitness) :
    finalLane assignment = compatible.lane terminal.finalNebula /\
      Closed emittedHashSemantics (compatible.lane terminal.finalNebula) /\
      authority.nebulaFinal terminal.finalNebula := by
  have openSound :=
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpenAlgebraRowSound.rows_sound
      assignment canonical one openAlgebraSatisfied
  have muxSound :=
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaMuxRowSound.rows_sound
      assignment canonical one openSound gammaTranscriptSatisfied
        gammaMuxSatisfied
  have openExact :
      delayedOpen assignment =
        compatible.openInput terminal.latest terminal.freshWitness :=
    (rows_bind_delayedOpen assignment canonical one
      sourceBindingSatisfied).trans openSourceExact
  have postPhaseExact :
      postPhaseLane assignment = compatible.lane terminal.nebula :=
    (rows_bind_postPhaseLane assignment canonical one
      sourceBindingSatisfied).trans postPhaseSourceExact
  have gammaExact :
      candidateGamma assignment =
        compatible.candidateGamma terminal.latest terminal.freshWitness :=
    (rows_bind_candidateGamma assignment canonical one
      gammaTranscriptSatisfied).trans
        (compatible.gammaTranscriptExact assignment canonical terminal.latest
          terminal.freshWitness gammaInputExact)
  have dPreExact :
      candidateDPre assignment =
        compatible.candidateDPre terminal.latest terminal.freshWitness :=
    (rows_bind_candidateDPre assignment canonical one sourceBindingSatisfied
      decodeSatisfied).trans dPreSourceExact
  have stepExact :
      stepInput assignment =
        compatible.step terminal.latest terminal.freshWitness :=
    (rows_bind_stepInput assignment canonical one sourceBindingSatisfied
      decodeSatisfied).trans stepSourceExact
  have leavesExact :
      assignedLeaves assignment =
        compatible.leaves terminal.latest terminal.freshWitness :=
    (Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge.rows_bind_assignedLeaves
      assignment canonical one opsLeafSatisfied isLeafSatisfied
        fsLeafSatisfied).trans
      (compatible.leafComputationExact assignment terminal.latest
        terminal.freshWitness leafInputExact)
  have openedTransition := rows_imply_maybeOpen assignment canonical one
    openSound muxSound
  have openedExact :
      openedLane assignment = compatible.opened terminal.nebula
        terminal.latest terminal.freshWitness := by
    calc
      openedLane assignment =
          maybeOpenLane (postPhaseLane assignment) (delayedOpen assignment)
            (candidateGamma assignment) (candidateDPre assignment) :=
        openedTransition.outputExact
      _ = maybeOpenLane (compatible.lane terminal.nebula)
            (compatible.openInput terminal.latest terminal.freshWitness)
            (compatible.candidateGamma terminal.latest terminal.freshWitness)
            (compatible.candidateDPre terminal.latest terminal.freshWitness) := by
        rw [postPhaseExact, openExact, gammaExact, dPreExact]
      _ = compatible.opened terminal.nebula terminal.latest
            terminal.freshWitness :=
        (compatible.openedExact terminal.nebula terminal.latest
          terminal.freshWitness).symm
  exact rows_bind_terminal_finalizer_from_opened_exact terminal compatible
    assignment canonical one decodeSatisfied advanceAlgebraSatisfied
    advanceChainSatisfied closeSatisfied terminalClosed openedExact stepExact
    leavesExact

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLifecycleBridge
