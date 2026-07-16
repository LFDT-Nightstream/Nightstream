import Nightstream.Assurance.FPrimeFullHistorySemantics
import Nightstream.Assurance.FPrimeTrace
import Nightstream.Assurance.ValidExecution
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryRows
import Nightstream.Assurance.FPrimeFullHistory.RecursiveShell
import Nightstream.Assurance.FPrimeFullHistory.TerminalShell

/-!
Contract: end-to-end circuit correspondence for the exact supported
plain/stateless two-step full-history profile.

The soundness theorem consumes only exact generated row predicates.  It
decodes one base edge, one recursive edge, and the complete terminal shell.
The two PiRLC sampled identities are coefficient-exact or returned as their
named `BatchBadRoot` events.  In the exact branch the result is a genuine M3
`ValidExecution` whose terminal predicate contains all fourteen direct CE
claims and every terminal authority link.

This file does not bound either root event; that probability reduction belongs
to M6.  It also does not broaden the supported artifact beyond the fixed
`[1, 1]` stateless profile.
-/

namespace Nightstream.Assurance.FPrimeFullHistoryCircuit

open Nightstream.HyperNova.Construction2
open Nightstream.Protocol.FPrime
open Nightstream.Implementation.R1CS

abbrev Digest := FPrimeConcreteNifs.Digest
abbrev Fresh := FPrimeConcreteNifs.Fresh
abbrev Accumulator := FPrimeConcreteNifs.Accumulator
abbrev Proof := FPrimeConcreteNifs.Proof

/-- The only verifier semantics represented by the exact M4 artifact. -/
def environment :
    Nightstream.Assurance.FPrimeTrace.Environment
      Unit Unit Digest Digest Accumulator Fresh Proof Unit Digest Unit where
  hashSemantics := FPrimeFullHistoryBaseStepSound.hashSemantics
  stepSemantics := FPrimeFullHistorySemantics.semantics
  mode := .stateless
  context := FPrimeFullHistoryBaseStepSound.context

def priorFresh (assignment : Nat → Nat) : Fresh :=
  FPrimeFullHistoryTranscriptSound.decodedFresh assignment

def nextFresh (assignment : Nat → Nat) : Fresh :=
  FPrimeFullHistoryOutputEncodingSound.decodedTerminalFresh assignment

def circuitProof
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Proof :=
  FPrimeConcreteNifs.proofOfAssignment assignment canonical

def initialState : State Digest Accumulator Fresh Unit :=
  FPrimeFullHistoryBaseGenericSound.prior

def middleState (assignment : Nat → Nat) :
    State Digest Accumulator Fresh Unit :=
  FPrimeFullHistoryBaseGenericSound.next
    FPrimeFullHistorySemantics.semantics (priorFresh assignment)

def finalState
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    State Digest Accumulator Fresh Unit :=
  FPrimeFullHistoryRecursiveShellSound.next assignment
    (FPrimeConcreteNifs.recursiveAccumulator
      (circuitProof assignment canonical))
    (nextFresh assignment)

def baseInput (assignment : Nat → Nat) : Step.Input Fresh Unit Unit :=
  FPrimeFullHistoryBaseGenericSound.input (priorFresh assignment)

def baseProof : Step.Proof Digest Proof Unit :=
  FPrimeFullHistoryBaseGenericSound.proof

def recursiveInput (assignment : Nat → Nat) : Step.Input Fresh Unit Unit :=
  FPrimeFullHistoryRecursiveShellSound.input (nextFresh assignment)

def recursiveProof
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Step.Proof Digest Proof Unit :=
  FPrimeFullHistoryRecursiveShellSound.proof assignment
    (circuitProof assignment canonical)

/-- Semantic edge relation consumed by the top-level `ValidExecution` target. -/
def Edge
    (prior next : State Digest Accumulator Fresh Unit) : Prop :=
  ∃ (input : Step.Input Fresh Unit Unit)
      (proof : Step.Proof Digest Proof Unit),
    Step.Holds environment.hashSemantics environment.stepSemantics
      environment.mode environment.context prior next input proof

/-- The exact terminal predicate retains the decoded final state as well as
all independently reconstructed terminal facts. -/
def TerminalValid
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (state : State Digest Accumulator Fresh Unit) : Prop :=
  state = finalState assignment canonical ∧
    FPrimeFullHistoryTerminalShellSound.TerminalFacts assignment

/-- Every generated row premise needed by the semantic compiler.  The exact
manifest-to-list bridge is kept separate so the protocol theorem cannot infer
meaning from a range name or hash. -/
structure OwnerRows (assignment : Nat → Nat) : Prop where
  base : Satisfies FPrimeFullHistoryBase.rows assignment
  recursivePrelude :
    Satisfies FPrimeFullHistoryRecursivePrelude.rows assignment
  stateLink : Satisfies FPrimeFullHistoryStateLink.rows assignment
  priorLink : Satisfies FPrimeFullHistoryPriorLink.rows assignment
  counter : Satisfies FPrimeFullHistoryCounterSound.globalRows assignment
  recursiveOutput :
    Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment
  recursiveNifs : FPrimeConcreteNifs.RecursiveRows assignment
  terminal : FPrimeFullHistoryTerminalShellSound.TerminalRows assignment

private theorem recursiveFeRows_in_owner :
    rowsIncluded FPrimeFullHistorySumcheckArtifact.recursiveFeRows
      FPrimeFullHistoryRecursivePiCcsFeSumcheck.rows = true := by
  native_decide

private theorem recursiveNcRows_in_owner :
    rowsIncluded FPrimeFullHistorySumcheckArtifact.recursiveNcRows
      FPrimeFullHistoryRecursivePiCcsNcSumcheck.rows = true := by
  native_decide

private theorem terminalFeRows_in_owner :
    rowsIncluded FPrimeFullHistorySumcheckArtifact.terminalFeRows
      FPrimeFullHistoryTerminalPiCcsFeSumcheck.rows = true := by
  native_decide

private theorem terminalNcRows_in_owner :
    rowsIncluded FPrimeFullHistorySumcheckArtifact.terminalNcRows
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.rows = true := by
  native_decide

/-- Satisfaction of the exact manifest-ordered 4,193,134-row artifact
reconstructs every semantic row owner.  This theorem is deliberately explicit:
no range label, count, or digest is allowed to stand in for sparse-row
satisfaction. -/
theorem ownerRows_of_satisfies
    {assignment : Nat → Nat}
    (satisfies : Satisfies FPrimeFullHistoryRows.fullRows assignment) :
    OwnerRows assignment := by
  have top := (FPrimeFullHistoryRows.full_satisfies_iff assignment).mp satisfies
  have base := top FPrimeFullHistoryBase.rows (by
    simp [FPrimeFullHistoryRows.topLevelPieces])
  have recursive := top FPrimeFullHistoryRows.recursiveRows (by
    simp [FPrimeFullHistoryRows.topLevelPieces])
  have stateLink := top FPrimeFullHistoryStateLink.rows (by
    simp [FPrimeFullHistoryRows.topLevelPieces])
  have terminal := top FPrimeFullHistoryRows.terminalRows (by
    simp [FPrimeFullHistoryRows.topLevelPieces])
  have continuity := top FPrimeFullHistoryTerminalContinuity.rows (by
    simp [FPrimeFullHistoryRows.topLevelPieces])
  have publicPins := top FPrimeFullHistoryPublicPins.rows (by
    simp [FPrimeFullHistoryRows.topLevelPieces])
  have terminalCe := top FPrimeFullHistoryTerminalCe.terminalCeRows (by
    simp [FPrimeFullHistoryRows.topLevelPieces])

  have recursivePieces :=
    (FPrimeFullHistoryRows.recursive_satisfies_iff assignment).mp recursive
  have recursivePrelude :=
    recursivePieces FPrimeFullHistoryRecursivePrelude.rows (by
      simp [FPrimeFullHistoryRows.recursivePieces])
  have recursiveTranscript :=
    recursivePieces FPrimeFullHistoryRecursiveTranscriptArtifact.ownerRows (by
      simp [FPrimeFullHistoryRows.recursivePieces])
  have recursiveNifs := recursivePieces FPrimeFullHistoryRows.recursiveNifsRows
    (by simp [FPrimeFullHistoryRows.recursivePieces])
  have priorLink := recursivePieces FPrimeFullHistoryPriorLink.rows (by
    simp [FPrimeFullHistoryRows.recursivePieces])
  have recursiveAccumulator :=
    recursivePieces FPrimeFullHistoryRecursiveAccumulator.rows (by
      simp [FPrimeFullHistoryRows.recursivePieces])
  have counterTransition :=
    recursivePieces FPrimeFullHistoryRows.counterTransitionRows (by
      simp [FPrimeFullHistoryRows.recursivePieces])
  have recursiveOutput :=
    recursivePieces FPrimeFullHistoryRecursiveOutput.rows (by
      simp [FPrimeFullHistoryRows.recursivePieces])

  have recursiveNifsPieces :=
    (FPrimeFullHistoryRows.recursiveNifs_satisfies_iff assignment).mp
      recursiveNifs
  have recursivePiCcs :=
    recursiveNifsPieces FPrimeFullHistoryNestedOwners.recursivePiCcsRows (by
      simp [FPrimeFullHistoryRows.recursiveNifsPieces])
  have recursivePiRlc :=
    recursiveNifsPieces FPrimeFullHistoryNestedOwners.recursivePiRlcRows (by
      simp [FPrimeFullHistoryRows.recursiveNifsPieces])
  have recursivePiDec :=
    recursiveNifsPieces FPrimeFullHistoryPiDec.recursiveRows (by
      simp [FPrimeFullHistoryRows.recursiveNifsPieces])
  have recursivePoint :=
    recursiveNifsPieces FPrimeFullHistoryRecursivePointBinding.rows (by
      simp [FPrimeFullHistoryRows.recursiveNifsPieces])
  have recursivePiCcsPieces :=
    (FPrimeFullHistoryNestedOwners.recursivePiCcs_satisfies_iff assignment).mp
      recursivePiCcs
  have recursivePiRlcPieces :=
    (FPrimeFullHistoryNestedOwners.recursivePiRlc_satisfies_iff assignment).mp
      recursivePiRlc
  have recursiveIdentity :=
    recursivePiRlcPieces
      FPrimeFullHistoryNestedOwners.recursiveProjectionIdentityRows (by
        simp [FPrimeFullHistoryNestedOwners.recursivePiRlcPieces])
  have recursiveIdentityPieces :=
    (satisfies_flatten_iff
      FPrimeFullHistoryNestedOwners.recursiveProjectionIdentityPieces
      assignment).mp recursiveIdentity
  have recursiveGlue := recursiveIdentityPieces
    FPrimeFullHistoryProjectionRoles.recursiveGlueRows (by
      exact List.mem_append_right _ (by simp
        [FPrimeFullHistoryNestedOwners.recursiveProjectionIdentityPieces]))
  have recursiveResidual : FPrimeConcreteNifs.OwnersRows
      FPrimeConcreteNifs.recursiveResidualOwners assignment := by
    intro owner ownerMember
    simp [FPrimeConcreteNifs.recursiveResidualOwners,
      FPrimeFullHistoryNestedOwners.recursivePiCcsResidualOwners,
      FPrimeFullHistoryNestedOwners.recursivePiRlcResidualOwners] at ownerMember
    rcases ownerMember with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl
    · simpa [FPrimeFullHistoryRecursivePiCcsFreshDigests.rows] using
        recursivePiCcsPieces FPrimeFullHistoryRecursivePiCcsFreshDigests.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
    · simpa [FPrimeFullHistoryRecursivePiCcsRunningAuthority.rows] using
        recursivePiCcsPieces
          FPrimeFullHistoryRecursivePiCcsRunningAuthority.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
    · simpa [FPrimeFullHistoryRecursivePiCcsTranscript.rows] using
        recursivePiCcsPieces FPrimeFullHistoryRecursivePiCcsTranscript.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
    · simpa [FPrimeFullHistoryRecursivePiCcsFeInitial.rows] using
        recursivePiCcsPieces FPrimeFullHistoryRecursivePiCcsFeInitial.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
    · simpa [FPrimeFullHistoryRecursivePiCcsFeOptionalClaim.rows] using
        recursivePiCcsPieces
          FPrimeFullHistoryRecursivePiCcsFeOptionalClaim.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
    · simpa [FPrimeFullHistoryRecursivePiCcsFeSumcheck.rows] using
        recursivePiCcsPieces FPrimeFullHistoryRecursivePiCcsFeSumcheck.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
    · simpa [FPrimeFullHistoryRecursivePiCcsNcSumcheck.rows] using
        recursivePiCcsPieces FPrimeFullHistoryRecursivePiCcsNcSumcheck.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
    · simpa [FPrimeFullHistoryRecursivePiCcsFeTerminal.rows] using
        recursivePiCcsPieces FPrimeFullHistoryRecursivePiCcsFeTerminal.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
    · simpa [FPrimeFullHistoryRecursivePiCcsNcTerminal.rows] using
        recursivePiCcsPieces FPrimeFullHistoryRecursivePiCcsNcTerminal.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
    · simpa [FPrimeFullHistoryRecursivePiCcsCatchup.rows] using
        recursivePiCcsPieces FPrimeFullHistoryRecursivePiCcsCatchup.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
    · simpa [FPrimeFullHistoryRecursivePiCcsOutputMessageHashes.rows] using
        recursivePiCcsPieces
          FPrimeFullHistoryRecursivePiCcsOutputMessageHashes.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
    · simpa [FPrimeFullHistoryRecursivePiRlcTranscriptRhos.rows] using
        recursivePiRlcPieces FPrimeFullHistoryRecursivePiRlcTranscriptRhos.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiRlcPieces])
    · simpa [FPrimeFullHistoryRecursivePiRlcProjectionBinding.rows] using
        recursivePiRlcPieces
          FPrimeFullHistoryRecursivePiRlcProjectionBinding.rows
          (by simp [FPrimeFullHistoryNestedOwners.recursivePiRlcPieces])

  have terminalPieces :=
    (FPrimeFullHistoryRows.terminal_satisfies_iff assignment).mp terminal
  have terminalNifs := terminalPieces FPrimeFullHistoryRows.terminalNifsRows
    (by simp [FPrimeFullHistoryRows.terminalPieces])
  have runningLink :=
    terminalPieces FPrimeFullHistoryTerminalRunningLink.rows (by
      simp [FPrimeFullHistoryRows.terminalPieces])
  have parentLink :=
    terminalPieces FPrimeFullHistoryTerminalParentLink.rows (by
      simp [FPrimeFullHistoryRows.terminalPieces])
  have latestLink := terminalPieces FPrimeFullHistoryTerminalLink.rows (by
    simp [FPrimeFullHistoryRows.terminalPieces])
  have terminalAccumulator :=
    terminalPieces FPrimeFullHistoryTerminalAccumulator.rows (by
      simp [FPrimeFullHistoryRows.terminalPieces])

  have terminalNifsPieces :=
    (FPrimeFullHistoryRows.terminalNifs_satisfies_iff assignment).mp
      terminalNifs
  have terminalTranscript :=
    terminalNifsPieces FPrimeFullHistoryTerminalTranscriptArtifact.ownerRows (by
      simp [FPrimeFullHistoryRows.terminalNifsPieces])
  have terminalPiCcs :=
    terminalNifsPieces FPrimeFullHistoryNestedOwners.terminalPiCcsRows (by
      simp [FPrimeFullHistoryRows.terminalNifsPieces])
  have terminalPiRlc :=
    terminalNifsPieces FPrimeFullHistoryNestedOwners.terminalPiRlcRows (by
      simp [FPrimeFullHistoryRows.terminalNifsPieces])
  have terminalPiDec := terminalNifsPieces FPrimeFullHistoryPiDec.terminalRows
    (by simp [FPrimeFullHistoryRows.terminalNifsPieces])
  have terminalPoint :=
    terminalNifsPieces FPrimeFullHistoryTerminalPointBinding.rows (by
      simp [FPrimeFullHistoryRows.terminalNifsPieces])
  have terminalPiCcsPieces :=
    (FPrimeFullHistoryNestedOwners.terminalPiCcs_satisfies_iff assignment).mp
      terminalPiCcs
  have terminalPiRlcPieces :=
    (FPrimeFullHistoryNestedOwners.terminalPiRlc_satisfies_iff assignment).mp
      terminalPiRlc
  have terminalIdentity :=
    terminalPiRlcPieces
      FPrimeFullHistoryNestedOwners.terminalProjectionIdentityRows (by
        simp [FPrimeFullHistoryNestedOwners.terminalPiRlcPieces])
  have terminalIdentityPieces :=
    (satisfies_flatten_iff
      FPrimeFullHistoryNestedOwners.terminalProjectionIdentityPieces
      assignment).mp terminalIdentity
  have terminalGlue := terminalIdentityPieces
    FPrimeFullHistoryProjectionRoles.terminalGlueRows (by
      exact List.mem_append_right _ (by simp
        [FPrimeFullHistoryNestedOwners.terminalProjectionIdentityPieces]))
  have terminalAuthorityRows := terminalPiCcsPieces
    FPrimeFullHistoryNestedOwners.terminalPiCcsAuthorityRows (by
      simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
  have terminalAuthorityPieces :=
    (satisfies_flatten_iff
      [FPrimeFullHistoryPiDec.terminalCeRows,
        FPrimeFullHistoryPiCcsTerminalAuthorityTail.rows] assignment).mp (by
      simpa [FPrimeFullHistoryNestedOwners.terminalPiCcsAuthorityRows] using
        terminalAuthorityRows)
  have terminalAuthority : FPrimeConcreteNifs.TerminalAuthorityRows assignment := {
    piDec := terminalAuthorityPieces FPrimeFullHistoryPiDec.terminalCeRows (by
      simp)
    tail := terminalAuthorityPieces
      FPrimeFullHistoryPiCcsTerminalAuthorityTail.rows (by simp) }
  have terminalResidual : FPrimeConcreteNifs.OwnersRows
      FPrimeConcreteNifs.terminalResidualOwners assignment := by
    intro owner ownerMember
    simp [FPrimeConcreteNifs.terminalResidualOwners,
      FPrimeFullHistoryNestedOwners.terminalPiCcsResidualOwners,
      FPrimeFullHistoryNestedOwners.terminalPiRlcResidualOwners] at ownerMember
    rcases ownerMember with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl
    · simpa [FPrimeFullHistoryTerminalPiCcsFreshDigests.rows] using
        terminalPiCcsPieces FPrimeFullHistoryTerminalPiCcsFreshDigests.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
    · simpa [FPrimeFullHistoryTerminalPiCcsRunningAuthority.rows] using
        terminalPiCcsPieces
          FPrimeFullHistoryTerminalPiCcsRunningAuthority.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
    · simpa [FPrimeFullHistoryTerminalPiCcsTranscript.rows] using
        terminalPiCcsPieces FPrimeFullHistoryTerminalPiCcsTranscript.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
    · simpa [FPrimeFullHistoryTerminalPiCcsFeInitial.rows] using
        terminalPiCcsPieces FPrimeFullHistoryTerminalPiCcsFeInitial.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
    · simpa [FPrimeFullHistoryTerminalPiCcsFeOptionalClaim.rows] using
        terminalPiCcsPieces
          FPrimeFullHistoryTerminalPiCcsFeOptionalClaim.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
    · simpa [FPrimeFullHistoryTerminalPiCcsFeSumcheck.rows] using
        terminalPiCcsPieces FPrimeFullHistoryTerminalPiCcsFeSumcheck.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
    · simpa [FPrimeFullHistoryTerminalPiCcsNcSumcheck.rows] using
        terminalPiCcsPieces FPrimeFullHistoryTerminalPiCcsNcSumcheck.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
    · simpa [FPrimeFullHistoryTerminalPiCcsFeTerminal.rows] using
        terminalPiCcsPieces FPrimeFullHistoryTerminalPiCcsFeTerminal.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
    · simpa [FPrimeFullHistoryTerminalPiCcsNcTerminal.rows] using
        terminalPiCcsPieces FPrimeFullHistoryTerminalPiCcsNcTerminal.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
    · simpa [FPrimeFullHistoryTerminalPiCcsCatchup.rows] using
        terminalPiCcsPieces FPrimeFullHistoryTerminalPiCcsCatchup.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
    · simpa [FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.rows] using
        terminalPiCcsPieces
          FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
    · simpa [FPrimeFullHistoryTerminalPiRlcTranscriptRhos.rows] using
        terminalPiRlcPieces FPrimeFullHistoryTerminalPiRlcTranscriptRhos.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiRlcPieces])
    · simpa [FPrimeFullHistoryTerminalPiRlcProjectionBinding.rows] using
        terminalPiRlcPieces FPrimeFullHistoryTerminalPiRlcProjectionBinding.rows
          (by simp [FPrimeFullHistoryNestedOwners.terminalPiRlcPieces])

  refine {
    base := base
    recursivePrelude := recursivePrelude
    stateLink := stateLink
    priorLink := priorLink
    counter := FPrimeFullHistoryRows.counter_satisfies_of_prelude_and_transition
      recursivePrelude counterTransition
    recursiveOutput := recursiveOutput
    recursiveNifs := ?_
    terminal := ?_ }
  · exact {
      transcript := recursiveTranscript
      affine := {
        piCcsAllocation := recursivePiCcsPieces
          FPrimeFullHistoryPiCcsRecursiveAllocation.rows (by
            simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
        piCcsAuthority := recursivePiCcsPieces
          FPrimeFullHistoryPiCcsRecursiveAuthority.rows (by
            simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
        piCcsOutputBinding := recursivePiCcsPieces
          FPrimeFullHistoryPiCcsRecursiveOutputBinding.rows (by
            simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
        piRlcShape := recursivePiRlcPieces
          FPrimeFullHistoryPiRlcRecursiveShape.rows (by
            simp [FPrimeFullHistoryNestedOwners.recursivePiRlcPieces])
        piRlcLinearFolds := recursivePiRlcPieces
          FPrimeFullHistoryPiRlcRecursiveLinearFolds.rows (by
            simp [FPrimeFullHistoryNestedOwners.recursivePiRlcPieces]) }
      projection :=
        FPrimeFullHistoryNestedOwners.recursivePiRlc_projectionHolds
          recursivePiRlc
      projectionGlue := recursiveGlue
      feSumcheck := by
        have ownerRows := recursivePiCcsPieces
          FPrimeFullHistoryRecursivePiCcsFeSumcheck.rows (by
            simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
        intro row rowMember
        exact ownerRows row
          (rowsIncluded_sound recursiveFeRows_in_owner row rowMember)
      ncSumcheck := by
        have ownerRows := recursivePiCcsPieces
          FPrimeFullHistoryRecursivePiCcsNcSumcheck.rows (by
            simp [FPrimeFullHistoryNestedOwners.recursivePiCcsPieces])
        intro row rowMember
        exact ownerRows row
          (rowsIncluded_sound recursiveNcRows_in_owner row rowMember)
      piDec := recursivePiDec
      pointBinding := recursivePoint
      accumulator := recursiveAccumulator
      residual := recursiveResidual }
  · exact {
      nifs := {
        transcript := terminalTranscript
        affine := {
          piCcsAllocation := terminalPiCcsPieces
            FPrimeFullHistoryPiCcsTerminalAllocation.rows (by
              simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
          piCcsOutputBinding := terminalPiCcsPieces
            FPrimeFullHistoryPiCcsTerminalOutputBinding.rows (by
              simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
          piRlcShape := terminalPiRlcPieces
            FPrimeFullHistoryPiRlcTerminalShape.rows (by
              simp [FPrimeFullHistoryNestedOwners.terminalPiRlcPieces])
          piRlcLinearFolds := terminalPiRlcPieces
            FPrimeFullHistoryPiRlcTerminalLinearFolds.rows (by
              simp [FPrimeFullHistoryNestedOwners.terminalPiRlcPieces]) }
        projection :=
          FPrimeFullHistoryNestedOwners.terminalPiRlc_projectionHolds
            terminalPiRlc
        projectionGlue := terminalGlue
        feSumcheck := by
          have ownerRows := terminalPiCcsPieces
            FPrimeFullHistoryTerminalPiCcsFeSumcheck.rows (by
              simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
          intro row rowMember
          exact ownerRows row
            (rowsIncluded_sound terminalFeRows_in_owner row rowMember)
        ncSumcheck := by
          have ownerRows := terminalPiCcsPieces
            FPrimeFullHistoryTerminalPiCcsNcSumcheck.rows (by
              simp [FPrimeFullHistoryNestedOwners.terminalPiCcsPieces])
          intro row rowMember
          exact ownerRows row
            (rowsIncluded_sound terminalNcRows_in_owner row rowMember)
        piDec := terminalPiDec
        pointBinding := terminalPoint
        authority := terminalAuthority
        residual := terminalResidual }
      runningLink := runningLink
      parentLink := parentLink
      latestLink := latestLink
      accumulator := terminalAccumulator
      continuity := continuity
      publicPins := publicPins
      terminalCe := terminalCe }

/-- The two deterministic circuit bad events.  Their probability bounds are
not assumptions hidden in CIR-SOUND. -/
inductive BadEvent (assignment : Nat → Nat) : Prop where
  | recursiveRoot :
      Nightstream.SuperNeo.ProjectionCheck.BatchBadRoot
        ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity
          FPrimeFullHistoryProjection.recursiveTraces assignment) →
      BadEvent assignment
  | terminalRoot :
      Nightstream.SuperNeo.ProjectionCheck.BatchBadRoot
        ProjectionProgram.K.ops
        (ProjectionProgram.BatchIdentity
          FPrimeFullHistoryProjection.terminalTraces assignment) →
      BadEvent assignment

private theorem normalized_base_step
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : OwnerRows assignment) :
    Step.Holds environment.hashSemantics environment.stepSemantics
      environment.mode environment.context initialState
      (middleState assignment) (baseInput assignment) baseProof := by
  have facts := FPrimeFullHistoryBaseFacts.sound prime canonical one rows.base
  have edge := FPrimeFullHistoryBaseGenericSound.step_holds
    prime canonical one rows.base rows.stateLink rows.priorLink
    FPrimeFullHistorySemantics.semantics (priorFresh assignment)
    (fun fresh : Fresh => fresh.publicXOut)
    (FPrimeFullHistorySemantics.baseLaws (priorFresh assignment))
    FPrimeFullHistorySemantics.freshLinkLaws rfl
  rw [FPrimeFullHistoryBaseGenericSound.decodedPrior_eq facts,
    FPrimeFullHistoryBaseGenericSound.decodedNext_eq facts
      FPrimeFullHistorySemantics.semantics (priorFresh assignment),
    FPrimeFullHistoryBaseGenericSound.decodedProof_eq facts] at edge
  exact edge

private theorem normalized_recursive_step_or_bad
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : OwnerRows assignment) :
    Step.Holds environment.hashSemantics environment.stepSemantics
        environment.mode environment.context (middleState assignment)
        (finalState assignment canonical) (recursiveInput assignment)
        (recursiveProof assignment canonical) ∨
      BadEvent assignment := by
  have core := FPrimeFullHistorySemantics.recursiveCoreLaws
    prime canonical one rows.base rows.stateLink rows.recursivePrelude
    (nextFresh assignment)
  have recursive := FPrimeFullHistoryRecursiveShellSound.step_holds_or_badRoot
    prime canonical one rows.base rows.stateLink rows.priorLink rows.counter
    rows.recursiveOutput rows.recursiveNifs rows.terminal.latestLink
    FPrimeFullHistorySemantics.chunkDigest
    FPrimeFullHistorySemantics.freshLink
    FPrimeFullHistorySemantics.applicationStep
    (priorFresh assignment) (nextFresh assignment)
    (FPrimeFullHistorySemantics.baseLaws (priorFresh assignment))
    FPrimeFullHistorySemantics.freshLinkLaws rfl rfl core
  rcases recursive with exact | bad
  · left
    have facts := FPrimeFullHistoryBaseFacts.sound prime canonical one rows.base
    have priorEq := FPrimeFullHistoryRecursiveShellSound.prior_eq_baseNext
      facts canonical one rows.stateLink FPrimeFullHistorySemantics.semantics
      (priorFresh assignment)
    have priorEq' :
        FPrimeFullHistoryRecursiveShellSound.prior assignment
            (FPrimeConcreteNifs.stepSemantics
              FPrimeFullHistorySemantics.chunkDigest
              FPrimeFullHistorySemantics.freshLink
              FPrimeFullHistorySemantics.applicationStep)
            (priorFresh assignment) =
          FPrimeFullHistoryBaseGenericSound.next
            FPrimeFullHistorySemantics.semantics (priorFresh assignment) := by
      simpa [FPrimeFullHistorySemantics.semantics] using priorEq
    rw [priorEq'] at exact
    simpa [environment, middleState, finalState, recursiveInput, recursiveProof,
      circuitProof, FPrimeFullHistorySemantics.semantics] using exact
  · exact Or.inr (.recursiveRoot bad)

/-- CIR-SOUND for the fixed full-history profile.  Exact rows establish two
closed M3 edges and the direct terminal relation, or expose one of the two
sampled-polynomial root events. -/
private theorem ownerRows_sound_or_bad
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : OwnerRows assignment) :
    Nightstream.Assurance.ValidExecution Edge
        (TerminalValid assignment canonical)
        initialState (finalState assignment canonical) 2 ∨
      BadEvent assignment := by
  have base := normalized_base_step prime canonical one rows
  rcases normalized_recursive_step_or_bad prime canonical one rows with
    recursive | bad
  · rcases FPrimeFullHistoryTerminalShellSound.sound_or_badRoot
      prime canonical one rows.terminal with terminal | terminalBad
    · left
      refine ⟨?_, rfl, terminal⟩
      simpa using Nightstream.Assurance.Reachable.succ
        (Nightstream.Assurance.Reachable.succ
          (Nightstream.Assurance.Reachable.zero :
            Nightstream.Assurance.Reachable Edge initialState 0 initialState)
          ⟨baseInput assignment, baseProof, base⟩)
        ⟨recursiveInput assignment, recursiveProof assignment canonical,
          recursive⟩
    · exact Or.inr (.terminalRoot terminalBad)
  · exact Or.inr bad

/-- Artifact-checked CIR-SOUND for the exact generated full-history row list.
The premise is satisfaction of all 4,193,134 sparse rows in manifest order;
semantic owner predicates are reconstructed by `ownerRows_of_satisfies`. -/
theorem fPrimeCircuit_sound_or_bad
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : Satisfies FPrimeFullHistoryRows.fullRows assignment) :
    Nightstream.Assurance.ValidExecution Edge
        (TerminalValid assignment canonical)
        initialState (finalState assignment canonical) 2 ∨
      BadEvent assignment :=
  ownerRows_sound_or_bad prime canonical one (ownerRows_of_satisfies rows)

end Nightstream.Assurance.FPrimeFullHistoryCircuit
