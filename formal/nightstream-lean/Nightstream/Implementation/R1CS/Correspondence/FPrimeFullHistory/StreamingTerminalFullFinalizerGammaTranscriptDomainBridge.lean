import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomain
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptRowSound

/-!
Contract: application-domain authority bridge for the terminal Nebula gamma
transcript.

Owns the equality between the eight accepted initial pin rows and the state
computed from the exact application-domain bytes. It then starts the existing
handwritten transcript schedule from that authoritative state. It does not own
the transcript rows, gamma output muxes, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptDomainBridge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomainModel
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaDomain
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptModel
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptExecution

namespace RowSound

abbrev Sound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptRowSound.Sound

abbrev semanticRun :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptRowSound.semanticRun

abbrev traceAccepted :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptRowSound.trace_accepted

abbrev rowsSound :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptRowSound.rows_sound

end RowSound

private theorem pin_count : trace.pins.length = 84 := by
  change
    (List.zip rawArtifact.gammaTranscriptPinColumns
      rawArtifact.gammaTranscriptPinValues).length = 84
  rw [List.length_zip, rawArtifact_valid.gammaTranscriptPinColumnCount,
    rawArtifact_valid.gammaTranscriptPinValueCount]
  rfl

private theorem getD_mem_of_lt {alpha : Type} [Inhabited alpha]
    {entries : List alpha} {index : Nat} (bounded : index < entries.length) :
    entries.getD index default ∈ entries := by
  have member := List.getElem_mem (l := entries) bounded
  rwa [List.getElem_eq_getD default] at member

private theorem initial_pin_shape :
    ∀ lane : Fin width,
      trace.pins.getD lane.val (0, 0) =
        (start.cursor.lanes lane,
          gammaInitialStateValues.getD lane.val 0) := by
  intro lane
  fin_cases lane <;> rfl

private theorem expected_domain_lane (lane : Fin width) :
    gammaInitialStateValues.getD lane.val 0 =
      (expectedDomainState.lanes lane).val := by
  fin_cases lane <;> rfl

private theorem state_ext {left right : State}
    (lanes : left.lanes = right.lanes)
    (absorbed : left.absorbed = right.absorbed) : left = right := by
  cases left
  cases right
  simp_all

private theorem semantic_run_ext
    {left right : SemanticRun}
    (state : left.state = right.state)
    (digests : left.digests = right.digests) : left = right := by
  cases left
  cases right
  simp_all

/-- The accepted initial pin rows decode to the state computed from the exact
gamma application-domain bytes. -/
theorem decoded_start_eq_domain
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (accepted : trace.Accepted assignment) :
    decodeRun assignment canonical start =
      { state := domainInitialState, digests := [] } := by
  apply semantic_run_ext
  · rw [domain_initial_state_exact]
    apply state_ext
    · funext lane
      apply Fin.ext
      have bounded : lane.val < trace.pins.length := by
        rw [pin_count]
        exact Nat.lt_trans lane.isLt (by decide)
      have pinEqual := accepted.1
        (trace.pins.getD lane.val default)
        (getD_mem_of_lt bounded)
      have pinShape := initial_pin_shape lane
      have columnShape := congrArg Prod.fst pinShape
      have valueShape := congrArg Prod.snd pinShape
      change assignment (start.cursor.lanes lane) =
        (expectedDomainState.lanes lane).val
      change assignment ((trace.pins.getD lane.val (0, 0)).1) =
        (trace.pins.getD lane.val (0, 0)).2 at pinEqual
      rw [columnShape, valueShape] at pinEqual
      exact pinEqual.trans (expected_domain_lane lane)
    · rfl
  · rfl

def authoritativeRun
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    SemanticRun :=
  semanticExecute assignment canonical
    { state := domainInitialState, digests := [] } operations

structure Sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) : Prop where
  transcript : RowSound.Sound assignment canonical
  initial : decodeRun assignment canonical start =
    { state := domainInitialState, digests := [] }
  authoritative : RowSound.semanticRun assignment canonical =
    authoritativeRun assignment canonical

/-- The exact gamma transcript rows start from the application-domain state
and refine the full handwritten operation schedule. -/
theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.GammaTranscriptSatisfied assignment) :
    Sound assignment canonical := by
  have accepted := RowSound.traceAccepted assignment canonical one satisfied
  have initial := decoded_start_eq_domain assignment canonical accepted
  refine {
    transcript := RowSound.rowsSound assignment canonical one satisfied
    initial := initial
    authoritative := ?_ }
  change semanticExecute assignment canonical
      (decodeRun assignment canonical start) operations =
    semanticExecute assignment canonical
      { state := domainInitialState, digests := [] } operations
  rw [initial]

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptDomainBridge
