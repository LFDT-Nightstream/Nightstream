import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Transcripts
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryBaseStepSound
import Nightstream.Protocol.FPrime.Step

/-!
Contract: independent transcript acceptance for the exact recursive F' prefix
and terminal-fold initialization owners.

The recursive checker binds the caller's complete fixed-profile `NifsContext`
and singleton latest claim to assignment-derived columns before accepting the
Poseidon2 replay.  Most context fields are direct recursive state/transcript
inputs. `initialSemanticState` is intentionally decomposed: production does
not re-absorb that immutable value on every recursive step, so the composed
checker reads it from the base state-in columns that the public/preprocessing
owner pins.  No fictitious recursive transcript column is introduced.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTranscriptSound

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.FPrime

abbrev Digest := List Nat
abbrev Fresh := FPrimeFullHistoryBaseStepSound.Fresh

/-- Complete context decoded from authoritative full-history columns.
`initialSemanticState` is verifier-bound indirectly through the exact base
and public-pin owners; every other field is a direct recursive input or the
new chunk-digest producer output. -/
def decodedContext (assignment : Nat → Nat) :
    Step.NifsContext Digest Unit :=
  let columns := FPrimeFullHistoryRecursiveTranscriptArtifact.contextColumns
  { chunkCount := assignment columns.chunkCount
    stepCount := assignment columns.stepCount
    z0 := columns.z0.map assignment
    zi := columns.zi.map assignment
    initialSemanticState := columns.initialSemanticState.map assignment
    semanticState := columns.semanticState.map assignment
    pc := assignment columns.pc
    accumulatorDigest := columns.accumulatorDigest.map assignment
    publicTrace := columns.publicTrace.map assignment
    nebula := none
    nextChunkDigest := columns.nextChunkDigest.map assignment }

/-- Generated singleton public projection of the fresh CCS claim. -/
def freshPublicColumns : List Nat :=
  FPrimeFullHistoryRecursiveTranscriptArtifact.freshPublicColumns.getD 0 []

def freshLaneValue (assignment : Nat → Nat) (lane : Nat) : Nat :=
  (List.range 64).foldl
    (fun total bit => total + 2 ^ bit * assignment
      (freshPublicColumns.getD (1 + lane * 64 + bit) 0)) 0

def decodedFresh (assignment : Nat → Nat) : Fresh :=
  { publicXOut := (List.range 4).map (freshLaneValue assignment) }

def decodedLatest (assignment : Nat → Nat) : List Fresh :=
  [decodedFresh assignment]

/-- Cross-owner context authority boundary used by the concrete NIFS
callback.  A self-consistent transcript with a different context is rejected. -/
structure ContextBinding
    (assignment : Nat → Nat)
    (context : Step.NifsContext Digest Unit)
    (latest : List Fresh) : Prop where
  contextEq : context = decodedContext assignment
  latestEq : latest = decodedLatest assignment

/-- Independent recursive transcript acceptance: exact caller context plus
the pure constant-pin/Poseidon2 replay semantics. -/
structure RecursiveTranscriptAccepted
    (assignment : Nat → Nat)
    (context : Step.NifsContext Digest Unit)
    (latest : List Fresh) : Prop where
  binding : ContextBinding assignment context latest
  transcript :
    FPrimeFullHistoryRecursiveTranscriptArtifact.trace.Accepted assignment

def recursiveCheck
    (assignment : Nat → Nat)
    (context : Step.NifsContext Digest Unit)
    (latest : List Fresh) : Bool :=
  decide (context = decodedContext assignment) &&
    decide (latest = decodedLatest assignment) &&
    FPrimeFullHistoryRecursiveTranscriptArtifact.trace.check assignment

theorem recursiveCheck_eq_true_iff
    (assignment : Nat → Nat)
    (context : Step.NifsContext Digest Unit)
    (latest : List Fresh) :
    recursiveCheck assignment context latest = true ↔
      RecursiveTranscriptAccepted assignment context latest := by
  simp only [recursiveCheck, Bool.and_eq_true, decide_eq_true_eq]
  rw [TranscriptCertificate.Trace.check_eq_true_iff]
  constructor
  · rintro ⟨⟨contextEq, latestEq⟩, transcript⟩
    exact ⟨⟨contextEq, latestEq⟩, transcript⟩
  · rintro ⟨⟨contextEq, latestEq⟩, transcript⟩
    exact ⟨⟨contextEq, latestEq⟩, transcript⟩

/-- Exact recursive-prefix rows accept their assignment-derived context and
fresh public projection. -/
theorem recursive_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      FPrimeFullHistoryRecursiveTranscriptArtifact.ownerRows assignment) :
    RecursiveTranscriptAccepted assignment (decodedContext assignment)
      (decodedLatest assignment) := by
  refine ⟨⟨rfl, rfl⟩, ?_⟩
  exact TranscriptCertificate.ordered_sound
    FPrimeFullHistoryRecursiveTranscriptArtifact.traceValid
    canonical one satisfies

/-- Independent recursive acceptance satisfies every exact prefix row. -/
theorem recursive_complete
    {assignment : Nat → Nat}
    {context : Step.NifsContext Digest Unit} {latest : List Fresh}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : RecursiveTranscriptAccepted assignment context latest) :
    Satisfies FPrimeFullHistoryRecursiveTranscriptArtifact.ownerRows
      assignment :=
  TranscriptCertificate.ordered_complete
    FPrimeFullHistoryRecursiveTranscriptArtifact.traceValid
    canonical one accepted.transcript

/-- Terminal initialization has no variable caller context: its eight exact
rows bind the fixed final-fold transcript state. -/
def TerminalTranscriptAccepted (assignment : Nat → Nat) : Prop :=
  FPrimeFullHistoryTerminalTranscriptArtifact.trace.Accepted assignment

def terminalCheck (assignment : Nat → Nat) : Bool :=
  FPrimeFullHistoryTerminalTranscriptArtifact.trace.check assignment

theorem terminalCheck_eq_true_iff (assignment : Nat → Nat) :
    terminalCheck assignment = true ↔
      TerminalTranscriptAccepted assignment :=
  TranscriptCertificate.Trace.check_eq_true_iff _ _

theorem terminal_sound
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      FPrimeFullHistoryTerminalTranscriptArtifact.ownerRows assignment) :
    TerminalTranscriptAccepted assignment :=
  TranscriptCertificate.ordered_sound
    FPrimeFullHistoryTerminalTranscriptArtifact.traceValid
    canonical one satisfies

theorem terminal_complete
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : TerminalTranscriptAccepted assignment) :
    Satisfies FPrimeFullHistoryTerminalTranscriptArtifact.ownerRows assignment :=
  TranscriptCertificate.ordered_complete
    FPrimeFullHistoryTerminalTranscriptArtifact.traceValid
    canonical one accepted

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTranscriptSound
