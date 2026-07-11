import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptArtifact

/-! Exact semantic checker for the fixed terminal-fold transcript state. -/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound

open Nightstream.Implementation.R1CS

def Accepted (assignment : Nat → Nat) : Prop :=
  FPrimeFullHistoryTerminalTranscriptArtifact.trace.Accepted assignment

def check (assignment : Nat → Nat) : Bool :=
  FPrimeFullHistoryTerminalTranscriptArtifact.trace.check assignment

theorem check_eq_true_iff (assignment : Nat → Nat) :
    check assignment = true ↔ Accepted assignment :=
  TranscriptCertificate.Trace.check_eq_true_iff _ _

theorem sound {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies
      FPrimeFullHistoryTerminalTranscriptArtifact.ownerRows assignment) :
    Accepted assignment :=
  TranscriptCertificate.ordered_sound
    FPrimeFullHistoryTerminalTranscriptArtifact.traceValid
    canonical one satisfies

theorem complete {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : Accepted assignment) :
    Satisfies FPrimeFullHistoryTerminalTranscriptArtifact.ownerRows assignment :=
  TranscriptCertificate.ordered_complete
    FPrimeFullHistoryTerminalTranscriptArtifact.traceValid
    canonical one accepted

end Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound
