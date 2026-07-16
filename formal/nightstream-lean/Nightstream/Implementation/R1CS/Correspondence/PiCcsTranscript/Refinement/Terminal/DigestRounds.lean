import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Schedule

/-!
Independent Poseidon2-call semantics for the terminal `Pi_CCS` call tree.

Assurance tier: implementation/R1CS correspondence. Every named structural
leaf is extracted from the accepted exact owner as a
`TranscriptCertificate.CallAccepted` proposition. That proposition invokes
the independent Poseidon2 SSA interpreter; neither row order nor a recorded
output value is accepted as transcript authority.

Owns: semantic acceptance of all seven instance-digest calls, six authority-
binding calls, seven main-challenge calls, and five `beta_m` calls.

Does not own: constant pins, inter-call state connectivity, the semantic
instance-digest preimage, dynamic message authority, initial transcript state,
challenge partitioning, SumCheck, Rust conformance, costs, or row removal.

Emits constraints: no.

Authority boundary: the generated owner only locates an exact call. This file
proves that the call's output lanes are the Poseidon2 permutation of its named
input lanes; later refinement must prove where every input lane came from.

| Protocol | Phase | Constraint family | Multiplicity | Lean guarantee |
|---|---|---|---:|---|
| `Pi_CCS` | instance authority | Poseidon2 call | 7 | every exact instance-digest call is independently accepted |
| `Pi_CCS` | authority binding | Poseidon2 call | 6 | every exact binding call is independently accepted |
| `Pi_CCS` | main challenges | Poseidon2 call | 7 | every exact main-challenge call is independently accepted |
| `Pi_CCS` | `beta_m` | Poseidon2 call | 5 | every exact NC challenge call is independently accepted |
| `Pi_CCS` | complete call tree | Poseidon2 calls | 25 | no current call remains an opaque unclassified leaf |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.DigestRounds

open Nightstream.Implementation.R1CS.OwnerCertificate

private theorem acceptedScheduledCall
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment)
    (piece : Piece)
    (member : piece ∈
      FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces)
    (call : Poseidon2Call.Call)
    (payload : piece.payload = .poseidon call) :
    TranscriptCertificate.CallAccepted call assignment := by
  have pieceAccepted := accepted piece member
  rw [Piece.Accepted, payload] at pieceAccepted
  exact pieceAccepted

theorem instanceDigestCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment)
    (index : Fin Schedule.instanceCount) :
    TranscriptCertificate.CallAccepted (Schedule.instanceCall index)
      assignment := by
  exact acceptedScheduledCall accepted (Schedule.instancePiece index)
    (Schedule.instancePiece_mem index) (Schedule.instanceCall index)
    (by rw [Schedule.instancePiece_eq]; rfl)

theorem bindingCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment)
    (index : Fin Schedule.bindingCount) :
    TranscriptCertificate.CallAccepted (Schedule.bindingCall index)
      assignment := by
  exact acceptedScheduledCall accepted (Schedule.bindingPiece index)
    (Schedule.bindingPiece_mem index) (Schedule.bindingCall index)
    (by rw [Schedule.bindingPiece_eq]; rfl)

theorem mainChallengeCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment)
    (index : Fin Schedule.mainChallengeCount) :
    TranscriptCertificate.CallAccepted (Schedule.mainChallengeCall index)
      assignment := by
  exact acceptedScheduledCall accepted (Schedule.mainChallengePiece index)
    (Schedule.mainChallengePiece_mem index)
    (Schedule.mainChallengeCall index)
    (by rw [Schedule.mainChallengePiece_eq]; rfl)

theorem betaMCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment)
    (index : Fin Schedule.betaMCount) :
    TranscriptCertificate.CallAccepted (Schedule.betaMCall index)
      assignment := by
  exact acceptedScheduledCall accepted (Schedule.betaMPiece index)
    (Schedule.betaMPiece_mem index) (Schedule.betaMCall index)
    (by rw [Schedule.betaMPiece_eq]; rfl)

/-- All 25 terminal calls have independent Poseidon2 semantics. This packages
call acceptance only and intentionally does not assert that adjacent calls
form one transcript execution. -/
structure ScheduledCallsAccepted (assignment : Nat -> Nat) : Prop where
  instanceDigest : forall index : Fin Schedule.instanceCount,
    TranscriptCertificate.CallAccepted (Schedule.instanceCall index)
      assignment
  binding : forall index : Fin Schedule.bindingCount,
    TranscriptCertificate.CallAccepted (Schedule.bindingCall index)
      assignment
  mainChallenge : forall index : Fin Schedule.mainChallengeCount,
    TranscriptCertificate.CallAccepted (Schedule.mainChallengeCall index)
      assignment
  betaM : forall index : Fin Schedule.betaMCount,
    TranscriptCertificate.CallAccepted (Schedule.betaMCall index)
      assignment

theorem scheduledCallsAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsTranscript.Accepted assignment) :
    ScheduledCallsAccepted assignment :=
  { instanceDigest := instanceDigestCallAccepted accepted
    binding := bindingCallAccepted accepted
    mainChallenge := mainChallengeCallAccepted accepted
    betaM := betaMCallAccepted accepted }

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.DigestRounds
