import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.DigestRounds
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.Schedule

/-!
Independent Poseidon2-call semantics for the terminal `Pi_RLC` sampler tree.

Assurance tier: implementation/R1CS correspondence. This file extracts every
scheduled terminal call from accepted owner pieces as a
`TranscriptCertificate.CallAccepted`. That certificate invokes the independent
Poseidon2 SSA interpreter; row order or a recorded output value is not treated
as transcript authority.

Owns: semantic acceptance of the scalar-entry call, scalar-zero-only
full-cursor call, and four digest calls for each of the fifteen rho scalars.

Does not own: constant pins, inter-call state connectivity, initial transcript
state, absorbed cursors, candidate decomposition, rejection selection,
coefficient assembly, Rust conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: the generated owner only locates an exact call descriptor.
`TranscriptCertificate.CallAccepted` independently proves its eight output
lanes are the Poseidon2 permutation of its eight named input lanes.

| Protocol | Phase | Constraint family | Multiplicity | Lean guarantee |
|---|---|---|---:|---|
| `Pi_RLC` | scalar entry | Poseidon2 boundary call | 15 | each exact entry piece exposes an independently accepted permutation |
| `Pi_RLC` | scalar 0 block 0 | Poseidon2 full-cursor call | 1 | the scalar-zero-only call exposes an independently accepted permutation |
| `Pi_RLC` | digest block 0 | Poseidon2 squeeze call | 15 | every first digest call exposes an independently accepted permutation |
| `Pi_RLC` | digest blocks 1-3 | Poseidon2 squeeze calls | 45 | every later digest call exposes an independently accepted permutation |
| `Pi_RLC` | complete transcript call tree | Poseidon2 calls | 76 | all calls required for terminal transcript replay are available semantically |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.DigestRounds

open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal

private theorem acceptedScheduledCall
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (piece : Piece)
    (member : piece ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces)
    (call : Poseidon2Call.Call)
    (payload : piece.payload = .poseidon call) :
    TranscriptCertificate.CallAccepted call assignment := by
  have pieceAccepted := accepted piece member
  rw [Piece.Accepted, payload] at pieceAccepted
  exact pieceAccepted

theorem entryBoundaryCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    TranscriptCertificate.CallAccepted (Schedule.entryBoundaryCall rho)
      assignment := by
  exact acceptedScheduledCall accepted (Schedule.entryPiece rho)
    (Schedule.entryPiece_mem rho) (Schedule.entryBoundaryCall rho)
    (by rw [Schedule.entryPiece_eq]; rfl)

theorem scalar0Block0FullCursorCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    TranscriptCertificate.CallAccepted
      Schedule.scalar0Block0FullCursorCall assignment := by
  exact acceptedScheduledCall accepted
    Schedule.scalar0Block0FullCursorPiece
    Schedule.scalar0Block0FullCursorPiece_mem
    Schedule.scalar0Block0FullCursorCall
    (by rw [Schedule.scalar0Block0FullCursorPiece_eq]; rfl)

theorem block0DigestCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    TranscriptCertificate.CallAccepted (Schedule.block0DigestCall rho)
      assignment := by
  exact acceptedScheduledCall accepted (Schedule.block0Piece rho)
    (Schedule.block0Piece_mem rho) (Schedule.block0DigestCall rho)
    (by rw [Schedule.block0Piece_eq]; rfl)

theorem block1DigestCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    TranscriptCertificate.CallAccepted (Schedule.block1DigestCall rho)
      assignment := by
  exact acceptedScheduledCall accepted (Schedule.block1Piece rho)
    (Schedule.block1Piece_mem rho) (Schedule.block1DigestCall rho)
    (by rw [Schedule.block1Piece_eq]; rfl)

theorem block2DigestCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    TranscriptCertificate.CallAccepted (Schedule.block2DigestCall rho)
      assignment := by
  exact acceptedScheduledCall accepted (Schedule.block2Piece rho)
    (Schedule.block2Piece_mem rho) (Schedule.block2DigestCall rho)
    (by rw [Schedule.block2Piece_eq]; rfl)

theorem block3DigestCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    TranscriptCertificate.CallAccepted (Schedule.block3DigestCall rho)
      assignment := by
  exact acceptedScheduledCall accepted (Schedule.block3Piece rho)
    (Schedule.block3Piece_mem rho) (Schedule.block3DigestCall rho)
    (by rw [Schedule.block3Piece_eq]; rfl)

/-- Uniform semantic acceptance for digest blocks one through three, indexed
by the same `Fin 3` address used by the structural schedule and pin tree. -/
theorem laterDigestCallAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) :
    TranscriptCertificate.CallAccepted
      (Schedule.laterDigestCall rho block) assignment := by
  have blockCases : block.val = 0 ∨ block.val = 1 ∨ block.val = 2 := by
    have blockLt := block.isLt
    omega
  rcases blockCases with zero | one | two
  · have blockEq : block = (⟨0, by decide⟩ : Fin 3) := Fin.ext zero
    subst block
    simpa [Schedule.block1DigestCall] using
      block1DigestCallAccepted accepted rho
  · have blockEq : block = (⟨1, by decide⟩ : Fin 3) := Fin.ext one
    subst block
    simpa [Schedule.block2DigestCall] using
      block2DigestCallAccepted accepted rho
  · have blockEq : block = (⟨2, by decide⟩ : Fin 3) := Fin.ext two
    subst block
    simpa [Schedule.block3DigestCall] using
      block3DigestCallAccepted accepted rho

/-- All 76 terminal sampler calls are independently accepted. This packages
call semantics only; it deliberately does not assert that adjacent calls form
one transcript execution. -/
structure ScheduledCallsAccepted (assignment : Nat -> Nat) : Prop where
  entryBoundary : forall rho : Fin ScalarRows.scalarCount,
    TranscriptCertificate.CallAccepted (Schedule.entryBoundaryCall rho)
      assignment
  scalar0Block0FullCursor : TranscriptCertificate.CallAccepted
    Schedule.scalar0Block0FullCursorCall assignment
  block0Digest : forall rho : Fin ScalarRows.scalarCount,
    TranscriptCertificate.CallAccepted (Schedule.block0DigestCall rho)
      assignment
  block1Digest : forall rho : Fin ScalarRows.scalarCount,
    TranscriptCertificate.CallAccepted (Schedule.block1DigestCall rho)
      assignment
  block2Digest : forall rho : Fin ScalarRows.scalarCount,
    TranscriptCertificate.CallAccepted (Schedule.block2DigestCall rho)
      assignment
  block3Digest : forall rho : Fin ScalarRows.scalarCount,
    TranscriptCertificate.CallAccepted (Schedule.block3DigestCall rho)
      assignment

theorem scheduledCallsAccepted
    {assignment : Nat -> Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    ScheduledCallsAccepted assignment :=
  { entryBoundary := entryBoundaryCallAccepted accepted
    scalar0Block0FullCursor :=
      scalar0Block0FullCursorCallAccepted accepted
    block0Digest := block0DigestCallAccepted accepted
    block1Digest := block1DigestCallAccepted accepted
    block2Digest := block2DigestCallAccepted accepted
    block3Digest := block3DigestCallAccepted accepted }

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.DigestRounds
