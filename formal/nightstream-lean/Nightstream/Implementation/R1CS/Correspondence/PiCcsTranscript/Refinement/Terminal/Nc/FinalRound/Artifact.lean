import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Artifact

/-!
Selected artifact view of terminal-NC round fourteen.

Assurance tier: implementation/R1CS structural correspondence.

Owns: only selection of the final index from the uniform later-round artifact
family and exact numeric audit statements for that selected leaf.

Does not own: another copy of the call formulas, pin soundness, semantic
execution, typed-message selection, complete replay, SumCheck algebra, costs,
necessity, or row removal.

Emits constraints: no.

Authority boundary: all physical owners, accepted-call facts, and constant
facts are inherited from the indexed `LaterRound.Artifact` family. This file
cannot drift into an independent final-round implementation.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.round.14.index` | select uniform later-round index thirteen | computed | `finalLaterRound` |
| `nifs.pi_ccs.nc_sumcheck.round.14.message` | selected message calls and length pin are the indexed family leaf | derived alias | `firstMessageCall`, `secondMessageCall`, `messageLengthPins` |
| `nifs.pi_ccs.nc_sumcheck.round.14.challenge` | selected marker and squeeze call are the indexed family leaf | derived alias | `finalSqueezeCall`, `squeezeMarkerPins` |
| `nifs.pi_ccs.nc_sumcheck.round.14.audit` | selected owner rows have the exact fixed numeric addresses | kernel evaluation | `firstMessagePiece_eq`, `secondMessagePiece_eq`, `finalSqueezePiece_eq` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate

def finalLaterRound : Fin LaterRound.Artifact.roundCount :=
  ⟨13, by decide⟩

abbrev firstMessageCall : Poseidon2Call.Call :=
  LaterRound.Artifact.firstMessageCall finalLaterRound

abbrev secondMessageCall : Poseidon2Call.Call :=
  LaterRound.Artifact.secondMessageCall finalLaterRound

abbrev finalSqueezeCall : Poseidon2Call.Call :=
  LaterRound.Artifact.squeezeCall finalLaterRound

abbrev messageLengthPins : List (Nat × Nat) :=
  LaterRound.Artifact.messageLengthPins finalLaterRound

abbrev squeezeMarkerPins : List (Nat × Nat) :=
  LaterRound.Artifact.squeezeMarkerPins finalLaterRound

abbrev messageLengthPiece : Piece :=
  LaterRound.Artifact.messageLengthPiece finalLaterRound

abbrev firstMessagePiece : Piece :=
  LaterRound.Artifact.firstMessagePiece finalLaterRound

abbrev secondMessagePiece : Piece :=
  LaterRound.Artifact.secondMessagePiece finalLaterRound

abbrev squeezeMarkerPiece : Piece :=
  LaterRound.Artifact.squeezeMarkerPiece finalLaterRound

abbrev finalSqueezePiece : Piece :=
  LaterRound.Artifact.squeezePiece finalLaterRound

theorem firstMessagePiece_eq :
    firstMessagePiece =
      { rowStart := 1640521
        rowEnd := 1641121
        payload := .poseidon firstMessageCall } := by
  decide

theorem secondMessagePiece_eq :
    secondMessagePiece =
      { rowStart := 1641121
        rowEnd := 1641721
        payload := .poseidon secondMessageCall } := by
  decide

theorem finalSqueezePiece_eq :
    finalSqueezePiece =
      { rowStart := 1641722
        rowEnd := 1642322
        payload := .poseidon finalSqueezeCall } := by
  decide

/-- The selected final squeeze exposes post-NC state columns beginning at the
fixed production base. -/
theorem finalSqueezeOutputBase_eq :
    LaterRound.Artifact.squeezeOutputBase finalLaterRound = 1692820 := by
  decide

theorem messageLengthPins_included :
    rowsIncluded (ConstantPins.rows messageLengthPins)
      messageLengthPiece.rows = true :=
  LaterRound.Artifact.messageLengthPins_included finalLaterRound

theorem squeezeMarkerPins_included :
    rowsIncluded (ConstantPins.rows squeezeMarkerPins)
      squeezeMarkerPiece.rows = true :=
  LaterRound.Artifact.squeezeMarkerPins_included finalLaterRound

theorem firstMessageCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    TranscriptCertificate.CallAccepted firstMessageCall assignment :=
  LaterRound.Artifact.firstMessageCallAccepted accepted finalLaterRound

theorem secondMessageCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    TranscriptCertificate.CallAccepted secondMessageCall assignment :=
  LaterRound.Artifact.secondMessageCallAccepted accepted finalLaterRound

theorem finalSqueezeCallAccepted
    {assignment : Nat → Nat}
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    TranscriptCertificate.CallAccepted finalSqueezeCall assignment :=
  LaterRound.Artifact.squeezeCallAccepted accepted finalLaterRound

abbrev Facts (assignment : Nat → Nat) : Prop :=
  LaterRound.Artifact.Facts finalLaterRound assignment

theorem facts
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.Accepted assignment) :
    Facts assignment :=
  LaterRound.Artifact.facts canonical one accepted finalLaterRound

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalRound.Artifact
