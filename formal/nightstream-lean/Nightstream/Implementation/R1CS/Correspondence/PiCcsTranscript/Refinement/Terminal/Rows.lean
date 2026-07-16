import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalPiCcsTranscriptArtifact

/-!
Exact terminal-owner address space for the production-shaped `Pi_CCS`
instance-digest and transcript rows.

Assurance tier: implementation/R1CS structural correspondence. This file only
provides proof-carrying addresses into the generated owner; it grants those
pieces no protocol meaning.

Owns: the exact 48-piece terminal owner spine, total indexed access, and owner
membership for every selected leaf.

Does not own: phase classification, Poseidon2 semantics, state connectivity,
constant values, transcript authority, Rust conformance, costs, or row removal.

Emits constraints: no.

Authority boundary: generated pieces are implementation evidence. Separate
semantic and refinement theorems must explain every selected piece.

| Protocol | Phase | Constraint family | Exact structural obligation |
|---|---|---|---|
| `Pi_CCS` | terminal fixed profile | owner spine | exactly 48 ordered pieces |
| `Pi_CCS` | any child phase | indexed leaf | selected piece is a member of the exact owner |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Rows

open Nightstream.Implementation.R1CS.OwnerCertificate

def pieceCount : Nat := 48

theorem ownerPieces_length :
    FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces.length =
      pieceCount := by
  decide

def ownerIndex (index : Fin pieceCount) :
    Fin FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces.length :=
  ⟨index.val, by
    rw [ownerPieces_length]
    exact index.isLt⟩

def pieceAt (index : Fin pieceCount) : Piece :=
  FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces.get
    (ownerIndex index)

theorem pieceAt_mem (index : Fin pieceCount) :
    pieceAt index ∈
      FPrimeFullHistoryTerminalPiCcsTranscript.owner.pieces := by
  exact List.get_mem _ _

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Rows
