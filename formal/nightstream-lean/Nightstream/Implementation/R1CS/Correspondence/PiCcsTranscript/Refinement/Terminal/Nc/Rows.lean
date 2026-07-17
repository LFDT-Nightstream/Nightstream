import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalPiCcsNcSumcheckArtifact

/-!
Exact owner address space for the terminal Split-NC `Pi_CCS` SumCheck phase.

Assurance tier: implementation/R1CS structural correspondence.

Owns: the exact 81-piece owner spine, total indexed access, payload-kind
classification, and owner membership for every selected piece.

Does not own: protocol phase classification, transcript semantics, SumCheck
equations, state connectivity, Rust conformance, cost totals, or row removal.

Emits constraints: no.

Authority boundary: the generated owner supplies implementation evidence only.
`pieceAt` gives a stable address; later modules must independently explain the
mathematical obligation of every addressed row.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc_sumcheck.owner` | exactly 81 ordered physical pieces cover the owner | artifact structure | `ownerPieces_length` |
| `nifs.pi_ccs.nc_sumcheck.owner.piece` | every finite index names one owner member | artifact structure | `pieceAt`, `pieceAt_mem` |
| `nifs.pi_ccs.nc_sumcheck.owner.kind` | distinguish ordinary and Poseidon leaves without assigning semantics | derived structure | `payloadKind` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Rows

open Nightstream.Implementation.R1CS.OwnerCertificate

/-- Exact number of compact physical pieces in the terminal NC owner. -/
def pieceCount : Nat := 81

theorem ownerPieces_length :
    FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces.length =
      pieceCount := by
  decide

def ownerIndex (index : Fin pieceCount) :
    Fin FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces.length :=
  ⟨index.val, by
    rw [ownerPieces_length]
    exact index.isLt⟩

def pieceAt (index : Fin pieceCount) : Piece :=
  FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces.get
    (ownerIndex index)

theorem pieceAt_mem (index : Fin pieceCount) :
    pieceAt index ∈
      FPrimeFullHistoryTerminalPiCcsNcSumcheck.owner.pieces := by
  exact List.get_mem _ _

/-- Coarse physical payload kind. `other` is retained so this classifier
fails closed if a future artifact introduces a new owner payload family. -/
inductive PayloadKind where
  | ordinary
  | poseidon
  | other
deriving DecidableEq, Repr

def payloadKind (piece : Piece) : PayloadKind :=
  match piece.payload with
  | .ordinary _ => .ordinary
  | .poseidon _ => .poseidon
  | _ => .other

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Rows
