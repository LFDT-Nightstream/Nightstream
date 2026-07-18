import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalPiCcsFeSumcheckArtifact

/-!
Exact owner address space for the legacy terminal FE SumCheck artifact.

Assurance tier: implementation/R1CS structural correspondence.

Owns: the exact 41-piece owner spine, total indexed access, payload-kind
classification, and owner membership for every selected piece.

Does not own: FE semantics, transcript replay, coefficient authority,
SumCheck equations, Rust conformance, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: the legacy generated owner is diagnostic implementation
evidence only. No artifact field becomes semantic authority by being indexed
here.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.fe_sumcheck.legacy_owner` | exactly 41 ordered physical pieces cover the owner | artifact structure | `ownerPieces_length` |
| `nifs.pi_ccs.fe_sumcheck.legacy_owner.piece` | every finite index names one owner member | artifact structure | `pieceAt`, `pieceAt_mem` |
| `nifs.pi_ccs.fe_sumcheck.legacy_owner.kind` | distinguish ordinary and Poseidon leaves without assigning semantics | derived structure | `payloadKind` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Fe.Rows

open Nightstream.Implementation.R1CS.OwnerCertificate

def pieceCount : Nat := 41

theorem ownerPieces_length :
    FPrimeFullHistoryTerminalPiCcsFeSumcheck.owner.pieces.length =
      pieceCount := by
  decide

def ownerIndex (index : Fin pieceCount) :
    Fin FPrimeFullHistoryTerminalPiCcsFeSumcheck.owner.pieces.length :=
  ⟨index.val, by
    rw [ownerPieces_length]
    exact index.isLt⟩

def pieceAt (index : Fin pieceCount) : Piece :=
  FPrimeFullHistoryTerminalPiCcsFeSumcheck.owner.pieces.get
    (ownerIndex index)

theorem pieceAt_mem (index : Fin pieceCount) :
    pieceAt index ∈
      FPrimeFullHistoryTerminalPiCcsFeSumcheck.owner.pieces := by
  exact List.get_mem _ _

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

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Fe.Rows
