import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Rows

/-!
Protocol-to-phase tree for the terminal Split-NC `Pi_CCS` SumCheck owner.

Assurance tier: implementation/R1CS structural correspondence.

Owns: the exact `prologue -> first round -> 14 later rounds` physical tree;
the 48 Poseidon/33 ordinary payload census; total coverage of all 81 pieces;
and the visible split between 30 round-equation rows and the next round's
length pin inside physically combined algebra pieces.

Does not own: the values of constant pins; Poseidon2 acceptance; message
serialization; claimed-chain or Horner semantics of the 30 equation rows;
inter-round state connectivity; Rust conformance; encoded columns; cost
totals beyond this owner; necessity; or row removal.

Emits constraints: no.

Authority boundary: this module classifies exact artifact locations but grants
them no protocol truth. In particular, an ordinary algebra piece is not
treated as one semantic leaf: every non-final one contains 30 SumCheck rows
plus one next-message length pin, which later refinement must split.

| Stage path | Mathematical obligation | Physical ownership | Lean owner |
|---|---|---:|---|
| `nifs.pi_ccs.nc_sumcheck.prologue.pins` | verifier domain/zero-claim/tag boundary pins | 3 ordinary pieces | `prologuePinPiece` |
| `nifs.pi_ccs.nc_sumcheck.prologue.permute` | two full-rate prologue boundaries | 2 Poseidon pieces | `prologueCallPiece` |
| `nifs.pi_ccs.nc_sumcheck.round.0.message` | first message begins at the prologue boundary | 3 Poseidon pieces | `firstMessageCallPiece` |
| `nifs.pi_ccs.nc_sumcheck.round.0.challenge` | marker then challenge permutation | 1 ordinary + 1 Poseidon piece | `firstSqueezePinPiece`, `firstSqueezeCallPiece` |
| `nifs.pi_ccs.nc_sumcheck.round.0.algebra` | 30 equations plus next length pin | 1 ordinary piece | `firstAlgebraPiece`, `firstAlgebra_row_formula` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.message` | two full-rate message permutations per round | 28 Poseidon pieces | `laterMessageCallPiece` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.challenge` | marker then challenge permutation per round | 14 ordinary + 14 Poseidon pieces | `laterSqueezePinPiece`, `laterSqueezeCallPiece` |
| `nifs.pi_ccs.nc_sumcheck.round.1_14.algebra` | 30 equations; non-final rounds also pin the next length | 14 ordinary pieces | `laterAlgebraPiece`, `laterAlgebra_row_formula` |
| `nifs.pi_ccs.nc_sumcheck` | every physical piece has exactly one tree address | 81 pieces | `phaseIndices_eq_ownerRange`, `familyCounts` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Schedule

open Nightstream.Implementation.R1CS.OwnerCertificate

def prologuePinCount : Nat := 3
def prologueCallCount : Nat := 2
def firstMessageCallCount : Nat := 3
def laterRoundCount : Nat := 14
def laterMessageCallCount : Nat := 2
def roundEquationRowCount : Nat := 30

def prologuePinIndex (index : Fin prologuePinCount) :
    Fin Rows.pieceCount :=
  ⟨2 * index.val, by
    have indexLt := index.isLt
    simp only [prologuePinCount, Rows.pieceCount] at indexLt ⊢
    omega⟩

def prologueCallIndex (index : Fin prologueCallCount) :
    Fin Rows.pieceCount :=
  ⟨1 + 2 * index.val, by
    have indexLt := index.isLt
    simp only [prologueCallCount, Rows.pieceCount] at indexLt ⊢
    omega⟩

def firstMessageCallIndex (index : Fin firstMessageCallCount) :
    Fin Rows.pieceCount :=
  ⟨5 + index.val, by
    have indexLt := index.isLt
    simp only [firstMessageCallCount, Rows.pieceCount] at indexLt ⊢
    omega⟩

def firstSqueezePinIndex : Fin Rows.pieceCount := ⟨8, by decide⟩
def firstSqueezeCallIndex : Fin Rows.pieceCount := ⟨9, by decide⟩
def firstAlgebraIndex : Fin Rows.pieceCount := ⟨10, by decide⟩

def laterRoundBase (round : Fin laterRoundCount) : Nat :=
  11 + 5 * round.val

def laterMessageCallIndex
    (round : Fin laterRoundCount)
    (call : Fin laterMessageCallCount) :
    Fin Rows.pieceCount :=
  ⟨laterRoundBase round + call.val, by
    have roundLt := round.isLt
    have callLt := call.isLt
    simp only [laterRoundBase, laterRoundCount, laterMessageCallCount,
      Rows.pieceCount] at roundLt callLt ⊢
    omega⟩

def laterSqueezePinIndex (round : Fin laterRoundCount) :
    Fin Rows.pieceCount :=
  ⟨laterRoundBase round + 2, by
    have roundLt := round.isLt
    simp only [laterRoundBase, laterRoundCount, Rows.pieceCount]
      at roundLt ⊢
    omega⟩

def laterSqueezeCallIndex (round : Fin laterRoundCount) :
    Fin Rows.pieceCount :=
  ⟨laterRoundBase round + 3, by
    have roundLt := round.isLt
    simp only [laterRoundBase, laterRoundCount, Rows.pieceCount]
      at roundLt ⊢
    omega⟩

def laterAlgebraIndex (round : Fin laterRoundCount) :
    Fin Rows.pieceCount :=
  ⟨laterRoundBase round + 4, by
    have roundLt := round.isLt
    simp only [laterRoundBase, laterRoundCount, Rows.pieceCount]
      at roundLt ⊢
    omega⟩

def prologuePinPiece (index : Fin prologuePinCount) : Piece :=
  Rows.pieceAt (prologuePinIndex index)

def prologueCallPiece (index : Fin prologueCallCount) : Piece :=
  Rows.pieceAt (prologueCallIndex index)

def firstMessageCallPiece (index : Fin firstMessageCallCount) : Piece :=
  Rows.pieceAt (firstMessageCallIndex index)

def firstSqueezePinPiece : Piece :=
  Rows.pieceAt firstSqueezePinIndex

def firstSqueezeCallPiece : Piece :=
  Rows.pieceAt firstSqueezeCallIndex

def firstAlgebraPiece : Piece :=
  Rows.pieceAt firstAlgebraIndex

def laterMessageCallPiece
    (round : Fin laterRoundCount)
    (call : Fin laterMessageCallCount) : Piece :=
  Rows.pieceAt (laterMessageCallIndex round call)

def laterSqueezePinPiece (round : Fin laterRoundCount) : Piece :=
  Rows.pieceAt (laterSqueezePinIndex round)

def laterSqueezeCallPiece (round : Fin laterRoundCount) : Piece :=
  Rows.pieceAt (laterSqueezeCallIndex round)

def laterAlgebraPiece (round : Fin laterRoundCount) : Piece :=
  Rows.pieceAt (laterAlgebraIndex round)

theorem prologuePin_payload :
    forall index : Fin prologuePinCount,
      Rows.payloadKind (prologuePinPiece index) =
        Rows.PayloadKind.ordinary := by
  decide

theorem prologueCall_payload :
    forall index : Fin prologueCallCount,
      Rows.payloadKind (prologueCallPiece index) =
        Rows.PayloadKind.poseidon := by
  decide

theorem firstMessageCall_payload :
    forall index : Fin firstMessageCallCount,
      Rows.payloadKind (firstMessageCallPiece index) =
        Rows.PayloadKind.poseidon := by
  decide

theorem firstSqueezePin_payload :
    Rows.payloadKind firstSqueezePinPiece =
      Rows.PayloadKind.ordinary := by
  decide

theorem firstSqueezeCall_payload :
    Rows.payloadKind firstSqueezeCallPiece =
      Rows.PayloadKind.poseidon := by
  decide

theorem firstAlgebra_payload :
    Rows.payloadKind firstAlgebraPiece =
      Rows.PayloadKind.ordinary := by
  decide

theorem laterMessageCall_payload :
    forall round : Fin laterRoundCount,
      forall call : Fin laterMessageCallCount,
        Rows.payloadKind (laterMessageCallPiece round call) =
          Rows.PayloadKind.poseidon := by
  decide

theorem laterSqueezePin_payload :
    forall round : Fin laterRoundCount,
      Rows.payloadKind (laterSqueezePinPiece round) =
        Rows.PayloadKind.ordinary := by
  decide

theorem laterSqueezeCall_payload :
    forall round : Fin laterRoundCount,
      Rows.payloadKind (laterSqueezeCallPiece round) =
        Rows.PayloadKind.poseidon := by
  decide

theorem laterAlgebra_payload :
    forall round : Fin laterRoundCount,
      Rows.payloadKind (laterAlgebraPiece round) =
        Rows.PayloadKind.ordinary := by
  decide

/-- The first round's ordinary tail contains 30 SumCheck equations and the
length pin that begins round one. This is a physical row-count decomposition;
the equation semantics remain a separate refinement obligation. -/
theorem firstAlgebra_row_formula :
    firstAlgebraPiece.payload.rowCount =
      roundEquationRowCount + 1 := by
  decide

/-- Every later round has 30 equation rows. Rounds one through thirteen also
carry the next message-length pin; round fourteen is final and has no such
row. -/
theorem laterAlgebra_row_formula :
    forall round : Fin laterRoundCount,
      (laterAlgebraPiece round).payload.rowCount =
        roundEquationRowCount +
          (if round.val + 1 < laterRoundCount then 1 else 0) := by
  decide

def firstPhaseIndices : List Nat :=
  List.range 11

def laterRoundIndices (round : Nat) : List Nat :=
  (List.range 5).map fun offset => 11 + 5 * round + offset

def phaseIndices : List Nat :=
  firstPhaseIndices ++
    (List.range laterRoundCount).flatMap laterRoundIndices

/-- The named tree covers the exact owner interval without gaps, duplicates,
or an unclassified tail. -/
theorem phaseIndices_eq_ownerRange :
    phaseIndices = List.range Rows.pieceCount := by
  decide

/-- Physical census reconciled from the protocol tree. -/
theorem familyCounts :
    prologuePinCount + 1 + 1 +
          laterRoundCount * 2 = 33 /\
      prologueCallCount + firstMessageCallCount + 1 +
          laterRoundCount * (laterMessageCallCount + 1) = 48 /\
      33 + 48 = Rows.pieceCount := by
  decide

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.Schedule
