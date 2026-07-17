import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Fe.Rows

/-!
Protocol-to-phase tree for the legacy terminal FE SumCheck artifact.

Assurance tier: implementation/R1CS structural correspondence.

Owns: the exact `prologue -> first round -> six later rounds` piece tree,
the 24 Poseidon/17 ordinary payload census, and complete coverage of all 41
legacy owner pieces.

Does not own: semantic FE message widths, transcript equivalence, SumCheck
algebra, inter-round state connectivity, Rust conformance, costs, necessity,
or row removal.

Emits constraints: no.

Authority boundary: this tree says where the old circuit spent rows, not what
the minimal verifier must accept. In particular, its uniform degree-four wire
format remains separate from the independently typed mixed-width FE language.

| Stage path | Mathematical obligation | Physical ownership | Lean owner |
|---|---|---:|---|
| `nifs.pi_ccs.fe_sumcheck.legacy.prologue` | FE domain, initial claim, and round tag enter two permutations | 2 ordinary + 2 Poseidon pieces | `prologuePinPiece`, `prologueCallPiece` |
| `nifs.pi_ccs.fe_sumcheck.legacy.round.0.message` | first degree-four wire message spans three permutations | 1 ordinary + 3 Poseidon pieces | `firstMessageLengthPiece`, `firstMessageCallPiece` |
| `nifs.pi_ccs.fe_sumcheck.legacy.round.0.challenge` | marker and challenge permutation | 1 ordinary + 1 Poseidon piece | `firstSqueezePinPiece`, `firstSqueezeCallPiece` |
| `nifs.pi_ccs.fe_sumcheck.legacy.round.0.algebra` | degree-four claimed-chain equations and next length pin | 1 ordinary piece | `firstAlgebraPiece` |
| `nifs.pi_ccs.fe_sumcheck.legacy.round.1_6` | six uniform degree-four wire rounds | 12 ordinary + 18 Poseidon pieces | indexed later-round owners |
| `nifs.pi_ccs.fe_sumcheck.legacy` | every physical piece has exactly one tree address | 41 pieces | `phaseIndices_eq_ownerRange`, `familyCounts` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Fe.Schedule

open Nightstream.Implementation.R1CS.OwnerCertificate

def prologuePinCount : Nat := 2
def prologueCallCount : Nat := 2
def firstMessageCallCount : Nat := 3
def laterRoundCount : Nat := 6
def laterMessageCallCount : Nat := 2

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

def firstMessageLengthIndex : Fin Rows.pieceCount := ⟨4, by decide⟩

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

def firstMessageLengthPiece : Piece :=
  Rows.pieceAt firstMessageLengthIndex

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
    ∀ index, Rows.payloadKind (prologuePinPiece index) =
      Rows.PayloadKind.ordinary := by
  decide

theorem prologueCall_payload :
    ∀ index, Rows.payloadKind (prologueCallPiece index) =
      Rows.PayloadKind.poseidon := by
  decide

theorem firstMessageLength_payload :
    Rows.payloadKind firstMessageLengthPiece =
      Rows.PayloadKind.ordinary := by
  decide

theorem firstMessageCall_payload :
    ∀ index, Rows.payloadKind (firstMessageCallPiece index) =
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
    ∀ round call,
      Rows.payloadKind (laterMessageCallPiece round call) =
        Rows.PayloadKind.poseidon := by
  decide

theorem laterSqueezePin_payload :
    ∀ round, Rows.payloadKind (laterSqueezePinPiece round) =
      Rows.PayloadKind.ordinary := by
  decide

theorem laterSqueezeCall_payload :
    ∀ round, Rows.payloadKind (laterSqueezeCallPiece round) =
      Rows.PayloadKind.poseidon := by
  decide

theorem laterAlgebra_payload :
    ∀ round, Rows.payloadKind (laterAlgebraPiece round) =
      Rows.PayloadKind.ordinary := by
  decide

def firstPhaseIndices : List Nat :=
  List.range 11

def laterRoundIndices (round : Nat) : List Nat :=
  (List.range 5).map fun offset => 11 + 5 * round + offset

def phaseIndices : List Nat :=
  firstPhaseIndices ++
    (List.range laterRoundCount).flatMap laterRoundIndices

theorem phaseIndices_eq_ownerRange :
    phaseIndices = List.range Rows.pieceCount := by
  decide

theorem familyCounts :
    prologuePinCount + 3 + laterRoundCount * 2 = 17 /\
      prologueCallCount + firstMessageCallCount + 1 +
          laterRoundCount * (laterMessageCallCount + 1) = 24 /\
      17 + 24 = Rows.pieceCount := by
  decide

end Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Fe.Schedule
