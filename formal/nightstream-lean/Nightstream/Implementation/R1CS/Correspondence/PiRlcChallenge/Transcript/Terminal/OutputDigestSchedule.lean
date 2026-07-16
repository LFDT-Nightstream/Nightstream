import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.ScalarRows
import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Ownership.FPrimeFullHistory.FPrimeFullHistoryTerminalPiCcsCatchupArtifact

/-!
Exact terminal-profile structure of the `Pi_CCS` output-digest handoff into
the `Pi_RLC` transcript.

Assurance tier: implementation/R1CS structural correspondence. This file
locates the relevant leaves but gives generated rows no protocol meaning.

Owns: the final `Pi_CCS` catch-up call address; the label and field-count pin
pieces; the two digest-binding Poseidon2 calls; and exact owner membership for
those five protocol leaves.

Does not own: constant semantics, Poseidon2 semantics, state continuity,
digest recomputation, prior `Pi_CCS` transcript correctness, Rust conformance,
row necessity, row removal, or cost totals.

Emits constraints: no.

Authority boundary: the generated owner is used only to locate exact leaves.
Separate modules must decode constants and prove the leaves form the
independently specified transcript transition.

| Protocol | Phase | Constraint family | Exact structural obligation |
|---|---|---|---|
| `Pi_CCS` | transcript catch-up | squeeze pin and Poseidon2 | locate the exact state-producing transition immediately before output hashing |
| `Pi_RLC` | output bind label | five constant rows | label length plus four packed limbs |
| `Pi_RLC` | output bind boundary 0 | Poseidon2 | first full-rate label boundary |
| `Pi_RLC` | output bind count | one constant row | digest field-count word `4` |
| `Pi_RLC` | output bind boundary 1 | Poseidon2 | remaining label, count, and first two digest lanes |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestSchedule

open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal

set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

def catchupCall : Poseidon2Call.Call :=
  { rowStart := 1
    rowEnd := 601
    inputColumns :=
      [1713693, 1692821, 1692822, 1692823,
       1692824, 1692825, 1692826, 1692827]
    firstAllocatedColumn := 1713694 }

def firstBoundaryCall : Poseidon2Call.Call :=
  { rowStart := 5
    rowEnd := 605
    inputColumns :=
      [2553445, 2553446, 2553447, 2553448,
       1714290, 1714291, 1714292, 1714293]
    firstAllocatedColumn := 2553450 }

def secondBoundaryCall : Poseidon2Call.Call :=
  { rowStart := 606
    rowEnd := 1206
    inputColumns :=
      [2553449, 2554050, 2553433, 2553434,
       2554046, 2554047, 2554048, 2554049]
    firstAllocatedColumn := 2554051 }

def labelPins : List (Nat × Nat) :=
  [(2553445, 26),
   (2553446, 13338641331874160),
   (2553447, 27970976485502569),
   (2553448, 28252447032566124),
   (2553449, 500152231785)]

def fieldCountPins : List (Nat × Nat) := [(2554050, 4)]

def catchupSqueezePins : List (Nat × Nat) := [(1713693, 1)]

def catchupPinPieceIndex :
    Fin FPrimeFullHistoryTerminalPiCcsCatchup.owner.pieces.length :=
  ⟨0, by decide⟩

def catchupPieceIndex :
    Fin FPrimeFullHistoryTerminalPiCcsCatchup.owner.pieces.length :=
  ⟨1, by decide⟩

def catchupPinPiece : Piece :=
  FPrimeFullHistoryTerminalPiCcsCatchup.owner.pieces.get catchupPinPieceIndex

def catchupPiece : Piece :=
  FPrimeFullHistoryTerminalPiCcsCatchup.owner.pieces.get catchupPieceIndex

def labelPiece : Piece := ScalarRows.pieceAt ⟨0, by decide⟩
def firstBoundaryPiece : Piece := ScalarRows.pieceAt ⟨1, by decide⟩
def fieldCountPiece : Piece := ScalarRows.pieceAt ⟨2, by decide⟩
def secondBoundaryPiece : Piece := ScalarRows.pieceAt ⟨3, by decide⟩

def expectedCatchupPiece : Piece :=
  { rowStart := 1872709
    rowEnd := 1873309
    payload := .poseidon catchupCall }

def expectedCatchupPinPiece : Piece :=
  { rowStart := 1872708
    rowEnd := 1872709
    payload := .ordinary (ConstantPins.rows catchupSqueezePins) }

def expectedLabelPiece : Piece :=
  { rowStart := 2726106
    rowEnd := 2726111
    payload := .ordinary (ConstantPins.rows labelPins) }

def expectedFirstBoundaryPiece : Piece :=
  { rowStart := 2726111
    rowEnd := 2726711
    payload := .poseidon firstBoundaryCall }

def expectedFieldCountPiece : Piece :=
  { rowStart := 2726711
    rowEnd := 2726712
    payload := .ordinary (ConstantPins.rows fieldCountPins) }

def expectedSecondBoundaryPiece : Piece :=
  { rowStart := 2726712
    rowEnd := 2727312
    payload := .poseidon secondBoundaryCall }

/-- Closed protocol/phase/family structure for the complete two-boundary
handoff and its immediately preceding state-producing call. -/
theorem scheduleTree_eq :
    catchupPinPiece = expectedCatchupPinPiece /\
    catchupPiece = expectedCatchupPiece /\
    labelPiece = expectedLabelPiece /\
    firstBoundaryPiece = expectedFirstBoundaryPiece /\
    fieldCountPiece = expectedFieldCountPiece /\
    secondBoundaryPiece = expectedSecondBoundaryPiece := by
  decide

theorem catchupPiece_eq : catchupPiece = expectedCatchupPiece :=
  scheduleTree_eq.2.1

theorem catchupPinPiece_eq :
    catchupPinPiece = expectedCatchupPinPiece :=
  scheduleTree_eq.1

theorem labelPiece_eq : labelPiece = expectedLabelPiece :=
  scheduleTree_eq.2.2.1

theorem firstBoundaryPiece_eq :
    firstBoundaryPiece = expectedFirstBoundaryPiece :=
  scheduleTree_eq.2.2.2.1

theorem fieldCountPiece_eq : fieldCountPiece = expectedFieldCountPiece :=
  scheduleTree_eq.2.2.2.2.1

theorem secondBoundaryPiece_eq :
    secondBoundaryPiece = expectedSecondBoundaryPiece :=
  scheduleTree_eq.2.2.2.2.2

theorem catchupPinPiece_mem :
    catchupPinPiece ∈ FPrimeFullHistoryTerminalPiCcsCatchup.owner.pieces :=
  List.get_mem _ _

theorem catchupPiece_mem :
    catchupPiece ∈ FPrimeFullHistoryTerminalPiCcsCatchup.owner.pieces :=
  List.get_mem _ _

theorem labelPiece_mem :
    labelPiece ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

theorem firstBoundaryPiece_mem :
    firstBoundaryPiece ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

theorem fieldCountPiece_mem :
    fieldCountPiece ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

theorem secondBoundaryPiece_mem :
    secondBoundaryPiece ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.OutputDigestSchedule
