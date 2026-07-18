import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.Schedule
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Exact verifier-owned constant schedule for all fifteen terminal `Pi_RLC`
scalar samplers.

Assurance tier: implementation/R1CS correspondence. This file groups every
ordinary constant row by scalar and transcript phase, proves the complete
grouping equals the generated owner pieces, and decodes accepted equations to
integer values under canonical-field assumptions.

Owns: scalar-domain words, block-domain words, scalar/counter coordinates,
squeeze words, accepted-count initialization pins, exact pin-piece ownership,
and semantic facts for every indexed pin.

Does not own: Poseidon2 call semantics, inter-call state connectivity,
candidate decomposition, first-accepted selection, coefficient assembly,
Rust conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: constants are not trusted because they occur in generated
rows. Each value follows from an independently accepted R1CS equation,
canonical assignment values, and the verifier's constant-one column.

| Protocol | Phase | Constraint family | Indexed values | Lean guarantee |
|---|---|---|---|---|
| `Pi_RLC` | scalar entry | raw-pair words | `[2, 0, rho]` | exact scalar-domain schedule for all 15 coordinates |
| `Pi_RLC` | block 0 | initialization | accepted count `0` | each scalar starts a fresh bounded selector chain |
| `Pi_RLC` | block 0 | raw-pair/squeeze words | `[2, 1, rho, 1]` | exact first digest counter schedule |
| `Pi_RLC` | blocks 1-3 | raw-pair/squeeze words | `[2, 1, rho+b, 1]` | exact later digest counter schedule |
| `Pi_RLC` | pin ownership | ordinary owner pieces | scalar/phase tree | every decoded value comes from its exact named piece |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.OwnerCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal

set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

namespace Artifact

def entryLengthColumn (rho : Fin ScalarRows.scalarCount) : Nat :=
  if rho.val = 0 then 2554651 else Schedule.domainBase rho

def entryDomainColumn (rho : Fin ScalarRows.scalarCount) : Nat :=
  entryLengthColumn rho + 1

def coordinateColumn (rho : Fin ScalarRows.scalarCount) : Nat :=
  entryLengthColumn rho + 2

def block0LengthColumn (rho : Fin ScalarRows.scalarCount) : Nat :=
  if rho.val = 0 then 2555255 else Schedule.domainBase rho + 4

def block0DomainColumn (rho : Fin ScalarRows.scalarCount) : Nat :=
  block0LengthColumn rho + 1

def block0CounterColumn (rho : Fin ScalarRows.scalarCount) : Nat :=
  if rho.val = 0 then 2555257
  else Schedule.entryFirstAllocated rho + 600

def block0SqueezeColumn (rho : Fin ScalarRows.scalarCount) : Nat :=
  block0CounterColumn rho + 1

/-- The exact first ordinary piece for one scalar. For successors it also
contains initialization and the first two block-zero words. -/
def entryPins (rho : Fin ScalarRows.scalarCount) : List (Nat × Nat) :=
  if rho.val = 0 then
    [(entryLengthColumn rho, 2),
     (entryDomainColumn rho, 0),
     (coordinateColumn rho, 0)]
  else
    [(entryLengthColumn rho, 2),
     (entryDomainColumn rho, 0),
     (coordinateColumn rho, rho.val),
     (ScalarRows.initialCountColumn rho, 0),
     (block0LengthColumn rho, 2),
     (block0DomainColumn rho, 1)]

/-- Scalar zero owns all block-zero pins here; successors own only counter and
squeeze because their length/domain words precede the shared boundary call. -/
def block0Pins (rho : Fin ScalarRows.scalarCount) : List (Nat × Nat) :=
  if rho.val = 0 then
    [(ScalarRows.initialCountColumn rho, 0),
     (block0LengthColumn rho, 2),
     (block0DomainColumn rho, 1),
     (block0CounterColumn rho, rho.val),
     (block0SqueezeColumn rho, 1)]
  else
    [(block0CounterColumn rho, rho.val),
     (block0SqueezeColumn rho, 1)]

/-- `block = 0,1,2` denotes digest blocks `1,2,3`. -/
def laterPins
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) :
    List (Nat × Nat) :=
  let base := Schedule.laterBlockPinBase rho block
  [(base, 2), (base + 1, 1),
   (base + 2, rho.val + block.val + 1), (base + 3, 1)]

def entryPieceIndex (rho : Fin ScalarRows.scalarCount) :
    Fin ScalarRows.pieceCount :=
  ⟨if rho.val = 0 then 4 else 5 + 43 * rho.val, by
    have rhoLt := rho.isLt
    simp only [ScalarRows.scalarCount, ScalarRows.pieceCount] at rhoLt ⊢
    split <;> omega⟩

def block0PieceIndex (rho : Fin ScalarRows.scalarCount) :
    Fin ScalarRows.pieceCount :=
  ⟨if rho.val = 0 then 6 else 7 + 43 * rho.val, by
    have rhoLt := rho.isLt
    simp only [ScalarRows.scalarCount, ScalarRows.pieceCount] at rhoLt ⊢
    split <;> omega⟩

def laterPieceIndex
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) :
    Fin ScalarRows.pieceCount :=
  ⟨17 + 43 * rho.val + 10 * block.val, by
    have rhoLt := rho.isLt
    have blockLt := block.isLt
    simp only [ScalarRows.scalarCount, ScalarRows.pieceCount] at rhoLt ⊢
    omega⟩

def entryPiece (rho : Fin ScalarRows.scalarCount) : Piece :=
  ScalarRows.pieceAt (entryPieceIndex rho)

def block0Piece (rho : Fin ScalarRows.scalarCount) : Piece :=
  ScalarRows.pieceAt (block0PieceIndex rho)

def laterPiece
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) : Piece :=
  ScalarRows.pieceAt (laterPieceIndex rho block)

def expectedEntryPiece (rho : Fin ScalarRows.scalarCount) : Piece :=
  { rowStart := if rho.val = 0 then 2727312
      else 2727912 + Schedule.scalarRowStride * rho.val
    rowEnd := if rho.val = 0 then 2727315
      else 2727918 + Schedule.scalarRowStride * rho.val
    payload := .ordinary (ConstantPins.rows (entryPins rho)) }

def expectedBlock0Piece (rho : Fin ScalarRows.scalarCount) : Piece :=
  { rowStart := if rho.val = 0 then 2727915
      else 2728518 + Schedule.scalarRowStride * rho.val
    rowEnd := if rho.val = 0 then 2727920
      else 2728520 + Schedule.scalarRowStride * rho.val
    payload := .ordinary (ConstantPins.rows (block0Pins rho)) }

def expectedLaterPiece
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) : Piece :=
  { rowStart := 2729812 + Schedule.scalarRowStride * rho.val +
        1296 * block.val
    rowEnd := 2729816 + Schedule.scalarRowStride * rho.val +
        1296 * block.val
    payload := .ordinary (ConstantPins.rows (laterPins rho block)) }

/-- Closed scalar/phase pin tree over all exact ordinary pieces. -/
theorem pinTree_eq : forall (rho : Fin ScalarRows.scalarCount),
    entryPiece rho = expectedEntryPiece rho /\
    block0Piece rho = expectedBlock0Piece rho /\
    forall block : Fin 3,
      laterPiece rho block = expectedLaterPiece rho block := by
  decide

theorem entryPiece_eq (rho : Fin ScalarRows.scalarCount) :
    entryPiece rho = expectedEntryPiece rho :=
  (pinTree_eq rho).1

theorem block0Piece_eq (rho : Fin ScalarRows.scalarCount) :
    block0Piece rho = expectedBlock0Piece rho :=
  (pinTree_eq rho).2.1

theorem laterPiece_eq
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) :
    laterPiece rho block = expectedLaterPiece rho block :=
  (pinTree_eq rho).2.2 block

theorem entryPiece_mem (rho : Fin ScalarRows.scalarCount) :
    entryPiece rho ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

theorem block0Piece_mem (rho : Fin ScalarRows.scalarCount) :
    block0Piece rho ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

theorem laterPiece_mem
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) :
    laterPiece rho block ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces :=
  ScalarRows.pieceAt_mem _

end Artifact

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  simp [rowsIncluded]

private theorem acceptedPins
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (pins : List (Nat × Nat))
    (valuesCanonical : ConstantPins.ValuesCanonical pins)
    (piece : Piece)
    (piecePayload : piece.payload = .ordinary (ConstantPins.rows pins))
    (pieceMember : piece ∈
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.owner.pieces) :
    forall pin, pin ∈ pins -> assignment pin.1 = pin.2 := by
  have pieceAccepted := accepted piece pieceMember
  rw [Piece.Accepted, piecePayload, Payload.Accepted] at pieceAccepted
  exact ConstantPins.sound valuesCanonical
    (rowsIncluded_self (ConstantPins.rows pins)) canonical one pieceAccepted

theorem entryPinsCanonical : forall rho : Fin ScalarRows.scalarCount,
    ConstantPins.ValuesCanonical (Artifact.entryPins rho) := by
  decide

theorem block0PinsCanonical : forall rho : Fin ScalarRows.scalarCount,
    ConstantPins.ValuesCanonical (Artifact.block0Pins rho) := by
  decide

theorem laterPinsCanonical : forall (rho : Fin ScalarRows.scalarCount)
    (block : Fin 3),
    ConstantPins.ValuesCanonical (Artifact.laterPins rho block) := by
  decide

/-- Exact constant facts grouped by terminal scalar and protocol phase. -/
structure Facts (assignment : Nat -> Nat) : Prop where
  entry : forall rho pin, pin ∈ Artifact.entryPins rho ->
    assignment pin.1 = pin.2
  block0 : forall rho pin, pin ∈ Artifact.block0Pins rho ->
    assignment pin.1 = pin.2
  later : forall rho block pin, pin ∈ Artifact.laterPins rho block ->
    assignment pin.1 = pin.2

theorem facts
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment) :
    Facts assignment := by
  refine {
    entry := ?_
    block0 := ?_
    later := ?_
  }
  · intro rho
    exact acceptedPins canonical one accepted (Artifact.entryPins rho)
      (entryPinsCanonical rho) (Artifact.entryPiece rho)
      (by rw [Artifact.entryPiece_eq]; rfl) (Artifact.entryPiece_mem rho)
  · intro rho
    exact acceptedPins canonical one accepted (Artifact.block0Pins rho)
      (block0PinsCanonical rho) (Artifact.block0Piece rho)
      (by rw [Artifact.block0Piece_eq]; rfl) (Artifact.block0Piece_mem rho)
  · intro rho block
    exact acceptedPins canonical one accepted (Artifact.laterPins rho block)
      (laterPinsCanonical rho block) (Artifact.laterPiece rho block)
      (by rw [Artifact.laterPiece_eq]; rfl)
      (Artifact.laterPiece_mem rho block)

variable {assignment : Nat -> Nat}

theorem Facts.entryLength (self : Facts assignment)
    (rho : Fin ScalarRows.scalarCount) :
    assignment (Artifact.entryLengthColumn rho) = 2 := by
  exact self.entry rho (Artifact.entryLengthColumn rho, 2) (by
    by_cases zero : rho.val = 0 <;>
      simp [Artifact.entryPins, zero])

theorem Facts.entryDomain (self : Facts assignment)
    (rho : Fin ScalarRows.scalarCount) :
    assignment (Artifact.entryDomainColumn rho) = 0 := by
  exact self.entry rho (Artifact.entryDomainColumn rho, 0) (by
    by_cases zero : rho.val = 0 <;>
      simp [Artifact.entryPins, zero])

theorem Facts.entryCoordinate (self : Facts assignment)
    (rho : Fin ScalarRows.scalarCount) :
    assignment (Artifact.coordinateColumn rho) = rho.val := by
  by_cases zero : rho.val = 0
  · have pinned := self.entry rho
      (Artifact.coordinateColumn rho, 0) (by
        simp [Artifact.entryPins, zero])
    simpa [zero] using pinned
  · exact self.entry rho (Artifact.coordinateColumn rho, rho.val) (by
      simp [Artifact.entryPins, zero])

theorem Facts.block0Length (self : Facts assignment)
    (rho : Fin ScalarRows.scalarCount) :
    assignment (Artifact.block0LengthColumn rho) = 2 := by
  by_cases zero : rho.val = 0
  · exact self.block0 rho (Artifact.block0LengthColumn rho, 2) (by
      simp [Artifact.block0Pins, zero])
  · exact self.entry rho (Artifact.block0LengthColumn rho, 2) (by
      simp [Artifact.entryPins, zero])

theorem Facts.block0Domain (self : Facts assignment)
    (rho : Fin ScalarRows.scalarCount) :
    assignment (Artifact.block0DomainColumn rho) = 1 := by
  by_cases zero : rho.val = 0
  · exact self.block0 rho (Artifact.block0DomainColumn rho, 1) (by
      simp [Artifact.block0Pins, zero])
  · exact self.entry rho (Artifact.block0DomainColumn rho, 1) (by
      simp [Artifact.entryPins, zero])

theorem Facts.block0Counter (self : Facts assignment)
    (rho : Fin ScalarRows.scalarCount) :
    assignment (Artifact.block0CounterColumn rho) = rho.val := by
  exact self.block0 rho (Artifact.block0CounterColumn rho, rho.val) (by
    by_cases zero : rho.val = 0 <;>
      simp [Artifact.block0Pins, zero])

theorem Facts.block0Squeeze (self : Facts assignment)
    (rho : Fin ScalarRows.scalarCount) :
    assignment (Artifact.block0SqueezeColumn rho) = 1 := by
  exact self.block0 rho (Artifact.block0SqueezeColumn rho, 1) (by
    by_cases zero : rho.val = 0 <;>
      simp [Artifact.block0Pins, zero])

theorem Facts.laterLength (self : Facts assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) :
    assignment (Schedule.laterBlockPinBase rho block) = 2 :=
  self.later rho block (Schedule.laterBlockPinBase rho block, 2)
    (by simp [Artifact.laterPins])

theorem Facts.laterDomain (self : Facts assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) :
    assignment (Schedule.laterBlockPinBase rho block + 1) = 1 :=
  self.later rho block (Schedule.laterBlockPinBase rho block + 1, 1)
    (by simp [Artifact.laterPins])

theorem Facts.laterCounter (self : Facts assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) :
    assignment (Schedule.laterBlockPinBase rho block + 2) =
      rho.val + block.val + 1 :=
  self.later rho block
    (Schedule.laterBlockPinBase rho block + 2,
      rho.val + block.val + 1)
    (by simp [Artifact.laterPins])

theorem Facts.laterSqueeze (self : Facts assignment)
    (rho : Fin ScalarRows.scalarCount) (block : Fin 3) :
    assignment (Schedule.laterBlockPinBase rho block + 3) = 1 :=
  self.later rho block (Schedule.laterBlockPinBase rho block + 3, 1)
    (by simp [Artifact.laterPins])

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.Terminal.PinSchedule
