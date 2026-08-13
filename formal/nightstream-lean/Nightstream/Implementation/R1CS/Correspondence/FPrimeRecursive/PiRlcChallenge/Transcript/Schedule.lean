import Nightstream.Implementation.R1CS.Artifacts.FPrimeRecursive.PiRlcChallenge.TranscriptLayout
import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcProjection.IndexedRows
import Nightstream.Implementation.R1CS.Core.TranscriptCertificate

/-!
Three-matrix diagnostic PiRLC transcript row schedule.

Owns: lossless conversion of the compact active artifact into generic
constant-pin/Poseidon2 call tables, exact emission-order row ownership, and
transport from explicitly embedded full-program rows to independent trace
acceptance.

Does not own: protocol meaning of any pin, transcript cursor semantics,
Poseidon2-to-sampler replay, the authority of the four PiCCS digest inputs,
whole-program embedding proof, costs, or row removal.

Emits constraints: no.

Authority boundary: the generated artifact supplies physical locations only.
`RowsEmbedded` is an explicit correspondence premise until the complete
production decoder proves these normalized rows are the active F-prime rows.

| Stage path | Mathematical obligation | Evidence |
|---|---|---|
| `nifs.pi_rlc.challenge.transcript.pins` | every emitted constant row fixes its listed canonical value | exact pin row plus full-row satisfaction |
| `nifs.pi_rlc.challenge.transcript.poseidon2` | every 600-row call satisfies the independent fixed SSA program | exact call slice plus full-row satisfaction |
| `nifs.pi_rlc.challenge.transcript.order` | all 772 pins and 257 calls occur exactly once in 1,029 emissions | phase-composed kernel certificate |
-/

namespace Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript.Schedule

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallengeTranscriptLayoutSchema
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge

namespace Layout

abbrev constantPins :=
  FPrimeRecursivePiRlcChallenge.TranscriptLayout.constantPins
abbrev calls := FPrimeRecursivePiRlcChallenge.TranscriptLayout.calls
abbrev emissionOrder :=
  FPrimeRecursivePiRlcChallenge.TranscriptLayout.emissionOrder

end Layout

/-- Lossless physical pin payload consumed by the generic checker. -/
def pinPair (pin : ConstantPin) : Nat × Nat :=
  (pin.column, pin.value)

/-- Lossless compact Poseidon2 call consumed by the generic checker. -/
def CompactCall.toCall (call : CompactCall) : Poseidon2Call.Call where
  rowStart := call.rowStart
  rowEnd := call.rowEnd
  inputColumns := call.inputColumns
  firstAllocatedColumn := call.firstAllocatedColumn

/-- Independent semantic checker instantiated by the active physical tables. -/
def trace : TranscriptCertificate.Trace where
  pins := Layout.constantPins.map pinPair
  calls := Layout.calls.map CompactCall.toCall

def pieceRef : EmissionRef → TranscriptCertificate.PieceRef
  | .pin index => .pin index
  | .call index => .call index

/-- Exact physical emission order, retained separately from protocol meaning. -/
def schedule : List TranscriptCertificate.PieceRef :=
  Layout.emissionOrder.map pieceRef

/-- Reconstructed rows owned by the compact active transcript schedule. -/
def ownerRows : List Row :=
  trace.orderedRows schedule

/-- Absolute source-row start for one scheduled pin or call. -/
def globalRowStart : TranscriptCertificate.PieceRef → Nat
  | .pin index => (Layout.constantPins.getD index default).row
  | .call index => (Layout.calls.getD index default).rowStart

/-- Explicit exact embedding of every compact transcript piece into the full
normalized source program. -/
def RowsEmbedded (fullRows : List Row) : Prop :=
  ∀ piece ∈ schedule,
    ActiveIndexedRows.RowsEmbeddedAt fullRows (globalRowStart piece)
      (piece.rows trace)

theorem pinValuesCanonical : ConstantPins.ValuesCanonical trace.pins := by
  intro pin member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  exact FPrimeRecursivePiRlcChallenge.TranscriptLayout.constant_pin_value_canonical
    source sourceMember

private theorem pin_mem_schedule_iff (index : Nat) :
    TranscriptCertificate.PieceRef.pin index ∈ schedule ↔
      EmissionRef.pin index ∈ Layout.emissionOrder := by
  constructor
  · intro member
    rcases List.mem_map.mp member with ⟨ref, refMember, mapped⟩
    cases ref with
    | pin sourceIndex =>
        simp only [pieceRef, TranscriptCertificate.PieceRef.pin.injEq] at mapped
        subst sourceIndex
        exact refMember
    | call sourceIndex =>
        simp [pieceRef] at mapped
  · intro member
    exact List.mem_map.mpr ⟨.pin index, member, rfl⟩

private theorem call_mem_schedule_iff (index : Nat) :
    TranscriptCertificate.PieceRef.call index ∈ schedule ↔
      EmissionRef.call index ∈ Layout.emissionOrder := by
  constructor
  · intro member
    rcases List.mem_map.mp member with ⟨ref, refMember, mapped⟩
    cases ref with
    | pin sourceIndex =>
        simp [pieceRef] at mapped
    | call sourceIndex =>
        simp only [pieceRef, TranscriptCertificate.PieceRef.call.injEq] at mapped
        subst sourceIndex
        exact refMember
  · intro member
    exact List.mem_map.mpr ⟨.call index, member, rfl⟩

private theorem scheduled_pin_bounded
    (index : Nat) (member : .pin index ∈ schedule) :
    index < trace.pins.length := by
  rw [pin_mem_schedule_iff,
    FPrimeRecursivePiRlcChallenge.TranscriptLayout.pin_mem_emissionOrder_iff]
    at member
  simpa [trace, Layout.constantPins] using member

private theorem scheduled_call_bounded
    (index : Nat) (member : .call index ∈ schedule) :
    index < trace.calls.length := by
  rw [call_mem_schedule_iff,
    FPrimeRecursivePiRlcChallenge.TranscriptLayout.call_mem_emissionOrder_iff]
    at member
  simpa [trace, Layout.calls] using member

private theorem every_pin_scheduled
    (index : Nat) (bounded : index < trace.pins.length) :
    .pin index ∈ schedule := by
  rw [pin_mem_schedule_iff,
    FPrimeRecursivePiRlcChallenge.TranscriptLayout.pin_mem_emissionOrder_iff]
  simpa [trace, Layout.constantPins] using bounded

private theorem every_call_scheduled
    (index : Nat) (bounded : index < trace.calls.length) :
    .call index ∈ schedule := by
  rw [call_mem_schedule_iff,
    FPrimeRecursivePiRlcChallenge.TranscriptLayout.call_mem_emissionOrder_iff]
  simpa [trace, Layout.calls] using bounded

/-- Kernel-clean exact schedule certificate for the active compact trace. -/
theorem orderedValid : trace.OrderedValid schedule ownerRows :=
  { pinIndicesBounded := scheduled_pin_bounded
    callIndicesBounded := scheduled_call_bounded
    everyPinScheduled := every_pin_scheduled
    everyCallScheduled := every_call_scheduled
    pinValuesCanonical := pinValuesCanonical
    exactRows := rfl }

/-- Full-program satisfaction transports through every explicit physical
piece embedding to satisfaction of the exact compact owner rows. -/
theorem ownerRows_satisfied
    {fullRows : List Row} {assignment : Nat → Nat}
    (embedded : RowsEmbedded fullRows)
    (satisfies : Satisfies fullRows assignment) :
    Satisfies ownerRows assignment := by
  unfold ownerRows TranscriptCertificate.Trace.orderedRows
  apply (satisfies_flatten_iff
    (schedule.map fun piece => piece.rows trace) assignment).mpr
  intro rows rowsMember
  rcases List.mem_map.mp rowsMember with ⟨piece, pieceMember, rfl⟩
  exact ActiveIndexedRows.rows_satisfied_of_embeddedAt
    (embedded piece pieceMember) satisfies

/-- Conditional physical-row refinement: exact embedded active rows and
full-program satisfaction imply acceptance by the independent pin/call
checker. This is model-level correspondence over an explicit embedding, not
yet a whole-program Rust-conformance theorem. -/
theorem accepted_of_embedded
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (embedded : RowsEmbedded fullRows)
    (satisfies : Satisfies fullRows assignment) :
    trace.Accepted assignment := by
  exact TranscriptCertificate.ordered_sound orderedValid canonical one
    (ownerRows_satisfied embedded satisfies)

end Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Transcript.Schedule
