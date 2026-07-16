import Nightstream.Implementation.R1CS.Core.ConstantPins
import Nightstream.Implementation.R1CS.Core.Poseidon2Call
import Nightstream.Implementation.R1CS.Core.CheckedProgram

/-!
Contract: compact semantic certificates for exact in-circuit Poseidon2
transcript owners.

`TranscriptGadget` emits only verifier-owned constant bindings and instances
of the fixed 600-row Poseidon2 permutation program.  A generated owner records
those pins and calls plus exact coverage certificates.  The executable checker
below independently replays every SSA-defined permutation wire; it does not
evaluate an R1CS row and carries no accepted bit from Rust.
-/

namespace Nightstream.Implementation.R1CS.TranscriptCertificate

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.CheckedProgram

set_option maxRecDepth 262144
set_option maxHeartbeats 8000000

private theorem poseidonRows_reference_known :
    ∀ row ∈ Poseidon2Permutation.rows,
      ∀ column ∈ rowRefs row,
        column ∈ knownAfter Poseidon2Permutation.inputColumns
          Poseidon2Permutation.definitions := by
  decide

/-- Independent executable semantics for one renamed production permutation:
the assignment agrees with the fixed SSA interpreter on all input and derived
columns. -/
def CallAccepted (call : Poseidon2Call.Call)
    (assignment : Nat → Nat) : Prop :=
  AgreeOn
    (Poseidon2PermutationSound.interpret
      (pullAssignment assignment call.columnMap))
    (pullAssignment assignment call.columnMap)
    (knownAfter Poseidon2Permutation.inputColumns
      Poseidon2Permutation.definitions)

def callCheck (call : Poseidon2Call.Call)
    (assignment : Nat → Nat) : Bool :=
  (knownAfter Poseidon2Permutation.inputColumns
      Poseidon2Permutation.definitions).all fun column =>
    decide
      (Poseidon2PermutationSound.interpret
          (pullAssignment assignment call.columnMap) column =
        pullAssignment assignment call.columnMap column)

theorem callCheck_eq_true_iff
    (call : Poseidon2Call.Call) (assignment : Nat → Nat) :
    callCheck call assignment = true ↔ CallAccepted call assignment := by
  constructor
  · intro accepted column member
    have checked := (List.all_eq_true.mp accepted) column member
    exact of_decide_eq_true checked
  · intro accepted
    apply List.all_eq_true.mpr
    intro column member
    exact decide_eq_true (accepted column member)

/-- Compiler completeness for one renamed permutation call.  Agreement with
the SSA interpreter, rather than row satisfaction, is the premise. -/
theorem call_complete
    (call : Poseidon2Call.Call) {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : CallAccepted call assignment) :
    Satisfies call.rows assignment := by
  intro mapped member
  rcases List.mem_map.mp member with ⟨row, rowMember, rfl⟩
  apply (rowHolds_pull_iff assignment call.columnMap row).mp
  have compiled := Poseidon2PermutationSound.poseidon2Permutation_complete
    (state := pullAssignment assignment call.columnMap)
    (fun column => canonical (call.columnMap column))
    (by simpa [pullAssignment] using one)
  exact (rowHolds_agree accepted row
    (poseidonRows_reference_known row rowMember)).mp
      (compiled row rowMember)

structure Trace where
  pins : List (Nat × Nat)
  calls : List Poseidon2Call.Call
deriving DecidableEq, Repr, Inhabited

def Trace.semanticRows (trace : Trace) : List Row :=
  ConstantPins.rows trace.pins ++ trace.calls.flatMap Poseidon2Call.Call.rows

/-- Structural-only artifact certificate.  Exact pins and call slices occur
in the owner, and every owner row is covered by one of those semantic pieces. -/
structure Trace.Valid (trace : Trace) (ownerRows : List Row) : Prop where
  pinValuesCanonical : ConstantPins.ValuesCanonical trace.pins
  pinsIncluded : rowsIncluded (ConstantPins.rows trace.pins) ownerRows = true
  callsMatch : ∀ call ∈ trace.calls, call.Matches ownerRows
  ownerCovered : rowsIncluded ownerRows trace.semanticRows = true

def Trace.Accepted (trace : Trace) (assignment : Nat → Nat) : Prop :=
  (∀ pin ∈ trace.pins, assignment pin.1 = pin.2) ∧
  ∀ call ∈ trace.calls, CallAccepted call assignment

def Trace.check (trace : Trace) (assignment : Nat → Nat) : Bool :=
  trace.pins.all (fun pin => decide (assignment pin.1 = pin.2)) &&
    trace.calls.all (fun call => callCheck call assignment)

theorem Trace.check_eq_true_iff
    (trace : Trace) (assignment : Nat → Nat) :
    trace.check assignment = true ↔ trace.Accepted assignment := by
  rw [Trace.check, Bool.and_eq_true]
  constructor
  · rintro ⟨pins, calls⟩
    constructor
    · intro pin member
      exact of_decide_eq_true ((List.all_eq_true.mp pins) pin member)
    · intro call member
      apply (callCheck_eq_true_iff call assignment).mp
      exact (List.all_eq_true.mp calls) call member
  · rintro ⟨pins, calls⟩
    constructor
    · apply List.all_eq_true.mpr
      intro pin member
      exact decide_eq_true (pins pin member)
    · apply List.all_eq_true.mpr
      intro call member
      exact (callCheck_eq_true_iff call assignment).mpr (calls call member)

/-- Exact owner rows reconstruct the independent transcript checker. -/
theorem sound
    {trace : Trace} {ownerRows : List Row} {assignment : Nat → Nat}
    (valid : trace.Valid ownerRows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies ownerRows assignment) :
    trace.Accepted assignment := by
  constructor
  · exact ConstantPins.sound valid.pinValuesCanonical valid.pinsIncluded
      canonical one satisfies
  · intro call member
    exact Poseidon2Call.sound call ownerRows (valid.callsMatch call member)
      canonical one satisfies

/-- Independent transcript acceptance satisfies every exact owner row. -/
theorem complete
    {trace : Trace} {ownerRows : List Row} {assignment : Nat → Nat}
    (valid : trace.Valid ownerRows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : trace.Accepted assignment) :
    Satisfies ownerRows assignment := by
  have semanticSatisfies : Satisfies trace.semanticRows assignment := by
    intro row member
    rw [Trace.semanticRows, List.mem_append] at member
    rcases member with pinMember | callMember
    · exact ConstantPins.complete valid.pinValuesCanonical one
        accepted.1 row pinMember
    · rcases List.mem_flatMap.mp callMember with
        ⟨call, callInTrace, rowInCall⟩
      exact call_complete call canonical one
        (accepted.2 call callInTrace) row rowInCall
  intro row member
  exact semanticSatisfies row
    (rowsIncluded_sound valid.ownerCovered row member)

/-- One emission-order reference into a transcript's compact pin/call tables. -/
inductive PieceRef where
  | pin (index : Nat)
  | call (index : Nat)
deriving DecidableEq, Repr, Inhabited

def PieceRef.rows (trace : Trace) : PieceRef → List Row
  | .pin index => ConstantPins.rows [trace.pins.getD index (0, 0)]
  | .call index => (trace.calls.getD index default).rows

def Trace.orderedRows (trace : Trace) (schedule : List PieceRef) : List Row :=
  (schedule.map fun piece => piece.rows trace).flatten

/-- Linear-size exact owner certificate.  Unlike unordered `ownerCovered`,
this representation never scans a many-thousand-row owner for every row: the
generated schedule names each pin/call once in emission order. -/
structure Trace.OrderedValid
    (trace : Trace) (schedule : List PieceRef) (ownerRows : List Row) : Prop where
  pinIndicesBounded : ∀ index, .pin index ∈ schedule → index < trace.pins.length
  callIndicesBounded : ∀ index, .call index ∈ schedule → index < trace.calls.length
  everyPinScheduled : ∀ index, index < trace.pins.length → .pin index ∈ schedule
  everyCallScheduled : ∀ index, index < trace.calls.length → .call index ∈ schedule
  pinValuesCanonical : ConstantPins.ValuesCanonical trace.pins
  exactRows : ownerRows = trace.orderedRows schedule

universe u

private theorem getD_mem_of_lt {α : Type u} [Inhabited α]
    {entries : List α} {index : Nat}
    (indexLt : index < entries.length) :
    entries.getD index default ∈ entries := by
  have member := List.getElem_mem (l := entries) indexLt
  rwa [List.getElem_eq_getD default] at member

private theorem piece_satisfies
    {trace : Trace} {schedule : List PieceRef} {ownerRows : List Row}
    {assignment : Nat → Nat}
    (valid : trace.OrderedValid schedule ownerRows)
    (satisfies : Satisfies ownerRows assignment)
    (piece : PieceRef) (member : piece ∈ schedule) :
    Satisfies (piece.rows trace) assignment := by
  rw [valid.exactRows] at satisfies
  unfold Trace.orderedRows at satisfies
  have pieces := (satisfies_flatten_iff
    (schedule.map fun current => current.rows trace) assignment).mp satisfies
  exact pieces (piece.rows trace) (List.mem_map.mpr ⟨piece, member, rfl⟩)

private theorem rowsIncluded_self (rows : List Row) :
    rowsIncluded rows rows = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true member

/-- Ordered exact rows reconstruct the same independent transcript semantics
without a quadratic set-inclusion certificate. -/
theorem ordered_sound
    {trace : Trace} {schedule : List PieceRef} {ownerRows : List Row}
    {assignment : Nat → Nat}
    (valid : trace.OrderedValid schedule ownerRows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies ownerRows assignment) :
    trace.Accepted assignment := by
  constructor
  · intro pin pinMember
    rcases List.mem_iff_getElem.mp pinMember with ⟨index, indexLt, pinEq⟩
    have getEq := List.getElem_eq_getD (l := trace.pins)
      (i := index) (h := indexLt) (0, 0)
    rw [getEq] at pinEq
    subst pin
    have localSatisfies := piece_satisfies valid satisfies (.pin index)
      (valid.everyPinScheduled index indexLt)
    have valuesCanonical : ConstantPins.ValuesCanonical
        [trace.pins.getD index (0, 0)] := by
      intro candidate candidateMember
      simp only [List.mem_singleton] at candidateMember
      subst candidate
      exact valid.pinValuesCanonical _ (getD_mem_of_lt indexLt)
    change Satisfies
      (ConstantPins.rows [trace.pins.getD index (0, 0)]) assignment
        at localSatisfies
    have facts := ConstantPins.sound valuesCanonical
      (rowsIncluded_self _) canonical one localSatisfies
    exact facts _ (by simp)
  · intro call callMember
    rcases List.mem_iff_getElem.mp callMember with ⟨index, indexLt, callEq⟩
    have getEq := List.getElem_eq_getD (l := trace.calls)
      (i := index) (h := indexLt) default
    rw [getEq] at callEq
    subst call
    have localSatisfies := piece_satisfies valid satisfies (.call index)
      (valid.everyCallScheduled index indexLt)
    exact Poseidon2PermutationSound.poseidon2Permutation_renamed_sound
      (trace.calls.getD index default).columnMap
      (trace.calls.getD index default).columnMap_zero canonical one
      localSatisfies

/-- Independent acceptance compiles to every row in an ordered exact owner. -/
theorem ordered_complete
    {trace : Trace} {schedule : List PieceRef} {ownerRows : List Row}
    {assignment : Nat → Nat}
    (valid : trace.OrderedValid schedule ownerRows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (accepted : trace.Accepted assignment) :
    Satisfies ownerRows assignment := by
  rw [valid.exactRows]
  unfold Trace.orderedRows
  apply (satisfies_flatten_iff
    (schedule.map fun piece => piece.rows trace) assignment).mpr
  intro rows rowsMember
  rcases List.mem_map.mp rowsMember with ⟨piece, pieceMember, rfl⟩
  cases piece with
  | pin index =>
      have indexLt := valid.pinIndicesBounded index pieceMember
      apply ConstantPins.complete
      · intro pin pinMember
        simp only [List.mem_singleton] at pinMember
        subst pin
        exact valid.pinValuesCanonical _ (getD_mem_of_lt indexLt)
      · exact one
      · intro pin pinMember
        simp only [List.mem_singleton] at pinMember
        subst pin
        exact accepted.1 _ (getD_mem_of_lt indexLt)
  | call index =>
      have indexLt := valid.callIndicesBounded index pieceMember
      exact call_complete _ canonical one
        (accepted.2 _ (getD_mem_of_lt indexLt))

end Nightstream.Implementation.R1CS.TranscriptCertificate
