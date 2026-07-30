import Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex
import Nightstream.Implementation.R1CS.Canonical.Poseidon2HonestFrom
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
import Nightstream.Implementation.R1CS.Canonical.KMul
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: a Lean-owned symbolic planner for the width-8 Poseidon2 duplex.

`TranscriptRecipe` is intentionally column-only: an overwritten lane is one
source column.  A complete protocol transcript also absorbs domain constants
and row-free linear encodings of extension values.  Those are expressions, not
new witness columns.  This module is the smallest generalization required to
represent them without allocating fake constant columns.

The planner mirrors `Poseidon2Duplex`:

* overwrite the current rate lane;
* when the rate is full, emit the pending permutation before the write;
* before every squeeze, absorb the constant one and emit one permutation;
* expose lanes zero and one as one carried quadratic-extension challenge.

Every emitted entry is lowered with `normalizedCanonicalProgramFrom`.  Counts
and soundness are consequently derived from the emitted list, not supplied by
a transcript profile or measured from Rust.

This module does not choose a protocol serialization.  PiCCS and PiRLC own
their typed field order and call this planner.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.SymbolicDuplex

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Honest
open Nightstream.Implementation.R1CS.Canonical.Poseidon2HonestFrom
open Nightstream.Implementation.R1CS.Canonical.KMul

/-- A carried-entry permutation allocates only its eight bound outputs and
344 S-box columns.  The eight standalone input ports are absent because the
entry is already a sparse linear-combination state. -/
def stride : Nat := 352

theorem stride_eq : stride = 352 := rfl

/-- Compact call-local layout.  Output ports occupy the first eight columns
and S-box auxiliaries the following 344.  `inputPort` is deliberately unused:
`normalizedCanonicalProgramFrom` consumes `Entry.state` directly. -/
def layoutAt (base call : Nat) : Layout where
  auxBase := base + call * stride + width
  inputPort := fun _ => 0
  outputPort := fun lane => base + call * stride + lane.val

/-- The symbolic state after call `call` consists exactly of that call's
output ports. -/
def outputState (base call : Nat) : State :=
  fun lane => [((layoutAt base call).outputPort lane, 1)]

/-- One emitted permutation entry carries its own positional call index. -/
structure Entry where
  call : Nat
  state : State

/-- A static duplex state together with the permutation entries emitted so
far.  `absorbed` is encoding-time control state, never a witness value. -/
structure Builder where
  entries : List Entry
  lanes : State
  absorbed : Nat

/-- Start from a caller-selected symbolic state and cursor. -/
def start (lanes : State) (absorbed : Nat := 0) : Builder :=
  { entries := [], lanes, absorbed }

/-- The all-zero transcript state. -/
def empty : Builder :=
  start (fun _ => []) 0

/-- Emit the pending permutation and carry its output ports forward. -/
def permute (base : Nat) (builder : Builder) : Builder :=
  let call := builder.entries.length
  { entries := builder.entries ++ [{ call, state := builder.lanes }]
    lanes := outputState base call
    absorbed := 0 }

/-- Restore a rate cursor before a write, exactly as the value-level duplex
guard does. -/
def guarded (base : Nat) (builder : Builder) : Builder :=
  if Poseidon2Sponge.rate ≤ builder.absorbed
  then permute base builder
  else builder

/-- Overwrite one symbolic rate lane. -/
def absorb (base : Nat) (value : LinCombNormal.LinComb)
    (builder : Builder) : Builder :=
  let ready := guarded base builder
  { entries := ready.entries
    lanes := fun lane =>
      if lane.val = ready.absorbed then value else ready.lanes lane
    absorbed := ready.absorbed + 1 }

/-- Absorb a typed field list from left to right. -/
def absorbMany (base : Nat) :
    List LinCombNormal.LinComb → Builder → Builder
  | [], builder => builder
  | value :: rest, builder =>
      absorbMany base rest (absorb base value builder)

/-- Constant one as a row-free expression on the shared constant wire. -/
def one : LinCombNormal.LinComb := [(0, 1)]

/-- Pre-squeeze domain gate: absorb one, then force one permutation. -/
def gate (base : Nat) (builder : Builder) : Builder :=
  permute base (absorb base one builder)

/-- One extension challenge and the successor builder.  The challenge is
definitionally the first two output lanes of the gate permutation. -/
def squeezeK (base : Nat) (builder : Builder) : Carried × Builder :=
  let next := gate base builder
  (⟨next.lanes ⟨0, by decide⟩, next.lanes ⟨1, by decide⟩⟩, next)

@[simp] theorem permute_entries_length (base : Nat) (builder : Builder) :
    (permute base builder).entries.length = builder.entries.length + 1 := by
  simp [permute]

@[simp] theorem permute_absorbed (base : Nat) (builder : Builder) :
    (permute base builder).absorbed = 0 := rfl

@[simp] theorem gate_absorbed (base : Nat) (builder : Builder) :
    (gate base builder).absorbed = 0 := rfl

@[simp] theorem squeezeK_absorbed (base : Nat) (builder : Builder) :
    (squeezeK base builder).2.absorbed = 0 := rfl

/-- Lower one self-indexed permutation entry. -/
def entryRows (base : Nat) (constants : Constants) (entry : Entry) : List Row :=
  normalizedCanonicalProgramFrom (layoutAt base entry.call)
    entry.state constants

/-- Lower an indexed list of permutation entries. -/
def rowsFrom (base : Nat) (constants : Constants) : List Entry → List Row
  | [] => []
  | entry :: rest =>
      entryRows base constants entry ++ rowsFrom base constants rest

/-- Lower every entry produced by a builder. -/
def rows (base : Nat) (constants : Constants) (builder : Builder) : List Row :=
  rowsFrom base constants builder.entries

theorem rowsFrom_length (base : Nat) (constants : Constants) :
    ∀ entries : List Entry,
      (rowsFrom base constants entries).length = entries.length * 352
  | [] => rfl
  | entry :: rest => by
      rw [rowsFrom, List.length_append,
        show (entryRows base constants entry).length = 352 by
          exact normalizedCanonicalProgramFrom_length
            (layoutAt base entry.call) entry.state constants,
        rowsFrom_length base constants rest]
      simp only [List.length_cons, Nat.succ_mul]
      omega

/-- Exact row count, derived from the emitted entry list. -/
theorem rows_length (base : Nat) (constants : Constants) (builder : Builder) :
    (rows base constants builder).length =
      builder.entries.length * 352 :=
  rowsFrom_length base constants builder.entries

/-- Restrict a lowered list's satisfaction to any member entry. -/
theorem rowsFrom_satisfies_entry
    (base : Nat) (constants : Constants) (assignment : Nat → Nat) :
    ∀ (entries : List Entry),
      Satisfies (rowsFrom base constants entries) assignment →
      ∀ entry ∈ entries, Satisfies (entryRows base constants entry) assignment
  | [], _, _, member => by cases member
  | head :: rest, satisfied, entry, member => by
      rcases List.mem_cons.1 member with rfl | inRest
      · intro row rowMember
        exact satisfied row (List.mem_append_left _ rowMember)
      · apply rowsFrom_satisfies_entry base constants assignment rest
          (fun row rowMember =>
            satisfied row (List.mem_append_right _ rowMember))
          entry inRest

/-- Restrict whole-program satisfaction to any emitted entry. -/
theorem satisfies_entry
    (base : Nat) (constants : Constants) (builder : Builder)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows base constants builder) assignment)
    (entry : Entry) (member : entry ∈ builder.entries) :
    Satisfies (entryRows base constants entry) assignment :=
  rowsFrom_satisfies_entry base constants assignment builder.entries
    (by simpa [rows] using satisfied) entry member

/-- Every emitted call computes the selected Poseidon2 permutation of its
symbolic entry. -/
theorem call_computes_reference
    (base : Nat) (constants : Constants) (builder : Builder)
    (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows base constants builder) assignment)
    (entry : Entry) (member : entry ∈ builder.entries)
    (entryValues : Values)
    (entryAgrees : ∀ lane : Fin width,
      lcEval assignment (entry.state lane) = entryValues lane)
    (lane : Fin width) :
    assignment ((layoutAt base entry.call).outputPort lane) =
      referencePermutation constants entryValues lane := by
  apply normalizedCanonicalProgramFrom_computes_reference
    (layoutAt base entry.call) constants assignment
    entry.state entryValues residues constantWire entryAgrees
  exact satisfies_entry base constants builder assignment satisfied entry member

/-! ## Honest completeness -/

/-- Per-call witness facts consumed by the existing permutation completeness
theorem. -/
structure EntryHonest
    (base : Nat) (constants : Constants) (assignment : Nat → Nat)
    (entry : Entry)
    (entryValues : Values) : Prop where
  entryAgrees : ∀ lane : Fin width,
    lcEval assignment (entry.state lane) = entryValues lane
  sboxAgrees : ∀ (sboxIndex : Fin sboxCount) (slot : Fin columnsPerSbox),
    assignment
        (sboxColumn (layoutAt base entry.call) sboxIndex slot) =
      chainSlot (sboxInputValue constants entryValues sboxIndex.val) slot.val
  outputAgrees : ∀ lane : Fin width,
    assignment ((layoutAt base entry.call).outputPort lane) =
      referencePermutation constants entryValues lane

theorem rowsFrom_honest
    (base : Nat) (constants : Constants) (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1) :
    ∀ (entries : List Entry) (values : Entry → Values),
      (∀ entry ∈ entries,
        EntryHonest base constants assignment entry (values entry)) →
      Satisfies (rowsFrom base constants entries) assignment
  | [], _, _ => by
      intro row member
      cases member
  | entry :: rest, values, honest => by
      have head := honest entry (by simp)
      have headSatisfied :
          Satisfies (entryRows base constants entry) assignment := by
        unfold entryRows
        apply honest_satisfies_normalizedFrom
          (layoutAt base entry.call) entry.state constants (values entry)
          assignment residues constantWire head.entryAgrees head.sboxAgrees
          head.outputAgrees
      have tailHonest : ∀ other ∈ rest,
          EntryHonest base constants assignment other (values other) :=
        fun other member => honest other (by simp [member])
      have tailSatisfied :=
        rowsFrom_honest base constants assignment residues constantWire
          rest values tailHonest
      intro row member
      rcases List.mem_append.1 member with inHead | inTail
      · exact headSatisfied row inHead
      · exact tailSatisfied row inTail

/-- Exact standalone structural cost of the emitted calls.

Each carried permutation owns its 344 S-box columns and its eight bound output
ports.  The latter are internal transcript state, not caller-owned call
outputs, so omitting them would count only an intrinsic permutation subtotal. -/
def cost (builder : Builder) : Lowering.Typed.Cost where
  recurringRows := builder.entries.length * 352
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := builder.entries.length * 352

theorem cost_rows (base : Nat) (constants : Constants) (builder : Builder) :
    (rows base constants builder).length = (cost builder).recurringRows :=
  rows_length base constants builder

end Nightstream.Implementation.R1CS.Canonical.SymbolicDuplex
