import Mathlib.Data.Int.Notation
import Mathlib.Data.Nat.Notation
import NightstreamFPrime.Spec.Algebra

/-! A Lean copy of the memory-event model that the Rocq proof of the WASM
relation assumes: `formal/wasm-rocq/MemoryTrace.v`, with `Column` and `Witness`
from `R1CS.v`, and the bounded-event premise of `validCircuitRun`. The names
are the Rocq names in Lean style, the events of a trace come in the same order
(rows, then catalog memories, then ports), and the semantics are the same. It
does not own the binding to Nebula (`EventBinding`).

`scripts/check_rocq_copy.py` fails when a Rocq source differs from the copied one:
- `MemoryTrace.v` SHA-256 `e855869aaef01a754c1d6d1d359f489234dc4c00cd796f46c8368ea88173c094`
- `R1CS.v` SHA-256 `54b1936c92777b63b441a9dc106a2b9e55d05e1427c804d6fc81df4de320b1e4` -/

namespace NightstreamFPrime.Spec.Nebula.MemoryTrace

open NightstreamFPrime.Spec

/-- A witness column. -/
abbrev Column := ℕ

/-- One row witness: a canonical base-field integer per column. -/
structure Witness where
  entries : Column → {z : ℤ // 0 ≤ z ∧ z < fieldModulus}

/-- The value of a column (`R1CS.v` `value`). -/
instance : CoeFun Witness fun _ => Column → ℤ := ⟨fun w column => (w.entries column).val⟩

/-- A read port, or a write port with an optional column for the prior value. -/
inductive PortKind
  | read
  | write (priorColumn : Option Column)
deriving DecidableEq

/-- When a port is active: always, when a column is 1, or unless a column is 0. -/
inductive Activation
  | always
  | when (column : Column)
  | «unless» (column : Column)
deriving DecidableEq

inductive MemoryKind
  | ram
  | rom
deriving DecidableEq

structure PortSpec where
  addressColumns : List Column
  valueColumn : Column
  portKind : PortKind
  activation : Activation
deriving DecidableEq

structure MemorySpec where
  memoryId : ℕ
  memoryKind : MemoryKind
  ports : List PortSpec
deriving DecidableEq

/-- An address: one integer per address column. -/
abbrev Address := List ℤ

inductive Event
  | read (memory : ℕ) (address : Address) (value : ℤ)
  | write (memory : ℕ) (address : Address) (value : ℤ) (prior : Option ℤ)
deriving DecidableEq

def active (w : Witness) : Activation → Bool
  | .always => true
  | .when column => w column == 1
  | .«unless» column => w column == 0

/-- The event of a port in a row, or `none` when the port is inactive. -/
def portEvent (memory : ℕ) (w : Witness) (port : PortSpec) : Option Event :=
  if active w port.activation then
    let address := port.addressColumns.map w
    let value := w port.valueColumn
    some (match port.portKind with
      | .read => .read memory address value
      | .write prior => .write memory address value (prior.map w))
  else none

/-- The events of a row: catalog memories in order, then each memory's ports
in order. -/
def eventsForRow (catalog : List MemorySpec) (w : Witness) : List Event :=
  catalog.flatMap fun m => m.ports.flatMap fun port =>
    match portEvent m.memoryId w port with
    | some event => [event]
    | none => []

/-- The events of a trace: rows in order. -/
def eventsForTrace (catalog : List MemorySpec) (rows : List Witness) : List Event :=
  rows.flatMap (eventsForRow catalog)

/-- The value of each address of each memory, if any. -/
abbrev MemoryState := ℕ → Address → Option ℤ

def update (state : MemoryState) (memory : ℕ) (address : Address) (value : ℤ) :
    MemoryState :=
  fun m a => if m = memory then (if a = address then some value else state m a) else state m a

def priorMatches (state : MemoryState) (memory : ℕ) (address : Address) :
    Option ℤ → Prop
  | none => True
  | some value => state memory address = some value

/-- A read returns the current value; a write with a prior value checks it. -/
inductive EventStep : MemoryState → Event → MemoryState → Prop
  | read {state : MemoryState} {memory : ℕ} {address : Address} {value : ℤ} :
      state memory address = some value → EventStep state (.read memory address value) state
  | write {state : MemoryState} {memory : ℕ} {address : Address} {value : ℤ}
      {prior : Option ℤ} : priorMatches state memory address prior →
      EventStep state (.write memory address value prior) (update state memory address value)

inductive Replay : MemoryState → List Event → MemoryState → Prop
  | nil (state : MemoryState) : Replay state [] state
  | cons {before middle after : MemoryState} {event : Event} {events : List Event} :
      EventStep before event middle → Replay middle events after →
      Replay before (event :: events) after

/-- The assumption of the Rocq proof: the events of the trace replay from the
initial state. -/
def MemoryConsistent (catalog : List MemorySpec) (initial : MemoryState)
    (rows : List Witness) : Prop :=
  ∃ final, Replay initial (eventsForTrace catalog rows) final

def u32Value (value : ℤ) : Prop := 0 ≤ value ∧ value < 4294967296

/-- The value of an event is 32-bit, and so is the prior value of a write. -/
def eventValuesBounded : Event → Prop
  | .read _ _ value => u32Value value
  | .write _ _ value prior => u32Value value ∧
      match prior with
      | some before => u32Value before
      | none => True

/-- The other assumption of the Rocq proof on events: each event has 32-bit
values. -/
def boundedMemoryEvents (events : List Event) : Prop := ∀ e ∈ events, eventValuesBounded e

end NightstreamFPrime.Spec.Nebula.MemoryTrace
