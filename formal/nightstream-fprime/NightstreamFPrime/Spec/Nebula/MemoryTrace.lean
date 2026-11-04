import Mathlib.Data.Int.Notation
import Mathlib.Data.Nat.Notation
import NightstreamFPrime.Spec.Algebra

/-! A Lean copy of the memory-event model that the Rocq proof of the WASM
relation assumes consistent: `formal/wasm-rocq/MemoryTrace.v`, with `Column`
and `Witness` from `R1CS.v`, at `main` commit
`5e7f5caa846bc2bf51d57d0fd3a1168cf19d98e2`. The names are the Rocq names in
Lean style, the events of a trace come in the same order (rows, then catalog
memories, then ports), and the semantics are the same. It does not own the
binding to Nebula (`EventBinding`). -/

namespace NightstreamFPrime.Spec.Nebula.MemoryTrace

open NightstreamFPrime.Spec

/-- A witness column. -/
abbrev Column := ℕ

/-- One row witness: a canonical Goldilocks integer per column. -/
structure Witness where
  entries : Column → {z : ℤ // 0 ≤ z ∧ z < goldilocksModulus}

/-- The value of a column (`R1CS.v` `value`). -/
instance : CoeFun Witness fun _ => Column → ℤ := ⟨fun w column => (w.entries column).val⟩

/-- A read port, or a write port with an optional column for the prior value. -/
inductive PortKind
  | read
  | write (priorColumn : Option Column)

/-- When a port is active: always, when a column is 1, or unless a column is 0. -/
inductive Activation
  | always
  | when (column : Column)
  | «unless» (column : Column)

inductive MemoryKind
  | ram
  | rom
deriving DecidableEq

structure PortSpec where
  addressColumns : List Column
  valueColumn : Column
  portKind : PortKind
  activation : Activation

structure MemorySpec where
  memoryId : ℕ
  memoryKind : MemoryKind
  ports : List PortSpec

/-- An address: one integer per address column. -/
abbrev Address := List ℤ

inductive Event
  | read (memory : ℕ) (address : Address) (value : ℤ)
  | write (memory : ℕ) (address : Address) (value : ℤ) (prior : Option ℤ)

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

end NightstreamFPrime.Spec.Nebula.MemoryTrace
