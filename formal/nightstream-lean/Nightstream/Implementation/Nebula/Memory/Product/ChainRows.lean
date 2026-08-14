import Nightstream.Implementation.Nebula.Memory.Product.RecordFactorRows
import Nightstream.Implementation.R1CS.Canonical.KEquality

/-!
Contract: exact sequential row program for one Nebula V2 fingerprint product.

Assurance tier: implementation model.

Owns left-to-right composition of record-factor updates, the final two-row
extension equality, exact row accounting, and arbitrary-assignment soundness
against an independently defined product recursion.

Does not own the fixed V2 slot census, typed record decoding, claim-column
placement, honest auxiliary allocation, or a generated artifact.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryProductChainRows

open Nightstream.Implementation.Nebula.MemoryRecordFactorRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

private theorem pair_ext
    {left right : Pair} (low : left.low = right.low)
    (high : left.high = right.high) : left = right := by
  cases left
  cases right
  simp_all

/-- One factor in a chain. Its semantic value does not depend on the running
product. The three frames are distinct generated allocations. -/
structure Entry where
  packed : LinComb
  value : LinComb
  activation : Activation
  scaleFrame : Frame
  gateFrame : Frame
  updateFrame : Frame

def Entry.layout (entry : Entry) (gamma1 gamma2 previous : Carried) :
    MemoryRecordFactorRows.Layout :=
  { gamma1 := gamma1
    gamma2 := gamma2
    previous := previous
    packed := entry.packed
    value := entry.value
    activation := entry.activation
    scaleFrame := entry.scaleFrame
    gateFrame := entry.gateFrame
    updateFrame := entry.updateFrame }

/-- Reference gate value. `zeroCarried` only fills the irrelevant running-
product field of the leaf layout. -/
def Entry.gateValue (assignment : Nat → Nat) (entry : Entry)
    (gamma1 gamma2 : Carried) : Pair :=
  semanticGate assignment (entry.layout gamma1 gamma2 KLinear.zeroCarried)

theorem Entry.semanticGate_layout
    (assignment : Nat → Nat) (entry : Entry)
    (gamma1 gamma2 previous : Carried) :
    semanticGate assignment (entry.layout gamma1 gamma2 previous) =
      entry.gateValue assignment gamma1 gamma2 := by
  cases entry.activation <;> rfl

def rowsFrom (gamma1 gamma2 : Carried) :
    Carried → List Entry → List Row
  | _, [] => []
  | previous, entry :: rest =>
      let leaf := entry.layout gamma1 gamma2 previous
      MemoryRecordFactorRows.rows leaf ++
        rowsFrom gamma1 gamma2 (MemoryRecordFactorRows.outputCarried leaf) rest

def carriedAfter (gamma1 gamma2 : Carried) :
    Carried → List Entry → Carried
  | previous, [] => previous
  | previous, entry :: rest =>
      let leaf := entry.layout gamma1 gamma2 previous
      carriedAfter gamma1 gamma2
        (MemoryRecordFactorRows.outputCarried leaf) rest

/-- Independent product recursion over the entry gate values. -/
def productValue (assignment : Nat → Nat) (gamma1 gamma2 : Carried) :
    Pair → List Entry → Pair
  | initial, [] => initial
  | initial, entry :: rest =>
      productValue assignment gamma1 gamma2
        (KHorner.mulPair initial (entry.gateValue assignment gamma1 gamma2)) rest

structure Layout where
  gamma1 : Carried
  gamma2 : Carried
  initial : Carried
  final : Carried
  entries : List Entry

def rows (layout : Layout) : List Row :=
  rowsFrom layout.gamma1 layout.gamma2 layout.initial layout.entries ++
    KEquality.rows
      (carriedAfter layout.gamma1 layout.gamma2 layout.initial layout.entries)
      layout.final

def Entry.rowCount (entry : Entry) : Nat :=
  match entry.activation with
  | .always => 6
  | .padded _ => 10

theorem rowsFrom_length (gamma1 gamma2 previous : Carried) :
    ∀ entries,
      (rowsFrom gamma1 gamma2 previous entries).length =
        (entries.map Entry.rowCount).sum
  | [] => rfl
  | entry :: rest => by
      simp only [rowsFrom, List.length_append, List.map_cons, List.sum_cons]
      rw [MemoryRecordFactorRows.rows_length,
        rowsFrom_length gamma1 gamma2
          (MemoryRecordFactorRows.outputCarried
            (entry.layout gamma1 gamma2 previous)) rest]
      cases entry.activation <;> rfl

theorem rows_length (layout : Layout) :
    (rows layout).length =
      (layout.entries.map Entry.rowCount).sum + 2 := by
  rw [rows, List.length_append,
    rowsFrom_length, KEquality.rows_length]

private theorem head_rows_hold
    {gamma1 gamma2 previous : Carried} {entry : Entry} {rest : List Entry}
    {assignment : Nat → Nat}
    (satisfies :
      Satisfies (rowsFrom gamma1 gamma2 previous (entry :: rest)) assignment) :
    Satisfies
      (MemoryRecordFactorRows.rows
        (entry.layout gamma1 gamma2 previous)) assignment := by
  intro row member
  exact satisfies row (by simp [rowsFrom, member])

private theorem tail_rows_hold
    {gamma1 gamma2 previous : Carried} {entry : Entry} {rest : List Entry}
    {assignment : Nat → Nat}
    (satisfies :
      Satisfies (rowsFrom gamma1 gamma2 previous (entry :: rest)) assignment) :
    Satisfies
      (rowsFrom gamma1 gamma2
        (MemoryRecordFactorRows.outputCarried
          (entry.layout gamma1 gamma2 previous)) rest) assignment := by
  intro row member
  exact satisfies row (by simp [rowsFrom, member])

theorem rowsFrom_sound
    (assignment : Nat → Nat) (one : assignment 0 = 1)
    (gamma1 gamma2 : Carried) :
    ∀ (previous : Carried) (entries : List Entry),
      Satisfies (rowsFrom gamma1 gamma2 previous entries) assignment →
      carriedValue assignment
          (carriedAfter gamma1 gamma2 previous entries) =
        productValue assignment gamma1 gamma2
          (carriedValue assignment previous) entries
  | _, [], _ => rfl
  | previous, entry :: rest, satisfies => by
      let leaf := entry.layout gamma1 gamma2 previous
      have head := MemoryRecordFactorRows.update_sound one
        (head_rows_hold satisfies)
      have tail := rowsFrom_sound assignment one gamma1 gamma2
        (MemoryRecordFactorRows.outputCarried leaf) rest
        (tail_rows_hold satisfies)
      have head' :
          carriedValue assignment (MemoryRecordFactorRows.outputCarried leaf) =
            mulPair (carriedValue assignment previous)
              (semanticGate assignment leaf) := by
        simpa [leaf, Entry.layout] using head
      simp only [carriedAfter, productValue]
      rw [tail]
      have gateEqual :
          semanticGate assignment leaf =
            entry.gateValue assignment gamma1 gamma2 := by
        simpa [leaf] using
          entry.semanticGate_layout assignment gamma1 gamma2 previous
      exact congrArg
        (fun initial : Pair =>
          productValue assignment gamma1 gamma2 initial rest)
        (head'.trans (congrArg
          (mulPair (carriedValue assignment previous)) gateEqual))

private theorem chain_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment) :
    Satisfies
      (rowsFrom layout.gamma1 layout.gamma2 layout.initial layout.entries)
      assignment := by
  intro row member
  exact satisfies row (by simp [rows, member])

private theorem equality_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows layout) assignment) :
    Satisfies
      (KEquality.rows
        (carriedAfter layout.gamma1 layout.gamma2 layout.initial layout.entries)
        layout.final) assignment := by
  intro row member
  exact satisfies row (by simp [rows, member])

/-- A satisfying chain fixes the declared final product to the exact
left-to-right product of all gate values. -/
theorem final_sound
    {layout : Layout} {assignment : Nat → Nat}
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment) :
    carriedValue assignment layout.final =
      productValue assignment layout.gamma1 layout.gamma2
        (carriedValue assignment layout.initial) layout.entries := by
  have accumulated := rowsFrom_sound assignment one layout.gamma1 layout.gamma2
    layout.initial layout.entries (chain_rows_hold satisfies)
  have equal := KEquality.rows_sound assignment
    (carriedAfter layout.gamma1 layout.gamma2 layout.initial layout.entries)
    layout.final one (equality_rows_hold satisfies)
  have carriedEqual :
      carriedValue assignment
          (carriedAfter layout.gamma1 layout.gamma2 layout.initial layout.entries) =
        carriedValue assignment layout.final := by
    exact pair_ext equal.1 equal.2
  exact carriedEqual.symm.trans accumulated

end Nightstream.Implementation.Nebula.MemoryProductChainRows
