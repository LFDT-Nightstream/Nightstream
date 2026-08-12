import Nightstream.Implementation.NebulaV2.FieldCodec
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: exact composition of one canonical-u64 R1CS block for every field
limb in a finite V2 schema.

Assurance tier: implementation model.

Owns the ordered row composition, the 64 public-bit links for each slot,
sound extraction for every listed slot, and the exact 133-row cost per slot.

Does not own a concrete claim or carry schema, absolute column allocation,
native container parsing, or the final generated relation.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.CanonicalFieldSchemaRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CanonicalU64
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Protocol.NebulaV2

variable {Slot : Type}

/-- The local canonical-u64 allocation and the 64 authority-bearing public
bit columns for one schema slot. -/
structure Layout (Slot : Type) where
  columnMap : Slot → List Nat
  rawColumns : Slot → List Nat
  rawColumnsLength : ∀ slot,
    (rawColumns slot).length = CanonicalFieldBits.bitCount
  mapsConstantOne : ∀ slot, Relabel.column (columnMap slot) 0 = 0

def linkPairs (layout : Layout Slot) (slot : Slot) : List (Nat × Nat) :=
  (List.range CanonicalFieldBits.bitCount).map fun index =>
    ((layout.rawColumns slot).getD index 0,
      Relabel.column (layout.columnMap slot) (bitCol index))

def slotRows (layout : Layout Slot) (slot : Slot) : List Row :=
  rows.map (Relabel.row (layout.columnMap slot)) ++
    EqualityPins.rows (linkPairs layout slot)

/-- Slots and rows occur in the same explicit schema order. -/
def schemaRows (slots : List Slot) (layout : Layout Slot) : List Row :=
  (slots.map fun slot => slotRows layout slot).flatten

abbrev RawWords (Slot : Type) := Slot → CanonicalFieldBits.Word

def rawDigits (layout : Layout Slot) (assignment : Nat → Nat)
    (slot : Slot) : List Nat :=
  (List.range CanonicalFieldBits.bitCount).map fun index =>
    assignment ((layout.rawColumns slot).getD index 0)

/-- Parser-to-assignment ownership boundary. It contains only the exact raw
bit placement. It does not contain canonicality or a decoded equality. -/
def Places (layout : Layout Slot) (assignment : Nat → Nat)
    (raw : RawWords Slot) : Prop :=
  ∀ slot, (raw slot).val = rawDigits layout assignment slot

private theorem rowsIncluded_append_left (left right : List Row) :
    rowsIncluded left (left ++ right) = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true (List.mem_append_left right member)

private theorem rowsIncluded_append_right (left right : List Row) :
    rowsIncluded right (left ++ right) = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true (List.mem_append_right left member)

def localCallSite (layout : Layout Slot) (slot : Slot) :
    FieldCodec.CallSite (slotRows layout slot) where
  columnMap := layout.columnMap slot
  rawColumns := layout.rawColumns slot
  rawColumnsLength := layout.rawColumnsLength slot
  mapsConstantOne := layout.mapsConstantOne slot
  canonicalRowsIncluded := by
    unfold slotRows
    exact rowsIncluded_append_left _ _
  linkRowsIncluded := by
    unfold slotRows
    exact rowsIncluded_append_right _ _

private theorem slot_rows_hold
    {slots : List Slot} {layout : Layout Slot}
    {assignment : Nat → Nat}
    (satisfies : Satisfies (schemaRows slots layout) assignment)
    {slot : Slot} (member : slot ∈ slots) :
    Satisfies (slotRows layout slot) assignment := by
  rw [schemaRows] at satisfies
  exact (satisfies_flatten_iff _ _).mp satisfies _
    (List.mem_map.mpr ⟨slot, member, rfl⟩)

/-- Construct the typed raw word of one listed slot from satisfying rows. -/
def rawWordOfRows
    {slots : List Slot} {layout : Layout Slot}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (schemaRows slots layout) assignment)
    (slot : Slot) (member : slot ∈ slots) : CanonicalFieldBits.Word :=
  (localCallSite layout slot).wordOfRows goldilocks_euclidPrime canonical one
    (slot_rows_hold satisfies member)

@[simp] theorem rawWordOfRows_val
    {slots : List Slot} {layout : Layout Slot}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (schemaRows slots layout) assignment)
    (slot : Slot) (member : slot ∈ slots) :
    (rawWordOfRows canonical one satisfies slot member).val =
      rawDigits layout assignment slot := rfl

/-- The row-derived raw word is accepted and decodes to the schema's native
value wire. -/
theorem rawWordOfRows_sound
    {slots : List Slot} {layout : Layout Slot}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (schemaRows slots layout) assignment)
    (slot : Slot) (member : slot ∈ slots) :
    ∃ value,
      FieldCodec.nativeDecode
          (rawWordOfRows canonical one satisfies slot member) = some value ∧
        value.val = assignment
          (Relabel.column (layout.columnMap slot) varCol) := by
  apply FieldCodec.CallSite.sound goldilocks_euclidPrime
    (localCallSite layout slot) canonical one
    (slot_rows_hold satisfies member)
  rfl

/-- Every listed raw word is forced to be canonical. Its native decoded value
is exactly the value in the corresponding canonical-u64 circuit wire. -/
theorem slot_sound
    {slots : List Slot} {layout : Layout Slot}
    {assignment : Nat → Nat} {raw : RawWords Slot}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (schemaRows slots layout) assignment)
    (placed : Places layout assignment raw)
    {slot : Slot} (member : slot ∈ slots) :
    ∃ value,
      FieldCodec.nativeDecode (raw slot) = some value ∧
        value.val = assignment
          (Relabel.column (layout.columnMap slot) varCol) := by
  apply FieldCodec.CallSite.sound goldilocks_euclidPrime
    (localCallSite layout slot) canonical one
    (slot_rows_hold satisfies member)
  exact placed slot

/-- Schema-wide fail-closed native decoding. No listed field limb can use a
noncanonical 64-bit representative. -/
theorem all_slots_sound
    {slots : List Slot} {layout : Layout Slot}
    {assignment : Nat → Nat} {raw : RawWords Slot}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (schemaRows slots layout) assignment)
    (placed : Places layout assignment raw) :
    ∀ slot ∈ slots, ∃ value,
      FieldCodec.nativeDecode (raw slot) = some value ∧
        value.val = assignment
          (Relabel.column (layout.columnMap slot) varCol) := by
  intro slot member
  exact slot_sound canonical one satisfies placed member

theorem linkPairs_length (layout : Layout Slot) (slot : Slot) :
    (linkPairs layout slot).length = CanonicalFieldBits.bitCount := by
  simp [linkPairs]

theorem slotRows_length (layout : Layout Slot) (slot : Slot) :
    (slotRows layout slot).length = 133 := by
  have canonicalRowsLength : rows.length = 69 := by decide
  simp [slotRows, EqualityPins.rows, linkPairs, canonicalRowsLength,
    CanonicalFieldBits.bitCount]

theorem schemaRows_length (slots : List Slot) (layout : Layout Slot) :
    (schemaRows slots layout).length = slots.length * 133 := by
  induction slots with
  | nil => simp [schemaRows]
  | cons head tail inductionHypothesis =>
      change (slotRows layout head ++ schemaRows tail layout).length =
        (head :: tail).length * 133
      rw [List.length_append, slotRows_length, inductionHypothesis]
      simp [Nat.add_mul, Nat.add_comm]

end Nightstream.Implementation.NebulaV2.CanonicalFieldSchemaRows
