import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: universal semantics of Rust `enforce_eq(var, constant)` row lists.

Generated artifacts may list column/value pairs and prove only exact row
inclusion. Satisfaction then derives every pinned value; the certificate does
not carry those equalities as assumptions.
-/

namespace Nightstream.Implementation.R1CS.ConstantPins

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

def pinRow (pin : Nat × Nat) : Row :=
  if pin.2 = 0 then builderLinearRow pin.1 []
  else builderLinearRow pin.1 [(0, pin.2)]

def rows (pins : List (Nat × Nat)) : List Row :=
  pins.map pinRow

def ValuesCanonical (pins : List (Nat × Nat)) : Prop :=
  ∀ pin ∈ pins, pin.2 < goldilocksP

def lookup : List (Nat × Nat) → Nat → Nat
  | [], _ => 0
  | pin :: rest, column =>
      if pin.1 = column then pin.2 else lookup rest column

def Covers (columns : List Nat) (pins : List (Nat × Nat)) : Prop :=
  ∀ column ∈ columns, ∃ pin ∈ pins, pin.1 = column

def keys (pins : List (Nat × Nat)) : List Nat :=
  pins.map Prod.fst

def KeysCover (columns sourceKeys : List Nat) : Prop :=
  ∀ column ∈ columns, column ∈ sourceKeys

instance (pins : List (Nat × Nat)) : Decidable (ValuesCanonical pins) := by
  unfold ValuesCanonical
  infer_instance

instance (columns : List Nat) (pins : List (Nat × Nat)) :
    Decidable (Covers columns pins) := by
  unfold Covers
  infer_instance

instance (columns sourceKeys : List Nat) :
    Decidable (KeysCover columns sourceKeys) := by
  unfold KeysCover
  infer_instance

theorem covers_iff_keys {columns : List Nat} {pins : List (Nat × Nat)} :
    Covers columns pins ↔ KeysCover columns (keys pins) := by
  constructor
  · intro covers column member
    rcases covers column member with ⟨pin, pinMember, key⟩
    exact List.mem_map.mpr ⟨pin, pinMember, key⟩
  · intro covers column member
    rcases List.mem_map.mp (covers column member) with
      ⟨pin, pinMember, key⟩
    exact ⟨pin, pinMember, key⟩

theorem lookup_pair_mem {pins : List (Nat × Nat)} {column : Nat}
    (present : ∃ pin ∈ pins, pin.1 = column) :
    (column, lookup pins column) ∈ pins := by
  induction pins with
  | nil => simp at present
  | cons pin rest inductionHypothesis =>
      by_cases matchesKey : pin.1 = column
      · rcases pin with ⟨key, value⟩
        simp only at matchesKey
        subst key
        simp [lookup]
      · right
        rw [show lookup (pin :: rest) column = lookup rest column by
          simp [lookup, matchesKey]]
        apply inductionHypothesis
        rcases present with ⟨candidate, member, key⟩
        simp only [List.mem_cons] at member
        rcases member with head | tail
        · subst candidate
          exact False.elim (matchesKey key)
        · exact ⟨candidate, tail, key⟩

theorem map_assignment_eq_lookup
    {pins : List (Nat × Nat)} {columns : List Nat}
    {assignment : Nat → Nat}
    (facts : ∀ pin ∈ pins, assignment pin.1 = pin.2)
    (covers : Covers columns pins) :
    columns.map assignment = columns.map (lookup pins) := by
  induction columns with
  | nil => rfl
  | cons column rest inductionHypothesis =>
      simp only [List.map_cons, List.cons.injEq]
      constructor
      · exact facts (column, lookup pins column)
          (lookup_pair_mem (covers column (by simp)))
      · apply inductionHypothesis
        intro candidate member
        exact covers candidate (by simp [member])

private theorem pinRow_sound
    {assignment : Nat → Nat} (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1) {pin : Nat × Nat}
    (valueCanonical : pin.2 < goldilocksP)
    (holds : RowHolds assignment (pinRow pin)) :
    assignment pin.1 = pin.2 := by
  rcases pin with ⟨column, value⟩
  by_cases zero : value = 0
  · subst value
    have defined := builderLinearRow_sound canonical one column []
      (by simp [CanonicalTerms]) (by simpa [pinRow] using holds)
    simpa [lcEval] using defined
  · have positive : 0 < value := Nat.pos_of_ne_zero zero
    have defined := builderLinearRow_sound canonical one column [(0, value)]
      (by simpa [CanonicalTerms] using And.intro positive valueCanonical)
      (by simpa [pinRow, zero] using holds)
    simpa [lcEval, one, Nat.mod_eq_of_lt valueCanonical] using defined

private theorem pinRow_complete
    {assignment : Nat → Nat}
    (one : assignment 0 = 1) {pin : Nat × Nat}
    (valueCanonical : pin.2 < goldilocksP)
    (equal : assignment pin.1 = pin.2) :
    RowHolds assignment (pinRow pin) := by
  rcases pin with ⟨column, value⟩
  by_cases zero : value = 0
  · subst value
    simp only [pinRow, ↓reduceIte]
    apply builderLinearRow_complete one column [] (by simp [CanonicalTerms])
    simpa [lcEval] using equal
  · simp only [pinRow, zero, ↓reduceIte]
    apply builderLinearRow_complete one column [(0, value)]
      (by simp [CanonicalTerms, Nat.pos_of_ne_zero zero, valueCanonical])
    simpa [lcEval, one, Nat.mod_eq_of_lt valueCanonical] using equal

theorem sound
    {pins : List (Nat × Nat)} {programRows : List Row}
    {assignment : Nat → Nat}
    (valuesCanonical : ValuesCanonical pins)
    (included : rowsIncluded (rows pins) programRows = true)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment) :
    ∀ pin ∈ pins, assignment pin.1 = pin.2 := by
  intro pin member
  apply pinRow_sound canonical one (valuesCanonical pin member)
  apply satisfies
  apply rowsIncluded_sound included
  exact List.mem_map.mpr ⟨pin, member, rfl⟩

/-- Pinned-value semantics directly satisfy every generated constant row.
This is the reverse compiler rule used by exact transcript certificates. -/
theorem complete
    {pins : List (Nat × Nat)} {assignment : Nat → Nat}
    (valuesCanonical : ValuesCanonical pins)
    (one : assignment 0 = 1)
    (facts : ∀ pin ∈ pins, assignment pin.1 = pin.2) :
    Satisfies (rows pins) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨pin, pinMember, rfl⟩
  exact pinRow_complete one
    (valuesCanonical pin pinMember) (facts pin pinMember)

end Nightstream.Implementation.R1CS.ConstantPins
