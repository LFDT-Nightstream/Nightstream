import Mathlib.Data.List.GetD
import NightstreamFPrime.Layout.LowNormSlot

/-!
Owns the proof-oriented reference encoding of a retained-source assignment.
Slots stay in canonical compiler order. Their widths determine the exact
logical width, and their encodings determine the exact low-norm coordinates.

Production export must use proved compact blocks equal to this reference; it
must not materialize an artifact-sized slot or coordinate list.
-/

namespace NightstreamFPrime.Layout.LowNormAssignment

open NightstreamFPrime.Spec

/-- One retained source value and its sole production encoding. -/
structure Slot (sourceWidth : Nat) where
  source : Fin sourceWidth
  kind : LowNormSlot.Kind
deriving Repr, DecidableEq

namespace Slot

def width {sourceWidth : Nat} (slot : Slot sourceWidth) : Nat :=
  slot.kind.width

def encode {sourceWidth : Nat} (source : Fin sourceWidth → F)
    (slot : Slot sourceWidth) : List F :=
  LowNormSlot.encode slot.kind (source slot.source)

def Valid {sourceWidth : Nat} (source : Fin sourceWidth → F)
    (slot : Slot sourceWidth) : Prop :=
  LowNormSlot.Valid slot.kind (source slot.source)

@[simp] theorem encode_length {sourceWidth : Nat}
    (source : Fin sourceWidth → F) (slot : Slot sourceWidth) :
    (slot.encode source).length = slot.width := by
  exact LowNormSlot.encode_length slot.kind (source slot.source)

theorem encode_norm {sourceWidth : Nat}
    (source : Fin sourceWidth → F) (slot : Slot sourceWidth)
    (valid : slot.Valid source) :
    normBounded 2 (slot.encode source) :=
  LowNormSlot.encode_norm slot.kind (source slot.source) valid

end Slot

/-- Exact logical assignment width of a canonical slot sequence. -/
def logicalWidth {sourceWidth : Nat} (slots : List (Slot sourceWidth)) : Nat :=
  (slots.map Slot.width).sum

/-- Reference coordinate sequence in exact slot and digit order. -/
def coordinates {sourceWidth : Nat} (slots : List (Slot sourceWidth))
    (source : Fin sourceWidth → F) : List F :=
  slots.flatMap (Slot.encode source)

/-- Direct coordinate lookup for executable production paths. It traverses
only the slots before the requested coordinate and does not build the full
flattened coordinate list. -/
def coordinateAt {sourceWidth : Nat} (source : Fin sourceWidth → F) :
    List (Slot sourceWidth) → Nat → F
  | [], _ => 0
  | slot :: rest, index =>
      if index < slot.width then
        (slot.encode source).getD index 0
      else
        coordinateAt source rest (index - slot.width)

/-- The direct executable lookup is exactly the proof-oriented flattened
coordinate sequence. -/
theorem coordinateAt_eq_getD {sourceWidth : Nat}
    (slots : List (Slot sourceWidth)) (source : Fin sourceWidth → F)
    (index : Nat) :
    coordinateAt source slots index = (coordinates slots source).getD index 0 := by
  induction slots generalizing index with
  | nil => rfl
  | cons slot rest inductionHypothesis =>
      unfold coordinateAt
      change
        (if inside : index < slot.width then
            (slot.encode source).getD index 0
          else coordinateAt source rest (index - slot.width)) =
          ((slot.encode source) ++ coordinates rest source).getD index 0
      split
      next inside =>
        rw [List.getD_append]
        simpa only [Slot.encode_length] using inside
      next outside =>
        rw [List.getD_append_right]
        · simpa only [Slot.encode_length] using
            inductionHypothesis (index - slot.width)
        · simpa only [Slot.encode_length] using Nat.le_of_not_gt outside

/-- Logical width before one indexed slot. -/
def prefixWidth {sourceWidth : Nat} (slots : List (Slot sourceWidth))
    (slot : Fin slots.length) : Nat :=
  logicalWidth (slots.take slot.val)

/-- The canonical coordinate of one slot lies inside the complete private
coordinate stream. -/
theorem prefixWidth_add_lt {sourceWidth : Nat}
    (slots : List (Slot sourceWidth)) (slot : Fin slots.length)
    (coordinate : Fin (slots.get slot).width) :
    prefixWidth slots slot + coordinate.val < logicalWidth slots := by
  induction slots with
  | nil => exact Fin.elim0 slot
  | cons head tail inductionHypothesis =>
      rcases slot with ⟨_ | index, bound⟩
      · simp [prefixWidth, logicalWidth]
        omega
      · let tailSlot : Fin tail.length := ⟨index, by simpa using bound⟩
        have inside := inductionHypothesis tailSlot coordinate
        simp [prefixWidth, logicalWidth, tailSlot] at inside ⊢
        omega

/-- Canonical private-coordinate index of one coordinate inside one slot. -/
def coordinateIndex {sourceWidth : Nat} (slots : List (Slot sourceWidth))
    (slot : Fin slots.length) (coordinate : Fin (slots.get slot).width) :
    Fin (logicalWidth slots) :=
  ⟨prefixWidth slots slot + coordinate.val,
    prefixWidth_add_lt slots slot coordinate⟩

private theorem coordinates_getD_prefix {sourceWidth : Nat}
    (slots : List (Slot sourceWidth)) (source : Fin sourceWidth → F)
    (slot : Fin slots.length) (coordinate : Fin (slots.get slot).width) :
    (coordinates slots source).getD
        (prefixWidth slots slot + coordinate.val) 0 =
      ((slots.get slot).encode source).getD coordinate.val 0 := by
  induction slots with
  | nil => exact Fin.elim0 slot
  | cons head tail inductionHypothesis =>
      rcases slot with ⟨_ | index, bound⟩
      · have inside : coordinate.val < (head.encode source).length := by
          simpa only [Slot.encode_length] using coordinate.isLt
        simpa [coordinates, prefixWidth, logicalWidth] using
          List.getD_append (head.encode source) (coordinates tail source)
            0 coordinate.val inside
      · let tailSlot : Fin tail.length := ⟨index, by simpa using bound⟩
        have subtract :
            head.width + prefixWidth tail tailSlot + coordinate.val -
                head.width =
              prefixWidth tail tailSlot + coordinate.val := by
          omega
        change
          ((head.encode source) ++ coordinates tail source).getD
              (head.width + prefixWidth tail tailSlot + coordinate.val) 0 =
            ((tail.get tailSlot).encode source).getD coordinate.val 0
        rw [List.getD_append_right]
        · simpa only [Slot.encode_length, subtract] using
            inductionHypothesis tailSlot coordinate
        · simp only [Slot.encode_length]
          omega

/-- Direct lookup at a canonical slot coordinate returns exactly that slot's
encoded coordinate. -/
theorem coordinateAt_coordinateIndex {sourceWidth : Nat}
    (slots : List (Slot sourceWidth)) (source : Fin sourceWidth → F)
    (slot : Fin slots.length) (coordinate : Fin (slots.get slot).width) :
    coordinateAt source slots (coordinateIndex slots slot coordinate).val =
      ((slots.get slot).encode source).getD coordinate.val 0 := by
  rw [coordinateAt_eq_getD]
  simpa only [coordinateIndex] using
    coordinates_getD_prefix slots source slot coordinate

@[simp] theorem logicalWidth_append {sourceWidth : Nat}
    (first second : List (Slot sourceWidth)) :
    logicalWidth (first ++ second) =
      logicalWidth first + logicalWidth second := by
  simp [logicalWidth, List.sum_append]

@[simp] theorem coordinates_append {sourceWidth : Nat}
    (first second : List (Slot sourceWidth))
    (source : Fin sourceWidth → F) :
    coordinates (first ++ second) source =
      coordinates first source ++ coordinates second source := by
  simp [coordinates]

@[simp] theorem coordinates_length {sourceWidth : Nat}
    (slots : List (Slot sourceWidth)) (source : Fin sourceWidth → F) :
    (coordinates slots source).length = logicalWidth slots := by
  simp [coordinates, logicalWidth, Slot.encode_length]

/-- A valid source for every retained slot yields the exact fresh-opening
bound over the complete coordinate stream. -/
theorem coordinates_norm {sourceWidth : Nat}
    (slots : List (Slot sourceWidth)) (source : Fin sourceWidth → F)
    (valid : ∀ slot ∈ slots, slot.Valid source) :
    normBounded 2 (coordinates slots source) := by
  intro coordinate member
  rw [coordinates, List.mem_flatMap] at member
  rcases member with ⟨slot, slotMember, coordinateMember⟩
  exact slot.encode_norm source (valid slot slotMember) coordinate
    coordinateMember

end NightstreamFPrime.Layout.LowNormAssignment
