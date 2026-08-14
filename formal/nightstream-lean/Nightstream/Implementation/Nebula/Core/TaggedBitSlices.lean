/-!
Contract: exact slicing of an ordered concatenation of variable-width words.

Assurance tier: implementation model.

Owns prefix-width offsets and the theorem that slicing a flattened tagged
encoding at one schema index returns exactly that tag's encoded word.

Does not own any concrete schema, digit language, or parser.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.TaggedBitSlices

variable {Tag Digit : Type}

def flatten (encode : Tag → List Digit) (tags : List Tag) : List Digit :=
  tags.flatMap encode

def offsetAt (width : Tag → Nat) (tags : List Tag) (index : Nat) : Nat :=
  ((tags.take index).map width).sum

theorem drop_flatten_at
    (encode : Tag → List Digit) (width : Tag → Nat)
    (lengthExact : ∀ tag, (encode tag).length = width tag)
    (tags : List Tag) (index : Nat) (bounded : index ≤ tags.length) :
    (flatten encode tags).drop (offsetAt width tags index) =
      flatten encode (tags.drop index) := by
  induction index generalizing tags with
  | zero => simp [flatten, offsetAt]
  | succ index inductionHypothesis =>
      cases tags with
      | nil => simp at bounded
      | cons head tail =>
          have tailBound : index ≤ tail.length := by simpa using bounded
          change
            (encode head ++ flatten encode tail).drop
                (width head + offsetAt width tail index) =
              flatten encode (tail.drop index)
          rw [← lengthExact head, List.drop_length_add_append]
          exact inductionHypothesis tail tailBound

/-- The word at one valid schema index is recovered exactly. -/
theorem slice_flatten_at
    (encode : Tag → List Digit) (width : Tag → Nat)
    (lengthExact : ∀ tag, (encode tag).length = width tag)
    (tags : List Tag) (index : Nat) (bounded : index < tags.length) :
    ((flatten encode tags).drop (offsetAt width tags index)).take
        (width (tags.get ⟨index, bounded⟩)) =
      encode (tags.get ⟨index, bounded⟩) := by
  rw [drop_flatten_at encode width lengthExact tags index
    (Nat.le_of_lt bounded)]
  have dropExact := List.drop_eq_getElem_cons (l := tags) bounded
  rw [dropExact]
  change
    (encode (tags.get ⟨index, bounded⟩) ++
        flatten encode (tags.drop (index + 1))).take
          (width (tags.get ⟨index, bounded⟩)) =
      encode (tags.get ⟨index, bounded⟩)
  rw [← lengthExact (tags.get ⟨index, bounded⟩),
    List.take_append_length]

end Nightstream.Implementation.Nebula.TaggedBitSlices
