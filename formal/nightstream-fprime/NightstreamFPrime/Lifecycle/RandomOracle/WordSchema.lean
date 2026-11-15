import Mathlib.Data.Vector.Basic
import NightstreamFPrime.Lifecycle.RandomOracle.ByteWords

/-! Encodings of records as `u64` words, for a hash input that Rust makes one word at a time.

An encoding is `PrefixFree` when its words fix its value before any later words, so that a
concatenation of such encodings fixes each part. The parts of a format are a word (`word`), a
field element (`fieldWord`), a flag (`flag`), two parts in a row (`pair`), a list with its count
first (`counted`), a list whose length the format fixes (`sized`, `array`), and a record through
an injective view of its fields (`PrefixFree.comap`).

The words are natural numbers. `fieldWords` reduces each one to a field element, as Rust
`Fq::from_u64` does, and two lists of words below `q` with one reduction are equal
(`fieldWords_injective`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracle.WordSchema

open NightstreamFPrime.Spec

variable {α β : Type}

/-- The words of an encoding fix its value before any later words. -/
def PrefixFree (encode : α → List ℕ) : Prop :=
  ∀ ⦃left right : α⦄ ⦃rest rest' : List ℕ⦄,
    encode left ++ rest = encode right ++ rest' → left = right ∧ rest = rest'

theorem PrefixFree.injective {encode : α → List ℕ} (free : PrefixFree encode) :
    Function.Injective encode := fun left right same =>
  (free (rest := []) (rest' := []) (by rw [List.append_nil, List.append_nil, same])).1

/-- The encoding of a record is the encoding of an injective view of its fields. -/
theorem PrefixFree.comap {encode : α → List ℕ} (free : PrefixFree encode) {view : β → α}
    (injective : Function.Injective view) : PrefixFree fun value => encode (view value) :=
  fun _ _ _ _ same => ⟨injective (free same).1, (free same).2⟩

def word (value : ℕ) : List ℕ := [value]

theorem word_prefixFree : PrefixFree word := fun _ _ _ _ same => by
  simpa [word] using same

/-- A field element as its canonical value. -/
def fieldWord (value : F) : List ℕ := [value.val]

theorem fieldWord_prefixFree : PrefixFree fieldWord := fun _ _ _ _ same => by
  simp only [fieldWord, List.cons_append, List.nil_append, List.cons.injEq] at same
  exact ⟨Fin.ext same.1, same.2⟩

/-- A flag as 0 or 1. -/
def flag (value : Bool) : List ℕ := [value.toNat]

theorem flag_prefixFree : PrefixFree flag := fun left right _ _ same => by
  simp only [flag, List.cons_append, List.nil_append, List.cons.injEq] at same
  refine ⟨?_, same.2⟩
  cases left <;> cases right <;> first | rfl | exact absurd same.1 (by decide)

/-- Two encodings in a row. -/
def pair (first : α → List ℕ) (second : β → List ℕ) (value : α × β) : List ℕ :=
  first value.1 ++ second value.2

theorem PrefixFree.pair {first : α → List ℕ} {second : β → List ℕ} (firstFree : PrefixFree first)
    (secondFree : PrefixFree second) : PrefixFree (pair first second) := by
  intro left right rest rest' same
  simp only [WordSchema.pair, List.append_assoc] at same
  obtain ⟨firsts, tail⟩ := firstFree same
  obtain ⟨seconds, rests⟩ := secondFree tail
  exact ⟨Prod.ext firsts seconds, rests⟩

/-- The words of a list whose length the format fixes. -/
def sized (encode : α → List ℕ) (values : List α) : List ℕ := values.flatMap encode

theorem PrefixFree.sized {encode : α → List ℕ} (free : PrefixFree encode) :
    ∀ {left right : List α} {rest rest' : List ℕ}, left.length = right.length →
      sized encode left ++ rest = sized encode right ++ rest' → left = right ∧ rest = rest'
  | [], [], _, _, _, same => ⟨rfl, by simpa [WordSchema.sized] using same⟩
  | [], _ :: _, _, _, lengths, _ => by simp at lengths
  | _ :: _, [], _, _, lengths, _ => by simp at lengths
  | head :: left, head' :: right, _, _, lengths, same => by
      simp only [WordSchema.sized, List.flatMap_cons, List.append_assoc] at same
      obtain ⟨heads, tail⟩ := free same
      obtain ⟨tails, rests⟩ := PrefixFree.sized free (by simpa using lengths) tail
      exact ⟨by rw [heads, tails], rests⟩

/-- A list with its count first. -/
def counted (encode : α → List ℕ) (values : List α) : List ℕ :=
  values.length :: sized encode values

theorem PrefixFree.counted {encode : α → List ℕ} (free : PrefixFree encode) :
    PrefixFree (counted encode) := by
  intro left right rest rest' same
  simp only [WordSchema.counted, List.cons_append, List.cons.injEq] at same
  exact free.sized same.1 same.2

/-- An array of a fixed size. -/
def array {size : ℕ} (encode : α → List ℕ) (values : List.Vector α size) : List ℕ :=
  sized encode values.toList

theorem PrefixFree.array {size : ℕ} {encode : α → List ℕ} (free : PrefixFree encode) :
    PrefixFree (array (size := size) encode) := by
  intro left right rest rest' same
  obtain ⟨values, rests⟩ := free.sized (by rw [left.toList_length, right.toList_length]) same
  exact ⟨List.Vector.toList_injective values, rests⟩

/-- Each word as a field element (Rust `Fq::from_u64`). -/
def fieldWords (words : List ℕ) : List F := words.map Poseidon2.ofNat

/-- No word is at or above `q`, so `fieldWords` reduces none. -/
def Canonical (words : List ℕ) : Prop := ∀ value ∈ words, value < fieldModulus

instance (words : List ℕ) : Decidable (Canonical words) := by
  unfold Canonical
  infer_instance

/-- The field elements of canonical words have the words as their values. -/
theorem fieldWords_val {words : List ℕ} (canonical : Canonical words) :
    (fieldWords words).map Fin.val = words := by
  rw [fieldWords, List.map_map]
  conv_rhs => rw [← List.map_id words]
  exact List.map_congr_left fun value member => by
    simp [Poseidon2.ofNat, Nat.mod_eq_of_lt (canonical value member)]

/-- Two lists of canonical words with one list of field elements are equal. -/
theorem fieldWords_injective {left right : List ℕ} (leftCanonical : Canonical left)
    (rightCanonical : Canonical right) (same : fieldWords left = fieldWords right) :
    left = right := by
  rw [← fieldWords_val leftCanonical, ← fieldWords_val rightCanonical, same]

end NightstreamFPrime.Lifecycle.RandomOracle.WordSchema
