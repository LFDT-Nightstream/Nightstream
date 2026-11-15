import Mathlib.Data.Nat.Digits.Defs
import NightstreamFPrime.Lifecycle.RandomOracle.Pieces

/-! Bytes in a Poseidon2 transcript (`SPEC.md` section 5.4).

`byteWords` is Rust `byte_words` (`crates/ns-compress/src/transcript.rs`): the byte count, then
one word for each 7 bytes, little-endian. Two byte strings with one list of words are equal
(`byteWords_injective`), and the count word fixes the length of the list, so the words of two
byte strings after two fronts of one length are alike pieces (`byteWords_alike`).

`sessionBytes` is the session of an Akita opening (Rust `session`): a label, then lanes 0 to 3 of
the sponge state, each as 8 bytes, little-endian. The session fixes these lanes
(`sessionBytes_injective`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.RandomOracle.Sponge

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle

abbrev Byte := Fin 256

/-- Up to 7 bytes as one word, little-endian. -/
def packWord (bytes : List Byte) : F := Poseidon2.ofNat (Nat.ofDigits 256 (bytes.map Fin.val))

/-- Bytes `7 · word` to `7 · word + 6`. -/
def byteChunk (bytes : List Byte) (word : ℕ) : List Byte := (bytes.drop (7 * word)).take 7

/-- The byte count, then one word for each 7 bytes, the last one short. -/
def byteWords (bytes : List Byte) : List F :=
  Poseidon2.ofNat bytes.length ::
    (List.range ((bytes.length + 6) / 7)).map fun word => packWord (byteChunk bytes word)

theorem byteWords_length (bytes : List Byte) :
    (byteWords bytes).length = 1 + (bytes.length + 6) / 7 := by
  simp [byteWords, Nat.add_comm]

theorem ofNat_injective {left right : ℕ} (leftSmall : left < fieldModulus)
    (rightSmall : right < fieldModulus) (same : Poseidon2.ofNat left = Poseidon2.ofNat right) :
    left = right := by
  have values := congrArg Fin.val same
  simpa [Poseidon2.ofNat, Nat.mod_eq_of_lt leftSmall, Nat.mod_eq_of_lt rightSmall] using values

private theorem digits_small (bytes : List Byte) (short : bytes.length ≤ 7) :
    Nat.ofDigits 256 (bytes.map Fin.val) < fieldModulus := by
  have bound := Nat.ofDigits_lt_base_pow_length (b := 256) (l := bytes.map Fin.val) (by decide)
    (by simp only [List.mem_map]; rintro _ ⟨byte, _, rfl⟩; exact byte.isLt)
  rw [List.length_map] at bound
  exact lt_of_lt_of_le bound (le_trans (Nat.pow_le_pow_right (by decide) short) (by decide))

theorem packWord_injective {left right : List Byte} (short : left.length ≤ 7)
    (lengthEqual : left.length = right.length) (same : packWord left = packWord right) :
    left = right := by
  have digits := ofNat_injective (digits_small left short)
    (digits_small right (lengthEqual ▸ short)) same
  have values := Nat.ofDigits_inj_of_len_eq (b := 256) (by decide)
    (by simp [lengthEqual])
    (by simp only [List.mem_map]; rintro _ ⟨byte, _, rfl⟩; exact byte.isLt)
    (by simp only [List.mem_map]; rintro _ ⟨byte, _, rfl⟩; exact byte.isLt) digits
  exact List.map_injective_iff.mpr Fin.val_injective values

/-- Two byte strings of one length with one list of words are equal. -/
theorem byteWords_injective_of_length {left right : List Byte}
    (lengthEqual : left.length = right.length) (same : byteWords left = byteWords right) :
    left = right := by
  have chunks : ∀ word ∈ List.range ((left.length + 6) / 7),
      packWord (byteChunk left word) = packWord (byteChunk right word) := by
    have tail := (List.cons.inj same).2
    rw [← lengthEqual] at tail
    exact List.map_inj_left.mp tail
  apply List.ext_getElem lengthEqual
  intro position leftBound rightBound
  have chunk := packWord_injective (List.length_take_le _ _)
    (by simp [byteChunk, lengthEqual])
    (chunks (position / 7) (List.mem_range.mpr (by omega)))
  have byte := congrArg (fun chunk : List Byte => chunk[position % 7]?) chunk
  have split : 7 * (position / 7) + position % 7 = position := Nat.div_add_mod position 7
  simp only [byteChunk, List.getElem?_take, Nat.mod_lt _ (show 0 < 7 by decide), ite_true,
    List.getElem?_drop, split, List.getElem?_eq_getElem leftBound,
    List.getElem?_eq_getElem rightBound] at byte
  exact Option.some.inj byte

/-- Two byte strings with one list of words are equal, when each has fewer than `q` bytes. -/
theorem byteWords_injective {left right : List Byte} (leftSmall : left.length < fieldModulus)
    (rightSmall : right.length < fieldModulus) (same : byteWords left = byteWords right) :
    left = right :=
  byteWords_injective_of_length (ofNat_injective leftSmall rightSmall (List.cons.inj same).1) same

/-- The words of two byte strings, each after a front of one length, are alike pieces: the count
word fixes the length. -/
theorem byteWords_alike {front front' : List F} {left right : List Byte}
    (fronts : front.length = front'.length) (leftSmall : left.length < fieldModulus)
    (rightSmall : right.length < fieldModulus) :
    Piece.Alike (.absorb (front ++ byteWords left)) (.absorb (front' ++ byteWords right)) :=
  Piece.alike_of_counted fronts fun counts => by
    simp [ofNat_injective leftSmall rightSmall counts]

/-- A word as 8 bytes, little-endian. -/
def le8 (word : F) : List Byte :=
  (List.range 8).map fun index => ⟨word.val / 256 ^ index % 256, Nat.mod_lt _ (by decide)⟩

theorem le8_injective {left right : F} (same : le8 left = le8 right) : left = right := by
  have small : ∀ word : F, word.val < 256 ^ 8 := fun word =>
    lt_of_lt_of_le word.isLt (by decide)
  have leftSmall := small left
  have rightSmall := small right
  simp only [le8, List.range, List.range.loop, List.map_cons, List.map_nil, List.cons.injEq,
    Fin.mk.injEq, and_true] at same
  norm_num at same leftSmall rightSmall
  exact Fin.ext (by omega)

private theorem flatMap_le8_injective : ∀ {left right : List F},
    left.flatMap le8 = right.flatMap le8 → left = right
  | [], [], _ => rfl
  | [], _ :: _, same => by simp [le8] at same
  | _ :: _, [], same => by simp [le8] at same
  | word :: left, word' :: right, same => by
      simp only [List.flatMap_cons] at same
      obtain ⟨head, tail⟩ := List.append_inj same (by simp [le8])
      rw [le8_injective head, flatMap_le8_injective tail]

/-- The session bytes of an Akita opening: `label`, then lanes 0 to 3 of the sponge state. -/
def sessionBytes (label : List Byte) (state : Transcript.State) : List Byte :=
  label ++ (state.take Poseidon2.digestLen).flatMap le8

/-- The session bytes fix lanes 0 to 3 of the sponge state. -/
theorem sessionBytes_injective {label : List Byte} {state state' : Transcript.State}
    (same : sessionBytes label state = sessionBytes label state') :
    state.take Poseidon2.digestLen = state'.take Poseidon2.digestLen :=
  flatMap_le8_injective (List.append_cancel_left same)

end NightstreamFPrime.Lifecycle.RandomOracle.Sponge
