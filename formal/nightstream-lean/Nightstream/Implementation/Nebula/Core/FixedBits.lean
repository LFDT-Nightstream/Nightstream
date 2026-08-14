import Mathlib.Data.Nat.Digits.Lemmas
import Nightstream.Implementation.Nebula.Application.Wasm.StateCodec

/-!
Contract: exact fixed-width little-endian binary words and safe slicing.

Assurance tier: implementation model.

Owns fixed length, binary digits, integer decoding, safe contiguous slices,
the width bound, and canonical re-encoding of every accepted word.

Does not own any V2 schema, byte packing, or R1CS rows.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.FixedBits

/-- An exact-width little-endian binary word. -/
def Word (width : Nat) :=
  { digits : List Nat //
    digits.length = width ∧ ∀ digit ∈ digits, digit < 2 }

def decode {width : Nat} (word : Word width) : Nat :=
  Nat.ofDigits 2 word.val

theorem decode_lt {width : Nat} (word : Word width) :
    decode word < 2 ^ width := by
  simpa [word.property.1] using
    Nat.ofDigits_lt_base_pow_length (by decide : 1 < 2) word.property.2

/-- A contiguous safe slice. The caller must prove that the slice ends in the
authority-bearing source word. -/
def slice {total : Nat} (word : Word total) (offset width : Nat)
    (fits : offset + width ≤ total) : Word width :=
  ⟨(word.val.drop offset).take width, by
    constructor
    · rw [List.length_take, List.length_drop, word.property.1]
      have widthLe : width ≤ total - offset := by omega
      rw [Nat.min_eq_left widthLe]
    · intro digit member
      apply word.property.2 digit
      exact List.mem_of_mem_drop (List.mem_of_mem_take member)⟩

theorem slice_val {total : Nat} (word : Word total) (offset width : Nat)
    (fits : offset + width ≤ total) :
    (slice word offset width fits).val =
      (word.val.drop offset).take width := rfl

/-- Pointwise slice recovery in a total form. The explicit bound prevents a
default zero from hiding an out-of-range coordinate. -/
theorem slice_getD {total : Nat} (word : Word total) (offset width : Nat)
    (fits : offset + width ≤ total) (index : Nat) (indexBound : index < width) :
    (slice word offset width fits).val.getD index 0 =
      word.val.getD (offset + index) 0 := by
  have sliceBound : index < (slice word offset width fits).val.length := by
    rw [(slice word offset width fits).property.1]
    exact indexBound
  have sourceBound : offset + index < word.val.length := by
    rw [word.property.1]
    omega
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD]
  change ((word.val.drop offset).take width)[index]?.getD 0 =
    word.val[offset + index]?.getD 0
  rw [List.getElem?_take_of_lt indexBound, List.getElem?_drop]

/-- Every accepted fixed binary word is the unique canonical fixed-width
encoding of its decoded integer. -/
theorem encode_decode {width : Nat} (word : Word width) :
    WasmStateCodec.encodeWord width (decode word) = word.val := by
  apply Nat.injOn_ofDigits (b := 2) (by decide) width
  · exact ⟨WasmStateCodec.encodeWord_length _ _,
      fun digit member =>
        WasmStateCodec.encodeWord_binary _ _ digit member⟩
  · exact word.property
  · rw [WasmStateCodec.ofDigits_encodeWord_of_bound (decode_lt word)]
    rfl

def ofEncoded (width value : Nat) : Word width :=
  ⟨WasmStateCodec.encodeWord width value,
    WasmStateCodec.encodeWord_length width value,
    fun digit member =>
      WasmStateCodec.encodeWord_binary width value digit member⟩

theorem decode_ofEncoded_of_bound {width value : Nat}
    (bounded : value < 2 ^ width) :
    decode (ofEncoded width value) = value :=
  WasmStateCodec.ofDigits_encodeWord_of_bound bounded

end Nightstream.Implementation.Nebula.FixedBits
