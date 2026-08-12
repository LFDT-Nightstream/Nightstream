import Nightstream.Implementation.NebulaV2.Core.U64HalvesRows
import Nightstream.Implementation.NebulaV2.Application.Wasm.StateCodec

/-!
Contract: lossless field-native encoding of the complete production WASM
state.

The public V2 codec remains the normative 2,293-bit image. Inside the
recursive relation, this encoding uses two little-endian 32-bit field limbs
for each 64-bit state word and one field for every smaller word. It therefore
uses 85 canonical Goldilocks fields instead of a bit-serial state carrier.

The encoding is injective before hashing. In particular, it never reduces a
64-bit word modulo the Goldilocks modulus. Canonical application states encode
only canonical Goldilocks representatives.

Does not own WASM transition rows, state-hash rows, absolute carrier columns,
external bytes, Rust refinement, or a verifier key.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionWasmStateFields

open Nightstream.Implementation.NebulaV2.U64HalvesRows
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open Nightstream.Protocol.NebulaV2.WasmStateEncoding

/-- One field for a word of at most 32 bits and two 32-bit limbs for a
64-bit word. -/
def nativeWidth (tag : FieldTag) : Nat :=
  if tag.bitWidth = 64 then 2 else 1

/-- Exact little-endian field block for one tagged state value. -/
def encodeTag (image : Image) (tag : FieldTag) : List Nat :=
  if tag.bitWidth = 64 then
    u64Halves (image.fieldValue tag)
  else
    [image.fieldValue tag]

theorem encodeTag_length (image : Image) (tag : FieldTag) :
    (encodeTag image tag).length = nativeWidth tag := by
  by_cases wide : tag.bitWidth = 64 <;>
    simp [encodeTag, nativeWidth, wide, u64Halves]

def encodeFor (tags : List FieldTag) (image : Image) : List Nat :=
  tags.flatMap (encodeTag image)

/-- Exact production state field image in the normative 55-tag order. -/
def encode (image : Image) : List Nat :=
  encodeFor schema image

theorem encodeFor_length (tags : List FieldTag) (image : Image) :
    (encodeFor tags image).length = (tags.map nativeWidth).sum := by
  induction tags with
  | nil => rfl
  | cons tag rest inductionHypothesis =>
      simp [encodeFor, encodeTag_length]

theorem native_width_sum_exact :
    (schema.map nativeWidth).sum = 85 := by
  decide

theorem encode_length (image : Image) : (encode image).length = 85 := by
  rw [encode, encodeFor_length, native_width_sum_exact]

/-- A tagged field block determines that tagged value. The 64-bit case uses
integer quotient and remainder, not a field reduction. -/
theorem fieldValue_eq_of_encodeTag_eq
    {left right : Image} (tag : FieldTag)
    (equal : encodeTag left tag = encodeTag right tag) :
    left.fieldValue tag = right.fieldValue tag := by
  by_cases wide : tag.bitWidth = 64
  · exact U64HalvesRows.u64Halves_injective (by
      simpa [encodeTag, wide] using equal)
  · simpa [encodeTag, wide] using equal

private theorem encodeFor_equal_at_member
    {left right : Image} {tags : List FieldTag}
    (equal : encodeFor tags left = encodeFor tags right)
    {tag : FieldTag} (member : tag ∈ tags) :
    left.fieldValue tag = right.fieldValue tag := by
  induction tags with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      rcases List.mem_cons.mp member with tagEqual | tailMember
      · subst tag
        have headEqual := congrArg (List.take (nativeWidth head)) equal
        have blockEqual : encodeTag left head = encodeTag right head := by
          simpa [encodeFor, encodeTag_length] using headEqual
        exact fieldValue_eq_of_encodeTag_eq head blockEqual
      · have tailEqual := congrArg (List.drop (nativeWidth head)) equal
        have exactTail : encodeFor tail left = encodeFor tail right := by
          simpa [encodeFor, encodeTag_length] using tailEqual
        exact inductionHypothesis exactTail tailMember

/-- The complete 85-field image recovers every one of the 55 typed state
values without a cryptographic assumption. -/
theorem encode_injective : Function.Injective encode := by
  intro left right equal
  apply Image.fieldValue_injective
  funext tag
  exact encodeFor_equal_at_member equal
    (WasmStateCodec.FieldTag.mem_schema tag)

private theorem u64Halves_member_lt_goldilocks
    {value limb : Nat} (bounded : value < 2 ^ 64)
    (member : limb ∈ u64Halves value) :
    limb < goldilocksP := by
  have alternatives :
      limb = value % 4294967296 ∨
        limb = value / 4294967296 := by
    simpa [u64Halves] using member
  rcases alternatives with rfl | rfl
  · exact Nat.lt_trans (Nat.mod_lt _ (by norm_num : 0 < 4294967296))
      (by norm_num [goldilocksP])
  · have highBound : value / 4294967296 < 4294967296 := by
      apply (Nat.div_lt_iff_lt_mul
        (by norm_num : 0 < 4294967296)).2
      norm_num at bounded ⊢
      exact bounded
    exact Nat.lt_trans highBound (by norm_num [goldilocksP])

private theorem pow_bitWidth_le_goldilocks_of_ne64
    (tag : FieldTag) (notWide : tag.bitWidth ≠ 64) :
    2 ^ tag.bitWidth ≤ goldilocksP := by
  cases tag <;>
    simp [FieldTag.bitWidth] at notWide ⊢ <;>
    norm_num [goldilocksP]

/-- A canonical semantic state maps to canonical Goldilocks representatives.
This is the range condition required before these values enter Poseidon2 or a
field-native recursive carrier. -/
theorem encode_fields_canonical
    {image : Image} (canonical : image.Canonical)
    {field : Nat} (member : field ∈ encode image) :
    field < goldilocksP := by
  simp only [encode, encodeFor, List.mem_flatMap] at member
  obtain ⟨tag, _, fieldMember⟩ := member
  by_cases wide : tag.bitWidth = 64
  · apply u64Halves_member_lt_goldilocks
      (by simpa [wide] using WasmStateCodec.fieldValue_lt_width canonical tag)
    simpa [encodeTag, wide] using fieldMember
  · have fieldExact : field = image.fieldValue tag := by
      simpa [encodeTag, wide] using fieldMember
    rw [fieldExact]
    exact Nat.lt_of_lt_of_le
      (WasmStateCodec.fieldValue_lt_width canonical tag)
      (pow_bitWidth_le_goldilocks_of_ne64 tag wide)

end Nightstream.Implementation.NebulaV2.ProductionWasmStateFields
