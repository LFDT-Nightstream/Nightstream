import Mathlib.Data.BitVec
import Mathlib.Data.List.OfFn
import Batteries.Data.BitVec.Lemmas
import Nightstream.Implementation.Nebula.Application.Wasm.PublicStatementCodec

/-!
Contract: exact byte packing for the V2 WASM public-statement section.

Assurance tier: implementation model.

Owns little-endian bit-to-byte packing, the exact 984-byte size, the four
zero high bits in the final byte, and injectivity on accepted public images.

Does not own the outer proof container, a raw byte parser, generated public
columns, Rust conformance, or proof verification.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.WasmStatementBytes

open Nightstream.Implementation.Nebula.WasmPublicStatementCodec
open Nightstream.Protocol.Nebula.WasmPublicStatementEncoding
open Nightstream.Protocol.Nebula.WasmStatement

def statementBitCount : Nat := 7868
def byteBitCount : Nat := 8
def statementByteCount : Nat := 984
def paddedBitCount : Nat := statementByteCount * byteBitCount
def highPaddingBitCount : Nat := paddedBitCount - statementBitCount

theorem exact_sizes :
    paddedBitCount = 7872 ∧ highPaddingBitCount = 4 := by
  decide

def bitBool (bit : Nat) : Bool := bit = 1

theorem bitBool_injective_below_two
    {left right : Nat} (leftBound : left < 2) (rightBound : right < 2)
    (equal : bitBool left = bitBool right) : left = right := by
  interval_cases left <;> interval_cases right <;> simp_all [bitBool]

/-- The 7,868 logical bits, with list index zero as the least-significant
bit of the complete bit vector. -/
def statementWord (image : PublicImage) : BitVec statementBitCount :=
  BitVec.ofFnLE fun index =>
    bitBool ((WasmPublicStatementCodec.encode image).get
      ⟨index.val, by
        rw [WasmPublicStatementCodec.encode_length]
        exact index.isLt⟩)

/-- Four zero most-significant bits complete the final byte. -/
def paddedWord (image : PublicImage) : BitVec paddedBitCount :=
  (statementWord image).setWidth paddedBitCount

/-- Split a complete 7,872-bit word into 984 consecutive little-endian
bytes. -/
def splitBytes (word : BitVec paddedBitCount) :
    Fin statementByteCount → BitVec byteBitCount :=
  fun index => word.extractLsb' (byteBitCount * index.val) byteBitCount

def byteWords (image : PublicImage) :
    Fin statementByteCount → BitVec byteBitCount :=
  splitBytes (paddedWord image)

/-- Wire byte values in increasing byte offset. -/
def encode (image : PublicImage) : List Nat :=
  List.ofFn fun index => (byteWords image index).toNat

theorem encode_length (image : PublicImage) :
    (encode image).length = statementByteCount := by
  simp [encode]

theorem encode_get
    (image : PublicImage) (index : Fin statementByteCount) :
    (encode image).get
        ⟨index.val, by rw [encode_length]; exact index.isLt⟩ =
      (byteWords image index).toNat := by
  simp only [encode, List.get_ofFn]
  rfl

theorem byte_value_lt_256
    (image : PublicImage) (index : Fin statementByteCount) :
    (byteWords image index).toNat < 256 := by
  simpa [byteBitCount] using
    BitVec.toNat_lt_twoPow_of_le (x := byteWords image index) (Nat.le_refl _)

theorem padded_high_bit_zero
    (image : PublicImage) (index : Nat)
    (logicalEnd : statementBitCount ≤ index) :
    (paddedWord image).getLsbD index = false := by
  simp only [paddedWord, BitVec.getLsbD_setWidth]
  by_cases inPadded : index < paddedBitCount
  · simp only [inPadded, decide_true, Bool.true_and]
    exact BitVec.getLsbD_of_ge _ _ logicalEnd
  · simp [inPadded]

theorem final_byte_lt_16 (image : PublicImage) :
    (byteWords image ⟨statementByteCount - 1, by decide⟩).toNat < 16 := by
  rw [show 16 = 2 ^ 4 by decide]
  apply (BitVec.toNat_lt_iff_getLsbD_eq_false 4 (by decide)).2
  intro offset
  simp only [byteWords, splitBytes, BitVec.getLsbD_extractLsb']
  by_cases insideByte : 4 + offset < byteBitCount
  · simp only [insideByte, decide_true, Bool.true_and]
    apply padded_high_bit_zero
    simp only [statementBitCount, statementByteCount, byteBitCount]
    omega
  · simp [insideByte]

/-- Recombine 984 byte words. This is also the semantic operation performed
by a raw parser after it rejects non-byte integers. -/
def joinBytes
    (bytes : Fin statementByteCount → BitVec byteBitCount) :
    BitVec paddedBitCount :=
  BitVec.ofFnLE fun index =>
    (bytes ⟨index.val / byteBitCount, by
      have bound := index.isLt
      simp only [paddedBitCount] at bound
      exact Nat.div_lt_of_lt_mul (by simpa [Nat.mul_comm] using bound)⟩).getLsb
      ⟨index.val % byteBitCount, Nat.mod_lt _ (by decide)⟩

theorem join_split (word : BitVec paddedBitCount) :
    joinBytes (splitBytes word) = word := by
  apply BitVec.eq_of_getLsbD_eq
  intro index indexBound
  simp only [joinBytes, BitVec.getLsbD_ofFnLE, splitBytes]
  rw [dif_pos indexBound]
  change
    (word.extractLsb' (byteBitCount * (index / byteBitCount))
      byteBitCount).getLsbD (index % byteBitCount) = word.getLsbD index
  rw [BitVec.getLsbD_extractLsb']
  have remainderBound : index % byteBitCount < byteBitCount :=
    Nat.mod_lt index (by decide)
  simp only [remainderBound, decide_true, Bool.true_and]
  have division := Nat.mod_add_div index byteBitCount
  have recombine :
      byteBitCount * (index / byteBitCount) + index % byteBitCount = index := by
    omega
  rw [recombine]

theorem splitBytes_injective : Function.Injective splitBytes := by
  intro left right equal
  rw [← join_split left, ← join_split right, equal]

theorem statementWord_injective_of_decodesFor
    {Program : Type}
    {left right : PublicImage}
    {leftStatement rightStatement : ProductionStatement Program}
    {expectedProfile : Nightstream.Protocol.Nebula.Profile.Identity}
    (leftDecoded : left.DecodesFor expectedProfile leftStatement)
    (rightDecoded : right.DecodesFor expectedProfile rightStatement)
    (equal : statementWord left = statementWord right) :
    left = right := by
  apply WasmPublicStatementCodec.encode_injective_of_decodesFor
    leftDecoded rightDecoded
  apply List.ext_get
  · rw [WasmPublicStatementCodec.encode_length,
      WasmPublicStatementCodec.encode_length]
  · intro index leftBound rightBound
    have wordBitEqual := congrArg (fun word => word.getLsbD index) equal
    simp only [statementWord, BitVec.getLsbD_ofFnLE] at wordBitEqual
    have indexBound : index < statementBitCount := by
      rw [WasmPublicStatementCodec.encode_length] at leftBound
      simpa [statementBitCount] using leftBound
    simp only [indexBound, dite_true] at wordBitEqual
    apply bitBool_injective_below_two
    · exact WasmPublicStatementCodec.encode_binary left _
        (List.get_mem _ ⟨index, leftBound⟩)
    · exact WasmPublicStatementCodec.encode_binary right _
        (List.get_mem _ ⟨index, rightBound⟩)
    · simpa using wordBitEqual

theorem statementWord_injective_of_decodes
    {Program : Type}
    {left right : PublicImage}
    {leftStatement rightStatement : ProductionStatement Program}
    (leftDecoded : left.Decodes leftStatement)
    (rightDecoded : right.Decodes rightStatement)
    (equal : statementWord left = statementWord right) :
    left = right :=
  statementWord_injective_of_decodesFor leftDecoded.toDecodesFor
    rightDecoded.toDecodesFor equal

/-- No two accepted public images have the same 984-byte statement section.
This does not give authority to any digest inside the section. -/
theorem encode_injective_of_decodesFor
    {Program : Type}
    {left right : PublicImage}
    {leftStatement rightStatement : ProductionStatement Program}
    {expectedProfile : Nightstream.Protocol.Nebula.Profile.Identity}
    (leftDecoded : left.DecodesFor expectedProfile leftStatement)
    (rightDecoded : right.DecodesFor expectedProfile rightStatement)
    (equal : encode left = encode right) :
    left = right := by
  have byteValueFunctions :
      (fun index => (byteWords left index).toNat) =
        fun index => (byteWords right index).toNat :=
    List.ofFn_injective equal
  have byteFunctions : byteWords left = byteWords right := by
    funext index
    apply BitVec.eq_of_toNat_eq
    exact congrFun byteValueFunctions index
  have paddedEqual : paddedWord left = paddedWord right :=
    splitBytes_injective byteFunctions
  have logicalEqual := congrArg (BitVec.setWidth statementBitCount) paddedEqual
  have statementEqual : statementWord left = statementWord right := by
    simpa [paddedWord, Nat.le_of_lt (by decide : statementBitCount < paddedBitCount)]
      using logicalEqual
  exact statementWord_injective_of_decodesFor leftDecoded rightDecoded
    statementEqual

theorem encode_injective_of_decodes
    {Program : Type}
    {left right : PublicImage}
    {leftStatement rightStatement : ProductionStatement Program}
    (leftDecoded : left.Decodes leftStatement)
    (rightDecoded : right.Decodes rightStatement)
    (equal : encode left = encode right) :
    left = right :=
  encode_injective_of_decodesFor leftDecoded.toDecodesFor
    rightDecoded.toDecodesFor equal

end Nightstream.Implementation.Nebula.WasmStatementBytes
