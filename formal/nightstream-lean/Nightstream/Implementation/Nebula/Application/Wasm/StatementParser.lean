import Nightstream.Implementation.Nebula.Application.Wasm.StatementBytes

/-!
Contract: fail-closed raw parser for the V2 WASM public-statement section.

Assurance tier: implementation model.

Owns exact length rejection, rejection of integers outside one byte,
rejection of nonzero high padding bits, and the byte-to-logical-bit decode.

Does not own typed field parsing, the outer proof container, Rust
conformance, generated public columns, or proof verification.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.WasmStatementParser

open Nightstream.Implementation.Nebula.WasmStatementBytes
open Nightstream.Protocol.Nebula.WasmPublicStatementEncoding

def bytesInRange (bytes : List Nat) : Bool :=
  bytes.all fun byte => decide (byte < 256)

def words
    (bytes : List Nat) (lengthExact : bytes.length = statementByteCount) :
    Fin statementByteCount → BitVec byteBitCount :=
  fun index => BitVec.ofNat byteBitCount
    (bytes.get ⟨index.val, by rw [lengthExact]; exact index.isLt⟩)

/-- Raw parser. It accepts no trailing bytes and no modulo-byte aliases. -/
def parse (bytes : List Nat) : Option (BitVec statementBitCount) :=
  if lengthExact : bytes.length = statementByteCount then
    if _range : bytesInRange bytes = true then
      if _padding :
          (words bytes lengthExact
            ⟨statementByteCount - 1, by decide⟩).toNat < 16 then
        some ((joinBytes (words bytes lengthExact)).setWidth statementBitCount)
      else none
    else none
  else none

theorem rejects_wrong_length
    (bytes : List Nat) (wrong : bytes.length ≠ statementByteCount) :
    parse bytes = none := by
  simp [parse, wrong]

theorem rejects_non_byte
    (bytes : List Nat)
    (lengthExact : bytes.length = statementByteCount)
    (outOfRange : bytesInRange bytes ≠ true) :
    parse bytes = none := by
  simp [parse, lengthExact, outOfRange]

theorem rejects_nonzero_high_padding
    (bytes : List Nat)
    (lengthExact : bytes.length = statementByteCount)
    (range : bytesInRange bytes = true)
    (badPadding :
      ¬ (words bytes lengthExact
        ⟨statementByteCount - 1, by decide⟩).toNat < 16) :
    parse bytes = none := by
  simp [parse, lengthExact, range, badPadding]

theorem parsed_words_encode
    (image : PublicImage)
    (lengthExact : (WasmStatementBytes.encode image).length =
      statementByteCount) :
    words (WasmStatementBytes.encode image) lengthExact = byteWords image := by
  funext index
  apply BitVec.eq_of_toNat_eq
  simp only [words, BitVec.toNat_ofNat]
  have byteBound := byte_value_lt_256 image index
  have byteBoundPow : (byteWords image index).toNat < 2 ^ byteBitCount := by
    norm_num [byteBitCount]
    exact byteBound
  rw [encode_get image index, Nat.mod_eq_of_lt byteBoundPow]

theorem encoded_bytes_in_range
    (image : PublicImage) :
    bytesInRange (WasmStatementBytes.encode image) = true := by
  rw [bytesInRange, List.all_eq_true]
  intro byte member
  rcases List.mem_ofFn.mp member with ⟨index, rfl⟩
  simp [byte_value_lt_256]

theorem parse_encode (image : PublicImage) :
    parse (WasmStatementBytes.encode image) = some (statementWord image) := by
  have lengthExact := WasmStatementBytes.encode_length image
  rw [parse, dif_pos lengthExact]
  have range := encoded_bytes_in_range image
  rw [dif_pos range]
  have wordsExact := parsed_words_encode image lengthExact
  have padding :
      (words (WasmStatementBytes.encode image) lengthExact
        ⟨statementByteCount - 1, by decide⟩).toNat < 16 := by
    rw [wordsExact]
    exact final_byte_lt_16 image
  rw [dif_pos padding]
  have decoded :
      (joinBytes (words (WasmStatementBytes.encode image) lengthExact)).setWidth
          statementBitCount = statementWord image := by
    rw [wordsExact]
    change
      (joinBytes (splitBytes (paddedWord image))).setWidth statementBitCount =
        statementWord image
    rw [join_split]
    simpa [paddedWord] using
      BitVec.setWidth_setWidth_of_le (statementWord image)
        (show statementBitCount ≤ paddedBitCount by decide)
  rw [decoded]

/-- Any accepted typed-image encoding parses to one logical bit word. -/
theorem encoded_section_has_no_padding_alias
    {Program : Type}
    {left right : PublicImage}
    {leftStatement rightStatement :
      Nightstream.Protocol.Nebula.WasmStatement.ProductionStatement Program}
    (leftDecoded : left.Decodes leftStatement)
    (rightDecoded : right.Decodes rightStatement)
    (equal : WasmStatementBytes.encode left = WasmStatementBytes.encode right) :
    left = right :=
  WasmStatementBytes.encode_injective_of_decodes
    leftDecoded rightDecoded equal

/-- The same no-alias result for one explicitly selected production profile.
This prevents an E=1 statement from being checked with the bit-serial V2
identity or with another batching factor. -/
theorem encoded_section_has_no_padding_aliasFor
    {Program : Type}
    {left right : PublicImage}
    {leftStatement rightStatement :
      Nightstream.Protocol.Nebula.WasmStatement.ProductionStatement Program}
    {expectedProfile : Nightstream.Protocol.Nebula.Profile.Identity}
    (leftDecoded : left.DecodesFor expectedProfile leftStatement)
    (rightDecoded : right.DecodesFor expectedProfile rightStatement)
    (equal : WasmStatementBytes.encode left = WasmStatementBytes.encode right) :
    left = right :=
  WasmStatementBytes.encode_injective_of_decodesFor
    leftDecoded rightDecoded equal

end Nightstream.Implementation.Nebula.WasmStatementParser
