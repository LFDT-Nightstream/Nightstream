import Mathlib.Data.Nat.Digits.Lemmas
import Nightstream.Protocol.Nebula.Encoding

/-!
Contract: exact canonical 64-bit Goldilocks word language for V2 public data.

Assurance tier: model-level.

Owns fixed little-endian binary words, integer decoding, the strict
`decoded < q` acceptance check, encoding injectivity, and the deterministic
`0` versus `q` modulo-alias countermodel.

Does not own Rust parsing, R1CS bitness/canonicality rows, or their refinement
to these definitions.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.CanonicalFieldBits

open Nightstream.Protocol.Nebula.ShiftedTernary41V1

def bitCount : Nat := 64

/-- Exactly 64 little-endian binary digits. -/
def Word :=
  { digits : List Nat //
    digits.length = bitCount ∧ ∀ digit ∈ digits, digit < 2 }

def decode (word : Word) : Nat :=
  Nat.ofDigits 2 word.val

def Canonical (word : Word) : Prop :=
  decode word < modulus

def AcceptedWord := { word : Word // Canonical word }

theorem modulus_lt_capacity : modulus < 2 ^ bitCount := by
  norm_num [modulus, bitCount]

def encode (value : CanonicalGoldilocks) : Word :=
  ⟨Nat.digitsAppend 2 bitCount value.val,
    Nat.mapsTo_digitsAppend (by decide) bitCount
      (Nat.lt_trans value.property modulus_lt_capacity)⟩

theorem decode_encode (value : CanonicalGoldilocks) :
    decode (encode value) = value.val := by
  exact (Nat.setInvOn_digitsAppend_ofDigits (by decide) bitCount).2
    (Nat.lt_trans value.property modulus_lt_capacity)

theorem encode_is_canonical (value : CanonicalGoldilocks) :
    Canonical (encode value) := by
  rw [Canonical, decode_encode]
  exact value.property

theorem decode_injective : Function.Injective decode := by
  intro left right equalDecode
  apply Subtype.ext
  exact Nat.ofDigits_inj_of_len_eq (by decide)
    (left.property.1.trans right.property.1.symm)
    left.property.2 right.property.2 equalDecode

def decodeAccepted (word : AcceptedWord) : CanonicalGoldilocks :=
  ⟨decode word.val, word.property⟩

theorem decodeAccepted_injective : Function.Injective decodeAccepted := by
  intro left right equal
  apply Subtype.ext
  apply decode_injective
  exact congrArg Subtype.val equal

def zero : CanonicalGoldilocks :=
  ⟨0, by decide⟩

def zeroWord : Word := encode zero

/-- The distinct 64-bit word for the modulus. It is binary and fixed-width,
but it is not a canonical field encoding. -/
def modulusWord : Word :=
  ⟨Nat.digitsAppend 2 bitCount modulus,
    Nat.mapsTo_digitsAppend (by decide) bitCount modulus_lt_capacity⟩

theorem decode_zeroWord : decode zeroWord = 0 := by
  exact decode_encode zero

theorem decode_modulusWord : decode modulusWord = modulus := by
  exact (Nat.setInvOn_digitsAppend_ofDigits (by decide) bitCount).2
    modulus_lt_capacity

theorem modulusWord_not_canonical : ¬ Canonical modulusWord := by
  rw [Canonical, decode_modulusWord]
  exact Nat.lt_irrefl modulus

theorem zeroWord_ne_modulusWord : zeroWord ≠ modulusWord := by
  intro equal
  have decoded := congrArg decode equal
  rw [decode_zeroWord, decode_modulusWord] at decoded
  norm_num [modulus] at decoded

/-- Modulo-q decoding alone accepts two distinct bit strings as the same
field element. The strict canonical predicate rejects the second string. -/
theorem zero_and_modulus_are_modulo_aliases :
    decode zeroWord % modulus = decode modulusWord % modulus := by
  rw [decode_zeroWord, decode_modulusWord]
  simp

end Nightstream.Protocol.Nebula.CanonicalFieldBits
