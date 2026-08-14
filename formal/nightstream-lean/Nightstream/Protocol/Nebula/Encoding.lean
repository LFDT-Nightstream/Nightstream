import Mathlib.Data.Nat.Digits.Lemmas
import Mathlib.Data.List.GetD

/-!
Contract: exact `ShiftedTernary41V1` field encoding for Nebula V2.

Assurance tier: model-level.

Owns the Goldilocks modulus, shift, target residue, fixed little-endian trit
word, centered-digit map, decoding rule, and injectivity on canonical field
integers.

Does not own R1CS rows, Rust witness generation, Ajtai maps, or verifier-key
serialization.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.ShiftedTernary41V1

def modulus : Nat := 18446744069414584321

def digitCount : Nat := 41

def shift : Nat := (3 ^ digitCount - 1) / 2

def CanonicalGoldilocks := { value : Nat // value < modulus }

deriving instance DecidableEq for CanonicalGoldilocks

def target (value : CanonicalGoldilocks) : Nat :=
  (value.val + shift) % modulus

/-- The unique fixed-width little-endian ordinary base-three word. -/
def trits (value : CanonicalGoldilocks) : List Nat :=
  Nat.digitsAppend 3 digitCount (target value)

/-- Exact ordinary-trit to centered-integer map. Values outside the checked
trit alphabet are not encodings. -/
def centeredDigit : Nat → Int
  | 0 => -1
  | 1 => 0
  | 2 => 1
  | _ => 0

/-- Exact ordinary-trit to canonical Goldilocks coordinate map. -/
def fieldDigit : Nat → Nat
  | 0 => modulus - 1
  | 1 => 0
  | 2 => 1
  | _ => 0

/-- Decode the ordinary trits by undoing the fixed shift modulo Goldilocks. -/
def decode (word : List Nat) : Nat :=
  (Nat.ofDigits 3 word + (modulus - shift)) % modulus

theorem shift_lt_modulus : shift < modulus := by
  norm_num [shift, digitCount, modulus]

theorem modulus_lt_wordCapacity : modulus < 3 ^ digitCount := by
  norm_num [digitCount, modulus]

theorem target_lt_modulus (value : CanonicalGoldilocks) :
    target value < modulus := by
  exact Nat.mod_lt _ (by norm_num [modulus])

theorem target_lt_wordCapacity (value : CanonicalGoldilocks) :
    target value < 3 ^ digitCount :=
  Nat.lt_trans (target_lt_modulus value) modulus_lt_wordCapacity

theorem trits_length (value : CanonicalGoldilocks) :
    (trits value).length = digitCount := by
  exact Nat.length_digitsAppend (by decide) digitCount
    (target_lt_wordCapacity value)

theorem trits_bounded (value : CanonicalGoldilocks) :
    ∀ trit ∈ trits value, trit < 3 := by
  intro trit member
  exact Nat.lt_of_mem_digitsAppend (by decide) digitCount trit member

/-- Coordinate formula for the fixed-width word. Appending zero digits does
not change the default-zero digit function, so this is the same formula used
by the independent centered-ternary compiler. -/
theorem trits_getD (value : CanonicalGoldilocks) (index : Nat) :
    (trits value).getD index 0 = target value / 3 ^ index % 3 := by
  have digitsFormula := Nat.getD_digits (target value) index
    (b := 3) (by decide)
  rw [← digitsFormula]
  unfold trits Nat.digitsAppend
  by_cases inDigits : index < (Nat.digits 3 (target value)).length
  · exact List.getD_append _ _ _ _ inDigits
  · have afterDigits : (Nat.digits 3 (target value)).length <= index :=
      Nat.le_of_not_gt inDigits
    rw [List.getD_append_right _ _ _ _ afterDigits]
    rw [List.getD_eq_default _ _ afterDigits]
    simp

/-- Finite coordinate form of `trits_getD`. -/
theorem trits_get (value : CanonicalGoldilocks) (index : Fin digitCount) :
    (trits value).get
        ⟨index.val, by rw [trits_length]; exact index.isLt⟩ =
      target value / 3 ^ index.val % 3 := by
  have indexLt : index.val < (trits value).length := by
    rw [trits_length]
    exact index.isLt
  change (trits value)[index.val] = target value / 3 ^ index.val % 3
  rw [← List.getD_eq_getElem (l := trits value) (d := 0) indexLt]
  exact trits_getD value index.val

theorem ofDigits_trits (value : CanonicalGoldilocks) :
    Nat.ofDigits 3 (trits value) = target value := by
  unfold trits Nat.digitsAppend
  rw [Nat.ofDigits_append_replicate_zero, Nat.ofDigits_digits]

private theorem add_shift_inverse_mod (value : Nat) :
    ((value + shift) + (modulus - shift)) % modulus =
      value % modulus := by
  calc
    ((value + shift) + (modulus - shift)) % modulus =
        (value + (shift + (modulus - shift))) % modulus := by
          rw [Nat.add_assoc]
    _ = (value + modulus) % modulus := by
          rw [Nat.add_sub_of_le (Nat.le_of_lt shift_lt_modulus)]
    _ = value % modulus := by simp

theorem decode_target (value : CanonicalGoldilocks) :
    (target value + (modulus - shift)) % modulus = value.val := by
  unfold target
  rw [Nat.mod_add_mod]
  rw [add_shift_inverse_mod]
  exact Nat.mod_eq_of_lt value.property

theorem decode_encode (value : CanonicalGoldilocks) :
    decode (trits value) = value.val := by
  unfold decode
  rw [ofDigits_trits]
  exact decode_target value

theorem target_injective : Function.Injective target := by
  intro left right equal
  apply Subtype.ext
  have recovered := congrArg (fun encoded =>
    (encoded + (modulus - shift)) % modulus) equal
  change (target left + (modulus - shift)) % modulus =
    (target right + (modulus - shift)) % modulus at recovered
  rw [decode_target left, decode_target right] at recovered
  exact recovered

theorem trits_injective : Function.Injective trits := by
  intro left right equal
  apply target_injective
  exact (Nat.bijOn_digitsAppend (b := 3) (by decide) digitCount).injOn
    (target_lt_wordCapacity left) (target_lt_wordCapacity right) equal

/-- Ordinary signed 41-digit balanced ternary cannot reconstruct every
canonical Goldilocks integer as an equal integer. V2 uses the modular shifted
encoding above instead. -/
theorem ordinarySignedMaximum_lt_goldilocksMaximum :
    (3 ^ digitCount - 1) / 2 < modulus - 1 := by
  norm_num [digitCount, modulus]

end Nightstream.Protocol.Nebula.ShiftedTernary41V1
