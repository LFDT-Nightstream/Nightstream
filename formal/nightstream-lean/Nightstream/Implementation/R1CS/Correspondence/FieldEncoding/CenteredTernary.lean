import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.CenteredZero
import Nightstream.Implementation.R1CS.Correspondence.ShiftedTernary.ReducedCore

/-!
Contract: model-level 41-coordinate centered-ternary encoding for ordinary
private Goldilocks fields.

Owns: the centered alphabet, deterministic shifted encoder, linear decoder,
low-norm proof, zero-word uniqueness, and the information-theoretic width
floor at norm bound `b = 2`. It also models an augmented source witness that
retains the exact accepted representation choice.

Does not own: public `enc_inst` bits, SIS-authoritative shifted-ternary
openings, Rust materialization, concrete matrix-row substitution, CE
commitments, selector emission, or HyperNova's end-to-end encoding theorem.

Emits constraints: no. `GateWord` models one `d^3 - d = 0` gate per
coordinate; it is not a production row artifact.

Authority boundary: digits are only a representation. Acceptance decodes
them to the source field before any source predicate is used. This component
proves that the old semantic witness has only a left-inverse encoder, and that
an augmented witness retaining the exact word has a two-sided parser/emitter.
It does not promote a decoded field or digest over the exact committed word.

| Branch | Mathematical obligation | Guarantee | Tier |
|---|---|---|---|
| `AlphabetWord` / `GateWord` | `d^3-d=0` over canonical Goldilocks residues | exactly `{-1,0,1}` | model-level |
| `encodeDigit_represents` | 41 little-endian centered trits | every canonical field has an accepted word | model-level |
| `decode_encodeDigit` | `sum d_i 3^i mod p` | `decode(encode(x)) = x` | model-level |
| `alphabetWord_low_norm` | SuperNeo `||z||_inf < 2` | every coordinate has centered magnitude at most one | model-level |
| `represents_zero_unique` | decoded inactive value is zero | every coordinate is zero | model-level |
| `width_floor` | three symbols per coordinate | 40 coordinates cannot cover Goldilocks | model-level |
| `duplicate_words_accepted` / `duplicate_words_decode_same` | targets `0` and `p` | distinct accepted words decode equally | model-level |
| `ChosenWitness` | exact 41-coordinate word is part of `w` | parse/re-emit is a two-sided inverse | model-level |
| `augmented_exists_iff_semantic_exists` | semantic relation consumes decoded fields | old and augmented witnesses are existentially equivalent | model-level |
| `ChosenPrivateWitness` | finite tuple of ordinary private fields | component contracts compose pointwise | model-level |
-/

namespace Nightstream.Implementation.R1CS.CenteredTernaryField

set_option maxRecDepth 262144

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.Implementation.R1CS.ShiftedTernarySound
open Nightstream.Implementation.R1CS.ShiftedTernaryCenteredZero
open Nightstream.Implementation.R1CS.ShiftedTernaryReducedCore
open Nightstream.Implementation.R1CS.ShiftedTernaryComplete

/-- Canonical Goldilocks residues for the centered alphabet. -/
def CenteredResidue (value : Nat) : Prop :=
  value = goldilocksP - 1 ∨ value = 0 ∨ value = 1

instance (value : Nat) : Decidable (CenteredResidue value) := by
  unfold CenteredResidue
  infer_instance

/-- Forty-one centered coordinates, before source-value binding. -/
def AlphabetWord (digits : Nat → Nat) : Prop :=
  ∀ index, index < digitCount → CenteredResidue (digits index)

/-- Exact common-gate schedule proposed for one ordinary private field. -/
def GateWord (digits : Nat → Nat) : Prop :=
  ∀ index, index < digitCount → CenteredUnitGateHolds (digits index)

/-- Linear source-field decoder used by substituted source constraints. -/
def decodeWord (digits : Nat → Nat) : Nat :=
  lowValue digits digitCount % goldilocksP

/-- One source field is represented by an alphabet word with the same residue. -/
def Represents (source : Nat) (digits : Nat → Nat) : Prop :=
  AlphabetWord digits ∧ decodeWord digits = source % goldilocksP

/-- The common cubic gate has exactly the centered roots. -/
theorem centeredUnitGate_iff
    (prime : EuclidPrime goldilocksP)
    {value : Nat} (canonical : value < goldilocksP) :
    CenteredUnitGateHolds value ↔ CenteredResidue value := by
  constructor
  · intro holds
    rcases centeredUnitGate_sound prime canonical holds with
      zero | one | negative
    · exact Or.inr (Or.inl zero)
    · exact Or.inr (Or.inr one)
    · exact Or.inl negative
  · intro centered
    rcases centered with negative | zero | one
    · exact centeredUnitGate_complete (Digit.neg negative rfl)
    · exact centeredUnitGate_complete (Digit.zero zero rfl)
    · exact centeredUnitGate_complete (Digit.pos one rfl)

theorem gateWord_iff_alphabetWord
    (prime : EuclidPrime goldilocksP)
    {digits : Nat → Nat}
    (canonical : ∀ index, index < digitCount → digits index < goldilocksP) :
    GateWord digits ↔ AlphabetWord digits := by
  constructor <;> intro holds index indexLt
  · exact (centeredUnitGate_iff prime (canonical index indexLt)).mp
      (holds index indexLt)
  · exact (centeredUnitGate_iff prime (canonical index indexLt)).mpr
      (holds index indexLt)

/-- Centered magnitude for one canonical residue. -/
def centeredMagnitude (value : Nat) : Nat :=
  min value (goldilocksP - value)

theorem centeredResidue_norm_lt_two
    {value : Nat} (centered : CenteredResidue value) :
    centeredMagnitude value < 2 := by
  rcases centered with rfl | rfl | rfl <;>
    decide

/-- Every committed coordinate obeys SuperNeo's strict `b = 2` norm bound. -/
theorem alphabetWord_low_norm
    {digits : Nat → Nat} (alphabet : AlphabetWord digits) :
    ∀ index, index < digitCount → centeredMagnitude (digits index) < 2 := by
  intro index indexLt
  exact centeredResidue_norm_lt_two (alphabet index indexLt)

/-- Negative-indicator witness used only to reuse the proved centered-zero lemma. -/
def negativeIndicator (value : Nat) : Nat :=
  if value = goldilocksP - 1 then 1 else 0

theorem digit_of_centeredResidue
    {value : Nat} (centered : CenteredResidue value) :
    Digit value (negativeIndicator value) := by
  rcases centered with negative | zero | one
  · exact .neg negative (by simp [negativeIndicator, negative])
  · exact .zero zero (by simp [negativeIndicator, zero, goldilocksP])
  · exact .pos one (by simp [negativeIndicator, one, goldilocksP])

/-- One decoded-zero selector equation forces the complete centered word to
the all-zero coordinate vector. -/
theorem represents_zero_unique
    {digits : Nat → Nat} (represented : Represents 0 digits) :
    ∀ index, index < digitCount → digits index = 0 := by
  apply centered_zero_unique
  · intro index indexLt
    exact digit_of_centeredResidue (represented.1 index indexLt)
  · simpa [Represents, decodeWord] using represented.2

/-- Deterministic shifted representative used for the model encoder. -/
def targetValue (source : Nat) : Nat :=
  (source + shift) % goldilocksP

/-- Ordinary trit at one little-endian position. -/
def encodeTrit (source index : Nat) : Nat :=
  targetValue source / 3 ^ index % 3

/-- Center an ordinary trit as a canonical Goldilocks residue. -/
def encodeDigit (source index : Nat) : Nat :=
  match encodeTrit source index with
  | 0 => goldilocksP - 1
  | 1 => 0
  | _ => 1

theorem encodeTrit_lt_three (source index : Nat) :
    encodeTrit source index < 3 := by
  unfold encodeTrit
  exact Nat.mod_lt _ (by decide)

theorem encodeDigit_centered (source index : Nat) :
    CenteredResidue (encodeDigit source index) := by
  have tritLt := encodeTrit_lt_three source index
  have cases : encodeTrit source index = 0 ∨
      encodeTrit source index = 1 ∨ encodeTrit source index = 2 := by
    omega
  rcases cases with zero | one | two
  · exact Or.inl (by simp [encodeDigit, zero])
  · exact Or.inr (Or.inl (by simp [encodeDigit, one]))
  · exact Or.inr (Or.inr (by simp [encodeDigit, two]))

theorem encodeDigit_add_one_mod (source index : Nat) :
    (encodeDigit source index + 1) % goldilocksP =
      encodeTrit source index := by
  have tritLt := encodeTrit_lt_three source index
  have cases : encodeTrit source index = 0 ∨
      encodeTrit source index = 1 ∨ encodeTrit source index = 2 := by
    omega
  rcases cases with zero | one | two
  · simp [encodeDigit, zero, goldilocksP]
  · simp [encodeDigit, one, goldilocksP]
  · simp [encodeDigit, two, goldilocksP]

theorem lowValue_encodeTrit (source : Nat) : ∀ count,
    lowValue (encodeTrit source) count = targetValue source % 3 ^ count := by
  intro count
  induction count with
  | zero => exact (Nat.mod_one _).symm
  | succ count inductionHypothesis =>
      rw [lowValue, inductionHypothesis, Nat.pow_succ, Nat.mod_mul]
      simp only [encodeTrit]
      rw [Nat.mul_comm]

private theorem targetValue_lt_radix (source : Nat) :
    targetValue source < 3 ^ digitCount := by
  exact Nat.lt_trans (Nat.mod_lt _ (by native_decide)) (by native_decide)

theorem lowValue_encodeTrit_full (source : Nat) :
    lowValue (encodeTrit source) digitCount = targetValue source := by
  rw [lowValue_encodeTrit]
  exact Nat.mod_eq_of_lt (targetValue_lt_radix source)

private theorem shifted_decode_congruence (source : Nat) :
    (lowValue (encodeDigit source) digitCount + shift) % goldilocksP =
      targetValue source := by
  calc
    (lowValue (encodeDigit source) digitCount + shift) % goldilocksP =
        lowValue (fun index => encodeDigit source index + 1)
          digitCount % goldilocksP := by
            rw [lowValue_pointwise_add, shift_eq_ones_lowValue]
    _ = lowValue (encodeTrit source) digitCount % goldilocksP := by
          apply lowValue_mod_congr
          intro index _indexLt
          rw [encodeDigit_add_one_mod]
          exact (Nat.mod_eq_of_lt
            (Nat.lt_trans (encodeTrit_lt_three source index)
              (by native_decide))).symm
    _ = targetValue source % goldilocksP := by
          rw [lowValue_encodeTrit_full]
    _ = targetValue source := Nat.mod_eq_of_lt
          (Nat.mod_lt _ (by native_decide))

private theorem shift_lt_modulus : shift < goldilocksP := by
  native_decide

private theorem add_shift_inverse_mod (value : Nat) :
    ((value + shift) + (goldilocksP - shift)) % goldilocksP =
      value % goldilocksP := by
  calc
    ((value + shift) + (goldilocksP - shift)) % goldilocksP =
        (value + (shift + (goldilocksP - shift))) % goldilocksP := by
          rw [Nat.add_assoc]
    _ = (value + goldilocksP) % goldilocksP := by
          rw [Nat.add_sub_of_le (Nat.le_of_lt shift_lt_modulus)]
    _ = value % goldilocksP := by simp

/-- The deterministic encoder is a left inverse of the linear decoder. -/
theorem decode_encodeDigit
    {source : Nat} (canonical : source < goldilocksP) :
    decodeWord (encodeDigit source) = source := by
  have shifted := shifted_decode_congruence source
  change
    (lowValue (encodeDigit source) digitCount + shift) % goldilocksP =
      (source + shift) % goldilocksP at shifted
  have appended := congrArg
    (fun value => (value + (goldilocksP - shift)) % goldilocksP) shifted
  change
    (((lowValue (encodeDigit source) digitCount + shift) % goldilocksP +
        (goldilocksP - shift)) % goldilocksP) =
      (((source + shift) % goldilocksP + (goldilocksP - shift)) %
        goldilocksP) at appended
  rw [Nat.mod_add_mod, Nat.mod_add_mod,
    add_shift_inverse_mod, add_shift_inverse_mod] at appended
  simpa [decodeWord, Nat.mod_eq_of_lt canonical] using appended

theorem encodeDigit_represents
    {source : Nat} (canonical : source < goldilocksP) :
    Represents source (encodeDigit source) := by
  constructor
  · intro index _indexLt
    exact encodeDigit_centered source index
  · rw [decode_encodeDigit canonical, Nat.mod_eq_of_lt canonical]

theorem encodeDigit_gateWord
    (prime : EuclidPrime goldilocksP) (source : Nat) :
    GateWord (encodeDigit source) := by
  intro index _indexLt
  apply (centeredUnitGate_iff prime ?_).mpr
  · exact encodeDigit_centered source index
  · rcases encodeDigit_centered source index with negative | zero | one
    · rw [negative]; native_decide
    · rw [zero]; native_decide
    · rw [one]; native_decide

/-- Three symbols cannot cover all Goldilocks residues in forty coordinates,
while forty-one coordinates have sufficient cardinality. -/
theorem width_floor :
    3 ^ 40 < goldilocksP ∧ goldilocksP < 3 ^ digitCount := by
  native_decide

/-! ## Exact finite words and the augmented witness contract -/

/-- The actual encoded object has exactly 41 coordinates. -/
abbrev FiniteWord := Fin digitCount → Nat

/-- Extend a finite encoded word only for reuse by the recursive `lowValue`
definition. The extension is never part of the emitted object. -/
def wordAt (digits : FiniteWord) (index : Nat) : Nat :=
  if indexLt : index < digitCount then digits ⟨index, indexLt⟩ else 0

/-- Acceptance predicate on the exact finite word. -/
def FiniteAlphabetWord (digits : FiniteWord) : Prop :=
  ∀ index, CenteredResidue (digits index)

/-- Source field decoded from an exact finite word. -/
def decodeFiniteWord (digits : FiniteWord) : Nat :=
  decodeWord (wordAt digits)

/-- Deterministic honest choice used to transport an old semantic witness. -/
def finiteEncode (source : Nat) : FiniteWord :=
  fun index => encodeDigit source index.val

theorem finiteEncode_alphabet (source : Nat) :
    FiniteAlphabetWord (finiteEncode source) := by
  intro index
  exact encodeDigit_centered source index.val

theorem decodeFiniteWord_finiteEncode
    {source : Nat} (canonical : source < goldilocksP) :
    decodeFiniteWord (finiteEncode source) = source := by
  have lowEq :
      lowValue (wordAt (finiteEncode source)) digitCount =
        lowValue (encodeDigit source) digitCount := by
    apply lowValue_congr
    intro index indexLt
    simp [wordAt, finiteEncode, indexLt]
  unfold decodeFiniteWord decodeWord
  rw [lowEq]
  exact decode_encodeDigit canonical

/-- A CCS-accepted component: the exact word plus its alphabet evidence. -/
abbrev AcceptedEncoding :=
  {digits : FiniteWord // FiniteAlphabetWord digits}

/-- HyperNova-compatible source-witness shape for this component. The
representation choice itself is retained in `w`; the semantic field is
derived, not separately authoritative. -/
structure ChosenWitness where
  digits : FiniteWord
  alphabet : FiniteAlphabetWord digits

def ChosenWitness.source (witness : ChosenWitness) : Nat :=
  decodeFiniteWord witness.digits

theorem ChosenWitness.source_canonical (witness : ChosenWitness) :
    witness.source < goldilocksP := by
  unfold ChosenWitness.source decodeFiniteWord decodeWord
  exact Nat.mod_lt _ (by native_decide)

/-- Component encoder on the augmented witness type. -/
def encodeChosen (witness : ChosenWitness) : AcceptedEncoding :=
  ⟨witness.digits, witness.alphabet⟩

/-- Component parser for every accepted word. -/
def decodeChosen (encoded : AcceptedEncoding) : ChosenWitness where
  digits := encoded.1
  alphabet := encoded.2

/-- H.2-shaped direction: parsing and re-encoding an arbitrary accepted CCS
word reproduces that exact word, not merely the same semantic residue. -/
theorem encodeChosen_decodeChosen (encoded : AcceptedEncoding) :
    encodeChosen (decodeChosen encoded) = encoded := by
  rfl

/-- Honest augmented witnesses also survive the opposite round trip. -/
theorem decodeChosen_encodeChosen (witness : ChosenWitness) :
    decodeChosen (encodeChosen witness) = witness := by
  cases witness
  rfl

/-- The augmented relation applies the old semantic predicate only after
decoding the exact committed representation. -/
def AugmentedRelation (sourcePredicate : Nat → Prop)
    (witness : ChosenWitness) : Prop :=
  sourcePredicate witness.source

/-- Any accepted augmented witness yields an ordinary canonical source
witness. This is the soundness half of source-row substitution. -/
theorem augmentedRelation_sound
    {sourcePredicate : Nat → Prop} {witness : ChosenWitness}
    (accepted : AugmentedRelation sourcePredicate witness) :
    ∃ source, source < goldilocksP ∧ sourcePredicate source := by
  exact ⟨witness.source, witness.source_canonical, accepted⟩

/-- Every old semantic witness has a deterministic chosen representation.
This proves existential completeness but does not identify the old and
augmented witness types literally. -/
theorem augmentedRelation_complete
    {sourcePredicate : Nat → Prop} {source : Nat}
    (canonical : source < goldilocksP)
    (accepted : sourcePredicate source) :
    ∃ witness, AugmentedRelation sourcePredicate witness := by
  let witness : ChosenWitness := {
    digits := finiteEncode source
    alphabet := finiteEncode_alphabet source
  }
  refine ⟨witness, ?_⟩
  unfold AugmentedRelation ChosenWitness.source witness
  rw [decodeFiniteWord_finiteEncode canonical]
  exact accepted

/-- Exact existential equivalence to the old semantic relation. -/
theorem augmented_exists_iff_semantic_exists
    (sourcePredicate : Nat → Prop) :
    (∃ witness, AugmentedRelation sourcePredicate witness) ↔
      ∃ source, source < goldilocksP ∧ sourcePredicate source := by
  constructor
  · rintro ⟨witness, accepted⟩
    exact augmentedRelation_sound accepted
  · rintro ⟨source, canonical, accepted⟩
    exact augmentedRelation_complete canonical accepted

/-! ## Finite ordinary-private-field tuples

This is the compositional H.2/H.3-shaped contract for the whole tuple of
ordinary private fields. The tuple length is deliberately a parameter: this
module has no verifier-owned fixed-F-prime slot manifest from which it could
derive the current eligible-field count. -/

/-- Exact finite tuple of ordinary private-field words. -/
abbrev PrivateWords (fieldCount : Nat) :=
  Fin fieldCount → FiniteWord

def PrivateWordsAccepted {fieldCount : Nat}
    (words : PrivateWords fieldCount) : Prop :=
  ∀ field, FiniteAlphabetWord (words field)

def decodePrivateWords {fieldCount : Nat}
    (words : PrivateWords fieldCount) : Fin fieldCount → Nat :=
  fun field => decodeFiniteWord (words field)

def encodePrivateWords {fieldCount : Nat}
    (sources : Fin fieldCount → Nat) : PrivateWords fieldCount :=
  fun field => finiteEncode (sources field)

theorem encodePrivateWords_accepted
    {fieldCount : Nat} (sources : Fin fieldCount → Nat) :
    PrivateWordsAccepted (encodePrivateWords sources) := by
  intro field
  exact finiteEncode_alphabet (sources field)

theorem decodePrivateWords_encodePrivateWords
    {fieldCount : Nat} {sources : Fin fieldCount → Nat}
    (canonical : ∀ field, sources field < goldilocksP) :
    decodePrivateWords (encodePrivateWords sources) = sources := by
  funext field
  exact decodeFiniteWord_finiteEncode (canonical field)

abbrev AcceptedPrivateEncoding (fieldCount : Nat) :=
  {words : PrivateWords fieldCount // PrivateWordsAccepted words}

/-- Augmented source witness for every ordinary private field in a finite
tuple. Public bits and existing SIS-authoritative openings are not members of
this tuple. -/
structure ChosenPrivateWitness (fieldCount : Nat) where
  words : PrivateWords fieldCount
  accepted : PrivateWordsAccepted words

def ChosenPrivateWitness.sources {fieldCount : Nat}
    (witness : ChosenPrivateWitness fieldCount) : Fin fieldCount → Nat :=
  decodePrivateWords witness.words

theorem ChosenPrivateWitness.sources_canonical
    {fieldCount : Nat} (witness : ChosenPrivateWitness fieldCount) :
    ∀ field, witness.sources field < goldilocksP := by
  intro field
  unfold ChosenPrivateWitness.sources decodePrivateWords
  exact Nat.mod_lt _ (by native_decide)

def encodeChosenPrivate {fieldCount : Nat}
    (witness : ChosenPrivateWitness fieldCount) :
    AcceptedPrivateEncoding fieldCount :=
  ⟨witness.words, witness.accepted⟩

def decodeChosenPrivate {fieldCount : Nat}
    (encoded : AcceptedPrivateEncoding fieldCount) :
    ChosenPrivateWitness fieldCount where
  words := encoded.1
  accepted := encoded.2

/-- Finite-tuple H.2 direction: every accepted ordinary-private-field tuple
is reproduced coordinate for coordinate. -/
theorem encodeChosenPrivate_decodeChosenPrivate
    {fieldCount : Nat} (encoded : AcceptedPrivateEncoding fieldCount) :
    encodeChosenPrivate (decodeChosenPrivate encoded) = encoded := by
  rfl

theorem decodeChosenPrivate_encodeChosenPrivate
    {fieldCount : Nat} (witness : ChosenPrivateWitness fieldCount) :
    decodeChosenPrivate (encodeChosenPrivate witness) = witness := by
  cases witness
  rfl

def AugmentedPrivateRelation {fieldCount : Nat}
    (sourcePredicate : (Fin fieldCount → Nat) → Prop)
    (witness : ChosenPrivateWitness fieldCount) : Prop :=
  sourcePredicate witness.sources

/-- Finite-tuple semantic soundness after extraction and decoding. -/
theorem augmentedPrivateRelation_sound
    {fieldCount : Nat}
    {sourcePredicate : (Fin fieldCount → Nat) → Prop}
    {witness : ChosenPrivateWitness fieldCount}
    (accepted : AugmentedPrivateRelation sourcePredicate witness) :
    ∃ sources, (∀ field, sources field < goldilocksP) ∧
      sourcePredicate sources := by
  exact ⟨witness.sources, witness.sources_canonical, accepted⟩

theorem augmentedPrivateRelation_complete
    {fieldCount : Nat}
    {sourcePredicate : (Fin fieldCount → Nat) → Prop}
    {sources : Fin fieldCount → Nat}
    (canonical : ∀ field, sources field < goldilocksP)
    (accepted : sourcePredicate sources) :
    ∃ witness, AugmentedPrivateRelation sourcePredicate witness := by
  let witness : ChosenPrivateWitness fieldCount := {
    words := encodePrivateWords sources
    accepted := encodePrivateWords_accepted sources
  }
  refine ⟨witness, ?_⟩
  unfold AugmentedPrivateRelation ChosenPrivateWitness.sources witness
  rw [decodePrivateWords_encodePrivateWords canonical]
  exact accepted

/-- Exact existential equivalence for any finite ordinary-private-field
tuple. Instantiating this theorem with fixed F-prime remains a separate
relation-and-layout refinement obligation. -/
theorem augmented_private_exists_iff_semantic_exists
    {fieldCount : Nat}
    (sourcePredicate : (Fin fieldCount → Nat) → Prop) :
    (∃ witness, AugmentedPrivateRelation sourcePredicate witness) ↔
      ∃ sources, (∀ field, sources field < goldilocksP) ∧
        sourcePredicate sources := by
  constructor
  · rintro ⟨witness, accepted⟩
    exact augmentedPrivateRelation_sound accepted
  · rintro ⟨sources, canonical, accepted⟩
    exact augmentedPrivateRelation_complete canonical accepted

/-! ## Formal counterexample to old-witness image canonicality -/

/-- Centered digit word for a raw integer target in `[0, 3^41)`. -/
def rawTargetDigit (target index : Nat) : Nat :=
  match target / 3 ^ index % 3 with
  | 0 => goldilocksP - 1
  | 1 => 0
  | _ => 1

def rawTargetWord (target : Nat) : FiniteWord :=
  fun index => rawTargetDigit target index.val

theorem rawTargetDigit_centered (target index : Nat) :
    CenteredResidue (rawTargetDigit target index) := by
  have tritLt : target / 3 ^ index % 3 < 3 :=
    Nat.mod_lt _ (by decide)
  have cases : target / 3 ^ index % 3 = 0 ∨
      target / 3 ^ index % 3 = 1 ∨ target / 3 ^ index % 3 = 2 := by
    omega
  rcases cases with zero | one | two
  · exact Or.inl (by simp [rawTargetDigit, zero])
  · exact Or.inr (Or.inl (by simp [rawTargetDigit, one]))
  · exact Or.inr (Or.inr (by simp [rawTargetDigit, two]))

theorem rawTargetWord_alphabet (target : Nat) :
    FiniteAlphabetWord (rawTargetWord target) := by
  intro index
  exact rawTargetDigit_centered target index.val

/-- `0` and `p` are distinct radix-three targets inside 41 digits, but they
differ by one field modulus. Their centered words are both accepted. -/
theorem duplicate_words_accepted :
    FiniteAlphabetWord (rawTargetWord 0) ∧
      FiniteAlphabetWord (rawTargetWord goldilocksP) := by
  exact ⟨rawTargetWord_alphabet 0, rawTargetWord_alphabet goldilocksP⟩

theorem duplicate_words_differ :
    rawTargetWord 0 ≠ rawTargetWord goldilocksP := by
  intro equal
  have coordinate := congrFun equal (⟨0, by native_decide⟩ : Fin digitCount)
  have unequal : rawTargetDigit 0 0 ≠ rawTargetDigit goldilocksP 0 := by
    native_decide
  exact unequal coordinate

/-- Concrete failure of `encode(decode(word)) = word` for an encoder whose
input is only the old semantic field: two distinct accepted finite words have
the same decoder output. -/
theorem duplicate_words_decode_same :
    decodeFiniteWord (rawTargetWord 0) =
      decodeFiniteWord (rawTargetWord goldilocksP) := by
  native_decide

/-- A generic source predicate is sound after decoding; no encoded word is
accepted as semantic authority. -/
def LoweredPredicate (sourcePredicate : Nat → Prop) (digits : Nat → Nat) : Prop :=
  AlphabetWord digits ∧ sourcePredicate (decodeWord digits)

theorem loweredPredicate_sound
    {sourcePredicate : Nat → Prop} {digits : Nat → Nat}
    (accepted : LoweredPredicate sourcePredicate digits) :
    sourcePredicate (decodeWord digits) :=
  accepted.2

theorem loweredPredicate_complete
    (prime : EuclidPrime goldilocksP)
    {sourcePredicate : Nat → Prop} {source : Nat}
    (canonical : source < goldilocksP)
    (sourceAccepted : sourcePredicate source) :
    GateWord (encodeDigit source) ∧
      LoweredPredicate sourcePredicate (encodeDigit source) := by
  constructor
  · exact encodeDigit_gateWord prime source
  · constructor
    · exact (encodeDigit_represents canonical).1
    · rw [decode_encodeDigit canonical]
      exact sourceAccepted

end Nightstream.Implementation.R1CS.CenteredTernaryField
