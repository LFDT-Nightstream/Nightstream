import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredTernary

/-!
Model-level 23-coordinate centered-septenary encoding for radix-four
ordinary private Goldilocks fields.

This file owns the seven-symbol alphabet, the deterministic Rust-shaped
shifted encoder, the linear decoder, the strict `b = 4` norm fact, the exact
width boundary, and semantic source-witness reconstruction. It does not prove
that the outer PiCCS norm polynomial supplies the alphabet premise. It also
does not identify a generated Rust assignment or a complete F-prime relation.
-/

namespace Nightstream.Implementation.R1CS.CenteredSeptenaryField

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler

set_option maxRecDepth 262144

def digitCount : Nat := 23
def radix : Nat := 7
def shift : Nat := 13684373670040458171

/-- Canonical Goldilocks residues for `{-3, -2, -1, 0, 1, 2, 3}`. -/
def CenteredResidue (value : Nat) : Prop :=
  value = goldilocksP - 3 ∨
    value = goldilocksP - 2 ∨
    value = goldilocksP - 1 ∨
    value = 0 ∨ value = 1 ∨ value = 2 ∨ value = 3

instance (value : Nat) : Decidable (CenteredResidue value) := by
  unfold CenteredResidue
  infer_instance

def AlphabetWord (digits : Nat → Nat) : Prop :=
  ∀ index, index < digitCount → CenteredResidue (digits index)

/-- Little-endian radix-seven value of the first `count` residues. -/
def lowValue (digits : Nat → Nat) : Nat → Nat
  | 0 => 0
  | count + 1 => lowValue digits count + digits count * radix ^ count

theorem lowValue_congr
    {left right : Nat → Nat} {count : Nat}
    (pointwise : ∀ index, index < count → left index = right index) :
    lowValue left count = lowValue right count := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [lowValue, lowValue,
        inductionHypothesis (fun index indexLt => pointwise index (by omega)),
        pointwise count (by omega)]

def decodeWord (digits : Nat → Nat) : Nat :=
  lowValue digits digitCount % goldilocksP

def Represents (source : Nat) (digits : Nat → Nat) : Prop :=
  AlphabetWord digits ∧ decodeWord digits = source % goldilocksP

def centeredMagnitude (value : Nat) : Nat :=
  min value (goldilocksP - value)

theorem centeredResidue_norm_lt_four
    {value : Nat} (centered : CenteredResidue value) :
    centeredMagnitude value < 4 := by
  rcases centered with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    decide

theorem alphabetWord_low_norm
    {digits : Nat → Nat} (alphabet : AlphabetWord digits) :
    ∀ index, index < digitCount → centeredMagnitude (digits index) < 4 := by
  intro index indexLt
  exact centeredResidue_norm_lt_four (alphabet index indexLt)

/-- Twenty-two seven-symbol coordinates cannot cover Goldilocks, while
twenty-three coordinates have sufficient cardinality. -/
theorem width_boundary :
    7 ^ 22 < goldilocksP ∧ goldilocksP < 7 ^ digitCount := by
  decide

def targetValue (source : Nat) : Nat :=
  (source + shift) % goldilocksP

def encodeHeptit (source index : Nat) : Nat :=
  targetValue source / radix ^ index % radix

/-- Map one ordinary radix-seven digit to its centered Goldilocks residue. -/
def encodeDigit (source index : Nat) : Nat :=
  match encodeHeptit source index with
  | 0 => goldilocksP - 3
  | 1 => goldilocksP - 2
  | 2 => goldilocksP - 1
  | 3 => 0
  | 4 => 1
  | 5 => 2
  | _ => 3

theorem encodeHeptit_lt_seven (source index : Nat) :
    encodeHeptit source index < 7 := by
  unfold encodeHeptit radix
  exact Nat.mod_lt _ (by decide)

theorem encodeDigit_centered (source index : Nat) :
    CenteredResidue (encodeDigit source index) := by
  have bounded := encodeHeptit_lt_seven source index
  have cases :
      encodeHeptit source index = 0 ∨
      encodeHeptit source index = 1 ∨
      encodeHeptit source index = 2 ∨
      encodeHeptit source index = 3 ∨
      encodeHeptit source index = 4 ∨
      encodeHeptit source index = 5 ∨
      encodeHeptit source index = 6 := by
    omega
  rcases cases with zero | one | two | three | four | five | six
  all_goals simp [CenteredResidue, encodeDigit, *]

theorem encodeDigit_add_three_mod (source index : Nat) :
    (encodeDigit source index + 3) % goldilocksP =
      encodeHeptit source index := by
  have bounded := encodeHeptit_lt_seven source index
  have cases :
      encodeHeptit source index = 0 ∨
      encodeHeptit source index = 1 ∨
      encodeHeptit source index = 2 ∨
      encodeHeptit source index = 3 ∨
      encodeHeptit source index = 4 ∨
      encodeHeptit source index = 5 ∨
      encodeHeptit source index = 6 := by
    omega
  rcases cases with zero | one | two | three | four | five | six
  all_goals simp [encodeDigit, goldilocksP, *]

theorem lowValue_encodeHeptit (source : Nat) : ∀ count,
    lowValue (encodeHeptit source) count =
      targetValue source % radix ^ count := by
  intro count
  induction count with
  | zero => exact (Nat.mod_one _).symm
  | succ count inductionHypothesis =>
      rw [lowValue, inductionHypothesis, Nat.pow_succ, Nat.mod_mul]
      simp only [encodeHeptit]
      rw [Nat.mul_comm]

private theorem targetValue_lt_radix (source : Nat) :
    targetValue source < radix ^ digitCount := by
  exact Nat.lt_trans (Nat.mod_lt _ (by decide)) (by decide)

theorem lowValue_encodeHeptit_full (source : Nat) :
    lowValue (encodeHeptit source) digitCount = targetValue source := by
  rw [lowValue_encodeHeptit]
  exact Nat.mod_eq_of_lt (targetValue_lt_radix source)

theorem lowValue_mod_congr
    {left right : Nat → Nat} {count : Nat}
    (pointwise : ∀ index, index < count →
      left index % goldilocksP = right index % goldilocksP) :
    lowValue left count % goldilocksP =
      lowValue right count % goldilocksP := by
  induction count with
  | zero => simp [lowValue]
  | succ count inductionHypothesis =>
      rw [lowValue, lowValue]
      have prefixEq := inductionHypothesis
        (fun index indexLt => pointwise index (by omega))
      have lastEq := pointwise count (by omega)
      calc
        (lowValue left count + left count * radix ^ count) % goldilocksP =
            (lowValue left count % goldilocksP +
              (left count % goldilocksP *
                (radix ^ count % goldilocksP)) % goldilocksP) %
              goldilocksP := by simp [Nat.add_mod, Nat.mul_mod]
        _ = (lowValue right count % goldilocksP +
              (right count % goldilocksP *
                (radix ^ count % goldilocksP)) % goldilocksP) %
              goldilocksP := by rw [prefixEq, lastEq]
        _ = (lowValue right count + right count * radix ^ count) %
              goldilocksP := by simp [Nat.add_mod, Nat.mul_mod]

theorem lowValue_pointwise_add (left right : Nat → Nat) : ∀ count,
    lowValue (fun index => left index + right index) count =
      lowValue left count + lowValue right count := by
  intro count
  induction count with
  | zero => simp [lowValue]
  | succ count inductionHypothesis =>
      rw [lowValue, lowValue, lowValue, inductionHypothesis, Nat.add_mul]
      omega

theorem shift_eq_threes_lowValue :
    shift = lowValue (fun _ => 3) digitCount := by
  decide

private theorem shifted_decode_congruence (source : Nat) :
    (lowValue (encodeDigit source) digitCount + shift) % goldilocksP =
      targetValue source := by
  calc
    (lowValue (encodeDigit source) digitCount + shift) % goldilocksP =
        lowValue (fun index => encodeDigit source index + 3)
          digitCount % goldilocksP := by
            rw [lowValue_pointwise_add, ← shift_eq_threes_lowValue]
    _ = lowValue (encodeHeptit source) digitCount % goldilocksP := by
          apply lowValue_mod_congr
          intro index _indexLt
          rw [encodeDigit_add_three_mod]
          exact (Nat.mod_eq_of_lt
            (Nat.lt_trans (encodeHeptit_lt_seven source index)
              (by decide))).symm
    _ = targetValue source % goldilocksP := by
          rw [lowValue_encodeHeptit_full]
    _ = targetValue source := Nat.mod_eq_of_lt
          (Nat.mod_lt _ (by decide))

private theorem shift_lt_modulus : shift < goldilocksP := by
  decide

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

abbrev FiniteWord := Fin digitCount → Nat

def wordAt (digits : FiniteWord) (index : Nat) : Nat :=
  if indexLt : index < digitCount then digits ⟨index, indexLt⟩ else 0

def FiniteAlphabetWord (digits : FiniteWord) : Prop :=
  ∀ index, CenteredResidue (digits index)

def decodeFiniteWord (digits : FiniteWord) : Nat :=
  decodeWord (wordAt digits)

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

structure ChosenWitness where
  digits : FiniteWord
  alphabet : FiniteAlphabetWord digits

def ChosenWitness.source (witness : ChosenWitness) : Nat :=
  decodeFiniteWord witness.digits

theorem ChosenWitness.source_canonical (witness : ChosenWitness) :
    witness.source < goldilocksP := by
  unfold ChosenWitness.source decodeFiniteWord decodeWord
  exact Nat.mod_lt _ (by decide)

def AugmentedRelation (sourcePredicate : Nat → Prop)
    (witness : ChosenWitness) : Prop :=
  sourcePredicate witness.source

/-- Every accepted septenary word reconstructs a canonical source witness.
This is the model-level soundness direction needed before source coordinates
can be replaced by their 23-coordinate low-norm word. -/
theorem augmentedRelation_sound
    {sourcePredicate : Nat → Prop} {witness : ChosenWitness}
    (accepted : AugmentedRelation sourcePredicate witness) :
    ∃ source, source < goldilocksP ∧ sourcePredicate source := by
  exact ⟨witness.source, witness.source_canonical, accepted⟩

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

theorem augmented_exists_iff_semantic_exists
    (sourcePredicate : Nat → Prop) :
    (∃ witness, AugmentedRelation sourcePredicate witness) ↔
      ∃ source, source < goldilocksP ∧ sourcePredicate source := by
  constructor
  · rintro ⟨witness, accepted⟩
    exact augmentedRelation_sound accepted
  · rintro ⟨source, canonical, accepted⟩
    exact augmentedRelation_complete canonical accepted

end Nightstream.Implementation.R1CS.CenteredSeptenaryField
