import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix

/-!
Owns the fixed 41-trit low-norm encoding of one Goldilocks field element.
The definition follows the centered base-three recurrence used by the
production selective compiler. It emits no constraints and chooses no
matrix layout.
-/

namespace NightstreamFPrime.Layout.BalancedTernary

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm

/-- Exact number of signed ternary coordinates used for one retained field
value. -/
def width : Nat := 41

/-- Largest natural magnitude representable by `count` balanced trits. -/
def radius : Nat → Nat
  | 0 => 0
  | count + 1 => 3 * radius count + 1

/-- The centered Goldilocks half interval fits in 41 balanced trits. -/
theorem halfModulus_le_radius : Centered.halfModulus ≤ radius width := by
  decide

def isNegative (value : F) : Prop :=
  Centered.halfModulus < value.val

instance (value : F) : Decidable (isNegative value) :=
  inferInstanceAs (Decidable (Centered.halfModulus < value.val))

/-- Nonnegative magnitude of the canonical centered representative. -/
def magnitude (value : F) : Nat :=
  if isNegative value then goldilocksModulus - value.val else value.val

theorem magnitude_le_halfModulus (value : F) :
    magnitude value ≤ Centered.halfModulus := by
  unfold magnitude isNegative
  split
  next negative =>
    have modulusShape := Centered.modulus_eq_two_half_add_one
    have valueBound := value.isLt
    omega
  next nonnegative =>
    omega

theorem magnitude_le_radius (value : F) :
    magnitude value ≤ radius width :=
  (magnitude_le_halfModulus value).trans halfModulus_le_radius

/-- One little-endian balanced trit in `{-1, 0, 1}`. -/
def digit (value : Nat) : F :=
  if value % 3 = 0 then 0
  else if value % 3 = 1 then 1
  else -1

/-- Remaining magnitude after removing one balanced trit. -/
def next (value : Nat) : Nat :=
  value / 3 + if value % 3 = 2 then 1 else 0

private theorem mod_three_cases (value : Nat) :
    value % 3 = 0 ∨ value % 3 = 1 ∨ value % 3 = 2 := by
  have bounded := Nat.mod_lt value (by decide : 0 < 3)
  omega

theorem digit_norm (value : Nat) : centeredMagnitude (digit value) < 2 := by
  rcases mod_three_cases value with zero | one | two
  · rw [show digit value = 0 by simp [digit, zero]]
    decide
  · rw [show digit value = 1 by simp [digit, one]]
    decide
  · have notZero : value % 3 ≠ 0 := by omega
    have notOne : value % 3 ≠ 1 := by omega
    rw [show digit value = (-1 : F) by simp [digit, notZero, notOne]]
    decide

/-- One balanced step reconstructs its input in the Goldilocks field. -/
theorem digit_add_three_mul_next (value : Nat) :
    digit value + fieldOfNat 3 * fieldOfNat (next value) =
      fieldOfNat value := by
  have decomposition := Nat.mod_add_div value 3
  rcases mod_three_cases value with zero | one | two
  · have valueEq : value = 3 * (value / 3) := by omega
    have digitEq : digit value = 0 := by simp [digit, zero]
    have nextEq : next value = value / 3 := by simp [next, zero]
    have rhsEq :
        fieldOfNat value = fieldOfNat 3 * fieldOfNat (value / 3) := by
      calc
        fieldOfNat value = fieldOfNat (3 * (value / 3)) :=
          congrArg fieldOfNat valueEq
        _ = fieldOfNat 3 * fieldOfNat (value / 3) := by
          rw [fieldOfNat_mul]
    rw [digitEq, nextEq, rhsEq]
    exact
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.zero_add _
  · have valueEq : value = 1 + 3 * (value / 3) := by omega
    have digitEq : digit value = 1 := by simp [digit, one]
    have nextEq : next value = value / 3 := by simp [next, one]
    have rhsEq :
        fieldOfNat value =
          (1 : F) + fieldOfNat 3 * fieldOfNat (value / 3) := by
      calc
        fieldOfNat value = fieldOfNat (1 + 3 * (value / 3)) :=
          congrArg fieldOfNat valueEq
        _ = (1 : F) + fieldOfNat 3 * fieldOfNat (value / 3) := by
          rw [fieldOfNat_add, fieldOfNat_mul, fieldOfNat_one]
    rw [digitEq, nextEq, rhsEq]
  · have notZero : value % 3 ≠ 0 := by omega
    have notOne : value % 3 ≠ 1 := by omega
    have valueSuccEq : value + 1 = 3 * (value / 3 + 1) := by omega
    have castEq :
        fieldOfNat value + 1 =
          fieldOfNat 3 * fieldOfNat (value / 3 + 1) := by
      rw [← fieldOfNat_one, ← fieldOfNat_add, valueSuccEq,
        fieldOfNat_mul]
    rw [show digit value = (-1 : F) by simp [digit, notZero, notOne]]
    rw [show next value = value / 3 + 1 by simp [next, two]]
    rw [← castEq]
    rw [Lean.Grind.Fin.add_comm (fieldOfNat value) 1,
      ← Lean.Grind.Fin.add_assoc]
    have cancel : (-1 : F) + 1 = 0 :=
      Lean.Grind.AddCommGroup.neg_add_cancel 1
    rw [cancel]
    exact
      NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.zero_add _

theorem next_le_radius {count value : Nat}
    (bounded : value ≤ radius (count + 1)) :
    next value ≤ radius count := by
  have decomposition := Nat.mod_add_div value 3
  rcases mod_three_cases value with zero | one | two
  · simp [radius, next, zero] at bounded ⊢
    omega
  · simp [radius, next, one] at bounded ⊢
    omega
  · simp [radius, next, two] at bounded ⊢
    omega

/-- Reference little-endian digit list. -/
def digitsNat : Nat → Nat → List F
  | 0, _ => []
  | count + 1, value => digit value :: digitsNat count (next value)

@[simp] theorem digitsNat_length (count value : Nat) :
    (digitsNat count value).length = count := by
  induction count generalizing value with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp [digitsNat, inductionHypothesis]

/-- Interpret little-endian balanced digits in the Goldilocks field. -/
def recompose : List F → F
  | [] => 0
  | head :: tail => head + fieldOfNat 3 * recompose tail

theorem recompose_digitsNat :
    ∀ count value, value ≤ radius count →
      recompose (digitsNat count value) = fieldOfNat value
  | 0, value, bounded => by
      have valueZero : value = 0 := by simpa [radius] using bounded
      subst value
      rfl
  | count + 1, value, bounded => by
      rw [digitsNat, recompose,
        recompose_digitsNat count (next value) (next_le_radius bounded),
        digit_add_three_mul_next]

private theorem recompose_map_neg (values : List F) :
    recompose (values.map fun value => -value) = -recompose values := by
  induction values with
  | nil => exact Lean.Grind.AddCommGroup.neg_zero.symm
  | cons value values inductionHypothesis =>
      simp only [List.map_cons, recompose, inductionHypothesis]
      have mulNeg :
          fieldOfNat 3 * -recompose values =
            -(fieldOfNat 3 * recompose values) := by
        calc
          fieldOfNat 3 * -recompose values =
              -recompose values * fieldOfNat 3 := Fin.mul_comm _ _
          _ = -(recompose values * fieldOfNat 3) :=
            Lean.Grind.Fin.neg_mul _ _
          _ = -(fieldOfNat 3 * recompose values) := by
            rw [Fin.mul_comm (recompose values) (fieldOfNat 3)]
      rw [mulNeg]
      exact (Lean.Grind.AddCommGroup.neg_add _ _).symm

private theorem fieldOfNat_val (value : F) : fieldOfNat value.val = value := by
  apply Fin.ext
  simp [fieldOfNat, Nat.mod_eq_of_lt value.isLt]

private theorem neg_fieldOfNat_complement (value : F) :
    -(fieldOfNat (goldilocksModulus - value.val)) = value := by
  by_cases valueZero : value = 0
  · subst value
    exact Lean.Grind.AddCommGroup.neg_zero
  · have valNonzero : value.val ≠ 0 := by
      intro equal
      apply valueZero
      apply Fin.ext
      simpa using equal
    have complementLt :
        goldilocksModulus - value.val < goldilocksModulus := by
      omega
    have embeddedNonzero :
        fieldOfNat (goldilocksModulus - value.val) ≠ 0 := by
      intro equal
      have valuesEqual := congrArg Fin.val equal
      simp [fieldOfNat, Nat.mod_eq_of_lt complementLt] at valuesEqual
      omega
    apply Fin.ext
    rw [Fin.val_neg, if_neg embeddedNonzero]
    simp [fieldOfNat, Nat.mod_eq_of_lt complementLt]
    omega

/-- Exact Rust-compatible 41-trit encoding of one canonical field value. -/
def digits (value : F) : List F :=
  let unsigned := digitsNat width (magnitude value)
  if isNegative value then unsigned.map (fun digit => -digit) else unsigned

@[simp] theorem digits_length (value : F) : (digits value).length = width := by
  unfold digits
  split <;> simp

private theorem digitsNat_norm :
    ∀ count value digitValue,
      digitValue ∈ digitsNat count value →
      centeredMagnitude digitValue < 2
  | 0, _, _, member => by simp [digitsNat] at member
  | count + 1, value, digitValue, member => by
      simp only [digitsNat, List.mem_cons] at member
      rcases member with rfl | member
      · exact digit_norm value
      · exact digitsNat_norm count (next value) digitValue member

theorem digits_norm (value : F) (digitValue : F)
    (member : digitValue ∈ digits value) :
    centeredMagnitude digitValue < 2 := by
  unfold digits at member
  split at member
  next negative =>
    rcases List.mem_map.mp member with ⟨unsigned, unsignedMember, rfl⟩
    simpa [Centered.centeredMagnitude_neg] using
      digitsNat_norm width (magnitude value) unsigned unsignedMember
  next nonnegative =>
    exact digitsNat_norm width (magnitude value) digitValue member

/-- The 41 low-norm coordinates reconstruct the exact field element. -/
theorem recompose_digits (value : F) : recompose (digits value) = value := by
  have unsigned := recompose_digitsNat width (magnitude value)
    (magnitude_le_radius value)
  unfold digits
  split
  next negative =>
    rw [recompose_map_neg, unsigned]
    have magnitudeEq : magnitude value = goldilocksModulus - value.val := by
      simp [magnitude, negative]
    rw [magnitudeEq]
    exact neg_fieldOfNat_complement value
  next nonnegative =>
    rw [unsigned]
    have magnitudeEq : magnitude value = value.val := by
      simp [magnitude, nonnegative]
    rw [magnitudeEq, fieldOfNat_val]

end NightstreamFPrime.Layout.BalancedTernary
