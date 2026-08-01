import Nightstream.Implementation.Lowering.Nebula.SourceProducts
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Canonical bit and word decoding for the Lean-owned Nebula compiler.

Assurance tier: model-level.

Owns the unconditional Goldilocks Boolean-root theorem, little-endian word
decoding, and the no-wrap bound used by 10-, 16-, 32-, and 44-bit fields.

Does not own public-port binding, a memory access, witness construction,
segment composition, Rust, or a security reduction.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.Nebula.BitSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics
open Nightstream.Implementation.Lowering.Nebula.SourceSemantics
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Little-endian natural value carried by a consecutive bit range. -/
def bitsValue (assignment : Nat -> F) (start : Nat) : Nat -> Nat
  | 0 => 0
  | width + 1 =>
      bitsValue assignment start width +
        2 ^ width * (assignment (start + width)).val

def CanonicalBits
    (assignment : Nat -> F) (start width : Nat) : Prop :=
  forall offset, offset < width ->
    assignment (start + offset) = 0 ∨ assignment (start + offset) = 1

private def baseNoZeroDivisors : NormRange.BaseFieldNoZeroDivisors :=
  NormRange.baseFieldNoZeroDivisors_of_modulusEuclid
    GoldilocksField.goldilocks_euclidPrime

@[simp] theorem fieldOfNat_zero : Compiler.fieldOfNat 0 = (0 : F) := by
  rfl

@[simp] theorem fieldOfNat_one : Compiler.fieldOfNat 1 = (1 : F) := by
  rfl

theorem fieldOfNat_add (left right : Nat) :
    Compiler.fieldOfNat (left + right) =
      Compiler.fieldOfNat left + Compiler.fieldOfNat right := by
  apply Fin.ext
  simp [Compiler.fieldOfNat, Fin.val_add, Nat.add_mod]

theorem fieldOfNat_mul (left right : Nat) :
    Compiler.fieldOfNat (left * right) =
      Compiler.fieldOfNat left * Compiler.fieldOfNat right := by
  apply Fin.ext
  simp [Compiler.fieldOfNat, Fin.val_mul, Nat.mul_mod]

theorem fieldOfNat_val_of_lt (value : Nat)
    (bound : value < goldilocksModulus) :
    (Compiler.fieldOfNat value).val = value := by
  exact Nat.mod_eq_of_lt bound

/-- A physical bit row has only the two canonical Goldilocks roots. -/
theorem isBit_iff_zero_or_one
    (assignment : Nat -> F) (column : Nat) :
    IsBit assignment column ↔
      assignment column = 0 ∨ assignment column = 1 := by
  let value := assignment column
  have factored :
      value * (value - 1) = value * value + -value := by
    rw [Fin.sub_eq_add_neg, Lean.Grind.Fin.left_distrib]
    have negOne : value * (-1) = -value := by
      calc
        value * (-1) = (-1) * value := Fin.mul_comm _ _
        _ = -(1 * value) := Lean.Grind.Fin.neg_mul _ _
        _ = -value := by rw [Fin.one_mul]
    rw [negOne]
  constructor
  · intro bit
    have zeroProduct : value * (value - 1) = 0 := by
      rw [factored]
      exact bit
    rcases baseNoZeroDivisors value (value - 1) zeroProduct with
      zero | one
    · exact Or.inl zero
    · exact Or.inr (Lean.Grind.AddCommGroup.sub_eq_zero_iff.mp one)
  · intro root
    rcases root with zero | one
    · change assignment column * assignment column + -assignment column = 0
      simp [zero]
    · change assignment column * assignment column + -assignment column = 0
      simp [one]

theorem canonicalBits_of_isBit
    (assignment : Nat -> F) (start width : Nat)
    (bits : forall offset, offset < width ->
      IsBit assignment (start + offset)) :
    CanonicalBits assignment start width := by
  intro offset offsetBound
  exact (isBit_iff_zero_or_one assignment (start + offset)).mp
    (bits offset offsetBound)

theorem fieldTwoPower_val (exponent : Nat) :
    (Rows.LinearCombination.fieldTwoPower exponent).val =
      2 ^ exponent % goldilocksModulus := by
  induction exponent with
  | zero => rfl
  | succ exponent inductionHypothesis =>
      simp only [Rows.LinearCombination.fieldTwoPower, Fin.val_mul,
        inductionHypothesis, pow_succ]
      exact Nat.mod_mul_mod _ _ _

theorem eval_word_succ
    (assignment : Nat -> F) (start width : Nat) :
    Rows.LinearCombination.eval assignment
        (Rows.LinearCombination.word start (width + 1)) =
      Rows.LinearCombination.eval assignment
          (Rows.LinearCombination.word start width) +
        Rows.LinearCombination.fieldTwoPower width *
          assignment (start + width) := by
  rw [show Rows.LinearCombination.word start (width + 1) =
      Rows.LinearCombination.add
        (Rows.LinearCombination.word start width)
        [{ column := start + width
           coefficient := Rows.LinearCombination.fieldTwoPower width }] by
    simp [Rows.LinearCombination.word, Rows.LinearCombination.wordScaled,
      Rows.LinearCombination.add, List.range_succ]]
  rw [Rows.LinearCombination.eval_add]
  simp [Rows.LinearCombination.eval]

theorem eval_word_val
    (assignment : Nat -> F) (start width : Nat) :
    (Rows.LinearCombination.eval assignment
      (Rows.LinearCombination.word start width)).val =
        bitsValue assignment start width % goldilocksModulus := by
  induction width with
  | zero => rfl
  | succ width inductionHypothesis =>
      rw [eval_word_succ]
      simp only [Fin.val_add, Fin.val_mul, inductionHypothesis,
        fieldTwoPower_val, bitsValue]
      simp [Nat.add_mod, Nat.mul_mod]

/-- If the little-endian source carries `value`, the emitted word evaluates
to the canonical field embedding of `value`. This statement also covers
64-bit extension limbs, where reduction can be necessary. -/
theorem eval_word_eq_fieldOfNat_of_bitsValue
    (assignment : Nat -> F) (start width value : Nat)
    (exact : bitsValue assignment start width = value) :
    Rows.LinearCombination.eval assignment
        (Rows.LinearCombination.word start width) =
      Compiler.fieldOfNat value := by
  apply Fin.eq_of_val_eq
  rw [eval_word_val, exact]
  rfl

theorem bitValue_le_one
    (assignment : Nat -> F) (start width offset : Nat)
    (canonical : CanonicalBits assignment start width)
    (offsetBound : offset < width) :
    (assignment (start + offset)).val <= 1 := by
  rcases canonical offset offsetBound with zero | one
  · rw [zero]
    decide
  · rw [one]
    decide

theorem bitsValue_lt_twoPower
    (assignment : Nat -> F) (start width : Nat)
    (canonical : CanonicalBits assignment start width) :
    bitsValue assignment start width < 2 ^ width := by
  induction width with
  | zero => simp [bitsValue]
  | succ width inductionHypothesis =>
      have prefixCanonical : CanonicalBits assignment start width := by
        intro offset offsetBound
        exact canonical offset (Nat.lt_succ_of_lt offsetBound)
      have prior := inductionHypothesis prefixCanonical
      have positive : 0 < 2 ^ width := pow_pos (by decide) width
      have oneModulus : 1 % goldilocksModulus = 1 := by decide
      rcases canonical width (Nat.lt_succ_self width) with zero | one
      · simp [bitsValue, zero, pow_succ]
        omega
      · simp [bitsValue, one, pow_succ, oneModulus]
        omega

/-- A canonical bit word below the modulus evaluates to its unreduced natural
value. -/
theorem eval_word_val_exact
    (assignment : Nat -> F) (start width : Nat)
    (canonical : CanonicalBits assignment start width)
    (fits : 2 ^ width <= goldilocksModulus) :
    (Rows.LinearCombination.eval assignment
      (Rows.LinearCombination.word start width)).val =
        bitsValue assignment start width := by
  rw [eval_word_val]
  apply Nat.mod_eq_of_lt
  exact Nat.lt_of_lt_of_le
    (bitsValue_lt_twoPower assignment start width canonical) fits

end Nightstream.Implementation.Lowering.Nebula.BitSemantics
