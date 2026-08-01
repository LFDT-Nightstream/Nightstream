import Nightstream.Implementation.Lowering.Nebula.BitSemantics

/-!
Canonical little-endian witness construction for Nebula words.

Assurance tier: model-level.

Owns the bit value at one position and reconstruction of any finite word from
those positions. It does not own a physical column layout, a memory trace,
products, application ports, Rust, or a security reduction.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Nebula.BitWitness

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics
open Nightstream.Implementation.Lowering.Nebula.BitSemantics

def bitField (value offset : Nat) : F :=
  if value.testBit offset then 1 else 0

theorem bitField_eq_fieldOfNat (value offset : Nat) :
    bitField value offset =
      Compiler.fieldOfNat ((value / 2 ^ offset) % 2) := by
  rw [← Nat.toNat_testBit]
  cases bitExact : value.testBit offset <;>
    simp [bitField, bitExact]

@[simp] theorem bitField_zero (value : Nat) :
    bitField value 0 = Compiler.fieldOfNat (value % 2) := by
  simpa using bitField_eq_fieldOfNat value 0

theorem bitField_isBit (value offset : Nat) :
    IsBit (fun _ => bitField value offset) 0 := by
  unfold IsBit bitField
  split <;> rfl

theorem bitField_zero_or_one (value offset : Nat) :
    bitField value offset = 0 ∨ bitField value offset = 1 := by
  unfold bitField
  split <;> simp_all

@[simp] theorem fieldOfNat_finVal (value : F) :
    Compiler.fieldOfNat value.val = value := by
  apply Fin.ext
  exact Nat.mod_eq_of_lt value.isLt

theorem canonical_of_get
    (assignment : Nat -> F) (start width value : Nat)
    (get : forall offset, offset < width ->
      assignment (start + offset) = bitField value offset) :
    CanonicalBits assignment start width := by
  intro offset offsetBound
  rw [get offset offsetBound]
  exact bitField_zero_or_one value offset

theorem isBit_of_get
    (assignment : Nat -> F) (start width value : Nat)
    (get : forall offset, offset < width ->
      assignment (start + offset) = bitField value offset)
    (offset : Nat) (offsetBound : offset < width) :
    IsBit assignment (start + offset) := by
  unfold IsBit
  rw [get offset offsetBound]
  unfold bitField
  split <;> rfl

/-- Exact reconstruction of the first `width` bits. -/
theorem bitsValue_of_get
    (assignment : Nat -> F) (start width value : Nat)
    (get : forall offset, offset < width ->
      assignment (start + offset) = bitField value offset) :
    bitsValue assignment start width = value % 2 ^ width := by
  induction width with
  | zero =>
      change 0 = value % 1
      exact (Nat.mod_one value).symm
  | succ width inductionHypothesis =>
      rw [bitsValue, get width (Nat.lt_succ_self width),
        bitField_eq_fieldOfNat]
      have digitLtTwo : (value / 2 ^ width) % 2 < 2 :=
        Nat.mod_lt _ (by decide)
      have digitLtModulus : (value / 2 ^ width) % 2 < goldilocksModulus :=
        Nat.lt_trans digitLtTwo (by decide)
      rw [fieldOfNat_val_of_lt _ digitLtModulus]
      rw [inductionHypothesis (fun offset offsetBound =>
        get offset (Nat.lt_succ_of_lt offsetBound))]
      exact Nat.mod_pow_succ.symm

theorem bitsValue_exact_of_get
    (assignment : Nat -> F) (start width value : Nat)
    (get : forall offset, offset < width ->
      assignment (start + offset) = bitField value offset)
    (bound : value < 2 ^ width) :
    bitsValue assignment start width = value := by
  rw [bitsValue_of_get assignment start width value get,
    Nat.mod_eq_of_lt bound]

theorem eval_word_of_get
    (assignment : Nat -> F) (start width value : Nat)
    (get : forall offset, offset < width ->
      assignment (start + offset) = bitField value offset) :
    Nightstream.Implementation.Lowering.Nebula.Rows.LinearCombination.eval
        assignment
        (Nightstream.Implementation.Lowering.Nebula.Rows.LinearCombination.word
          start width) =
      Compiler.fieldOfNat (value % 2 ^ width) := by
  apply eval_word_eq_fieldOfNat_of_bitsValue
  exact bitsValue_of_get assignment start width value get

theorem eval_word_exact_of_get
    (assignment : Nat -> F) (start width value : Nat)
    (get : forall offset, offset < width ->
      assignment (start + offset) = bitField value offset)
    (bound : value < 2 ^ width) :
    Nightstream.Implementation.Lowering.Nebula.Rows.LinearCombination.eval
        assignment
        (Nightstream.Implementation.Lowering.Nebula.Rows.LinearCombination.word
          start width) =
      Compiler.fieldOfNat value := by
  rw [eval_word_of_get assignment start width value get,
    Nat.mod_eq_of_lt bound]

end Nightstream.Implementation.Lowering.Nebula.BitWitness
