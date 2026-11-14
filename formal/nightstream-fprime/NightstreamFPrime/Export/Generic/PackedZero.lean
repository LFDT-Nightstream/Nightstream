import NightstreamFPrime.Export.Generic.PackedProduct

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.PackedZero

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.RingRelation.PiDECAlgebra.Radix
open Fin.CommRing

structure Values where
  left : F
  right : F
  sum : F

def Values.residual (values : Values) : F := values.sum - values.left - values.right

def Values.Norm (values : Values) : Prop :=
  LowNormSlot.Valid .bit values.left ∧ LowNormSlot.Valid .bit values.right ∧
    centeredMagnitude values.sum < 2

def Values.digit (values : Values) : Nat :=
  PackedBoolean.unitDigit values.sum + PackedProduct.complementDigit values.left +
    PackedProduct.complementDigit values.right

theorem Values.digit_le (values : Values) : values.digit ≤ 4 := by
  have := PackedBoolean.unitDigit_le values.sum
  have := PackedProduct.complementDigit_le values.left
  have := PackedProduct.complementDigit_le values.right
  unfold Values.digit
  omega

theorem Values.digit_field (values : Values) (norm : values.Norm) :
    fieldOfNat values.digit = values.residual + 3 := by
  rw [Values.digit, fieldOfNat_add, fieldOfNat_add,
    PackedBoolean.unitDigit_field norm.2.2,
    PackedProduct.complementDigit_field norm.1,
    PackedProduct.complementDigit_field norm.2.1]
  unfold Values.residual
  ring

def pack : List Values → F
  | [] => 0
  | values :: rest => values.residual + 4 * pack rest

theorem pack_field (values : List Values) (norm : ∀ value ∈ values, value.Norm) :
    fieldOfNat (PackedBoolean.packNat (values.map Values.digit)) =
      pack values + fieldOfNat (PackedBoolean.target values.length) := by
  induction values with
  | nil => simp [PackedBoolean.packNat, pack, PackedBoolean.target]
  | cons value rest ih =>
    have first := value.digit_field (norm value List.mem_cons_self)
    have remaining := ih (fun v member => norm v (List.mem_cons_of_mem _ member))
    simp only [List.map_cons, List.length_cons, PackedBoolean.packNat, PackedBoolean.target,
      fieldOfNat_add, fieldOfNat_mul]
    rw [first, remaining]
    change value.residual + 3 + 4 * (pack rest + fieldOfNat (PackedBoolean.target rest.length)) =
      value.residual + 4 * pack rest + (3 + 4 * fieldOfNat (PackedBoolean.target rest.length))
    ring

theorem packed_implies_residuals (values : List Values)
    (count : values.length ≤ PackedBoolean.capacity)
    (norm : ∀ value ∈ values, value.Norm) (zero : pack values = 0) :
    ∀ value ∈ values, value.residual = 0 := by
  have small : ∀ d ∈ values.map Values.digit, d ≤ 4 := by
    intro d member
    obtain ⟨value, _, rfl⟩ := List.mem_map.mp member
    exact value.digit_le
  have packedBound := PackedBoolean.packNat_below_modulus (values.map Values.digit)
    (by simpa using count) small
  have limit := PackedBoolean.target_mono count
  have maximum := PackedBoolean.target_capacity_lt_modulus
  have targetBound : PackedBoolean.target values.length < fieldModulus := by omega
  have field := pack_field values norm
  rw [zero, zero_add] at field
  have equal : PackedBoolean.packNat (values.map Values.digit) =
      PackedBoolean.target values.length := by
    have := congrArg Fin.val field
    simpa only [fieldOfNat, Nat.mod_eq_of_lt packedBound, Nat.mod_eq_of_lt targetBound] using this
  have allThree := PackedBoolean.packNat_eq_target (values.map Values.digit) small (by simpa using equal)
  intro value member
  have digit := allThree _ (List.mem_map.mpr ⟨value, member, rfl⟩)
  have result := value.digit_field (norm value member)
  rw [digit] at result
  change 3 = value.residual + 3 at result
  exact add_right_cancel (result.symm.trans (zero_add 3).symm)

theorem residuals_imply_packed (values : List Values)
    (zero : ∀ value ∈ values, value.residual = 0) : pack values = 0 := by
  induction values with
  | nil => rfl
  | cons value rest ih =>
    have first := zero value List.mem_cons_self
    have remaining := ih (fun v member => zero v (List.mem_cons_of_mem _ member))
    simp only [pack, first, remaining, mul_zero, add_zero]

theorem zero_product {left right sum : F}
    (leftBit : LowNormSlot.Valid .bit left) (rightBit : LowNormSlot.Valid .bit right)
    (sumNorm : centeredMagnitude sum < 2) (holds : sum - left - right = 0) :
    left * right = 0 ∧ LowNormSlot.Valid .bit sum := by
  rcases leftBit with rfl | rfl <;> rcases rightBit with rfl | rfl <;>
    rcases PackedBoolean.unit_cases sumNorm with rfl | rfl | rfl
  all_goals unfold LowNormSlot.Valid; revert holds; decide

theorem honest_values {left right : F}
    (leftBit : LowNormSlot.Valid .bit left) (rightBit : LowNormSlot.Valid .bit right)
    (product : left * right = 0) :
    (Values.mk left right (left + right)).Norm ∧
      (Values.mk left right (left + right)).residual = 0 := by
  rcases leftBit with rfl | rfl <;> rcases rightBit with rfl | rfl
  all_goals unfold Values.Norm Values.residual LowNormSlot.Valid; revert product; decide

end NightstreamFPrime.Export.Generic.PackedZero
