import NightstreamFPrime.Export.Generic.PackedBoolean

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.PackedProduct

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.RingRelation.PiDECAlgebra.Radix
open Fin.CommRing

def capacity : Nat := 24

structure Values where
  left : F
  right : F
  output : F
  parity : F

def Values.residual (values : Values) : F :=
  values.left + values.right - 2 * values.output - values.parity

def Values.Norm (values : Values) : Prop :=
  centeredMagnitude values.left < 2 ∧ centeredMagnitude values.right < 2 ∧
    centeredMagnitude values.output < 2 ∧ LowNormSlot.Valid .bit values.parity

def complementDigit (value : F) : Nat := if value = 0 then 1 else 0

theorem complementDigit_le (value : F) : complementDigit value ≤ 1 := by
  unfold complementDigit
  split <;> omega

theorem complementDigit_field {value : F} (bit : LowNormSlot.Valid .bit value) :
    fieldOfNat (complementDigit value) = 1 - value := by
  rcases bit with rfl | rfl <;> decide

theorem negative_unitDigit_field {value : F} (norm : centeredMagnitude value < 2) :
    fieldOfNat (PackedBoolean.unitDigit (-value)) = 1 - value := by
  rcases PackedBoolean.unit_cases norm with rfl | rfl | rfl <;> decide

def Values.digit (values : Values) : Nat :=
  PackedBoolean.unitDigit values.left + PackedBoolean.unitDigit values.right +
    2 * PackedBoolean.unitDigit (-values.output) + complementDigit values.parity

theorem Values.digit_le (values : Values) : values.digit ≤ 9 := by
  have := PackedBoolean.unitDigit_le values.left
  have := PackedBoolean.unitDigit_le values.right
  have := PackedBoolean.unitDigit_le (-values.output)
  have := complementDigit_le values.parity
  unfold Values.digit
  omega

theorem Values.digit_field (values : Values) (norm : values.Norm) :
    fieldOfNat values.digit = values.residual + 5 := by
  rw [Values.digit, fieldOfNat_add, fieldOfNat_add, fieldOfNat_add, fieldOfNat_mul,
    PackedBoolean.unitDigit_field norm.1, PackedBoolean.unitDigit_field norm.2.1,
    negative_unitDigit_field norm.2.2.1, complementDigit_field norm.2.2.2]
  change values.left + 1 + (values.right + 1) + 2 * (1 - values.output) +
    (1 - values.parity) = values.residual + 5
  unfold Values.residual
  ring

def target : Nat → Nat
  | 0 => 0
  | n + 1 => 5 + 6 * target n

def packNat : List Nat → Nat
  | [] => 0
  | x :: xs => x + 6 * packNat xs

def pack : List Values → F
  | [] => 0
  | values :: rest => values.residual + 6 * pack rest

theorem target_mono : Monotone target := by
  apply monotone_nat_of_le_succ
  intro n
  simp only [target]
  omega

theorem packNat_bound (digits : List Nat) (small : ∀ d ∈ digits, d ≤ 9) :
    packNat digits ≤ 2 * target digits.length := by
  induction digits with
  | nil => simp [packNat, target]
  | cons d ds ih =>
    have head := small d List.mem_cons_self
    have tail := ih (fun x member => small x (List.mem_cons_of_mem _ member))
    simp only [packNat, List.length_cons, target]
    omega

theorem target_capacity_lt_modulus : 2 * target capacity < fieldModulus := by
  decide

theorem packNat_eq_target (digits : List Nat) (small : ∀ d ∈ digits, d ≤ 9)
    (equal : packNat digits = target digits.length) : ∀ d ∈ digits, d = 5 := by
  induction digits with
  | nil => simp
  | cons d ds ih =>
    have head := small d List.mem_cons_self
    simp only [packNat, List.length_cons, target] at equal
    have five : d = 5 := by omega
    have rest : packNat ds = target ds.length := by omega
    intro value member
    rcases List.mem_cons.mp member with rfl | member
    · exact five
    · exact ih (fun x inside => small x (List.mem_cons_of_mem _ inside)) rest value member

theorem pack_field (values : List Values) (norm : ∀ value ∈ values, value.Norm) :
    fieldOfNat (packNat (values.map Values.digit)) =
      pack values + fieldOfNat (target values.length) := by
  induction values with
  | nil => simp [packNat, pack, target]
  | cons value rest ih =>
    have first := value.digit_field (norm value List.mem_cons_self)
    have remaining := ih (fun v member => norm v (List.mem_cons_of_mem _ member))
    simp only [List.map_cons, List.length_cons, packNat, target, fieldOfNat_add, fieldOfNat_mul]
    rw [first, remaining]
    change value.residual + 5 + 6 * (pack rest + fieldOfNat (target rest.length)) =
      value.residual + 6 * pack rest + (5 + 6 * fieldOfNat (target rest.length))
    ring

theorem packed_implies_residuals (values : List Values) (count : values.length ≤ capacity)
    (norm : ∀ value ∈ values, value.Norm) (zero : pack values = 0) :
    ∀ value ∈ values, value.residual = 0 := by
  have small : ∀ d ∈ values.map Values.digit, d ≤ 9 := by
    intro d member
    obtain ⟨value, _, rfl⟩ := List.mem_map.mp member
    exact value.digit_le
  have bound := packNat_bound (values.map Values.digit) small
  have limit := target_mono count
  have maximum := target_capacity_lt_modulus
  have packedBound : packNat (values.map Values.digit) < fieldModulus := by
    simp only [List.length_map] at bound
    exact lt_of_le_of_lt bound (lt_of_le_of_lt (Nat.mul_le_mul_left 2 limit) maximum)
  have targetBound : target values.length < fieldModulus := by omega
  have field := pack_field values norm
  rw [zero, zero_add] at field
  have equal : packNat (values.map Values.digit) = target values.length := by
    have := congrArg Fin.val field
    simpa only [fieldOfNat, Nat.mod_eq_of_lt packedBound, Nat.mod_eq_of_lt targetBound] using this
  have allFive := packNat_eq_target (values.map Values.digit) small (by simpa using equal)
  intro value member
  have digit := allFive _ (List.mem_map.mpr ⟨value, member, rfl⟩)
  have result := value.digit_field (norm value member)
  rw [digit] at result
  change 5 = value.residual + 5 at result
  exact add_right_cancel (result.symm.trans (zero_add 5).symm)

theorem residuals_imply_packed (values : List Values)
    (zero : ∀ value ∈ values, value.residual = 0) : pack values = 0 := by
  induction values with
  | nil => rfl
  | cons value rest ih =>
    have first := zero value List.mem_cons_self
    have remaining := ih (fun v member => zero v (List.mem_cons_of_mem _ member))
    simp only [pack, first, remaining, mul_zero, add_zero]

theorem half_adder {left right output parity : F}
    (leftBit : LowNormSlot.Valid .bit left) (rightBit : LowNormSlot.Valid .bit right)
    (outputNorm : centeredMagnitude output < 2) (parityBit : LowNormSlot.Valid .bit parity)
    (holds : left + right - 2 * output - parity = 0) :
    output = left * right ∧ parity = left + right - 2 * (left * right) := by
  rcases leftBit with rfl | rfl <;> rcases rightBit with rfl | rfl <;>
    rcases PackedBoolean.unit_cases outputNorm with rfl | rfl | rfl <;>
    rcases parityBit with rfl | rfl
  all_goals revert holds; decide

theorem honest_parity {left right : F}
    (leftBit : LowNormSlot.Valid .bit left) (rightBit : LowNormSlot.Valid .bit right) :
    LowNormSlot.Valid .bit (left + right - 2 * (left * right)) := by
  rcases leftBit with rfl | rfl <;> rcases rightBit with rfl | rfl <;>
    unfold LowNormSlot.Valid <;> decide

structure Gate where
  row : Nat
  left : Nat
  right : Nat
  output : Nat
  parity : Nat

def Gate.values (gate : Gate) (env : Env) : Values :=
  ⟨env gate.left, env gate.right, env gate.output, env gate.parity⟩

def Gate.Matches (gate : Gate) (row : R1CS.Row) : Prop :=
  ∀ env, row.Holds env ↔ env gate.left * env gate.right = env gate.output

def Gate.Ready (gate : Gate) (original seeds : List R1CS.Row) : Prop :=
  BooleanInference.Derived (seeds ++ original.take gate.row) gate.left ∧
    BooleanInference.Derived (seeds ++ original.take gate.row) gate.right

theorem restore_rows (original kept seeds : List R1CS.Row) (gates : List Gate)
    (cover : ∀ i (inside : i < original.length), original[i] ∈ kept ∨
      ∃ gate ∈ gates, gate.row = i ∧ gate.Matches original[i])
    (ready : ∀ gate ∈ gates, gate.Ready original seeds)
    {env : Env} (one : env 0 = 1)
    (seedsHold : R1CS.RowsHold env seeds) (keptHold : R1CS.RowsHold env kept)
    (norm : ∀ gate ∈ gates, (gate.values env).Norm)
    (residuals : ∀ gate ∈ gates, (gate.values env).residual = 0) :
    R1CS.RowsHold env original := by
  have holdsAt : ∀ i (inside : i < original.length), original[i].Holds env := by
    intro i
    induction i using Nat.strong_induction_on with
    | h i ih =>
      intro inside
      rcases cover i inside with member | ⟨gate, member, position, matched⟩
      · exact keptHold _ member
      · have before : R1CS.RowsHold env (seeds ++ original.take i) := by
          apply (R1CS.rowsHold_append env seeds (original.take i)).mpr
          refine ⟨seedsHold, ?_⟩
          change ∀ row ∈ original.take i, row.Holds env
          refine (List.forall_mem_iff_getElem (l := original.take i)
            (p := fun row => row.Holds env)).mpr ?_
          intro j hj
          have bounds : j < i ∧ j < original.length := by
            simpa only [List.length_take, Nat.lt_min] using hj
          simpa only [List.getElem_take] using ih j bounds.1 bounds.2
        obtain ⟨left, right⟩ := ready gate member
        rw [position] at left right
        have leftBit := BooleanInference.derived_valid left one before
        have rightBit := BooleanInference.derived_valid right one before
        have bounded := norm gate member
        have result := half_adder leftBit rightBit bounded.2.2.1 bounded.2.2.2
          (residuals gate member)
        exact (matched env).mpr result.1.symm
  intro row member
  obtain ⟨i, inside, rfl⟩ := List.mem_iff_getElem.mp member
  exact holdsAt i inside

theorem honest_values {left right output : F}
    (leftBit : LowNormSlot.Valid .bit left) (rightBit : LowNormSlot.Valid .bit right)
    (product : output = left * right) :
    (Values.mk left right output (left + right - 2 * output)).Norm ∧
      (Values.mk left right output (left + right - 2 * output)).residual = 0 := by
  subst output
  rcases leftBit with rfl | rfl <;> rcases rightBit with rfl | rfl <;>
    unfold Values.Norm Values.residual LowNormSlot.Valid <;> decide

end NightstreamFPrime.Export.Generic.PackedProduct
