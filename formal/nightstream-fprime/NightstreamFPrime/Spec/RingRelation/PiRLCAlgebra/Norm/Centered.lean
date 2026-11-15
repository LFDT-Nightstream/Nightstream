import NightstreamFPrime.Spec.RingStrongSet

namespace NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Norm.Centered

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RingStrongSet

def halfModulus : Nat := fieldModulus / 2

def distance (value : Nat) : Nat :=
  if value <= halfModulus then value else fieldModulus - value

theorem modulus_eq_two_half_add_one :
    fieldModulus = 2 * halfModulus + 1 := by
  decide

theorem centeredMagnitude_eq_distance (value : F) :
    centeredMagnitude value = distance value.val := by
  unfold centeredMagnitude distance
  by_cases low : value.val <= halfModulus
  · have selfLe : value.val <= fieldModulus - value.val := by
      unfold halfModulus fieldModulus at low
      unfold fieldModulus
      omega
    simp [low, selfLe]
  · have complementLe : fieldModulus - value.val <= value.val := by
      unfold halfModulus fieldModulus at low
      unfold fieldModulus
      omega
    simp [low, complementLe]

@[simp] theorem centeredMagnitude_zero : centeredMagnitude (0 : F) = 0 := by
  simp [centeredMagnitude]

private theorem distance_le_self {value : Nat}
    (valueLt : value < fieldModulus) :
    distance value <= value := by
  unfold distance
  by_cases low : value <= halfModulus
  · simp [low]
  · have modulusLeTwice : fieldModulus <= 2 * value := by
      rw [modulus_eq_two_half_add_one]
      omega
    have complementLe : fieldModulus - value <= value := by omega
    simp [low, complementLe]

private theorem distance_le_complement {value : Nat}
    (valueLt : value < fieldModulus) :
    distance value <= fieldModulus - value := by
  unfold distance
  by_cases low : value <= halfModulus
  · have twiceLeModulus : 2 * value <= fieldModulus := by
      rw [modulus_eq_two_half_add_one]
      omega
    have selfLe : value <= fieldModulus - value := by omega
    simp [low, selfLe]
  · simp [low]

private theorem distance_sub_triangle
    (left right : Nat)
    (leftLt : left < fieldModulus)
    (rightLt : right < fieldModulus) :
    distance ((left + fieldModulus - right) % fieldModulus) <=
      distance left + distance right := by
  by_cases rightLeLeft : right <= left
  · have reduced :
        ((left + fieldModulus - right) % fieldModulus) =
          left - right := by
      have decompose : left + fieldModulus - right =
          fieldModulus + (left - right) := by omega
      calc
        ((left + fieldModulus - right) % fieldModulus) =
            ((fieldModulus + (left - right)) % fieldModulus) := by
              rw [decompose]
        _ = (left - right) % fieldModulus := by
          simp
        _ = left - right := by
          exact Nat.mod_eq_of_lt
            (Nat.lt_of_le_of_lt (Nat.sub_le _ _) leftLt)
    by_cases leftLow : left <= halfModulus
    · have rightLow : right <= halfModulus := Nat.le_trans rightLeLeft leftLow
      have resultLe : distance (left - right) <= left - right :=
        distance_le_self (Nat.lt_of_le_of_lt (Nat.sub_le _ _) leftLt)
      calc
        distance ((left + fieldModulus - right) % fieldModulus) =
            distance (left - right) := by rw [reduced]
        _ <= left - right := resultLe
        _ <= left + right := by omega
        _ = distance left + distance right := by
          simp [distance, leftLow, rightLow]
    · by_cases rightLow : right <= halfModulus
      · have resultLe : distance (left - right) <=
            fieldModulus - (left - right) :=
          distance_le_complement
            (Nat.lt_of_le_of_lt (Nat.sub_le _ _) leftLt)
        calc
          distance ((left + fieldModulus - right) % fieldModulus) =
              distance (left - right) := by rw [reduced]
          _ <= fieldModulus - (left - right) := resultLe
          _ = (fieldModulus - left) + right := by omega
          _ = distance left + distance right := by
            simp [distance, leftLow, rightLow]
      · have resultLe : distance (left - right) <= left - right :=
          distance_le_self
            (Nat.lt_of_le_of_lt (Nat.sub_le _ _) leftLt)
        have differenceLe :
            left - right <=
              (fieldModulus - left) +
                (fieldModulus - right) := by
          omega
        calc
          distance ((left + fieldModulus - right) % fieldModulus) =
              distance (left - right) := by rw [reduced]
          _ <= left - right := resultLe
          _ <= (fieldModulus - left) +
              (fieldModulus - right) := differenceLe
          _ = distance left + distance right := by
            simp [distance, leftLow, rightLow]
  · have leftLtRight : left < right := Nat.lt_of_not_ge rightLeLeft
    have reduced :
        ((left + fieldModulus - right) % fieldModulus) =
          fieldModulus + left - right := by
      have resultLt : left + fieldModulus - right < fieldModulus := by
        omega
      have rearrange : left + fieldModulus - right =
          fieldModulus + left - right := by omega
      calc
        ((left + fieldModulus - right) % fieldModulus) =
            left + fieldModulus - right :=
          Nat.mod_eq_of_lt resultLt
        _ = fieldModulus + left - right := rearrange
    by_cases leftLow : left <= halfModulus
    · by_cases rightLow : right <= halfModulus
      · have resultLe : distance (fieldModulus + left - right) <=
            fieldModulus -
              (fieldModulus + left - right) :=
          distance_le_complement (by omega)
        calc
          distance ((left + fieldModulus - right) % fieldModulus) =
              distance (fieldModulus + left - right) := by rw [reduced]
          _ <= fieldModulus -
              (fieldModulus + left - right) := resultLe
          _ <= left + right := by omega
          _ = distance left + distance right := by
            simp [distance, leftLow, rightLow]
      · have resultLe : distance (fieldModulus + left - right) <=
            fieldModulus + left - right :=
          distance_le_self (by omega)
        calc
          distance ((left + fieldModulus - right) % fieldModulus) =
              distance (fieldModulus + left - right) := by rw [reduced]
          _ <= fieldModulus + left - right := resultLe
          _ = left + (fieldModulus - right) := by omega
          _ = distance left + distance right := by
            simp [distance, leftLow, rightLow]
    · have rightLow : ¬ right <= halfModulus := by
        intro rightLow
        exact leftLow (Nat.le_trans (Nat.le_of_lt leftLtRight) rightLow)
      have resultLe : distance (fieldModulus + left - right) <=
          fieldModulus - (fieldModulus + left - right) :=
        distance_le_complement (by omega)
      have resultBound :
          fieldModulus - (fieldModulus + left - right) <=
            (fieldModulus - left) +
              (fieldModulus - right) := by
        omega
      calc
        distance ((left + fieldModulus - right) % fieldModulus) =
            distance (fieldModulus + left - right) := by rw [reduced]
        _ <= fieldModulus -
            (fieldModulus + left - right) := resultLe
        _ <= (fieldModulus - left) +
            (fieldModulus - right) := resultBound
        _ = distance left + distance right := by
          simp [distance, leftLow, rightLow]

private theorem val_sub (left right : F) :
    (left - right).val =
      (left.val + fieldModulus - right.val) % fieldModulus := by
  have rearrange : fieldModulus + left.val - right.val =
      left.val + (fieldModulus - right.val) := by omega
  calc
    (left - right).val =
        (left.val + (fieldModulus - right.val)) %
          fieldModulus := by
      simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using
        Fin.val_sub left right
    _ = (fieldModulus + left.val - right.val) %
          fieldModulus := by rw [rearrange]
    _ = (left.val + fieldModulus - right.val) %
          fieldModulus := by
      simp [Nat.add_comm]

theorem centeredMagnitude_sub_le (left right : F) :
    centeredMagnitude (left - right) <=
      centeredMagnitude left + centeredMagnitude right := by
  rw [centeredMagnitude_eq_distance, val_sub,
    centeredMagnitude_eq_distance, centeredMagnitude_eq_distance]
  exact distance_sub_triangle left.val right.val left.isLt right.isLt

private theorem zero_sub_eq_neg (value : F) : (0 : F) - value = -value := by
  rw [Fin.sub_eq_add_neg]
  calc
    (0 : F) + -value = -value + 0 := Lean.Grind.Fin.add_comm _ _
    _ = -value := Lean.Grind.Fin.add_zero _

private theorem neg_neg_eq (value : F) : -(-value) = value := by
  exact Lean.Grind.AddCommGroup.neg_neg value

@[simp] theorem centeredMagnitude_neg (value : F) :
    centeredMagnitude (-value) = centeredMagnitude value := by
  apply Nat.le_antisymm
  · have triangle := centeredMagnitude_sub_le (0 : F) value
    simpa [zero_sub_eq_neg, centeredMagnitude_zero] using triangle
  · have triangle := centeredMagnitude_sub_le (0 : F) (-value)
    have restore : (0 : F) - (-value) = value := by
      calc
        (0 : F) - (-value) = -(-value) := zero_sub_eq_neg (-value)
        _ = value := neg_neg_eq value
    simpa [restore, centeredMagnitude_zero] using triangle

theorem centeredMagnitude_add_le (left right : F) :
    centeredMagnitude (left + right) <=
      centeredMagnitude left + centeredMagnitude right := by
  have triangle := centeredMagnitude_sub_le left (-right)
  have subtractNegative : left - (-right) = left + right := by
    calc
      left - (-right) = left + -(-right) := Fin.sub_eq_add_neg left (-right)
      _ = left + right := by rw [neg_neg_eq]
  simpa [subtractNegative, centeredMagnitude_neg] using triangle

private theorem embedCoefficient_cases (coefficient : Coefficient) :
    embedCoefficient coefficient = (0 : F) - 2 \/
    embedCoefficient coefficient = (0 : F) - 1 \/
    embedCoefficient coefficient = 0 \/
    embedCoefficient coefficient = 1 \/
    embedCoefficient coefficient = 2 := by
  revert coefficient
  decide

private theorem zero_sub_mul (factor value : F) :
    ((0 : F) - factor) * value = -(factor * value) := by
  calc
    ((0 : F) - factor) * value = (-factor) * value := by
      rw [Fin.sub_eq_add_neg]
      simp
    _ = -(factor * value) := Lean.Grind.Fin.neg_mul factor value

private theorem two_mul (value : F) :
    (2 : F) * value = value + value := by
  have twoEq : (2 : F) = 1 + 1 := by decide
  rw [twoEq]
  calc
    ((1 : F) + 1) * value = value * ((1 : F) + 1) :=
      Lean.Grind.Fin.mul_comm _ _
    _ = value * 1 + value * 1 := Lean.Grind.Fin.left_distrib _ _ _
    _ = value + value := by
      congr 1 <;> exact Lean.Grind.Fin.mul_one value

theorem embedCoefficient_mul_le_two
    (coefficient : Coefficient) (value : F) :
    centeredMagnitude (embedCoefficient coefficient * value) <=
      2 * centeredMagnitude value := by
  rcases embedCoefficient_cases coefficient with
    negativeTwo | negativeOne | zero | one | two
  · rw [negativeTwo, zero_sub_mul, centeredMagnitude_neg, two_mul]
    simpa [Nat.two_mul] using centeredMagnitude_add_le value value
  · rw [negativeOne, zero_sub_mul, centeredMagnitude_neg]
    have oneMul : (1 : F) * value = value := Fin.one_mul value
    rw [oneMul]
    omega
  · rw [zero, Fin.zero_mul, centeredMagnitude_zero]
    omega
  · rw [one, Fin.one_mul]
    omega
  · rw [two, two_mul]
    simpa [Nat.two_mul] using centeredMagnitude_add_le value value

end NightstreamFPrime.Spec.RingRelation.PiRLCAlgebra.Norm.Centered
