import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Challenge

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/PiRLCAlgebra/Norm/Centered.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Centered Goldilocks inequalities used by concrete Phi81 norm growth.

Protocol: SuperNeo `Pi_RLC`.
Phase: one field coefficient before quotient-ring support accounting.
Constraint family: semantic norm only; this file emits no rows.

Owns: the exact cyclic distance induced by the active `Fin q` carrier,
centered triangle inequalities, and the two-times bound for multiplication by
one production five-symbol coefficient.

Does not own: quotient-ring support, assignment folding, arity arithmetic,
Rust/R1CS refinement, or row removal.

Emits constraints: no.

Authority boundary: every inequality is derived from the executable
Goldilocks carrier and the independently defined five-symbol embedding. There
is no caller-supplied norm law.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.norm_growth.centered.distance` | `min(x,q-x)` is the cyclic distance of the canonical residue | derived | `centeredMagnitude_eq_distance` |
| `nifs.pi_rlc.verify.norm_growth.centered.triangle` | centered addition/subtraction obey the triangle inequality | derived | `centeredMagnitude_add_le`, `centeredMagnitude_sub_le` |
| `nifs.pi_rlc.verify.norm_growth.centered.symbol` | one embedded symbol in `{-2,-1,0,1,2}` expands magnitude by at most two | derived | `embedCoefficient_mul_le_two` |
-/

namespace NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81StrongSet
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet

/-- Floor of half the odd Goldilocks modulus. -/
def halfModulus : Nat := goldilocksModulus / 2

/-- Executable cyclic distance of one canonical natural residue. -/
def distance (value : Nat) : Nat :=
  if value <= halfModulus then value else goldilocksModulus - value

theorem modulus_eq_two_half_add_one :
    goldilocksModulus = 2 * halfModulus + 1 := by
  decide

/-- The active `min` definition is the same executable centered branch. -/
theorem centeredMagnitude_eq_distance (value : F) :
    centeredMagnitude value = distance value.val := by
  unfold centeredMagnitude distance
  by_cases low : value.val <= halfModulus
  · have selfLe : value.val <= goldilocksModulus - value.val := by
      unfold halfModulus goldilocksModulus at low
      unfold goldilocksModulus
      omega
    simp [low, selfLe]
  · have complementLe : goldilocksModulus - value.val <= value.val := by
      unfold halfModulus goldilocksModulus at low
      unfold goldilocksModulus
      omega
    simp [low, complementLe]

@[simp] theorem centeredMagnitude_zero : centeredMagnitude (0 : F) = 0 := by
  simp [centeredMagnitude]

private theorem distance_le_self {value : Nat}
    (valueLt : value < goldilocksModulus) :
    distance value <= value := by
  unfold distance
  by_cases low : value <= halfModulus
  · simp [low]
  · have modulusLeTwice : goldilocksModulus <= 2 * value := by
      rw [modulus_eq_two_half_add_one]
      omega
    have complementLe : goldilocksModulus - value <= value := by omega
    simp [low, complementLe]

private theorem distance_le_complement {value : Nat}
    (valueLt : value < goldilocksModulus) :
    distance value <= goldilocksModulus - value := by
  unfold distance
  by_cases low : value <= halfModulus
  · have twiceLeModulus : 2 * value <= goldilocksModulus := by
      rw [modulus_eq_two_half_add_one]
      omega
    have selfLe : value <= goldilocksModulus - value := by omega
    simp [low, selfLe]
  · simp [low]

/- The four centered-sign cases below prove the cyclic subtraction triangle
directly on canonical natural representatives. This is the only substantial
field-metric argument used by norm growth. -/
private theorem distance_sub_triangle
    (left right : Nat)
    (leftLt : left < goldilocksModulus)
    (rightLt : right < goldilocksModulus) :
    distance ((left + goldilocksModulus - right) % goldilocksModulus) <=
      distance left + distance right := by
  by_cases rightLeLeft : right <= left
  · have reduced :
        ((left + goldilocksModulus - right) % goldilocksModulus) =
          left - right := by
      have decompose : left + goldilocksModulus - right =
          goldilocksModulus + (left - right) := by omega
      calc
        ((left + goldilocksModulus - right) % goldilocksModulus) =
            ((goldilocksModulus + (left - right)) % goldilocksModulus) := by
              rw [decompose]
        _ = (left - right) % goldilocksModulus := by
          simp
        _ = left - right := by
          exact Nat.mod_eq_of_lt
            (Nat.lt_of_le_of_lt (Nat.sub_le _ _) leftLt)
    by_cases leftLow : left <= halfModulus
    · have rightLow : right <= halfModulus := Nat.le_trans rightLeLeft leftLow
      have resultLe : distance (left - right) <= left - right :=
        distance_le_self (Nat.lt_of_le_of_lt (Nat.sub_le _ _) leftLt)
      calc
        distance ((left + goldilocksModulus - right) % goldilocksModulus) =
            distance (left - right) := by rw [reduced]
        _ <= left - right := resultLe
        _ <= left + right := by omega
        _ = distance left + distance right := by
          simp [distance, leftLow, rightLow]
    · by_cases rightLow : right <= halfModulus
      · have resultLe : distance (left - right) <=
            goldilocksModulus - (left - right) :=
          distance_le_complement
            (Nat.lt_of_le_of_lt (Nat.sub_le _ _) leftLt)
        calc
          distance ((left + goldilocksModulus - right) % goldilocksModulus) =
              distance (left - right) := by rw [reduced]
          _ <= goldilocksModulus - (left - right) := resultLe
          _ = (goldilocksModulus - left) + right := by omega
          _ = distance left + distance right := by
            simp [distance, leftLow, rightLow]
      · have resultLe : distance (left - right) <= left - right :=
          distance_le_self
            (Nat.lt_of_le_of_lt (Nat.sub_le _ _) leftLt)
        have differenceLe :
            left - right <=
              (goldilocksModulus - left) +
                (goldilocksModulus - right) := by
          omega
        calc
          distance ((left + goldilocksModulus - right) % goldilocksModulus) =
              distance (left - right) := by rw [reduced]
          _ <= left - right := resultLe
          _ <= (goldilocksModulus - left) +
              (goldilocksModulus - right) := differenceLe
          _ = distance left + distance right := by
            simp [distance, leftLow, rightLow]
  · have leftLtRight : left < right := Nat.lt_of_not_ge rightLeLeft
    have reduced :
        ((left + goldilocksModulus - right) % goldilocksModulus) =
          goldilocksModulus + left - right := by
      have resultLt : left + goldilocksModulus - right < goldilocksModulus := by
        omega
      have rearrange : left + goldilocksModulus - right =
          goldilocksModulus + left - right := by omega
      calc
        ((left + goldilocksModulus - right) % goldilocksModulus) =
            left + goldilocksModulus - right :=
          Nat.mod_eq_of_lt resultLt
        _ = goldilocksModulus + left - right := rearrange
    by_cases leftLow : left <= halfModulus
    · by_cases rightLow : right <= halfModulus
      · have resultLe : distance (goldilocksModulus + left - right) <=
            goldilocksModulus -
              (goldilocksModulus + left - right) :=
          distance_le_complement (by omega)
        calc
          distance ((left + goldilocksModulus - right) % goldilocksModulus) =
              distance (goldilocksModulus + left - right) := by rw [reduced]
          _ <= goldilocksModulus -
              (goldilocksModulus + left - right) := resultLe
          _ <= left + right := by omega
          _ = distance left + distance right := by
            simp [distance, leftLow, rightLow]
      · have resultLe : distance (goldilocksModulus + left - right) <=
            goldilocksModulus + left - right :=
          distance_le_self (by omega)
        calc
          distance ((left + goldilocksModulus - right) % goldilocksModulus) =
              distance (goldilocksModulus + left - right) := by rw [reduced]
          _ <= goldilocksModulus + left - right := resultLe
          _ = left + (goldilocksModulus - right) := by omega
          _ = distance left + distance right := by
            simp [distance, leftLow, rightLow]
    · have rightLow : ¬ right <= halfModulus := by
        intro rightLow
        exact leftLow (Nat.le_trans (Nat.le_of_lt leftLtRight) rightLow)
      have resultLe : distance (goldilocksModulus + left - right) <=
          goldilocksModulus - (goldilocksModulus + left - right) :=
        distance_le_complement (by omega)
      have resultBound :
          goldilocksModulus - (goldilocksModulus + left - right) <=
            (goldilocksModulus - left) +
              (goldilocksModulus - right) := by
        omega
      calc
        distance ((left + goldilocksModulus - right) % goldilocksModulus) =
            distance (goldilocksModulus + left - right) := by rw [reduced]
        _ <= goldilocksModulus -
            (goldilocksModulus + left - right) := resultLe
        _ <= (goldilocksModulus - left) +
            (goldilocksModulus - right) := resultBound
        _ = distance left + distance right := by
          simp [distance, leftLow, rightLow]

/-- Canonical representative of subtraction in the active `Fin q` carrier. -/
private theorem val_sub (left right : F) :
    (left - right).val =
      (left.val + goldilocksModulus - right.val) % goldilocksModulus := by
  have rearrange : goldilocksModulus + left.val - right.val =
      left.val + (goldilocksModulus - right.val) := by omega
  calc
    (left - right).val =
        (left.val + (goldilocksModulus - right.val)) %
          goldilocksModulus := by
      simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using
        Fin.val_sub left right
    _ = (goldilocksModulus + left.val - right.val) %
          goldilocksModulus := by rw [rearrange]
    _ = (left.val + goldilocksModulus - right.val) %
          goldilocksModulus := by
      simp [Nat.add_comm]

/-- Centered modular subtraction obeys the triangle inequality. -/
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

/-- Centered modular addition is subtraction of an equal-magnitude negative. -/
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

/-- Exact five-case proof: multiplication by one sampled coefficient expands
centered magnitude by at most two. -/
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

end NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm.Centered
