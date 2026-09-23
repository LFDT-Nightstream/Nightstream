import NightstreamFPrime.Gadgets.Sampling.WideReduction.HintProgram
import NightstreamFPrime.Gadgets.Sampling.WideReduction.Values

/-! The biased check expression computes an integer quotient. The bias
makes subtraction nonnegative, and both sides stay below the field modulus.
No modular representative is used as an integer without these bounds. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.CheckQuotient

open NightstreamFPrime.Spec NightstreamFPrime.Circuit

def left (env : Env) (offset : Nat) (check : Fin checkCount) : Nat :=
  linearValue env (reduceTerms (modulus check) (drawTerms offset)) + modulus check * checkBias

def right (env : Env) (offset : Nat) (check : Fin checkCount) : Nat :=
  linearValue env (reduceTerms (modulus check) (resultTerms offset))

theorem integer_bounds (env : Env) (offset : Nat) (check : Fin checkCount)
    (drawBits : ∀ term ∈ drawTerms offset, (term.2.eval env).val ≤ 1)
    (resultBits : ∀ term ∈ resultTerms offset, (term.2.eval env).val ≤ 1)
    (sameInteger : linearValue env (drawTerms offset) = linearValue env (resultTerms offset)) :
    right env offset check ≤ left env offset check ∧
      left env offset check < goldilocksModulus ∧
      modulus check ∣ left env offset check - right env offset check ∧
      (left env offset check - right env offset check) / modulus check < 549 := by
  let m := modulus check
  have positive := modulus_pos check
  have small := modulus_lt check
  have lower : 293 * m ≤ left env offset check := by
    unfold left checkBias quotientBitCount digitCount digitBitCount
    omega
  have upper : left env offset check ≤ 256 * (m - 1) + 293 * m := by
    have bound := linearValue_le env (reduceTerms m (drawTerms offset)) (m - 1)
      (reduceTerms_bound m positive _) (reduceTerms_bits env m _ drawBits)
    rw [reduceTerms_length, length_drawTerms] at bound
    dsimp only [m] at bound ⊢
    unfold left checkBias quotientBitCount digitCount digitBitCount
    omega
  have rightUpper : right env offset check ≤ 293 * (m - 1) := by
    have bound := linearValue_le env (reduceTerms m (resultTerms offset)) (m - 1)
      (reduceTerms_bound m positive _) (reduceTerms_bits env m _ resultBits)
    rwa [reduceTerms_length, length_resultTerms] at bound
  have ordered : right env offset check ≤ left env offset check := by omega
  have leftCongruence : left env offset check ≡ linearValue env (drawTerms offset) [MOD m] := by
    have biasZero : m * checkBias ≡ 0 [MOD m] :=
      (Nat.modEq_zero_iff_dvd).mpr (Dvd.intro _ rfl)
    simpa only [Nat.add_zero] using
      (linearValue_reduce_modEq env m (drawTerms offset)).add biasZero
  have rightCongruence : right env offset check ≡ linearValue env (drawTerms offset) [MOD m] := by
    rw [sameInteger]
    exact linearValue_reduce_modEq env m (resultTerms offset)
  refine ⟨ordered, ?_, ?_, ?_⟩
  · change m < 2 ^ 50 at small
    norm_num [goldilocksModulus] at *
    omega
  · exact (Nat.modEq_iff_dvd' ordered).mp (rightCongruence.trans leftCongruence.symm)
  · apply Nat.div_lt_of_lt_mul
    change m * _ > _
    omega

theorem modulus_inverse (check : Fin checkCount) :
    fieldOfNat (modulus check) * Hint.inverse (fieldOfNat (modulus check)) = 1 := by
  fin_cases check <;> decide

theorem field_value (env : Env) (offset : Nat) (check : Fin checkCount)
    (ordered : right env offset check ≤ left env offset check)
    (divides : modulus check ∣ left env offset check - right env offset check) :
    (HintProgram.checkQuotient offset check).eval env =
      fieldOfNat ((left env offset check - right env offset check) / modulus check) := by
  have subtraction : fieldOfNat (left env offset check) - fieldOfNat (right env offset check) =
      fieldOfNat (left env offset check - right env offset check) := by
    have add := fieldOfNat_add (left env offset check - right env offset check) (right env offset check)
    rw [Nat.sub_add_cancel ordered] at add
    exact sub_eq_iff_eq_add.mpr add.symm
  unfold HintProgram.checkQuotient
  rw [Expr.eval_hmul, Expr.eval_const, Expr.eval_sub, Expr.eval_hadd, Expr.eval_const,
    linearExpr_eval, linearExpr_eval, fieldOfNat_add]
  change (fieldOfNat (left env offset check) - fieldOfNat (right env offset check)) * _ = _
  have factor : fieldOfNat (left env offset check - right env offset check) =
      fieldOfNat (modulus check) *
        fieldOfNat ((left env offset check - right env offset check) / modulus check) := by
    rw [fieldOfNat_mul, Nat.mul_div_cancel' divides]
  rw [subtraction, factor]
  have inverse := modulus_inverse check
  change (fieldOfNat (modulus check) : ZMod goldilocksModulus) *
    fieldOfNat ((left env offset check - right env offset check) / modulus check) * _ = _
  calc
    _ = fieldOfNat ((left env offset check - right env offset check) / modulus check) *
        (fieldOfNat (modulus check) * Hint.inverse (fieldOfNat (modulus check))) := by
          rw [mul_comm (fieldOfNat (modulus check)), mul_assoc]
    _ = _ := by rw [inverse, mul_one]

end NightstreamFPrime.Gadgets.Sampling.WideReduction.CheckQuotient
