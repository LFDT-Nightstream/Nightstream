import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFFrobenius
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet

/-!
Counted binary-power inverse candidate in the fixed Phi81 ring. Every
intermediate ring value is a stored array. The exponent and loop depth are
derived from Goldilocks and the 27th Frobenius identity. Counts measure named
operations, including natural arithmetic on at most 1728-bit exponents;
they are not machine instruction counts or elapsed time.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingPowerInverse

open NightstreamFPrime.Spec
open StoredRingArithmetic (StoredRing multiply multiplyWork one oneWork)
open RingFPolynomial (image image_mul image_one QuotientRing)
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkAlgebra (UnitWitness)

/-- Binary recursion materializes each square and each selected odd product.
The step envelope includes dispatch, exponent tests, division, parity,
recursive call, index descent, projections, branches, and return. -/
private def power : Nat → StoredRing → Nat → Result StoredRing
  | 0, _, _ =>
      let result := one ()
      ⟨result.value, result.work + 2⟩
  | fuel + 1, base, exponent =>
      if exponent = 0 then
        let result := one ()
        ⟨result.value, result.work + 5⟩
      else
        let squared := multiply base base
        let half := power fuel squared.value (exponent / 2)
        if exponent % 2 = 1 then
          let result := multiply half.value base
          ⟨result.value, squared.work + half.work + result.work + 16⟩
        else ⟨half.value, squared.work + half.work + 14⟩

attribute [irreducible] power

private theorem multiply_image (left right : StoredRing) :
    image (multiply left right).value.get = image left.get * image right.get :=
  (congrArg image (StoredRingArithmetic.multiply_value left right)).trans
    (image_mul left.get right.get)

private theorem one_image : image (one ()).value.get = 1 :=
  (congrArg image StoredRingArithmetic.one_value).trans image_one

private theorem power_image (fuel : Nat) (base : StoredRing) (exponent : Nat)
    (bounded : exponent < 2 ^ fuel) :
    image (power (fuel + 1) base exponent).value.get = image base.get ^ exponent := by
  induction fuel generalizing base exponent with
  | zero =>
      have zero : exponent = 0 := by simpa using bounded
      subst exponent
      simp only [power, ↓reduceIte, one_image, pow_zero]
  | succ fuel ih =>
      by_cases zero : exponent = 0
      · subst exponent
        simp only [power, ↓reduceIte, one_image, pow_zero]
      · have halfBound : exponent / 2 < 2 ^ fuel := by
          apply (Nat.div_lt_iff_lt_mul (by decide : 0 < 2)).2
          simpa only [pow_succ, Nat.mul_comm] using bounded
        rw [power, if_neg zero]
        dsimp only
        have squareImage := multiply_image base base
        have halfImage := ih (multiply base base).value (exponent / 2) halfBound
        generalize squareResult : (multiply base base).value = squared at squareImage halfImage ⊢
        generalize halfResult : (power (fuel + 1) squared (exponent / 2)).value = half at halfImage ⊢
        rcases Nat.mod_two_eq_zero_or_one exponent with even | odd
        · rw [if_neg (by omega : ¬exponent % 2 = 1)]
          dsimp only
          rw [halfImage, squareImage, ← pow_two, ← pow_mul]
          congr 1
          omega
        · rw [if_pos odd]
          dsimp only
          rw [multiply_image half base, halfImage, squareImage, ← pow_two, ← pow_mul, ← pow_succ]
          congr 1
          omega

private theorem power_work_le (fuel : Nat) (base : StoredRing) (exponent : Nat) :
    (power fuel base exponent).work ≤ fuel * (2 * multiplyWork + 16) + oneWork + 5 := by
  induction fuel generalizing base exponent with
  | zero =>
      have identity := StoredRingArithmetic.one_work_le
      rw [power]
      change (one ()).work + 2 ≤ _
      omega
  | succ fuel ih =>
      have identity := StoredRingArithmetic.one_work_le
      have square := StoredRingArithmetic.multiply_work_le base base
      have half := ih (multiply base base).value (exponent / 2)
      have product := StoredRingArithmetic.multiply_work_le
        (power fuel (multiply base base).value (exponent / 2)).value base
      rw [power]
      split
      · dsimp only
        omega
      · split <;> dsimp only <;> rw [Nat.add_mul] <;> omega

private def naturalPower (base : Nat) : Nat → Result Nat
  | 0 => ⟨1, 2⟩
  | exponent + 1 =>
      let previous := naturalPower base exponent
      ⟨previous.value * base, previous.work + 4⟩

private theorem naturalPower_value (base exponent : Nat) :
    (naturalPower base exponent).value = base ^ exponent := by
  induction exponent with
  | zero => rfl
  | succ exponent ih => simpa only [naturalPower, ih, pow_succ]

private theorem naturalPower_work (base exponent : Nat) :
    (naturalPower base exponent).work = exponent * 4 + 2 := by
  induction exponent with
  | zero => rfl
  | succ exponent ih => simp only [naturalPower, ih, Nat.add_mul, Nat.one_mul]

-- Callers use the symbolic value and work laws. In particular they must not
-- normalize the fixed-depth executable ring loop during type inference.
attribute [irreducible] naturalPower

private theorem exponent_bound : goldilocksModulus ^ 27 - 2 < 2 ^ (64 * 27) := by
  have fieldBound : goldilocksModulus < 2 ^ 64 := by decide
  have bound : goldilocksModulus ^ 27 < (2 ^ 64) ^ 27 :=
    Nat.pow_lt_pow_left fieldBound (by decide : 27 ≠ 0)
  rw [← pow_mul] at bound
  exact lt_of_le_of_lt (Nat.sub_le _ _) bound

private def evaluated (value : StoredRing) (fuel : Nat) (exponent : Result Nat) : Result StoredRing :=
  let result := power (fuel + 1) value (exponent.value - 2)
  ⟨result.value, exponent.work + result.work + 8⟩

private theorem evaluated_image (value : StoredRing) (fuel : Nat) (exponent : Result Nat)
    (expected : Nat) (recorded : exponent.value = expected) (bounded : expected - 2 < 2 ^ fuel) :
    image (evaluated value fuel exponent).value.get = image value.get ^ (expected - 2) := by
  have actualBound : exponent.value - 2 < 2 ^ fuel := by rw [recorded]; exact bounded
  unfold evaluated
  exact (power_image fuel value (exponent.value - 2) actualBound).trans
    (congrArg (fun power : Nat => image value.get ^ (power - 2)) recorded)

private theorem evaluated_work_le (value : StoredRing) (fuel : Nat) (exponent : Result Nat)
    (exponentWork : Nat) (bounded : exponent.work ≤ exponentWork) :
    (evaluated value fuel exponent).work ≤
      exponentWork + ((fuel + 1) * (2 * multiplyWork + 16) + oneWork + 5) + 8 := by
  unfold evaluated
  exact Nat.add_le_add_right
    (Nat.add_le_add bounded (power_work_le (fuel + 1) value (exponent.value - 2))) 8

private theorem stored_power_product (value output : StoredRing)
    (powered : image output.get = image value.get ^ (goldilocksModulus ^ 27 - 2))
    (unit : UnitWitness Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring value.get) :
    ringFMul output.get value.get = ringFOne := by
  have unitImage : image value.get * image unit.inverse = 1 :=
    (image_mul value.get unit.inverse).symm.trans
      ((congrArg image unit.mul_inverse).trans image_one)
  have mapped : image (ringFMul output.get value.get) = image ringFOne := by
    rw [image_mul, powered, image_one]
    exact RingFFrobenius.unit_inverse_product (image value.get) (image unit.inverse) unitImage
  exact RingFPolynomial.image_injective mapped

private theorem stored_inverse_unique (value output : StoredRing)
    (unit : UnitWitness Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring value.get)
    (product : ringFMul output.get value.get = ringFOne) : output.get = unit.inverse := by
  have unitProduct : ringFMul value.get unit.inverse = ringFOne := unit.mul_inverse
  calc
    output.get = ringFMul output.get ringFOne := (RingFLaws.ringFMul_one_right output.get).symm
    _ = ringFMul output.get (ringFMul value.get unit.inverse) :=
      congrArg (ringFMul output.get) unitProduct.symm
    _ = ringFMul (ringFMul output.get value.get) unit.inverse :=
      (RingFLaws.ringFMul_assoc output.get value.get unit.inverse).symm
    _ = unit.inverse := (congrArg (fun result : RingF => ringFMul result unit.inverse) product).trans
      (RingFLaws.ringFMul_one_left unit.inverse)

attribute [irreducible] evaluated

/-- The wrapper computes the exponent and depth. Eight named operations
cover constant reads, subtraction, depth arithmetic, call, and return. -/
def inverse (value : StoredRing) : Result StoredRing :=
  evaluated value (64 * 27) (naturalPower goldilocksModulus 27)

/-- The executable candidate receives only its input array. Unit evidence
is used in this proof to cancel in the existing quotient ring. -/
theorem inverse_mul_value (value : StoredRing)
    (unit : UnitWitness Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring value.get) :
    ringFMul (inverse value).value.get value.get = ringFOne :=
  stored_power_product value (inverse value).value
    (evaluated_image value (64 * 27) (naturalPower goldilocksModulus 27)
      (goldilocksModulus ^ 27) (naturalPower_value goldilocksModulus 27) exponent_bound) unit

theorem inverse_eq_unitInverse (value : StoredRing)
    (unit : UnitWitness Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring value.get) :
    (inverse value).value.get = unit.inverse :=
  stored_inverse_unique value (inverse value).value unit (inverse_mul_value value unit)

def inverseWork : Nat :=
  (27 * 4 + 2) + ((64 * 27 + 1) * (2 * multiplyWork + 16) + oneWork + 5) + 8

theorem inverse_work_le (value : StoredRing) : (inverse value).work ≤ inverseWork :=
  evaluated_work_le value (64 * 27) (naturalPower goldilocksModulus 27) (27 * 4 + 2)
    (le_of_eq (naturalPower_work goldilocksModulus 27))

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingPowerInverse
