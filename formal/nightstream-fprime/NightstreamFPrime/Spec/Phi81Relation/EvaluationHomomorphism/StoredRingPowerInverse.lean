import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial

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
open RingFPolynomial (image image_mul QuotientRing)
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

private def naturalPower (base : Nat) : Nat → Result Nat
  | 0 => ⟨1, 2⟩
  | exponent + 1 =>
      let previous := naturalPower base exponent
      ⟨previous.value * base, previous.work + 4⟩

private theorem naturalPower_work (base exponent : Nat) :
    (naturalPower base exponent).work = exponent * 4 + 2 := by
  induction exponent with
  | zero => rfl
  | succ exponent ih => simp only [naturalPower, ih, Nat.add_mul, Nat.one_mul]

-- Callers use the symbolic value and work laws. In particular they must not
-- normalize the fixed-depth executable ring loop during type inference.
attribute [irreducible] naturalPower

private def evaluated (value : StoredRing) (fuel : Nat) (exponent : Result Nat) : Result StoredRing :=
  let result := power (fuel + 1) value (exponent.value - 2)
  ⟨result.value, exponent.work + result.work + 8⟩

attribute [irreducible] evaluated

/-- The wrapper computes the exponent and depth. Eight named operations
cover constant reads, subtraction, depth arithmetic, call, and return. -/
def inverse (value : StoredRing) : Result StoredRing :=
  evaluated value (64 * 27) (naturalPower goldilocksModulus 27)

def inverseWork : Nat :=
  (27 * 4 + 2) + ((64 * 27 + 1) * (2 * multiplyWork + 16) + oneWork + 5) + 8

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingPowerInverse
