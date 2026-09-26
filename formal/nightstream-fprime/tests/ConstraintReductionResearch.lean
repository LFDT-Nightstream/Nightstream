import Mathlib.Data.Fintype.BigOperators
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Algebra.Polynomial.Roots
import NightstreamFPrime.Spec.GoldilocksPrime
import NightstreamFPrime.Layout.BalancedTernary
import NightstreamFPrime.Export.Stage1.PoseidonRetainedBlock
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package
import tests.AxiomAudit

/-!
Research bounds for the selected Lean layout and two specific encoding choices.
These do not give a lower bound for all Poseidon circuits or correlated
multi-field encodings. No production layout or security parameter changes.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Tests.ConstraintReductionResearch

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm

private abbrev Base := ZMod goldilocksModulus
local instance : Fact (Nat.Prime goldilocksModulus) :=
  ⟨GoldilocksPrime.goldilocks_natPrime⟩

/-- The new owner target, exactly half the checkpoint carrier, is ring aligned. -/
theorem target_aligned : 92179782 = 1707033 * ringDegree := by decide

theorem poseidon_alone_exceeds_target :
    92179782 < PoseidonRetainedBlock.retainedCoordinateCount := by
  rw [PoseidonRetainedBlock.retainedCoordinateCount_eq]
  decide

theorem other_logical_coordinates :
    PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application -
      PoseidonRetainedBlock.retainedCoordinateCount = 34849617 := by
  rw [Poseidon2HashChainV1Package.logicalWidth,
    PoseidonRetainedBlock.retainedCoordinateCount_eq]

/-- A necessary budget condition only; it is not a proposed construction. -/
theorem poseidon_budget_if_other_blocks_fixed (candidate : Nat)
    (fits : PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application -
        PoseidonRetainedBlock.retainedCoordinateCount + candidate ≤ 92179782) :
    candidate ≤ 57330165 := by
  rw [other_logical_coordinates] at fits
  omega

/-- Any independent decoder from at most 40 three-valued coordinates misses
some Goldilocks value, regardless of the chosen coordinate weights. -/
theorem no_forty_trit_decoder {count : Nat} (bounded : count ≤ 40)
    (decode : (Fin count → Fin 3) → F) : ¬ Function.Surjective decode := by
  intro complete
  have cardinal := Fintype.card_le_of_surjective decode complete
  have capacity : goldilocksModulus ≤ 3 ^ count := by
    simpa only [F, Fintype.card_fin, Fintype.card_fun] using cardinal
  have powerBound := Nat.pow_le_pow_right (by decide : 0 < 3) bounded
  have tooSmall : 3 ^ 40 < goldilocksModulus := by decide
  omega

private theorem magnitude_three_mul (value : F) :
    centeredMagnitude ((3 : F) * value) ≤ 3 * centeredMagnitude value := by
  have expand : (3 : F) * value = value + value + value := by
    have three : (3 : F) = 1 + 1 + 1 := by decide
    rw [three]
    calc
      ((1 : F) + 1 + 1) * value = value * ((1 : F) + 1 + 1) :=
        Lean.Grind.Fin.mul_comm _ _
      _ = (value * 1 + value * 1) + value * 1 := by
        rw [Lean.Grind.Fin.left_distrib, Lean.Grind.Fin.left_distrib]
      _ = value + value + value := by rw [Lean.Grind.Fin.mul_one]
  rw [expand]
  have first := Centered.centeredMagnitude_add_le value value
  have second := Centered.centeredMagnitude_add_le (value + value) value
  omega

theorem signed_recomposition_bound (digits : List F)
    (signed : ∀ digit ∈ digits, centeredMagnitude digit < 2) :
    centeredMagnitude (BalancedTernary.recompose digits) ≤
      BalancedTernary.radius digits.length := by
  induction digits with
  | nil => decide
  | cons digit rest ih =>
      have headBound := signed digit (by simp)
      have tailBound := ih (fun item member => signed item (by simp [member]))
      have multiplied := magnitude_three_mul (BalancedTernary.recompose rest)
      have added := Centered.centeredMagnitude_add_le digit
        ((3 : F) * BalancedTernary.recompose rest)
      change centeredMagnitude (digit + (3 : F) * BalancedTernary.recompose rest) ≤
        3 * BalancedTernary.radius rest.length + 1
      omega

def missingValue : F := 6078832729528464401
def sboxInput : F := 3194645001229403778

theorem missingValue_is_sbox_output : Poseidon2.sbox sboxInput = missingValue := by
  decide

/-- This is the exact missing value used by the cvc5 integer-lift experiment.
The proof works directly with field recomposition and the protocol norm. -/
theorem forty_trits_miss_sbox_output (digits : List F)
    (length : digits.length = 40)
    (signed : ∀ digit ∈ digits, centeredMagnitude digit < 2) :
    BalancedTernary.recompose digits ≠ Poseidon2.sbox sboxInput := by
  intro same
  have bounded := signed_recomposition_bound digits signed
  rw [length, same, missingValue_is_sbox_output] at bounded
  have gap : BalancedTernary.radius 40 < centeredMagnitude missingValue := by decide
  omega

/-- Exponents in a bivariate equation under the current strict degree bound. -/
abbrev PowerPair := {pair : Fin 9 × Fin 9 //
  pair.1.val + pair.2.val < ProductionRelation.polynomial.degreeBound}

private def substitutedPower (pair : PowerPair) : Nat :=
  pair.val.1.val + 49 * pair.val.2.val

private theorem substitutedPower_injective : Function.Injective substitutedPower := by
  intro left right same
  apply Subtype.ext
  apply Prod.ext <;> apply Fin.ext
  all_goals
    have leftBound : left.val.1.val + left.val.2.val < 9 := by
      simpa only [ProductionRelation.polynomial_degreeBound] using left.property
    have rightBound : right.val.1.val + right.val.2.val < 9 := by
      simpa only [ProductionRelation.polynomial_degreeBound] using right.property
    change left.val.1.val + 49 * left.val.2.val =
      right.val.1.val + 49 * right.val.2.val at same
    omega

private theorem substitutedPower_le (pair : PowerPair) : substitutedPower pair ≤ 392 := by
  have bounded : pair.val.1.val + pair.val.2.val < 9 := by
    simpa only [ProductionRelation.polynomial_degreeBound] using pair.property
  unfold substitutedPower
  omega

def pairEquation (coefficients : PowerPair → Base) (x y : Base) : Base :=
  ∑ pair, coefficients pair * x ^ pair.val.1.val * y ^ pair.val.2.val

private noncomputable def substituted (coefficients : PowerPair → Base) : Polynomial Base :=
  ∑ pair, Polynomial.monomial (substitutedPower pair) (coefficients pair)

private theorem substituted_eval (coefficients : PowerPair → Base) (x : Base) :
    (substituted coefficients).eval x = pairEquation coefficients x (x ^ 49) := by
  simp [substituted, pairEquation, substitutedPower, Polynomial.eval_finsetSum,
    Polynomial.eval_monomial, pow_add, pow_mul, mul_assoc]

private theorem substituted_coefficient (coefficients : PowerPair → Base) (pair : PowerPair) :
    (substituted coefficients).coeff (substitutedPower pair) = coefficients pair := by
  classical
  simp [substituted, Polynomial.finsetSum_coeff, Polynomial.coeff_monomial,
    substitutedPower_injective.eq_iff]

private theorem substituted_degree (coefficients : PowerPair → Base) :
    (substituted coefficients).natDegree ≤ 392 := by
  apply Polynomial.natDegree_sum_le_of_forall_le
  intro pair _
  exact (Polynomial.natDegree_monomial_le _).trans (substitutedPower_le pair)

/-- No nonzero bivariate equation of total degree below nine can replace
the scalar chain z=x^7, y=z^7 with no auxiliary variable. This is a local
candidate-class bound, not a lower bound for the full Poseidon permutation. -/
theorem no_low_degree_pair_equation (coefficients : PowerPair → Base)
    (complete : ∀ x, pairEquation coefficients x (x ^ 49) = 0) :
    ∀ pair, coefficients pair = 0 := by
  have degree := substituted_degree coefficients
  have zero : substituted coefficients = 0 :=
    Polynomial.eq_zero_of_natDegree_lt_card_of_eval_eq_zero _ Function.injective_id
      (fun x => (substituted_eval coefficients x).trans (complete x)) (by
        have card : Fintype.card Base = goldilocksModulus := by simp [Base]
        rw [card]
        have bound : 392 < goldilocksModulus := by decide
        omega)
  intro pair
  rw [← substituted_coefficient coefficients pair, zero, Polynomial.coeff_zero]

theorem repeated_sbox_not_single_sbox :
    Poseidon2.sbox (Poseidon2.sbox (2 : F)) ≠ Poseidon2.sbox (2 : F) := by decide

#audit_axioms target_aligned
#audit_axioms poseidon_alone_exceeds_target
#audit_axioms other_logical_coordinates
#audit_axioms poseidon_budget_if_other_blocks_fixed
#audit_axioms no_forty_trit_decoder
#audit_axioms signed_recomposition_bound
#audit_axioms missingValue_is_sbox_output
#audit_axioms forty_trits_miss_sbox_output
#audit_axioms no_low_degree_pair_equation
#audit_axioms repeated_sbox_not_single_sbox

end NightstreamFPrime.Tests.ConstraintReductionResearch
