import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverseCorrect
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.ForkStrongSet

/-!
The executed stored inverse agrees with every unit witness of the selected
PiRLC scalar ring. Unit witnesses appear only in correctness statements;
the candidate computes its own inverse from the stored input coefficients.
This module supplies no work bound.
-/

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverseUnit

open NightstreamFPrime.Spec
open StoredRingInverse
open CompPoly (CPolynomial)
open Folding.PiRLC.PaperForkAlgebra

private abbrev Base := ZMod goldilocksModulus
local instance : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩

private theorem view_degree (value : RingF) :
    (RingFPolynomial.toPolynomial value).degree < (ringDegree : WithBot Nat) :=
  Polynomial.degree_sum_fin_lt (R := Base) value

private theorem view_eq_polynomial (value : RingF) (polynomial : Polynomial Base)
    (degree : polynomial.degree < (ringDegree : WithBot Nat))
    (coefficients : ∀ index : Fin ringDegree, polynomial.coeff index.val = value index) :
    RingFPolynomial.toPolynomial value = polynomial := by
  ext index
  by_cases inside : index < ringDegree
  · rw [RingFPolynomial.coeff_toPolynomial _ ⟨index, inside⟩]
    exact (coefficients ⟨index, inside⟩).symm
  · have outside : (ringDegree : WithBot Nat) ≤ index := by exact_mod_cast Nat.le_of_not_gt inside
    rw [Polynomial.coeff_eq_zero_of_degree_lt ((view_degree value).trans_le outside),
      Polynomial.coeff_eq_zero_of_degree_lt (degree.trans_le outside)]

private theorem encode_degree (value : StoredRing) :
    (encode value).toPoly.degree < (ringDegree : WithBot Nat) := by
  apply (Polynomial.degree_lt_iff_coeff_zero _ _).mpr
  intro index outside
  rw [← CPolynomial.coeff_toPoly, encode, CPolynomial.coeff_ofArray]
  simp [Array.getD, Nat.not_lt.mpr outside]

private theorem encode_view (value : StoredRing) :
    (encode value).toPoly = RingFPolynomial.toPolynomial value.get := by
  symm
  apply view_eq_polynomial _ _ (encode_degree value)
  intro index
  rw [← CPolynomial.coeff_toPoly]
  exact encode_coeff value index

private theorem candidate_view (value : StoredRing) :
    RingFPolynomial.toPolynomial (candidate value).get = (candidatePolynomial value).toPoly := by
  apply view_eq_polynomial _ _ (StoredRingInverseCorrect.candidatePolynomial_degree_lt value)
  intro index
  rw [← CPolynomial.coeff_toPoly]
  exact (candidate_coeff value index).symm

private theorem modulus_view : (modulus ()).toPoly = RingFPolynomial.modulus :=
  modulus_toPolynomial

private theorem coprime_of_unit (value : StoredRing)
    (unit : UnitWitness Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring value.get) :
    IsCoprime (encode value).toPoly (modulus ()).toPoly := by
  have inverse : ringFMul unit.inverse value.get = ringFOne := unit.inverse_mul
  have remainder :
      (RingFPolynomial.toPolynomial unit.inverse * RingFPolynomial.toPolynomial value.get) %ₘ
        RingFPolynomial.modulus = 1 := by
    rw [← RingFPolynomial.toPolynomial_ringFMul_mod, inverse, RingFPolynomial.toPolynomial_one]
  rw [encode_view, modulus_view]
  refine ⟨RingFPolynomial.toPolynomial unit.inverse,
    -((RingFPolynomial.toPolynomial unit.inverse * RingFPolynomial.toPolynomial value.get) /ₘ
      RingFPolynomial.modulus), ?_⟩
  rw [Polynomial.modByMonic_eq_sub_mul_div] at remainder
  calc
    _ = RingFPolynomial.toPolynomial unit.inverse * RingFPolynomial.toPolynomial value.get -
        RingFPolynomial.modulus *
          ((RingFPolynomial.toPolynomial unit.inverse * RingFPolynomial.toPolynomial value.get) /ₘ
            RingFPolynomial.modulus) := by ring
    _ = 1 := remainder

/-- The candidate from the actual stored call is a ring inverse whenever
that input is a unit of the existing selected PiRLC ring. -/
theorem candidate_mul_value (value : StoredRing)
    (unit : UnitWitness Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring value.get) :
    ringFMul (candidate value).get value.get = ringFOne := by
  apply RingFPolynomial.toPolynomial_injective
  rw [RingFPolynomial.toPolynomial_ringFMul_mod, candidate_view, ← encode_view,
    ← modulus_view, RingFPolynomial.toPolynomial_one]
  exact StoredRingInverseCorrect.candidatePolynomial_mul_mod value (coprime_of_unit value unit)

/-- This is the inverse equality required by the extraction primitive's
Correct.unitInverse field. The unit's inverse is not used by the program. -/
theorem candidate_eq_unitInverse (value : StoredRing)
    (unit : UnitWitness Phi81Relation.PiRLCAlgebra.ForkStrongSet.ring value.get) :
    (candidate value).get = unit.inverse := by
  have inverse : ringFMul value.get unit.inverse = ringFOne := unit.mul_inverse
  calc
    (candidate value).get = ringFMul (candidate value).get ringFOne :=
      (RingFLaws.ringFMul_one_right _).symm
    _ = ringFMul (candidate value).get (ringFMul value.get unit.inverse) := by rw [inverse]
    _ = ringFMul (ringFMul (candidate value).get value.get) unit.inverse :=
      (RingFLaws.ringFMul_assoc _ _ _).symm
    _ = unit.inverse := by rw [candidate_mul_value value unit, RingFLaws.ringFMul_one_left]

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverseUnit
