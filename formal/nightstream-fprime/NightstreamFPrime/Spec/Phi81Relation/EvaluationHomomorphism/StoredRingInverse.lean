import CompPoly.Univariate.EuclideanAlgorithm
import Mathlib.Algebra.Field.ZMod
import NightstreamFPrime.Spec.GoldilocksPrime
import Init.Data.Vector.OfFn

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverse

open NightstreamFPrime.Spec
open CompPoly (CPolynomial)

private abbrev Base := ZMod goldilocksModulus
local instance : Fact (Nat.Prime goldilocksModulus) := ⟨GoldilocksPrime.goldilocks_natPrime⟩

abbrev StoredRing := Vector F ringDegree

def encode (value : StoredRing) : CPolynomial Base :=
  CPolynomial.ofArray (R := Base) value.toArray

def modulus (_ : Unit) : CPolynomial Base :=
  CPolynomial.X ^ ringDegree + (CPolynomial.X ^ ringMiddleDegree + 1)

def candidatePolynomial (value : StoredRing) : CPolynomial Base :=
  let divisor := modulus ()
  let bezout := CPolynomial.normXgcd (encode value) divisor 0
  bezout.2.1.modByMonic divisor

def candidate (value : StoredRing) : StoredRing :=
  let polynomial := candidatePolynomial value
  Vector.ofFn fun index => polynomial.coeff index.val

set_option backward.isDefEq.respectTransparency false in
private theorem get_ofFn {count : Nat} (value : Fin count → F) (index : Fin count) :
    (Vector.ofFn value).get index = value index := by
  simp [Vector.get, Vector.ofFn]

set_option backward.isDefEq.respectTransparency false in
theorem encode_coeff (value : StoredRing) (index : Fin ringDegree) :
    (encode value).coeff index.val = value.get index := by
  rw [encode, CPolynomial.coeff_ofArray]
  simp [Array.getD, Vector.get, index.isLt]
  rfl

theorem candidate_coeff (value : StoredRing) (index : Fin ringDegree) :
    (candidate value).get index = (candidatePolynomial value).coeff index.val :=
  get_ofFn (fun index => (candidatePolynomial value).coeff index.val) index

theorem modulus_toPolynomial :
    (modulus ()).toPoly =
      (Polynomial.X : Polynomial Base) ^ ringDegree +
        (Polynomial.X ^ ringMiddleDegree + 1) := by
  simp only [modulus, CPolynomial.toPoly_add, CPolynomial.toPoly_pow,
    CPolynomial.X_toPoly, CPolynomial.toPoly_one]

theorem divisor_monic : (modulus ()).monic := by
  apply (CPolynomial.monic_toPoly_iff _).mpr
  rw [modulus_toPolynomial]
  apply Polynomial.monic_X_pow_add
  have lower : (Polynomial.X ^ ringMiddleDegree + 1 : Polynomial Base).degree =
      (ringMiddleDegree : WithBot Nat) := by
    simpa only [ringMiddleDegree] using!
      (Polynomial.degree_X_pow_add_C (R := Base) (n := 27) (by decide) (1 : Base))
  rw [lower]
  decide

theorem candidatePolynomial_toPolynomial (value : StoredRing) :
    (candidatePolynomial value).toPoly =
      (CPolynomial.normXgcd (encode value) (modulus ()) 0).2.1.toPoly %ₘ (modulus ()).toPoly :=
  CPolynomial.modByMonic_toPoly_eq_modByMonic _ _ divisor_monic

theorem normalized_bezout (value : StoredRing) :
    let result := CPolynomial.normXgcd (encode value) (modulus ()) 0
    result.1.toPoly = result.2.1.toPoly * (encode value).toPoly +
      result.2.2.toPoly * (modulus ()).toPoly := by
  have identity := CPolynomial.normXgcd_bezout (encode value) (modulus ()) 0
  change _ = _ * _ + _ * _ at identity
  exact congrArg CPolynomial.toPoly identity |>.trans
    (by
      rw [CPolynomial.toPoly_add, CPolynomial.toPoly_mul, CPolynomial.toPoly_mul]
      rfl)

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverse
