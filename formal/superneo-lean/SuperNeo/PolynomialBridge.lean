import Mathlib.Data.ZMod.Basic
import Mathlib.Algebra.Polynomial.BigOperators
import SuperNeo.GoldilocksPrime
import SuperNeo.Ring

namespace SuperNeo
open scoped BigOperators

/-- `ZMod q` presentation of the concrete Goldilocks field. -/
abbrev Fq := ZMod Goldilocks.q

/-- View a field element through the canonical `ZMod q` representative. -/
def fToZMod (a : F) : Fq :=
  (a.val : Fq)

@[simp] theorem fToZMod_zero :
    fToZMod 0 = 0 := by
  simp [fToZMod]

@[simp] theorem fToZMod_one :
    fToZMod 1 = 1 := by
  simp [fToZMod]

/-- Rebuild a field element from the canonical `ZMod q` representative. -/
noncomputable def zmodToF (z : Fq) : F :=
  ⟨z.val, z.val_lt⟩

theorem fToZMod_add (a b : F) :
    fToZMod (a + b) = fToZMod a + fToZMod b := by
  change (((a + b).val : Fq) = ((a.val : Fq) + (b.val : Fq)))
  rw [F.val_add]
  simp

theorem fToZMod_mul (a b : F) :
    fToZMod (a * b) = fToZMod a * fToZMod b := by
  change (((a * b).val : Fq) = ((a.val : Fq) * (b.val : Fq)))
  rw [F.val_mul]
  simp

theorem fToZMod_neg (a : F) :
    fToZMod (-a) = -fToZMod a := by
  change (((-a).val : Fq) = -((a.val : Fq)))
  rw [F.val_neg]
  simp

theorem fToZMod_sub (a b : F) :
    fToZMod (a - b) = fToZMod a - fToZMod b := by
  rw [sub_eq_add_neg, fToZMod_add, fToZMod_neg, sub_eq_add_neg]

@[simp] theorem zmodToF_fToZMod (a : F) :
    zmodToF (fToZMod a) = a := by
  apply Fin.ext
  simp [zmodToF, fToZMod, Nat.mod_eq_of_lt a.isLt]

@[simp] theorem fToZMod_zmodToF (z : Fq) :
    fToZMod (zmodToF z) = z := by
  simpa [fToZMod, zmodToF] using (ZMod.cast_eq_val (R := Fq) z)

theorem fToZMod_injective : Function.Injective fToZMod := by
  intro a b h
  have hz : zmodToF (fToZMod a) = zmodToF (fToZMod b) := by
    simpa [h]
  simpa using hz

@[simp] theorem zmodToF_zero :
    zmodToF 0 = 0 := by
  exact fToZMod_injective (by simp)

@[simp] theorem zmodToF_one :
    zmodToF 1 = 1 := by
  exact fToZMod_injective (by simp)

theorem zmodToF_add (x y : Fq) :
    zmodToF (x + y) = zmodToF x + zmodToF y := by
  apply fToZMod_injective
  simp [fToZMod_add]

theorem zmodToF_mul (x y : Fq) :
    zmodToF (x * y) = zmodToF x * zmodToF y := by
  apply fToZMod_injective
  simp [fToZMod_mul]

theorem zmodToF_neg (x : Fq) :
    zmodToF (-x) = -zmodToF x := by
  apply fToZMod_injective
  simp [fToZMod_neg]

theorem zmodToF_sub (x y : Fq) :
    zmodToF (x - y) = zmodToF x - zmodToF y := by
  apply fToZMod_injective
  simp [fToZMod_sub]

/-- Coefficient-vector view as a degree-`< d` polynomial over `ZMod q`. -/
noncomputable def coeffsToPolynomial (a : Coeffs) : Polynomial Fq :=
  Finset.sum (Finset.range d) fun i =>
    Polynomial.monomial i (fToZMod (coeffAt a i))

/-- Truncate a polynomial to the canonical degree-`< d` coefficient vector view. -/
noncomputable def polynomialToCoeffs (p : Polynomial Fq) : Coeffs :=
  Array.ofFn (fun i : Fin d => zmodToF (p.coeff i.1))

@[simp] theorem polynomialToCoeffs_size (p : Polynomial Fq) :
    (polynomialToCoeffs p).size = d := by
  simp [polynomialToCoeffs]

theorem hasRingDegreeShape_polynomialToCoeffs (p : Polynomial Fq) :
    hasRingDegreeShape (polynomialToCoeffs p) := by
  simp [hasRingDegreeShape, polynomialToCoeffs_size]

theorem coeff_coeffsToPolynomial (a : Coeffs) (i : Nat) :
    (coeffsToPolynomial a).coeff i =
      if i < d then fToZMod (coeffAt a i) else 0 := by
  by_cases hi : i < d
  · have hiMem : i ∈ Finset.range d := by simp [hi]
    calc
      (coeffsToPolynomial a).coeff i
          = Finset.sum (Finset.range d) (fun b =>
              (Polynomial.monomial b (fToZMod (coeffAt a b))).coeff i) := by
                simp [coeffsToPolynomial]
      _ = (Polynomial.monomial i (fToZMod (coeffAt a i))).coeff i := by
            apply Finset.sum_eq_single i
            · intro b hb hbi
              simp [Polynomial.coeff_monomial, hbi]
            · intro hNotMem
              exact False.elim (hNotMem hiMem)
      _ = fToZMod (coeffAt a i) := by simp [Polynomial.coeff_monomial]
      _ = if i < d then fToZMod (coeffAt a i) else 0 := by simp [hi]
  · have hiNotMem : i ∉ Finset.range d := by simp [hi]
    calc
      (coeffsToPolynomial a).coeff i
          = Finset.sum (Finset.range d) (fun b =>
              (Polynomial.monomial b (fToZMod (coeffAt a b))).coeff i) := by
                simp [coeffsToPolynomial]
      _ = 0 := by
            apply Finset.sum_eq_zero
            intro b hb
            have hbne : b ≠ i := by
              intro hEq
              exact hiNotMem (by simpa [hEq] using hb)
            simp [Polynomial.coeff_monomial, hbne]
      _ = if i < d then fToZMod (coeffAt a i) else 0 := by simp [hi]

theorem coeffAt_polynomialToCoeffs (p : Polynomial Fq) (i : Nat) (hi : i < d) :
    coeffAt (polynomialToCoeffs p) i = zmodToF (p.coeff i) := by
  unfold coeffAt polynomialToCoeffs
  simp [hi, Array.getD]

@[simp] theorem coeffsToPolynomial_zeroRq :
    coeffsToPolynomial zeroRq = 0 := by
  ext i
  by_cases hi : i < d
  · simp [coeff_coeffsToPolynomial, hi, coeffAt_zeroRq]
  · simp [coeff_coeffsToPolynomial, hi]

@[simp] theorem coeffsToPolynomial_oneRq :
    coeffsToPolynomial oneRq = 1 := by
  ext i
  by_cases hi : i = 0
  · subst hi
    have h0 : (0 : Nat) < d := by decide
    simp [coeff_coeffsToPolynomial, h0, oneRq, coeffAt, h0]
  · by_cases hlt : i < d
    · have hget : oneRq[i] = (0 : F) := by
        simpa [oneRq] using
          (Array.getElem_setIfInBounds_ne
            (xs := Array.replicate d (0 : F))
            (i := 0) (j := i) (a := (1 : F)) hlt (by simpa [eq_comm] using hi))
      have hgetD : oneRq.getD i 0 = 0 := by
        simpa [oneRq, Array.getD, hlt, hi]
      simp [coeff_coeffsToPolynomial, hlt, coeffAt, hlt, hgetD, Polynomial.coeff_one, hi]
    · simp [coeff_coeffsToPolynomial, hlt, hi, Polynomial.coeff_one]

theorem coeffsToPolynomial_injective_of_size_d
    {a b : Coeffs}
    (ha : a.size = d)
    (hb : b.size = d)
    (hEq : coeffsToPolynomial a = coeffsToPolynomial b) :
    a = b := by
  apply Array.ext
  · simpa [ha, hb]
  · intro i hiA hiB
    have hi : i < d := by simpa [ha] using hiA
    have hCoeff :
        fToZMod (coeffAt a i) = fToZMod (coeffAt b i) := by
      simpa [coeff_coeffsToPolynomial, hi] using
        congrArg (fun p : Polynomial Fq => p.coeff i) hEq
    have hCoeffF : coeffAt a i = coeffAt b i := fToZMod_injective hCoeff
    have hGetA : a[i]'hiA = coeffAt a i := by
      simp [coeffAt, hi, ha, Array.getD]
    have hGetB : b[i]'hiB = coeffAt b i := by
      simp [coeffAt, hi, hb, Array.getD]
    calc
      a[i]'hiA = coeffAt a i := hGetA
      _ = coeffAt b i := hCoeffF
      _ = b[i]'hiB := hGetB.symm

theorem coeffsToPolynomial_polynomialToCoeffs_of_degree_lt
    {p : Polynomial Fq}
    (hDeg : p.degree < d) :
    coeffsToPolynomial (polynomialToCoeffs p) = p := by
  ext i
  by_cases hi : i < d
  · simp [coeff_coeffsToPolynomial, hi, coeffAt_polynomialToCoeffs]
  · have hCoeffZero : p.coeff i = 0 := by
      exact Polynomial.coeff_eq_zero_of_degree_lt
        (lt_of_lt_of_le hDeg (by exact_mod_cast Nat.le_of_not_lt hi))
    simp [coeff_coeffsToPolynomial, hi, hCoeffZero]

theorem coeffsToPolynomial_vecAdd_of_size_d
    {a b : Coeffs}
    (ha : a.size = d)
    (hb : b.size = d) :
    coeffsToPolynomial (vecAdd a b) = coeffsToPolynomial a + coeffsToPolynomial b := by
  ext i
  by_cases hi : i < d
  · have hSize : (vecAdd a b).size = d := by
      simpa [ha] using (vecAdd_size_of_eq (show a.size = b.size by simpa [ha, hb]))
    have hCoeff :
        coeffAt (vecAdd a b) i = coeffAt a i + coeffAt b i := by
      exact coeffAt_vecAdd_of_size_d a b ha hb i hi
    simp [coeff_coeffsToPolynomial, hi, hSize, hCoeff, Polynomial.coeff_add, fToZMod_add]
  · simp [coeff_coeffsToPolynomial, hi, Polynomial.coeff_add]

theorem coeffsToPolynomial_vecScale_of_size_d
    {s : F} {a : Coeffs}
    (ha : a.size = d) :
    coeffsToPolynomial (vecScale s a) = Polynomial.C (fToZMod s) * coeffsToPolynomial a := by
  ext i
  by_cases hi : i < d
  · have hCoeff :
        coeffAt (vecScale s a) i = s * coeffAt a i := by
      exact coeffAt_vecScale_of_size_d (s := s) (x := a) (hx := ha) (k := i) hi
    simp [coeff_coeffsToPolynomial, hi, hCoeff, Polynomial.coeff_C_mul, fToZMod_mul]
  · simp [coeff_coeffsToPolynomial, hi, Polynomial.coeff_C_mul]

/-- Concrete cyclotomic modulus `Φ(X) = X^54 + X^27 + 1` over `ZMod q`. -/
noncomputable def phiPolynomial : Polynomial Fq :=
  Polynomial.X ^ d + Polynomial.X ^ (d / 2) + 1

theorem phiPolynomial_def :
    phiPolynomial = Polynomial.X ^ 54 + Polynomial.X ^ 27 + 1 := by
  simp [phiPolynomial, d]

end SuperNeo
