import SuperNeo.InvertibilityGoldilocksBase

namespace SuperNeo

open Polynomial

noncomputable section

private theorem coeffsToPolynomial_basisVecNat
    {i : Nat}
    (hi : i < d) :
    coeffsToPolynomial (basisVecNat i) = X ^ i := by
  ext n
  by_cases hn : n < d
  · by_cases hEq : n = i
    · subst hEq
      simp [coeff_coeffsToPolynomial, basisVecNat, coeffAt, hi]
    · simp [coeff_coeffsToPolynomial, basisVecNat, coeffAt, hn, hEq, Polynomial.coeff_X_pow]
  · have hNe : n ≠ i := by omega
    simp [coeff_coeffsToPolynomial, hn, hNe, Polynomial.coeff_X_pow]

private theorem coeffsToPolynomial_monomialReduce_of_lt_d
    {n : Nat}
    (hn : n < d) :
    coeffsToPolynomial (monomialReduce n) = X ^ n := by
  have h54 : n < 54 := by simpa [d] using hn
  have h81 : n < 81 := by omega
  unfold monomialReduce
  simp [Nat.mod_eq_of_lt h81, hn, coeffsToPolynomial_basisVecNat hn]

private theorem coeffsToPolynomial_monomialReduce_of_ge54_lt81
    {n : Nat}
    (h54 : 54 ≤ n)
    (h81 : n < 81) :
    coeffsToPolynomial (monomialReduce n) =
      -(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27) := by
  have hNotLt : ¬ n < d := by simpa [d] using (show ¬ n < 54 by omega)
  have h54lt : n - 54 < d := by simpa [d] using (show n - 54 < 54 by omega)
  have h27lt : n - 27 < d := by simpa [d] using (show n - 27 < 54 by omega)
  calc
    coeffsToPolynomial (monomialReduce n)
        = coeffsToPolynomial
            (vecAdd
              (vecScale (-1 : F) (basisVecNat (n - 54)))
              (vecScale (-1 : F) (basisVecNat (n - 27)))) := by
              simp [monomialReduce, Nat.mod_eq_of_lt h81, hNotLt]
    _ = coeffsToPolynomial (vecScale (-1 : F) (basisVecNat (n - 54))) +
          coeffsToPolynomial (vecScale (-1 : F) (basisVecNat (n - 27))) := by
            rw [coeffsToPolynomial_vecAdd_of_size_d]
            · simp [basisVecNat]
            · simp [basisVecNat]
    _ = -coeffsToPolynomial (basisVecNat (n - 54)) +
          -coeffsToPolynomial (basisVecNat (n - 27)) := by
            rw [coeffsToPolynomial_vecScale_of_size_d, coeffsToPolynomial_vecScale_of_size_d]
            · simp [basisVecNat, fToZMod_neg, fToZMod_one]
            · simp [basisVecNat]
            · simp [basisVecNat]
    _ = -(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27) := by
          rw [coeffsToPolynomial_basisVecNat h54lt, coeffsToPolynomial_basisVecNat h27lt]
          ring

private theorem coeffsToPolynomial_monomialReduce_of_ge81_lt108
    {n : Nat}
    (h81 : 81 ≤ n)
    (h108 : n < 2 * d) :
    coeffsToPolynomial (monomialReduce n) = X ^ (n - 81) := by
  have hn108 : n < 108 := by simpa [d] using h108
  have hSub : n - 81 < d := by simpa [d] using (show n - 81 < 54 by omega)
  have hLt81 : n - 81 < 81 := by omega
  have hmod : n % 81 = n - 81 := by
    calc
      n % 81 = (81 + (n - 81)) % 81 := by congr; omega
      _ = (n - 81) % 81 := by simp
      _ = n - 81 := Nat.mod_eq_of_lt hLt81
  unfold monomialReduce
  simp [hmod, hSub, coeffsToPolynomial_basisVecNat hSub]

private theorem X_pow_mod_phiPolynomial_of_lt_d
    {n : Nat}
    (hn : n < d) :
    ((X ^ n : Polynomial Fq) %ₘ phiPolynomial) = X ^ n := by
  exact (Polynomial.modByMonic_eq_self_iff phiPolynomial_monic).2 <|
    by simpa [phiPolynomial_degree] using degree_X_pow_lt_d hn

private theorem degree_neg_Xpow_sub_Xpow_lt_d
    {n : Nat}
    (h54 : 54 ≤ n)
    (h81 : n < 81) :
    (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)).degree < (d : WithBot Nat) := by
  rw [Polynomial.degree_lt_iff_coeff_zero]
  intro m hm
  have hmGe : d ≤ m := hm
  have h54lt : n - 54 < d := by simpa [d] using (show n - 54 < 54 by omega)
  have h27lt : n - 27 < d := by simpa [d] using (show n - 27 < 54 by omega)
  have hm1 : m ≠ n - 54 := by
    intro hEq
    subst hEq
    exact (Nat.not_lt_of_ge hmGe h54lt).elim
  have hm2 : m ≠ n - 27 := by
    intro hEq
    subst hEq
    exact (Nat.not_lt_of_ge hmGe h27lt).elim
  simp [Polynomial.coeff_sub, Polynomial.coeff_neg, Polynomial.coeff_X_pow, hm1, hm2]

  private theorem X_pow_mod_phiPolynomial_of_ge54_lt81
    {n : Nat}
    (h54 : 54 ≤ n)
    (h81 : n < 81) :
    ((X ^ n : Polynomial Fq) %ₘ phiPolynomial) =
      (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)) := by
  have h1 : (X ^ 54 : Polynomial Fq) * X ^ (n - 54) = X ^ n := by
    rw [← pow_add]
    congr
    omega
  have h2 : (X ^ 27 : Polynomial Fq) * X ^ (n - 54) = X ^ (n - 27) := by
    rw [← pow_add]
    congr
    omega
  refine (Polynomial.div_modByMonic_unique (q := X ^ (n - 54))
    (r := (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)))
    phiPolynomial_monic ?_).2
  constructor
  · calc
      (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)) + phiPolynomial * X ^ (n - 54)
          = (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)) +
              ((X ^ 54 : Polynomial Fq) * X ^ (n - 54) +
                (X ^ 27 : Polynomial Fq) * X ^ (n - 54) +
                X ^ (n - 54)) := by
                  rw [phiPolynomial_def, add_mul, add_mul, one_mul]
      _ = (-(X ^ (n - 54) : Polynomial Fq) - X ^ (n - 27)) +
            (X ^ n + X ^ (n - 27) + X ^ (n - 54)) := by rw [h1, h2]
      _ = X ^ n := by ring_nf
  · exact by simpa [phiPolynomial_degree] using degree_neg_Xpow_sub_Xpow_lt_d h54 h81

  private theorem X_pow_mod_phiPolynomial_of_ge81_lt108
    {n : Nat}
    (h81 : 81 ≤ n)
    (h108 : n < 2 * d) :
    ((X ^ n : Polynomial Fq) %ₘ phiPolynomial) = X ^ (n - 81) := by
  have h54 : 54 ≤ n := by omega
  have h1 : (X ^ 54 : Polynomial Fq) * X ^ (n - 54) = X ^ n := by
    rw [← pow_add]
    congr
    omega
  have h2 : (X ^ 27 : Polynomial Fq) * X ^ (n - 54) = X ^ (n - 27) := by
    rw [← pow_add]
    congr
    omega
  have h3 : (X ^ 54 : Polynomial Fq) * X ^ (n - 81) = X ^ (n - 27) := by
    rw [← pow_add]
    congr
    omega
  have h4 : (X ^ 27 : Polynomial Fq) * X ^ (n - 81) = X ^ (n - 54) := by
    rw [← pow_add]
    congr
    omega
  refine (Polynomial.div_modByMonic_unique
    (q := X ^ (n - 54) - X ^ (n - 81))
    (r := X ^ (n - 81))
    phiPolynomial_monic ?_).2
  constructor
  · calc
      X ^ (n - 81) + phiPolynomial * (X ^ (n - 54) - X ^ (n - 81))
          = X ^ (n - 81) +
              (((X ^ 54 : Polynomial Fq) + X ^ 27 + 1) *
                (X ^ (n - 54) - X ^ (n - 81))) := by
                  rw [phiPolynomial_def]
      _ = X ^ (n - 81) +
            (((X ^ 54 : Polynomial Fq) * (X ^ (n - 54) - X ^ (n - 81))) +
              (X ^ 27 * (X ^ (n - 54) - X ^ (n - 81))) +
              (X ^ (n - 54) - X ^ (n - 81))) := by
                rw [add_mul, add_mul, one_mul]
      _ = X ^ (n - 81) +
            ((X ^ n - X ^ (n - 27)) + (X ^ (n - 27) - X ^ (n - 54)) +
              (X ^ (n - 54) - X ^ (n - 81))) := by
                rw [mul_sub, mul_sub, h1, h3, h2, h4]
      _ = X ^ n := by ring_nf
  · have hn108 : n < 108 := by simpa [d] using h108
    have hSub54 : n - 81 < 54 := by omega
    have hSub' : n - 81 < d := by simpa [d] using hSub54
    simpa [phiPolynomial_degree] using degree_X_pow_lt_d hSub'

set_option maxRecDepth 4096 in
private theorem coeffsToPolynomial_basis_basis_mod :
    ∀ i j : Fin d,
      ((coeffsToPolynomial (basisVecNat i.1) * coeffsToPolynomial (basisVecNat j.1)) %ₘ phiPolynomial) =
        coeffsToPolynomial (monomialReduce (i.1 + j.1)) := by
  intro i j
  let n := i.1 + j.1
  have hn108 : n < 2 * d := by
    dsimp [n]
    omega
  calc
    ((coeffsToPolynomial (basisVecNat i.1) * coeffsToPolynomial (basisVecNat j.1)) %ₘ phiPolynomial)
        = ((X ^ i.1 : Polynomial Fq) * X ^ j.1) %ₘ phiPolynomial := by
            rw [coeffsToPolynomial_basisVecNat i.2, coeffsToPolynomial_basisVecNat j.2]
    _ = (X ^ n : Polynomial Fq) %ₘ phiPolynomial := by
          dsimp [n]
          rw [← pow_add]
    _ = coeffsToPolynomial (monomialReduce n) := by
          by_cases hLt : n < d
          · rw [X_pow_mod_phiPolynomial_of_lt_d hLt, coeffsToPolynomial_monomialReduce_of_lt_d hLt]
          · by_cases hLt81 : n < 81
            · have h54 : 54 ≤ n := by simpa [d] using Nat.le_of_not_lt hLt
              rw [X_pow_mod_phiPolynomial_of_ge54_lt81 h54 hLt81,
                coeffsToPolynomial_monomialReduce_of_ge54_lt81 h54 hLt81]
            · have h81 : 81 ≤ n := by omega
              rw [X_pow_mod_phiPolynomial_of_ge81_lt108 h81 hn108,
                coeffsToPolynomial_monomialReduce_of_ge81_lt108 h81 hn108]

set_option maxHeartbeats 1200000 in
private theorem mulRq_basis_basis_eq_polyMulCoeffs
    (i j : Nat)
    (hi : i < d)
    (hj : j < d) :
    mulRq (basisVecNat i) (basisVecNat j) = polyMulCoeffs (basisVecNat i) (basisVecNat j) := by
  let ii : Fin d := ⟨i, hi⟩
  let jj : Fin d := ⟨j, hj⟩
  calc
    mulRq (basisVecNat i) (basisVecNat j)
        = monomialReduce (i + j) := by simpa [ii, jj] using mulRqPhi_basis_basis ii jj
    _ = polyMulCoeffs (basisVecNat i) (basisVecNat j) := by
          apply coeffsToPolynomial_injective_of_size_d
          · simp [monomialReduce_size]
          · simp [polyMulCoeffs_size]
          calc
            coeffsToPolynomial (monomialReduce (i + j))
                = ((coeffsToPolynomial (basisVecNat i) * coeffsToPolynomial (basisVecNat j)) %ₘ phiPolynomial) := by
                    symm
                    simpa [ii, jj] using coeffsToPolynomial_basis_basis_mod ii jj
            _ = coeffsToPolynomial (polyMulCoeffs (basisVecNat i) (basisVecNat j)) := by
                    symm
                    simp [coeffsToPolynomial_polyMulCoeffs]

set_option maxHeartbeats 4000000 in
private theorem mulRq_eq_polyMulCoeffs
    {a b : Coeffs}
    (ha : a.size = d)
    (hb : b.size = d) :
    mulRq a b = polyMulCoeffs a b := by
  exact bilinear_eq_of_basis
    (K := mulRq)
    (L := polyMulCoeffs)
    (hKSize := fun x y _ _ => by simp [mulRq_size])
    (hLSize := fun x y _ _ => by simp [polyMulCoeffs_size])
    (hKAddLeft := fun x y z hx hy _ => mulRqPhi_vecAdd_left_of_size_d x y z hx hy)
    (hLAddLeft := fun x y z hx hy hz => polyMulCoeffs_vecAdd_left hx hy hz)
    (hKScaleLeft := fun s x z hx _ => mulRqPhi_vecScale_left_of_size_d s x z hx)
    (hLScaleLeft := fun s x z hx hz => polyMulCoeffs_vecScale_left hx hz)
    (hKAddRight := fun z x y _ hx hy => mulRqPhi_vecAdd_right_of_size_d z x y hx hy)
    (hLAddRight := fun z x y hz hx hy => polyMulCoeffs_vecAdd_right hz hx hy)
    (hKScaleRight := fun s z x _ hx => mulRqPhi_vecScale_right_of_size_d s z x hx)
    (hLScaleRight := fun s z x hz hx => polyMulCoeffs_vecScale_right hz hx)
    (hBasis := fun i j hi hj => mulRq_basis_basis_eq_polyMulCoeffs i j hi hj)
    a b ha hb

private theorem coeffsToPolynomial_mulRq
    {a b : Coeffs}
    (ha : a.size = d)
    (hb : b.size = d) :
    coeffsToPolynomial (mulRq a b) =
      ((coeffsToPolynomial a * coeffsToPolynomial b) %ₘ phiPolynomial) := by
  rw [mulRq_eq_polyMulCoeffs ha hb]
  exact coeffsToPolynomial_polyMulCoeffs a b

private theorem isCoprime_phiPolynomial_coeffsToPolynomial
    {a : Coeffs}
    (ha : hasRingDegreeShape a)
    (hNorm : normInfCoeffs a ≤ 4)
    (hNe : a ≠ zeroRq) :
    IsCoprime phiPolynomial (coeffsToPolynomial a) := by
  have hA : IsCoprime phiA (coeffsToPolynomial a) := by
    exact (phiA_irreducible.coprime_iff_not_dvd).2
      (not_dvd_phiA_of_shape_norm_le_four_ne_zeroRq ha hNorm hNe)
  have hB : IsCoprime phiB (coeffsToPolynomial a) := by
    exact (phiB_irreducible.coprime_iff_not_dvd).2
      (not_dvd_phiB_of_shape_norm_le_four_ne_zeroRq ha hNorm hNe)
  simpa [phi_factor] using hA.mul_left hB

private theorem isCoprime_phiPolynomial_coeffsToPolynomial_lt_omegaNat
    {a : Coeffs}
    (ha : hasRingDegreeShape a)
    (hNorm : normInfCoeffs a < goldilocksOmegaNat)
    (hNe : a ≠ zeroRq) :
    IsCoprime phiPolynomial (coeffsToPolynomial a) := by
  have hA : IsCoprime phiA (coeffsToPolynomial a) := by
    exact (phiA_irreducible.coprime_iff_not_dvd).2
      (not_dvd_phiA_of_shape_norm_lt_omegaNat_ne_zeroRq ha hNorm hNe)
  have hB : IsCoprime phiB (coeffsToPolynomial a) := by
    exact (phiB_irreducible.coprime_iff_not_dvd).2
      (not_dvd_phiB_of_shape_norm_lt_omegaNat_ne_zeroRq ha hNorm hNe)
  simpa [phi_factor] using hA.mul_left hB

private theorem invertibleRq_of_isCoprime_coeffsToPolynomial
    {a : Coeffs}
    (ha : hasRingDegreeShape a)
    (hcop : IsCoprime (coeffsToPolynomial a) phiPolynomial) :
    invertibleRq a := by
  rcases hcop with ⟨u, v, huv⟩
  let ubar : Polynomial Fq := u %ₘ phiPolynomial
  refine ⟨polynomialToCoeffs ubar, ?_⟩
  apply coeffsToPolynomial_injective_of_size_d
  · simpa [hasRingDegreeShape] using ha
  · exact oneRq_size
  · calc
      coeffsToPolynomial (mulRq a (polynomialToCoeffs ubar))
          = ((coeffsToPolynomial a * coeffsToPolynomial (polynomialToCoeffs ubar)) %ₘ phiPolynomial) := by
              exact coeffsToPolynomial_mulRq (by simpa [hasRingDegreeShape] using ha) (by simp [polynomialToCoeffs_size])
      _ = ((coeffsToPolynomial a * ubar) %ₘ phiPolynomial) := by
            have hubarDeg : ubar.degree < d := by
              simpa [ubar, phiPolynomial_degree] using Polynomial.degree_modByMonic_lt u phiPolynomial_monic
            simpa [ubar] using congrArg
              (fun p => ((coeffsToPolynomial a * p) %ₘ phiPolynomial))
              (coeffsToPolynomial_polynomialToCoeffs_of_degree_lt hubarDeg)
      _ = ((coeffsToPolynomial a * (u %ₘ phiPolynomial)) %ₘ phiPolynomial) := by
            simp [ubar]
      _ = ((coeffsToPolynomial a * u) %ₘ phiPolynomial) := by
            have hpmod : coeffsToPolynomial a %ₘ phiPolynomial = coeffsToPolynomial a := by
              exact (Polynomial.modByMonic_eq_self_iff phiPolynomial_monic).2 <|
                by simpa [phiPolynomial_degree] using degree_coeffsToPolynomial_lt_d a
            have hubarMod : (u %ₘ phiPolynomial) %ₘ phiPolynomial = u %ₘ phiPolynomial := by
              exact (Polynomial.modByMonic_eq_self_iff phiPolynomial_monic).2 <|
                by simpa [phiPolynomial_degree] using Polynomial.degree_modByMonic_lt u phiPolynomial_monic
            rw [Polynomial.mul_modByMonic (p₁ := coeffsToPolynomial a) (p₂ := u) (q := phiPolynomial)]
            simp [hpmod, hubarMod]
      _ = 1 := by
            have huv' : coeffsToPolynomial a * u + v * phiPolynomial = 1 := by
              simpa [mul_comm] using huv
            have hrepr : (1 : Polynomial Fq) + phiPolynomial * (-v) = coeffsToPolynomial a * u := by
              calc
                (1 : Polynomial Fq) + phiPolynomial * (-v)
                    = (coeffsToPolynomial a * u + v * phiPolynomial) + phiPolynomial * (-v) := by
                        rw [huv']
                _ = coeffsToPolynomial a * u := by ring_nf
            have hdeg : ((1 : Polynomial Fq).degree) < phiPolynomial.degree := by
              simpa [phiPolynomial_degree, d]
            exact (Polynomial.div_modByMonic_unique
              (q := -v)
              (r := (1 : Polynomial Fq))
              phiPolynomial_monic
              ⟨hrepr, hdeg⟩).2
      _ = coeffsToPolynomial oneRq := by simp

theorem invertibleRq_of_shape_norm_le_four_ne_zeroRq
    {a : Coeffs}
    (ha : hasRingDegreeShape a)
    (hNorm : normInfCoeffs a ≤ 4)
    (hNe : a ≠ zeroRq) :
    invertibleRq a := by
  exact invertibleRq_of_isCoprime_coeffsToPolynomial ha
    (isCoprime_phiPolynomial_coeffsToPolynomial ha hNorm hNe).symm

private theorem invertibleRq_of_shape_norm_lt_omegaNat_ne_zeroRq
    {a : Coeffs}
    (ha : hasRingDegreeShape a)
    (hNorm : normInfCoeffs a < goldilocksOmegaNat)
    (hNe : a ≠ zeroRq) :
    invertibleRq a := by
  exact invertibleRq_of_isCoprime_coeffsToPolynomial ha
    (isCoprime_phiPolynomial_coeffsToPolynomial_lt_omegaNat ha hNorm hNe).symm

private theorem lowNormInvertibilityAssumption_omegaNat_goldilocks :
    lowNormInvertibilityAssumption goldilocksOmegaNat := by
  intro a hShape hWin
  have hNe : a ≠ zeroRq := by
    intro hEq
    have hPos : 0 < normInfCoeffs a := hWin.1
    rw [hEq, normInfCoeffs_zeroRq] at hPos
    omega
  exact invertibleRq_of_shape_norm_lt_omegaNat_ne_zeroRq hShape hWin.2 hNe

/--
Concrete Goldilocks low-norm invertibility theorem at the Appendix B.2 paper
floor: every ring-shaped coefficient vector with `0 < ‖a‖∞ < 383` is invertible.
-/
theorem lowNormInvertibilityAssumption_paperBInv_goldilocks :
    lowNormInvertibilityAssumption goldilocksPaperBInv := by
  intro a hShape hWin
  exact invertibleRq_of_lowNormAssumption
    lowNormInvertibilityAssumption_omegaNat_goldilocks
    hShape
    (strictInvertibilityWindowProp_mono (by native_decide) hWin)

/--
Concrete Goldilocks low-norm invertibility theorem on actual ring elements:
every ring-shaped coefficient vector with `0 < ‖a‖∞ < 5` is invertible.
-/
theorem lowNormInvertibilityAssumption_five_goldilocks :
    lowNormInvertibilityAssumption 5 := by
  intro a hShape hWin
  exact invertibleRq_of_lowNormAssumption
    lowNormInvertibilityAssumption_paperBInv_goldilocks
    hShape
    (strictInvertibilityWindowProp_mono (by native_decide) hWin)

theorem paperCarrierDiffInvertibilityAssumption_goldilocks :
    paperCarrierDiffInvertibilityAssumption := by
  exact paperCarrierDiffInvertibilityAssumption_of_lowNormPaperBInv
    lowNormInvertibilityAssumption_paperBInv_goldilocks

end

end SuperNeo
