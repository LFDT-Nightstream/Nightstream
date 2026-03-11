import Mathlib.Data.Nat.Prime.Basic
import SuperNeo.Norm
import SuperNeo.SamplingSet

/-!
Low-norm invertibility boundary (Theorem 8 style).

This module separates two notions:
- `invertibilityWindowProp` is the weak norm-window bookkeeping predicate used by
  arithmetic bundles.
- `strictInvertibilityWindowProp` is the actual Theorem-8-style premise
  `0 < ‖a‖∞ < B`.

The old weak reading `normInfCoeffs a ≤ B -> invertibleRq a` is false because it
includes `a = 0`; the module proves that counterexample explicitly.
-/

namespace SuperNeo

/-- Ring-level invertibility witness predicate. -/
def invertibleRq (a : Coeffs) : Prop :=
  ∃ aInv : Coeffs, mulRq a aInv = oneRq

/-- Weak norm-window bookkeeping predicate used by arithmetic bundles. -/
def invertibilityWindowProp (B : Nat) (a : Coeffs) : Prop :=
  normInfCoeffs a ≤ B

/-- Paper-faithful strict norm window `0 < ‖a‖∞ < B`. -/
def strictInvertibilityWindowProp (B : Nat) (a : Coeffs) : Prop :=
  0 < normInfCoeffs a ∧ normInfCoeffs a < B

/-- External Theorem-8-style boundary: strictly low-norm ring elements are invertible in `Rq`. -/
def lowNormInvertibilityAssumption (B : Nat) : Prop :=
  ∀ a : Coeffs, hasRingDegreeShape a → strictInvertibilityWindowProp B a → invertibleRq a

/--
Active protocol-path invertibility boundary: every nonzero difference of two
paper-carrier elements is invertible in `Rq`.
-/
def paperCarrierDiffInvertibilityAssumption : Prop :=
  ∀ δ : Coeffs, samplingDiffSet paperCarrier δ → δ ≠ zeroRq → invertibleRq δ

/-- The strict paper-style window implies the weak bookkeeping bound. -/
theorem invertibilityWindowProp_of_strictWindow
  {B : Nat} {a : Coeffs}
  (h : strictInvertibilityWindowProp B a) :
  invertibilityWindowProp B a := by
  exact Nat.le_of_lt h.2

/-- Monotonicity of the strict paper-style norm window in the upper bound. -/
theorem strictInvertibilityWindowProp_mono
  {B B' : Nat} {a : Coeffs}
  (hBB' : B ≤ B')
  (h : strictInvertibilityWindowProp B a) :
  strictInvertibilityWindowProp B' a := by
  exact ⟨h.1, Nat.lt_of_lt_of_le h.2 hBB'⟩

/-- `zeroRq` has norm `0`. -/
theorem normInfCoeffs_zeroRq : normInfCoeffs zeroRq = 0 := by
  apply Nat.le_antisymm
  · apply normInfCoeffs_le_of_forall_coeffAt zeroRq_size
    intro i hi
    simp [coeffAt_zeroRq, normInfF_zero]
  · exact Nat.zero_le _

/-- Any ring-shaped coefficient vector with zero norm is the zero ring element. -/
theorem eq_zeroRq_of_hasRingDegreeShape_of_normInfCoeffs_eq_zero
  {a : Coeffs}
  (ha : hasRingDegreeShape a)
  (hNorm : normInfCoeffs a = 0) :
  a = zeroRq := by
  apply Array.ext
  · simpa [hasRingDegreeShape] using ha
  · intro i hiA hiZ
    have hi : i < d := by
      rw [ha] at hiA
      exact hiA
    have hCoeffLe : normInfF (coeffAt a i) ≤ 0 := by
      simpa [hNorm] using (normInfF_coeffAt_le_normInfCoeffs a i)
    have hCoeffZero : coeffAt a i = 0 := by
      exact (normInfF_eq_zero_iff (coeffAt a i)).mp (Nat.eq_zero_of_le_zero hCoeffLe)
    have hAi : a[i]'hiA = 0 := by
      simpa [coeffAt, hi, Array.getD, hiA] using hCoeffZero
    have hZi : zeroRq[i]'hiZ = 0 := by
      simp [zeroRq]
    exact hAi.trans hZi.symm

/-- A nonzero ring-shaped coefficient vector has positive infinity norm. -/
theorem normInfCoeffs_pos_of_hasRingDegreeShape_of_ne_zeroRq
  {a : Coeffs}
  (ha : hasRingDegreeShape a)
  (hNe : a ≠ zeroRq) :
  0 < normInfCoeffs a := by
  by_cases hZero : normInfCoeffs a = 0
  · exact False.elim (hNe (eq_zeroRq_of_hasRingDegreeShape_of_normInfCoeffs_eq_zero ha hZero))
  · exact Nat.pos_of_ne_zero hZero

/-- Concrete strict window extracted from the proved paper-carrier diff bound `< 5`. -/
theorem strictInvertibilityWindowProp_five_of_shape_norm_le_four_of_ne_zeroRq
  {a : Coeffs}
  (ha : hasRingDegreeShape a)
  (hNorm : normInfCoeffs a ≤ 4)
  (hNe : a ≠ zeroRq) :
  strictInvertibilityWindowProp 5 a := by
  constructor
  · exact normInfCoeffs_pos_of_hasRingDegreeShape_of_ne_zeroRq ha hNe
  · exact Nat.lt_of_le_of_lt hNorm (by decide)

/-- The weak bookkeeping window always contains `zeroRq`. -/
theorem invertibilityWindowProp_zeroRq (B : Nat) :
  invertibilityWindowProp B zeroRq := by
  simp [invertibilityWindowProp, normInfCoeffs_zeroRq]

private theorem vecScale_zero_of_size_d
  (x : Coeffs)
  (hx : x.size = d) :
  vecScale (0 : F) x = zeroRq := by
  apply Array.ext
  · simp [vecScale, zeroRq, hx]
  · intro i hiL hiR
    have hi : i < d := by
      simpa [vecScale, hx, zeroRq] using hiL
    calc
      (vecScale (0 : F) x)[i]'hiL = (0 : F) * x[i] := by
        simp [vecScale]
      _ = 0 := by
        simpa using (Lean.Grind.Fin.zero_mul (n := Goldilocks.q) x[i])
      _ = zeroRq[i]'hiR := by
        simp [zeroRq]

private theorem mulRq_zero_left (a : Coeffs) :
  mulRq zeroRq a = zeroRq := by
  calc
    mulRq zeroRq a = mulRq (vecScale (0 : F) oneRq) a := by
      rw [vecScale_zero_of_size_d oneRq oneRq_size]
    _ = vecScale (0 : F) (mulRq oneRq a) := by
      exact mulRq_vecScale_left (s := (0 : F)) (x := oneRq) (b := a) oneRq_size
    _ = zeroRq := by
      exact vecScale_zero_of_size_d (mulRq oneRq a) (mulRq_size oneRq a)

/-- `0` is not invertible in `Rq`. -/
theorem not_invertibleRq_zeroRq : ¬ invertibleRq zeroRq := by
  intro hInv
  rcases hInv with ⟨aInv, hMul⟩
  have hCt : (0 : F) = 1 := by
    simpa [mulRq_zero_left, ct_zeroRq, ct_oneRq] using congrArg ct hMul
  exact (by decide : (0 : F) ≠ 1) hCt

/-- The old weak reading `‖a‖∞ ≤ B -> invertibleRq a` is false because it includes `zeroRq`. -/
theorem not_all_window_elements_invertible (B : Nat) :
  ¬ (∀ a : Coeffs, invertibilityWindowProp B a → invertibleRq a) := by
  intro hAll
  exact not_invertibleRq_zeroRq (hAll zeroRq (invertibilityWindowProp_zeroRq B))

/-- Use the low-norm invertibility boundary to extract an inverse witness. -/
theorem invertibleRq_of_lowNormAssumption
  {B : Nat} {a : Coeffs}
  (hInv : lowNormInvertibilityAssumption B)
  (hShape : hasRingDegreeShape a)
  (hWin : strictInvertibilityWindowProp B a) :
  invertibleRq a := by
  exact hInv a hShape hWin

/--
Any strict low-norm invertibility theorem with threshold at least `5` implies
the narrower active protocol-path boundary on nonzero paper-carrier
differences.
-/
theorem paperCarrierDiffInvertibilityAssumption_of_lowNormAtLeastFive
  {B : Nat}
  (hFive : 5 ≤ B)
  (hInv : lowNormInvertibilityAssumption B) :
  paperCarrierDiffInvertibilityAssumption := by
  intro δ hDiff hNe
  rcases samplingDiffSet_paperCarrier_hasRingDegreeShape_and_norm_le_four hDiff with
    ⟨hShape, hNorm⟩
  exact invertibleRq_of_lowNormAssumption hInv hShape
    (strictInvertibilityWindowProp_mono hFive
      (strictInvertibilityWindowProp_five_of_shape_norm_le_four_of_ne_zeroRq
        hShape hNorm hNe))

/--
The strict low-norm Theorem-8 boundary at `B = 5` implies the narrower active
protocol-path boundary on nonzero paper-carrier differences.
-/
theorem paperCarrierDiffInvertibilityAssumption_of_lowNormFive
  (hInv : lowNormInvertibilityAssumption 5) :
  paperCarrierDiffInvertibilityAssumption := by
  exact paperCarrierDiffInvertibilityAssumption_of_lowNormAtLeastFive (B := 5) (by decide) hInv

/-!
Concrete Goldilocks arithmetic side-conditions for the paper's low-norm
invertibility theorem.

For the active SuperNeo Goldilocks path, the paper cites Theorem 8 with
`η = 81`, `q = Goldilocks.q`, and the splitting parameter `z = 3`, which gives
`ord_η(q) = η / z = 27`. These lemmas discharge the arithmetic checks entirely
in-repo. What remains open is the external theorem itself, not its concrete
instantiation.
-/

/-- Goldilocks/SuperNeo Theorem-8 splitting parameter `z = 3`. -/
def goldilocksTheorem8Z : Nat := 3

/-- Goldilocks/SuperNeo witness for `ord_η(q) = 27 = η / z`. -/
def goldilocksTheorem8Order : Nat := 27

/-- Appendix B.2 paper route uses `b_inv ≈ 383`; keep the floor explicit. -/
def goldilocksPaperBInv : Nat := 383

theorem goldilocksTheorem8Z_dvd_eta :
  goldilocksTheorem8Z ∣ Parameters.Goldilocks.eta := by
  native_decide

theorem goldilocksModulus_mod_theorem8Z_eq_one :
  Parameters.Goldilocks.modulus % goldilocksTheorem8Z = 1 := by
  native_decide

theorem goldilocksModulus_pow_order_eq_one_mod_eta :
  (Parameters.Goldilocks.modulus ^ goldilocksTheorem8Order) % Parameters.Goldilocks.eta = 1 := by
  native_decide

/-- The only proper positive divisors of `27` are `1`, `3`, and `9`. -/
private theorem properPositiveDivisorsOfTwentySeven :
  ∀ k : Nat, k ∣ 27 → 0 < k → k < 27 → k = 1 ∨ k = 3 ∨ k = 9 := by
  intro k hk hpos hlt
  have hPrime3 : Nat.Prime 3 := by
    native_decide
  have hkPow : ∃ i ≤ 3, k = 3 ^ i := by
    have hk' : k ∣ 3 ^ 3 := by
      simpa using hk
    exact (Nat.dvd_prime_pow hPrime3).1 hk'
  rcases hkPow with ⟨i, hi, rfl⟩
  have hiCases : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by
    omega
  rcases hiCases with rfl | rfl | rfl | rfl
  · exact Or.inl rfl
  · exact Or.inr (Or.inl rfl)
  · exact Or.inr (Or.inr rfl)
  · have hFalse : ¬ (27 < 27) := Nat.lt_irrefl 27
    exact False.elim (hFalse (by simpa using hlt))

/--
Among the proper positive divisors of `27`, none sends the Goldilocks modulus
to `1` modulo `η = 81`; together with `q^27 ≡ 1`, this is the exact-order side
condition needed by the paper's theorem.
-/
theorem goldilocksModulus_order_mod_eta :
  ∀ k : Nat, k ∣ goldilocksTheorem8Order → 0 < k → k < goldilocksTheorem8Order →
    (Parameters.Goldilocks.modulus ^ k) % Parameters.Goldilocks.eta ≠ 1 := by
  intro k hk hpos hlt
  have hk' : k ∣ 27 := by
    simpa [goldilocksTheorem8Order] using hk
  have hlt' : k < 27 := by
    simpa [goldilocksTheorem8Order] using hlt
  have hkCases : k = 1 ∨ k = 3 ∨ k = 9 := by
    exact properPositiveDivisorsOfTwentySeven k hk' hpos hlt'
  rcases hkCases with rfl | rfl | rfl <;> native_decide

/--
For the Goldilocks Appendix-B.2 route with `z = 3`, the paper's explicit
invertibility threshold is well above `5`, so the active paper-carrier
difference path is covered if the cited low-norm theorem is available.
-/
theorem goldilocksTheorem8Bound_gt_five :
  3 * 5 ^ 2 < Parameters.Goldilocks.modulus := by
  native_decide

/-- The Goldilocks Appendix-B.2 paper threshold `b_inv ≈ 383` also lies below the concrete modulus bound. -/
theorem goldilocksTheorem8Bound_gt_paperBInv :
  3 * goldilocksPaperBInv ^ 2 < Parameters.Goldilocks.modulus := by
  native_decide

/-- Specialized bridge from the paper's concrete Goldilocks bound `b_inv = 383`. -/
theorem paperCarrierDiffInvertibilityAssumption_of_lowNormPaperBInv
  (hInv : lowNormInvertibilityAssumption goldilocksPaperBInv) :
  paperCarrierDiffInvertibilityAssumption := by
  exact paperCarrierDiffInvertibilityAssumption_of_lowNormAtLeastFive
    (B := goldilocksPaperBInv) (by decide) hInv

/-! Compatibility precondition package retained for protocol arithmetic glue. -/

/-- Compact invertibility precondition placeholder used by protocol bundle constructors. -/
def invertibilityPreconditionsProp : Prop := True

/-- Canonical constructor for compact invertibility preconditions. -/
theorem invertibilityPreconditions_from_constants : invertibilityPreconditionsProp := by
  trivial


end SuperNeo
