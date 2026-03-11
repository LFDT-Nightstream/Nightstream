import SuperNeo.InvertibilityAxioms

/-!
Contract interface for `SuperNeo.InvertibilityAxioms`.

Spec: `specs/InvertibilityAxioms.spec.md`

Paper anchors:
- Theorem 8 (Low-norm invertibility), Section 5/6, lines 375-378.
-/

namespace SuperNeo

namespace InvertibilityAxiomsInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `invertibleRq`. -/
abbrev invertibleRq := SuperNeo.invertibleRq

/-- [Role: Theorem-Target] Curated re-export of `invertibilityWindowProp`. -/
abbrev invertibilityWindowProp := SuperNeo.invertibilityWindowProp

/-- [Role: Theorem-Target] Curated re-export of `strictInvertibilityWindowProp`. -/
abbrev strictInvertibilityWindowProp := SuperNeo.strictInvertibilityWindowProp

/-- [Role: Theorem-Target] Curated re-export of `invertibilityPreconditionsProp`. -/
abbrev invertibilityPreconditionsProp := SuperNeo.invertibilityPreconditionsProp

/-- [Role: Theorem-Target] Concrete Goldilocks/SuperNeo Theorem-8 splitting parameter. -/
abbrev goldilocksTheorem8Z := SuperNeo.goldilocksTheorem8Z

/-- [Role: Theorem-Target] Concrete Goldilocks/SuperNeo order witness `27 = η / z`. -/
abbrev goldilocksTheorem8Order := SuperNeo.goldilocksTheorem8Order

/-- [Role: Theorem-Target] Appendix B.2 paper floor for the invertibility threshold. -/
abbrev goldilocksPaperBInv := SuperNeo.goldilocksPaperBInv

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `invertibilityPreconditions_from_constants`. -/
abbrev invertibilityPreconditions_from_constants := SuperNeo.invertibilityPreconditions_from_constants

/-- [Role: Theorem-Target] Goldilocks Theorem-8 splitting parameter divides `η = 81`. -/
theorem goldilocksTheorem8Z_dvd_eta :
  goldilocksTheorem8Z ∣ Parameters.Goldilocks.eta :=
  SuperNeo.goldilocksTheorem8Z_dvd_eta

/-- [Role: Theorem-Target] Goldilocks modulus satisfies `q ≡ 1 (mod z)` for `z = 3`. -/
theorem goldilocksModulus_mod_theorem8Z_eq_one :
  Parameters.Goldilocks.modulus % goldilocksTheorem8Z = 1 :=
  SuperNeo.goldilocksModulus_mod_theorem8Z_eq_one

/-- [Role: Theorem-Target] Goldilocks modulus satisfies `q^27 ≡ 1 (mod η)`. -/
theorem goldilocksModulus_pow_order_eq_one_mod_eta :
  (Parameters.Goldilocks.modulus ^ goldilocksTheorem8Order) % Parameters.Goldilocks.eta = 1 :=
  SuperNeo.goldilocksModulus_pow_order_eq_one_mod_eta

/-- [Role: Theorem-Target] No proper positive divisor of `27` gives residue `1` modulo `η`. -/
theorem goldilocksModulus_order_mod_eta
  {k : Nat}
  (hk : k ∣ goldilocksTheorem8Order)
  (hpos : 0 < k)
  (hlt : k < goldilocksTheorem8Order) :
  (Parameters.Goldilocks.modulus ^ k) % Parameters.Goldilocks.eta ≠ 1 :=
  SuperNeo.goldilocksModulus_order_mod_eta k hk hpos hlt

/-- [Role: Theorem-Target] The concrete Goldilocks Theorem-8 bound exceeds the active threshold `5`. -/
theorem goldilocksTheorem8Bound_gt_five :
  3 * 5 ^ 2 < Parameters.Goldilocks.modulus :=
  SuperNeo.goldilocksTheorem8Bound_gt_five

/-- [Role: Theorem-Target] The concrete Goldilocks Theorem-8 bound exceeds the Appendix B.2 floor `383`. -/
theorem goldilocksTheorem8Bound_gt_paperBInv :
  3 * goldilocksPaperBInv ^ 2 < Parameters.Goldilocks.modulus :=
  SuperNeo.goldilocksTheorem8Bound_gt_paperBInv

/-- [Role: Theorem-Target] Curated theorem surface `invertibilityWindowProp_of_strictWindow`. -/
theorem invertibilityWindowProp_of_strictWindow
  {B : Nat} {a : Coeffs}
  (h : strictInvertibilityWindowProp B a) :
  invertibilityWindowProp B a :=
  SuperNeo.invertibilityWindowProp_of_strictWindow h

/-- [Role: Theorem-Target] Curated theorem surface `normInfCoeffs_zeroRq`. -/
theorem normInfCoeffs_zeroRq :
  normInfCoeffs zeroRq = 0 :=
  SuperNeo.normInfCoeffs_zeroRq

/-- [Role: Theorem-Target] Curated theorem surface `invertibilityWindowProp_zeroRq`. -/
theorem invertibilityWindowProp_zeroRq (B : Nat) :
  invertibilityWindowProp B zeroRq :=
  SuperNeo.invertibilityWindowProp_zeroRq B

/-- [Role: Theorem-Target] Curated theorem surface `not_invertibleRq_zeroRq`. -/
theorem not_invertibleRq_zeroRq :
  ¬ invertibleRq zeroRq :=
  SuperNeo.not_invertibleRq_zeroRq

/-- [Role: Theorem-Target] Curated theorem surface `not_all_window_elements_invertible`. -/
theorem not_all_window_elements_invertible (B : Nat) :
  ¬ (∀ a : Coeffs, invertibilityWindowProp B a → invertibleRq a) :=
  SuperNeo.not_all_window_elements_invertible B

/-- [Role: Theorem-Target] Zero norm at ring shape implies the zero ring element. -/
theorem eq_zeroRq_of_hasRingDegreeShape_of_normInfCoeffs_eq_zero
  {a : Coeffs}
  (ha : hasRingDegreeShape a)
  (hNorm : normInfCoeffs a = 0) :
  a = zeroRq :=
  SuperNeo.eq_zeroRq_of_hasRingDegreeShape_of_normInfCoeffs_eq_zero ha hNorm

/-- [Role: Theorem-Target] Nonzero ring-shaped elements have positive infinity norm. -/
theorem normInfCoeffs_pos_of_hasRingDegreeShape_of_ne_zeroRq
  {a : Coeffs}
  (ha : hasRingDegreeShape a)
  (hNe : a ≠ zeroRq) :
  0 < normInfCoeffs a :=
  SuperNeo.normInfCoeffs_pos_of_hasRingDegreeShape_of_ne_zeroRq ha hNe

/-- [Role: Theorem-Target] Ring shape + `‖a‖∞ ≤ 4` + nonzero implies the strict window `< 5`. -/
theorem strictInvertibilityWindowProp_five_of_shape_norm_le_four_of_ne_zeroRq
  {a : Coeffs}
  (ha : hasRingDegreeShape a)
  (hNorm : normInfCoeffs a ≤ 4)
  (hNe : a ≠ zeroRq) :
  strictInvertibilityWindowProp 5 a :=
  SuperNeo.strictInvertibilityWindowProp_five_of_shape_norm_le_four_of_ne_zeroRq
    ha hNorm hNe

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `lowNormInvertibilityAssumption` on ring-shaped elements. -/
abbrev lowNormInvertibilityAssumption := SuperNeo.lowNormInvertibilityAssumption

/-- [Role: Boundary] Active protocol-path invertibility boundary on nonzero paper-carrier differences. -/
abbrev paperCarrierDiffInvertibilityAssumption :=
  SuperNeo.paperCarrierDiffInvertibilityAssumption

/-- [Role: Boundary] Use a shape-aware low-norm boundary assumption to extract invertibility. -/
theorem invertibleRq_of_lowNormAssumption
  {B : Nat} {a : Coeffs}
  (hInv : lowNormInvertibilityAssumption B)
  (hShape : hasRingDegreeShape a)
  (hWin : strictInvertibilityWindowProp B a) :
  invertibleRq a :=
  SuperNeo.invertibleRq_of_lowNormAssumption hInv hShape hWin

/-- [Role: Theorem-Target] Derive the active paper-carrier-difference boundary from Theorem-8 at `B = 5`. -/
theorem paperCarrierDiffInvertibilityAssumption_of_lowNormFive
  (hInv : lowNormInvertibilityAssumption 5) :
  paperCarrierDiffInvertibilityAssumption :=
  SuperNeo.paperCarrierDiffInvertibilityAssumption_of_lowNormFive hInv

/-- [Role: Theorem-Target] Specialized bridge from the concrete Goldilocks Appendix B.2 floor `383`. -/
theorem paperCarrierDiffInvertibilityAssumption_of_lowNormPaperBInv
  (hInv : lowNormInvertibilityAssumption goldilocksPaperBInv) :
  paperCarrierDiffInvertibilityAssumption :=
  SuperNeo.paperCarrierDiffInvertibilityAssumption_of_lowNormPaperBInv hInv

end InvertibilityAxiomsInterface

end SuperNeo
