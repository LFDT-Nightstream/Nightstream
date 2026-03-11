import SuperNeo.DecompBase2

/-!
Contract interface for `SuperNeo.Decomp`.

Spec: `specs/Decomp.spec.md`

Paper anchors:
- Section 4, lines 294-296: `split_b` decomposition into base-`b` digits.
- Section 7.5 (Π_DEC), lines 490-520: decomposition check in folding protocol.
- Appendix B.2, lines 709-727: concrete `b = 2`, `k = 14`.
-/

namespace SuperNeo

namespace DecompInterface

/-! ## Core Surfaces — Base-2 -/

abbrev bitAt := SuperNeo.bitAt
abbrev splitBase2Scalar := SuperNeo.splitBase2Scalar
abbrev recomposeBase2Scalar := SuperNeo.recomposeBase2Scalar
abbrev splitBase2TerminalZeroProp := SuperNeo.splitBase2TerminalZeroProp
abbrev splitBase2Coeffs := SuperNeo.splitBase2Coeffs
abbrev recomposeBase2Coeffs := SuperNeo.recomposeBase2Coeffs

/-! ## Core Surfaces — Balanced -/

abbrev splitBalancedScalar := SuperNeo.splitBalancedScalar
abbrev splitBalancedVec := SuperNeo.splitBalancedVec
abbrev recomposeSplitDigits := SuperNeo.recomposeSplitDigits
abbrev digitsWithinBase := SuperNeo.digitsWithinBase
abbrev digitsWithinBaseProp := SuperNeo.digitsWithinBaseProp
abbrev splitBalancedRoundTripProp := SuperNeo.splitBalancedRoundTripProp
abbrev splitRoundTrip := SuperNeo.splitRoundTrip
abbrev splitBalancedVecDigitBoundProp := SuperNeo.splitBalancedVecDigitBoundProp

/-! ## Base-2 Theorem Surfaces -/

abbrev bitAt_lt_two := SuperNeo.bitAt_lt_two
abbrev splitBase2DecompositionNat := SuperNeo.splitBase2DecompositionNat
abbrev splitBase2DigitsWithinBound := SuperNeo.splitBase2DigitsWithinBound
abbrev splitBase2RowsWithinBound := SuperNeo.splitBase2RowsWithinBound
abbrev recomposeBase2Coeffs_splitBase2Coeffs_entry :=
  SuperNeo.recomposeBase2Coeffs_splitBase2Coeffs_entry
abbrev recomposeBase2Coeffs_splitBase2Coeffs_eq_of_terminal_zero :=
  SuperNeo.recomposeBase2Coeffs_splitBase2Coeffs_eq_of_terminal_zero
abbrev recomposeBase2Coeffs_splitBase2Coeffs_eq_of_val_lt_pow :=
  SuperNeo.recomposeBase2Coeffs_splitBase2Coeffs_eq_of_val_lt_pow

/-! ## Balanced Decomposition Theorem Surfaces -/

abbrev splitBalancedDecompositionInt := SuperNeo.splitBalancedDecompositionInt
abbrev splitBalancedDecompositionInt_of_terminal_zero := SuperNeo.splitBalancedDecompositionInt_of_terminal_zero
abbrev splitBalancedVecFieldLiftProp_holds_of_base_ge_two :=
  SuperNeo.splitBalancedVecFieldLiftProp_holds_of_base_ge_two
abbrev splitBalancedRoundTripProp_of_constructive_boundaries :=
  SuperNeo.splitBalancedRoundTripProp_of_constructive_boundaries

/-! ## Bound/Round-Trip Bridge Surfaces -/

abbrev splitBalancedVecDigitBoundProp_iff_digitsWithinBaseProp :=
  SuperNeo.splitBalancedVecDigitBoundProp_iff_digitsWithinBaseProp

theorem digitsWithinBase_eq_true_iff_prop
  {digits : Array (Array F)} {b : Nat} :
  digitsWithinBase digits b = true ↔ digitsWithinBaseProp digits b :=
  SuperNeo.digitsWithinBase_eq_true_iff_prop

theorem splitRoundTrip_eq_true_iff_prop
  {z : Array F} {b k : Nat} :
  splitRoundTrip z b k = true ↔ splitBalancedRoundTripProp z b k :=
  SuperNeo.splitRoundTrip_eq_true_iff_prop

theorem splitRoundTrip_sound_prop
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  splitBalancedRoundTripProp z b k :=
  SuperNeo.splitRoundTrip_sound_prop hOk

theorem splitRoundTrip_complete_prop
  {z : Array F} {b k : Nat}
  (hProp : splitBalancedRoundTripProp z b k) :
  splitRoundTrip z b k = true :=
  SuperNeo.splitRoundTrip_complete_prop hProp

theorem splitRoundTrip_splitBalancedVecDigitBoundProp
  {z : Array F} {b k : Nat}
  (hOk : splitRoundTrip z b k = true) :
  splitBalancedVecDigitBoundProp z b k :=
  SuperNeo.splitRoundTrip_splitBalancedVecDigitBoundProp hOk

theorem digitsWithinBase_eq_true_of_splitBalancedVecDigitBoundProp
  {z : Array F} {b k : Nat}
  (hBounds : splitBalancedVecDigitBoundProp z b k) :
  digitsWithinBase (splitBalancedVec z b k) b = true :=
  SuperNeo.digitsWithinBase_eq_true_of_splitBalancedVecDigitBoundProp hBounds

end DecompInterface

end SuperNeo
