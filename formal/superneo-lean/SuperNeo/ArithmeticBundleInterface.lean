import SuperNeo.ArithmeticBundle

/-!
Contract interface for `SuperNeo.ArithmeticBundle`.

Spec: ./formal/superneo-lean/specs/ArithmeticBundle.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Section 4 (Preliminaries): decomposition, matrix transform, evaluation homomorphism.
- Section 5 (Embedding products): Theorem 4 (Mz = ct(M̄z)), Theorem 5 (Evaluation homomorphism).
- Section 7 (Folding scheme): arithmetic obligations composed for protocol context, lines 449-596.
-/

namespace SuperNeo

namespace ArithmeticBundleInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `arithmeticEvalHomProp`. -/
abbrev arithmeticEvalHomProp := SuperNeo.arithmeticEvalHomProp

/-- [Role: Theorem-Target] Curated re-export of `arithmeticVecModuleProp`. -/
abbrev arithmeticVecModuleProp := SuperNeo.arithmeticVecModuleProp

/-- [Role: Theorem-Target] Curated re-export of `arithmeticScalarModuleProp`. -/
abbrev arithmeticScalarModuleProp := SuperNeo.arithmeticScalarModuleProp

/-- [Role: Theorem-Target] Curated re-export of `arithmeticSamplingProp`. -/
abbrev arithmeticSamplingProp := SuperNeo.arithmeticSamplingProp

/-- [Role: Theorem-Target] Curated re-export of `arithmeticPolyProp`. -/
abbrev arithmeticPolyProp := SuperNeo.arithmeticPolyProp

/-- [Role: Theorem-Target] Curated re-export of `arithmeticDecompProp`. -/
abbrev arithmeticDecompProp := SuperNeo.arithmeticDecompProp

/-- [Role: Theorem-Target] Curated re-export of `arithmeticInterpProp`. -/
abbrev arithmeticInterpProp := SuperNeo.arithmeticInterpProp

/-- [Role: Theorem-Target] Curated re-export of `arithmeticBundleProp`. -/
abbrev arithmeticBundleProp := SuperNeo.arithmeticBundleProp

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `arithmeticDecompProp_iff_splitRoundTrip_true`. -/
abbrev arithmeticDecompProp_iff_splitRoundTrip_true := @SuperNeo.arithmeticDecompProp_iff_splitRoundTrip_true

/-- [Role: Theorem-Target] Curated theorem surface `arithmeticBundleProp_of_props`. -/
abbrev arithmeticBundleProp_of_props := @SuperNeo.arithmeticBundleProp_of_props

/-- [Role: Theorem-Target] Theorem-native constructor threading `P10` into P20 obligations. -/
abbrev arithmeticBundleProp_of_theorem_stack := @SuperNeo.arithmeticBundleProp_of_theorem_stack

/-- [Role: Theorem-Target] Curated theorem surface `arithmeticBundleProp_checks_imply_props`. -/
abbrev arithmeticBundleProp_checks_imply_props := @SuperNeo.arithmeticBundleProp_checks_imply_props

/-- [Role: Theorem-Target] Curated theorem surface `arithmeticBundleProp_props_imply_check_subset`. -/
abbrev arithmeticBundleProp_props_imply_check_subset := @SuperNeo.arithmeticBundleProp_props_imply_check_subset

/-- [Role: Theorem-Target] Curated theorem surface `arithmeticBundleProp_props_imply_module_checks`. -/
abbrev arithmeticBundleProp_props_imply_module_checks := @SuperNeo.arithmeticBundleProp_props_imply_module_checks

/-- [Role: Theorem-Target] Curated theorem surface `arithmeticBundleProp_of_checks`. -/
abbrev arithmeticBundleProp_of_checks := @SuperNeo.arithmeticBundleProp_of_checks

end ArithmeticBundleInterface

end SuperNeo
