import SuperNeo.Primitives.ExtensionMLE

/-!
Contract interface for `SuperNeo.ExtensionMLE`.

Spec: `specs/ExtensionMLE.spec.md`

Paper anchors:
- Section 4: equality polynomial / multilinear extension surfaces.
- Opening convergence Phase 1: the relevant carrier is `SuperNeo.KExt`.
-/

namespace SuperNeo

namespace ExtensionMLEInterface

/-- [Role: Theorem-Target] Curated re-export of the extension carrier. -/
abbrev KExt := SuperNeo.KExt

/-- [Role: Theorem-Target] Curated re-export of the Boolean-cube embedding. -/
abbrev bitsToKExtArray := SuperNeo.bitsToKExtArray

/-- [Role: Theorem-Target] Curated re-export of the extension-field equality term. -/
abbrev eqTermK := SuperNeo.eqTermK

/-- [Role: Theorem-Target] Curated re-export of the extension-field equality polynomial. -/
abbrev eqPolyK := SuperNeo.eqPolyK

/-- [Role: Theorem-Target] Curated re-export of the extension-field MLE evaluator. -/
abbrev mleEvalK := SuperNeo.mleEvalK

/-- [Role: Theorem-Target] Curated re-export of the extension-field inner-product form. -/
abbrev mleInnerProductFormK := SuperNeo.mleInnerProductFormK

/-- [Role: Theorem-Target] Curated re-export of the extension-field MLE identity boundary. -/
abbrev mleIdentityAssumptionK := SuperNeo.mleIdentityAssumptionK

/-- [Role: Theorem-Target] Curated theorem surface `mleEvalK_eq_innerProductForm_of_size`. -/
abbrev mleEvalK_eq_innerProductForm_of_size := @SuperNeo.mleEvalK_eq_innerProductForm_of_size

/-- [Role: Theorem-Target] Curated theorem surface `mleIdentityAssumptionK_holds`. -/
abbrev mleIdentityAssumptionK_holds := SuperNeo.mleIdentityAssumptionK_holds

/-- [Role: Theorem-Target] Curated re-export of the folding layer. -/
abbrev foldLayerK := SuperNeo.foldLayerK

/-- [Role: Theorem-Target] Curated theorem surface `foldLayerK_size`. -/
abbrev foldLayerK_size := @SuperNeo.foldLayerK_size

/-- [Role: Theorem-Target] Curated theorem surface `foldLayerK_get`. -/
abbrev foldLayerK_get := @SuperNeo.foldLayerK_get

/-- [Role: Theorem-Target] Curated re-export of the executable folding evaluator. -/
abbrev mleByFoldingExecK := SuperNeo.mleByFoldingExecK

/-- [Role: Theorem-Target] Curated re-export of the theorem-facing folding evaluator. -/
abbrev mleByFoldingK := SuperNeo.mleByFoldingK

/-- [Role: Theorem-Target] Curated theorem surface `mleByFoldingK_step`. -/
abbrev mleByFoldingK_step := @SuperNeo.mleByFoldingK_step

/-- [Role: Theorem-Target] Curated theorem surface `mleByFoldingK_empty`. -/
abbrev mleByFoldingK_empty := @SuperNeo.mleByFoldingK_empty

/-- [Role: Theorem-Target] Curated theorem surface `mleInnerProductFormK_eq_mleByFoldingK_of_size`. -/
abbrev mleInnerProductFormK_eq_mleByFoldingK_of_size :=
  @SuperNeo.mleInnerProductFormK_eq_mleByFoldingK_of_size

/-- [Role: Theorem-Target] Curated re-export of extension-field pointwise linear combination. -/
abbrev linCombK := SuperNeo.linCombK

/-- [Role: Theorem-Target] Curated theorem surface `linCombK_size`. -/
abbrev linCombK_size := @SuperNeo.linCombK_size

/-- [Role: Definitional] Package target for extension-field inner-product linearity. -/
abbrev mleInnerProductLinearityAssumptionK := SuperNeo.mleInnerProductLinearityAssumptionK

/-- [Role: Theorem-Target] Curated theorem surface `mleInnerProductLinearityAssumptionK_holds`. -/
abbrev mleInnerProductLinearityAssumptionK_holds :=
  SuperNeo.mleInnerProductLinearityAssumptionK_holds

/-- [Role: Definitional] Package target for guarded extension-field MLE linearity. -/
abbrev mleEvalLinearityAssumptionK := SuperNeo.mleEvalLinearityAssumptionK

/-- [Role: Theorem-Target] Curated theorem surface `mleEvalK_linComb_of_assumption`. -/
abbrev mleEvalK_linComb_of_assumption := @SuperNeo.mleEvalK_linComb_of_assumption

/-- [Role: Theorem-Target] Curated theorem surface `mleEvalK_linComb_of_assumptions`. -/
abbrev mleEvalK_linComb_of_assumptions := @SuperNeo.mleEvalK_linComb_of_assumptions

/-- [Role: Theorem-Target] Curated theorem surface `mleEvalLinearityAssumptionK_holds`. -/
abbrev mleEvalLinearityAssumptionK_holds :=
  SuperNeo.mleEvalLinearityAssumptionK_holds

/-- [Role: Theorem-Target] Curated theorem surface `mleEvalK_eq_mleByFoldingK_of_size`. -/
abbrev mleEvalK_eq_mleByFoldingK_of_size :=
  @SuperNeo.mleEvalK_eq_mleByFoldingK_of_size

end ExtensionMLEInterface

end SuperNeo
