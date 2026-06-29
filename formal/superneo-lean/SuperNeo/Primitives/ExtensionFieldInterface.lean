import SuperNeo.Primitives.ExtensionField

/-!
Contract interface for `SuperNeo.ExtensionField`.

Spec: `specs/ExtensionField.spec.md`

Paper anchors:
- Definition 1, Section 4, lines 275-282: extension field `K` of degree `2`.
- The opening-convergence target uses the quadratic extension carrier for
  multilinear evaluation points and packed-column evaluations.
-/

namespace SuperNeo

namespace ExtensionFieldInterface

/-- [Role: Theorem-Target] Curated re-export of the extension carrier. -/
abbrev KExt := SuperNeo.KExt

/-- [Role: Theorem-Target] Curated re-export of the quadratic relation constant. -/
abbrev KExt_w := SuperNeo.KExt.w

/-- [Role: Theorem-Target] Curated re-export of the base-field embedding. -/
abbrev KExt_ofF := SuperNeo.KExt.ofF

/-- [Role: Theorem-Target] Curated re-export of the coefficient constructor. -/
abbrev KExt_ofCoeffs := SuperNeo.KExt.ofCoeffs

/-- [Role: Theorem-Target] Curated re-export of the coefficient view. -/
abbrev KExt_coeffs := SuperNeo.KExt.coeffs

/-- [Role: Theorem-Target] Curated re-export of base-field scaling. -/
abbrev KExt_scaleBase := SuperNeo.KExt.scaleBase

/-- [Role: Theorem-Target] Curated re-export of exponentiation. -/
abbrev KExt_pow := SuperNeo.KExt.pow

/-- [Role: Theorem-Target] Recover the extension degree parameter. -/
theorem extDegreeK_eq_two : SuperNeo.Parameters.Goldilocks.extDegreeK = 2 :=
  SuperNeo.extDegreeK_eq_two

/-- [Role: Theorem-Target] Exact embedding coordinates for `ofF`. -/
abbrev KExt_ofF_re := SuperNeo.KExt.ofF_re
abbrev KExt_ofF_im := SuperNeo.KExt.ofF_im

/-- [Role: Theorem-Target] Exact multiplication coordinates under `u^2 = 7`. -/
abbrev KExt_mul_re := SuperNeo.KExt.mul_re
abbrev KExt_mul_im := SuperNeo.KExt.mul_im

/-- [Role: Theorem-Target] Exact base-scaling coordinates. -/
abbrev KExt_scaleBase_re := SuperNeo.KExt.scaleBase_re
abbrev KExt_scaleBase_im := SuperNeo.KExt.scaleBase_im

/-- [Role: Theorem-Target] Coordinate extensionality for `KExt`. -/
theorem KExt_ext
    {x y : KExt}
    (hre : x.re = y.re)
    (him : x.im = y.im) : x = y :=
  SuperNeo.KExt.ext hre him

end ExtensionFieldInterface

end SuperNeo
