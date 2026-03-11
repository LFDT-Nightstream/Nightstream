import SuperNeo.InvertibilityGoldilocks

/-!
Contract interface for `SuperNeo.InvertibilityGoldilocks`.

Spec: `specs/InvertibilityGoldilocks.spec.md`

Paper anchors:
- Theorem 8 (Low-norm invertibility), Section 5/6, lines 375-378.
- Appendix B.2 concrete floor `b_inv = 383`.
-/

namespace SuperNeo

namespace InvertibilityGoldilocksInterface

/-- [Role: Theorem-Target] Concrete Goldilocks low-norm invertibility at the Appendix B.2 floor. -/
theorem lowNormInvertibilityAssumption_paperBInv_goldilocks :
  lowNormInvertibilityAssumption goldilocksPaperBInv :=
  SuperNeo.lowNormInvertibilityAssumption_paperBInv_goldilocks

/-- [Role: Theorem-Target] Concrete Goldilocks low-norm invertibility at the narrower threshold `5`. -/
theorem lowNormInvertibilityAssumption_five_goldilocks :
  lowNormInvertibilityAssumption 5 :=
  SuperNeo.lowNormInvertibilityAssumption_five_goldilocks

/-- [Role: Theorem-Target] Active protocol-path invertibility for nonzero `paperCarrier` differences. -/
theorem paperCarrierDiffInvertibilityAssumption_goldilocks :
  paperCarrierDiffInvertibilityAssumption :=
  SuperNeo.paperCarrierDiffInvertibilityAssumption_goldilocks

end InvertibilityGoldilocksInterface

end SuperNeo
