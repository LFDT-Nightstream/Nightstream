import SuperNeo.Checks

/-!
Contract interface for `SuperNeo.Checks`.

Spec: `./formal/superneo-lean/specs/Checks.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
  - Definition 1 (Fields), lines 275–282: sanity checks validate Definition 1 and later definitions.
-/

namespace SuperNeo

namespace ChecksInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `checkSuperNeoCases`. -/
abbrev checkSuperNeoCases := SuperNeo.checkSuperNeoCases

/-- [Role: Theorem-Target] Curated re-export of `checkRingMulCases`. -/
abbrev checkRingMulCases := SuperNeo.checkRingMulCases

/-- [Role: Theorem-Target] Curated re-export of `checkNormCases`. -/
abbrev checkNormCases := SuperNeo.checkNormCases

/-- [Role: Theorem-Target] Curated re-export of `checkSplitCases`. -/
abbrev checkSplitCases := SuperNeo.checkSplitCases

/-- [Role: Theorem-Target] Curated re-export of `checkEqCases`. -/
abbrev checkEqCases := SuperNeo.checkEqCases

/-- [Role: Theorem-Target] Curated re-export of `checkMleCases`. -/
abbrev checkMleCases := SuperNeo.checkMleCases

/-- [Role: Theorem-Target] Curated re-export of `checkEmbeddingVecCases`. -/
abbrev checkEmbeddingVecCases := SuperNeo.checkEmbeddingVecCases

/-- [Role: Theorem-Target] Curated re-export of `checkEmbeddingMatrixCases`. -/
abbrev checkEmbeddingMatrixCases := SuperNeo.checkEmbeddingMatrixCases

end ChecksInterface

end SuperNeo
