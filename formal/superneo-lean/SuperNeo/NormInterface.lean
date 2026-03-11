import SuperNeo.Norm

/-!
Contract interface for `SuperNeo.Norm`.

Spec: `specs/Norm.spec.md`

Paper anchors:
- Definition 3, Section 4, lines 290-291: `‖·‖_∞` on ring elements via coefficients.
- Theorem 8, Section 6, lines 375-378: norm bounds enter security parameter.
-/

namespace SuperNeo

namespace NormInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `normInfF`. -/
abbrev normInfF := SuperNeo.normInfF

/-- [Role: Theorem-Target] Curated re-export of `normInfCoeffs`. -/
abbrev normInfCoeffs := SuperNeo.normInfCoeffs

/-- [Role: Theorem-Target] Curated re-export of `maxRhoNorm`. -/
abbrev maxRhoNorm := SuperNeo.maxRhoNorm

/-- [Role: Theorem-Target] Curated re-export of `vecAddNormBoundFromOperands`. -/
abbrev vecAddNormBoundFromOperands := SuperNeo.vecAddNormBoundFromOperands

/-- [Role: Theorem-Target] Curated re-export of `vecScaleNormBoundFromOperands`. -/
abbrev vecScaleNormBoundFromOperands := SuperNeo.vecScaleNormBoundFromOperands

/-- [Role: Theorem-Target] Curated re-export of `mulRqNormBoundFromOperands`. -/
abbrev mulRqNormBoundFromOperands := SuperNeo.mulRqNormBoundFromOperands

/-- [Role: Theorem-Target] Curated re-export of `coeffSubNormBoundFromOperands`. -/
abbrev coeffSubNormBoundFromOperands := SuperNeo.coeffSubNormBoundFromOperands

/-- [Role: Theorem-Target] Curated re-export of `AllChallengeCoeffs`. -/
abbrev AllChallengeCoeffs := SuperNeo.AllChallengeCoeffs

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `normInfF_zero`. -/
theorem normInfF_zero : normInfF (0 : F) = 0 := SuperNeo.normInfF_zero

/-- [Role: Theorem-Target] Curated theorem surface `normInfCoeffs_empty`. -/
theorem normInfCoeffs_empty : normInfCoeffs (#[] : Coeffs) = 0 := SuperNeo.normInfCoeffs_empty

/-- [Role: Theorem-Target] Curated theorem surface `normInfCoeffs_nonneg`. -/
theorem normInfCoeffs_nonneg (a : Coeffs) : 0 ≤ normInfCoeffs a := SuperNeo.normInfCoeffs_nonneg a

/-- [Role: Theorem-Target] Curated theorem surface `maxRhoNorm_nonneg`. -/
theorem maxRhoNorm_nonneg (a : Coeffs) : 0 ≤ maxRhoNorm a := SuperNeo.maxRhoNorm_nonneg a

/-- [Role: Theorem-Target] Curated theorem surface `allChallengeCoeffs_empty`. -/
theorem allChallengeCoeffs_empty : AllChallengeCoeffs (#[] : Coeffs) := SuperNeo.allChallengeCoeffs_empty

/-- [Role: Theorem-Target] Curated theorem surface `allChallengeCoeffs_mono`. -/
theorem allChallengeCoeffs_mono
    {a : Coeffs}
    {B C : Nat}
    (hB : ∀ i : Fin a.size, normInfF a[i] ≤ B)
    (hBC : B ≤ C) :
    ∀ i : Fin a.size, normInfF a[i] ≤ C :=
  SuperNeo.allChallengeCoeffs_mono hB hBC

/-- [Role: Theorem-Target] Curated theorem surface `allChallengeCoeffs_of_bound`. -/
theorem allChallengeCoeffs_of_bound
    {a : Coeffs}
    (hB : ∀ i : Fin a.size, normInfF a[i] ≤ 2) :
    AllChallengeCoeffs a :=
  SuperNeo.allChallengeCoeffs_of_bound hB

/-- [Role: Theorem-Target] Curated theorem surface `allChallengeCoeffs_weaken`. -/
theorem allChallengeCoeffs_weaken
    {a : Coeffs}
    (h : AllChallengeCoeffs a) :
    ∀ i : Fin a.size, normInfF a[i] ≤ 2 :=
  SuperNeo.allChallengeCoeffs_weaken h

/-- [Role: Theorem-Target] Curated theorem surface `vecAddNormBoundFromOperands_of_global`. -/
theorem vecAddNormBoundFromOperands_of_global
    {BA BB B : Nat}
    (hGlobal : ∀ a b : Coeffs, a.size = b.size →
      normInfCoeffs (vecAdd a b) ≤ B) :
    vecAddNormBoundFromOperands BA BB B :=
  SuperNeo.vecAddNormBoundFromOperands_of_global hGlobal

/-- [Role: Theorem-Target] Curated theorem surface `vecScaleNormBoundFromOperands_of_global`. -/
theorem vecScaleNormBoundFromOperands_of_global
    {BS BA B : Nat}
    (hGlobal : ∀ s : F, ∀ a : Coeffs, normInfCoeffs (vecScale s a) ≤ B) :
    vecScaleNormBoundFromOperands BS BA B :=
  SuperNeo.vecScaleNormBoundFromOperands_of_global hGlobal

/-- [Role: Theorem-Target] Curated theorem surface `mulRqNormBoundFromOperands_of_global`. -/
theorem mulRqNormBoundFromOperands_of_global
    {BA BB B : Nat}
    (hGlobal : ∀ a b : Coeffs, normInfCoeffs (mulRq a b) ≤ B) :
    mulRqNormBoundFromOperands BA BB B :=
  SuperNeo.mulRqNormBoundFromOperands_of_global hGlobal

/-- [Role: Theorem-Target] Curated theorem surface `coeffSubNormBoundFromOperands_of_global`. -/
theorem coeffSubNormBoundFromOperands_of_global
    {BA BB B : Nat}
    (hGlobal : ∀ a b : Coeffs, a.size = b.size →
      normInfCoeffs (vecAdd a (vecScale (-1) b)) ≤ B) :
    coeffSubNormBoundFromOperands BA BB B :=
  SuperNeo.coeffSubNormBoundFromOperands_of_global hGlobal

end NormInterface

end SuperNeo
