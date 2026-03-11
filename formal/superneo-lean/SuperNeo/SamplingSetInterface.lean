import SuperNeo.SamplingSet

/-!
Contract interface for `SuperNeo.SamplingSet`.

Spec: `specs/SamplingSet.spec.md`

Paper anchors:
- Definition 17 (Strong sampling sets), Theorem 9 (Expansion factors), lines 379-383.
-/

namespace SuperNeo

namespace SamplingSetInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `samplingNormBoundProp`. -/
abbrev samplingNormBoundProp := SuperNeo.samplingNormBoundProp

/-- [Role: Theorem-Target] Curated re-export of `samplingExpansionProp`. -/
abbrev samplingExpansionProp := SuperNeo.samplingExpansionProp

/-- [Role: Theorem-Target] Curated re-export of `SamplingCarrier`. -/
abbrev SamplingCarrier := SuperNeo.SamplingCarrier

/-- [Role: Theorem-Target] Curated re-export of `samplingDiffSet`. -/
abbrev samplingDiffSet := SuperNeo.samplingDiffSet

/-- [Role: Theorem-Target] Curated re-export of `strongSamplingExpansionProp`. -/
abbrev strongSamplingExpansionProp := SuperNeo.strongSamplingExpansionProp

/-- [Role: Theorem-Target] Curated re-export of `ringNormCarrier`. -/
abbrev ringNormCarrier := SuperNeo.ringNormCarrier

/-- [Role: Theorem-Target] Curated re-export of `paperCarrier`. -/
abbrev paperCarrier := SuperNeo.paperCarrier

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `samplingExpansionProp_of_bounds`. -/
theorem samplingExpansionProp_of_bounds
  {cset samples : Array Coeffs}
  {B : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ B) :
  samplingExpansionProp cset samples :=
  SuperNeo.samplingExpansionProp_of_bounds hCset hSamples

/-- [Role: Theorem-Target] Curated theorem surface `samplingNormBoundProp_left`. -/
theorem samplingNormBoundProp_left
  {cset samples : Array Coeffs}
  {B : Nat}
  (h : samplingNormBoundProp cset samples B) :
  ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ B :=
  SuperNeo.samplingNormBoundProp_left h

/-- [Role: Theorem-Target] Curated theorem surface `samplingNormBoundProp_right`. -/
theorem samplingNormBoundProp_right
  {cset samples : Array Coeffs}
  {B : Nat}
  (h : samplingNormBoundProp cset samples B) :
  ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ B :=
  SuperNeo.samplingNormBoundProp_right h

/-- [Role: Theorem-Target] Curated theorem surface `samplingNormBoundProp_mono`. -/
theorem samplingNormBoundProp_mono
  {cset samples : Array Coeffs}
  {B B' : Nat}
  (h : samplingNormBoundProp cset samples B)
  (hBB' : B ≤ B') :
  samplingNormBoundProp cset samples B' :=
  SuperNeo.samplingNormBoundProp_mono h hBB'

/-- [Role: Theorem-Target] Curated theorem surface `samplingExpansionProp_mono`. -/
theorem samplingExpansionProp_mono
  {cset samples : Array Coeffs}
  {B B' : Nat}
  (h : samplingNormBoundProp cset samples B)
  (hBB' : B ≤ B') :
  samplingExpansionProp cset samples :=
  SuperNeo.samplingExpansionProp_mono h hBB'

/-- [Role: Theorem-Target] Curated theorem surface `samplingExpansionProp_empty`. -/
theorem samplingExpansionProp_empty :
  samplingExpansionProp (#[] : Array Coeffs) (#[] : Array Coeffs) :=
  SuperNeo.samplingExpansionProp_empty

/-- [Role: Theorem-Target] Curated theorem surface `samplingSetBoundCheck_sound`. -/
theorem samplingSetBoundCheck_sound
  {cset samples : Array Coeffs}
  (hOk : SuperNeo.samplingSetBoundCheck cset samples = true) :
  samplingExpansionProp cset samples :=
  SuperNeo.samplingSetBoundCheck_sound hOk

/-- [Role: Theorem-Target] Curated theorem surface `samplingSetBoundCheck_complete`. -/
theorem samplingSetBoundCheck_complete
  {cset samples : Array Coeffs}
  (hProp : samplingExpansionProp cset samples) :
  SuperNeo.samplingSetBoundCheck cset samples = true :=
  SuperNeo.samplingSetBoundCheck_complete hProp

/-- [Role: Theorem-Target] Curated theorem surface `samplingSetBoundCheck_iff`. -/
theorem samplingSetBoundCheck_iff
  {cset samples : Array Coeffs} :
  SuperNeo.samplingSetBoundCheck cset samples = true ↔ samplingExpansionProp cset samples :=
  SuperNeo.samplingSetBoundCheck_iff

/-- [Role: Theorem-Target] Curated theorem surface `strongSamplingExpansionProp_mono`. -/
theorem strongSamplingExpansionProp_mono
  {C : SamplingCarrier}
  {T T' : Nat}
  (h : strongSamplingExpansionProp C T)
  (hTT' : T ≤ T') :
  strongSamplingExpansionProp C T' :=
  SuperNeo.strongSamplingExpansionProp_mono h hTT'

/-- [Role: Theorem-Target] Curated theorem surface `expansionFactor_of_strongSampling`. -/
theorem expansionFactor_of_strongSampling
  {C : SamplingCarrier}
  {T : Nat}
  (h : strongSamplingExpansionProp C T)
  {δ z : Coeffs}
  (hδ : samplingDiffSet C δ)
  {B : Nat}
  (hB : normInfCoeffs z ≤ B) :
  normInfCoeffs (mulRq δ z) ≤ 4 * T * B :=
  SuperNeo.expansionFactor_of_strongSampling h hδ hB

/-- [Role: Theorem-Target] Curated theorem surface `strongSamplingExpansionProp_of_ringNormCarrier`. -/
theorem strongSamplingExpansionProp_of_ringNormCarrier
  {K T D : Nat}
  (hSub : coeffSubNormBoundFromOperands K K D)
  (hMul : ∀ B : Nat, mulRqNormBoundFromOperands D B (4 * T * B)) :
  strongSamplingExpansionProp (ringNormCarrier K) T :=
  SuperNeo.strongSamplingExpansionProp_of_ringNormCarrier hSub hMul

/-- [Role: Theorem-Target] Curated theorem surface `strongSamplingExpansionProp_of_paperCarrier`. -/
theorem strongSamplingExpansionProp_of_paperCarrier
  {T D : Nat}
  (hSub : coeffSubNormBoundFromOperands 2 2 D)
  (hMul : ∀ B : Nat, mulRqNormBoundFromOperands D B (4 * T * B)) :
  strongSamplingExpansionProp paperCarrier T :=
  SuperNeo.strongSamplingExpansionProp_of_paperCarrier hSub hMul

/-- [Role: Theorem-Target] Paper-carrier differences stay ring-shaped and have `‖·‖∞ ≤ 4`. -/
theorem samplingDiffSet_paperCarrier_hasRingDegreeShape_and_norm_le_four
  {δ : Coeffs}
  (hδ : samplingDiffSet paperCarrier δ) :
  hasRingDegreeShape δ ∧ normInfCoeffs δ ≤ 4 :=
  SuperNeo.samplingDiffSet_paperCarrier_hasRingDegreeShape_and_norm_le_four hδ

end SamplingSetInterface

end SuperNeo
