import SuperNeo.SamplingSet.Core

/-! Sampling-set check/prop bridges and check-driven wrappers. -/

namespace SuperNeo

theorem samplingSetBoundCheck_sound
  {cset : Array Coeffs} {samples : Array Coeffs}
  (hOk : samplingSetBoundCheck cset samples = true) :
  empiricalExpansionFactor cset samples <= theorem9UpperBound (maxRhoNorm cset) := by
  unfold samplingSetBoundCheck at hOk
  exact decide_eq_true_eq.mp hOk

theorem samplingSetBoundCheck_complete
  {cset : Array Coeffs} {samples : Array Coeffs}
  (hBound : empiricalExpansionFactor cset samples <= theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  unfold samplingSetBoundCheck
  exact decide_eq_true hBound

theorem samplingSetBoundCheck_iff_prop
  {cset : Array Coeffs} {samples : Array Coeffs} :
  samplingSetBoundCheck cset samples = true ↔ samplingSetBoundProp cset samples := by
  constructor
  · intro hOk
    exact samplingSetBoundCheck_sound (cset := cset) (samples := samples) hOk
  · intro h
    exact samplingSetBoundCheck_complete (cset := cset) (samples := samples) h

/-- Convenience: recover the regression check flag from theorem-native hypotheses. -/
theorem samplingSetBoundCheck_true_of_operand_norm_assumptions
  {cset samples : Array Coeffs} {BB BRaw : Nat}
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands (maxRhoNorm cset) BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions
    (cset := cset) (samples := samples) (BB := BB) (BRaw := BRaw)
    hSamples hRaw hAddSub hSub

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_inRange
  {cset samples : Array Coeffs} {BB BRaw : Nat}
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawInRange : mulRqRawInRangeBoundFromOperands (maxRhoNorm cset) BB BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_inRange
    (cset := cset) (samples := samples) (BB := BB) (BRaw := BRaw)
    hSamples hRawInRange hAddSub hSub

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_schoolbook
  {cset samples : Array Coeffs} {BB BTerm BRaw : Nat}
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul :
    ∀ x y : F,
      normInfF x ≤ maxRhoNorm cset →
      normInfF y ≤ BB →
      normInfF (x * y) ≤ BTerm)
  (hAdd :
    ∀ x y : F,
      normInfF x ≤ BRaw →
      normInfF y ≤ BTerm →
      normInfF (x + y) ≤ BRaw)
  (hZero : normInfF (0 : F) ≤ BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_schoolbook
    (cset := cset) (samples := samples) (BB := BB) (BTerm := BTerm) (BRaw := BRaw)
    hSamples hMul hAdd hZero hAddSub hSub

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_schoolbook_sum
  {cset samples : Array Coeffs} {BB BTerm : Nat}
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul :
    ∀ x y : F,
      normInfF x ≤ maxRhoNorm cset →
      normInfF y ≤ BB →
      normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hAddSub :
    rawAddSubCollapseBound ((D * D) * BTerm) (theorem9UpperBound (maxRhoNorm cset)))
  (hSub :
    rawSubCollapseBound ((D * D) * BTerm) (theorem9UpperBound (maxRhoNorm cset))) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_schoolbook_sum
    (cset := cset) (samples := samples) (BB := BB) (BTerm := BTerm)
    hSamples hMul hAddTri hAddSub hSub

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_schoolbook_sum_fieldOp
  {cset samples : Array Coeffs} {BB BTerm : Nat}
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul :
    ∀ x y : F,
      normInfF x ≤ maxRhoNorm cset →
      normInfF y ≤ BB →
      normInfF (x * y) ≤ BTerm)
  (hAddTri : ∀ x y : F, normInfF (x + y) ≤ normInfF x + normInfF y)
  (hOps : rawFieldOpCollapseBound ((D * D) * BTerm) ((D * D) * BTerm))
  (hRawLe : ((D * D) * BTerm) ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_schoolbook_sum_fieldOp
    (cset := cset) (samples := samples) (BB := BB) (BTerm := BTerm)
    hSamples hMul hAddTri hOps hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_schoolbook_of_term_le
  {cset samples : Array Coeffs} {BB BTerm BRaw : Nat}
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul :
    ∀ x y : F,
      normInfF x ≤ maxRhoNorm cset →
      normInfF y ≤ BB →
      normInfF (x * y) ≤ BTerm)
  (hTermLe : BTerm ≤ BRaw)
  (hAddCollapse : rawAddCollapseBound BRaw BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_schoolbook_of_term_le
    (cset := cset) (samples := samples) (BB := BB) (BTerm := BTerm) (BRaw := BRaw)
    hSamples hMul hTermLe hAddCollapse hAddSub hSub

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_schoolbook_sameBound
  {cset samples : Array Coeffs} {BB BRaw : Nat}
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMul :
    ∀ x y : F,
      normInfF x ≤ maxRhoNorm cset →
      normInfF y ≤ BB →
      normInfF (x * y) ≤ BRaw)
  (hAddCollapse : rawAddCollapseBound BRaw BRaw)
  (hAddSub : rawAddSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset)))
  (hSub : rawSubCollapseBound BRaw (theorem9UpperBound (maxRhoNorm cset))) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_schoolbook_sameBound
    (cset := cset) (samples := samples) (BB := BB) (BRaw := BRaw)
    hSamples hMul hAddCollapse hAddSub hSub

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_universal_blockers
  {cset samples : Array Coeffs} {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_blockers
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples hMulUniv hAddTri hSubTri hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_universal_blockers_tight
  {cset samples : Array Coeffs} {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_blockers_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples hMulUniv hAddTri hSubTri hRawLe

/--
Triangle-bundle variant of
`samplingSetBoundCheck_true_of_operand_norm_assumptions_via_universal_blockers_tight`.
-/
theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
  {cset samples : Array Coeffs} {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hTri : schoolbookTriangleBounds)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_mul_and_triangles_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples hMulUniv hTri hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_universal_mul_and_add_tight
  {cset samples : Array Coeffs} {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulUniv : schoolbookMulUniversalBound)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_mul_and_add_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples hMulUniv hAddTri hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
  {cset samples : Array Coeffs} {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples hMulRep hAddRep hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_centeredRep_mul_tight
  {cset samples : Array Coeffs} {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hMulRep :
    ∀ x y : F, Int.natAbs (centeredRep (x * y))
      ≤ Int.natAbs (centeredRep x) * Int.natAbs (centeredRep y))
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples hMulRep centeredRepAddTriangleBound_theorem hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_centeredRepMulAddBounds_tight
  {cset samples : Array Coeffs} {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRep : centeredRepMulAddBounds)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_centeredRep_mul_and_add_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples
    (centeredRepMulAddBounds_mul hRep)
    (centeredRepMulAddBounds_add hRep)
    hRawLe

/-- Assumption-free native check bridge (`D^2` schoolbook path). -/
theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_native
  {cset samples : Array Coeffs} {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawLe :
    ((D * D) * (BA * BB)) + ((D * D) * (BA * BB)) + ((D * D) * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_native
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples hRawLe

/-- Assumption-free native-tight check bridge (`3 * D * BA * BB` path). -/
theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_native_tight
  {cset samples : Array Coeffs} {BA BB : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawLe :
    (D * (BA * BB)) + (D * (BA * BB)) + (D * (BA * BB))
      ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_native_tight
    (cset := cset) (samples := samples) (BA := BA) (BB := BB)
    hCset hSamples hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_universal_blockers_and_raw
  {cset samples : Array Coeffs} {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_blockers_and_raw
    (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
    hCset hSamples hRaw hAddTri hSubTri hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_universal_blockers_and_rawCoeff
  {cset samples : Array Coeffs} {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hSubTri : schoolbookSubTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_blockers_and_rawCoeff
    (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
    hCset hSamples hRawCoeff hAddTri hSubTri hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
  {cset samples : Array Coeffs} {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_mul_and_add_and_raw
    (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
    hCset hSamples hRaw hAddTri hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_universal_mul_and_add_and_rawCoeff
  {cset samples : Array Coeffs} {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddTri : schoolbookAddTriangleBound)
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_universal_mul_and_add_and_rawCoeff
    (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
    hCset hSamples hRawCoeff hAddTri hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
  {cset samples : Array Coeffs} {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRaw : mulRqRawNormBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact empiricalExpansionFactor_le_theorem9UpperBound_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
    (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
    hCset hSamples hRaw hAddRep hRawLe

theorem samplingSetBoundCheck_true_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_rawCoeff
  {cset samples : Array Coeffs} {BA BB BRaw : Nat}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ BA)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ BB)
  (hRawCoeff : mulRqRawCoeffBoundFromOperands BA BB BRaw)
  (hAddRep :
    ∀ x y : F, Int.natAbs (centeredRep (x + y))
      ≤ Int.natAbs (centeredRep x) + Int.natAbs (centeredRep y))
  (hRawLe : BRaw + BRaw + BRaw ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_operand_norm_assumptions_via_centeredRep_mul_and_add_and_raw
    (cset := cset) (samples := samples) (BA := BA) (BB := BB) (BRaw := BRaw)
    hCset hSamples
    (mulRqRawNormBoundFromOperands_of_rawCoeff hRawCoeff)
    hAddRep hRawLe

theorem samplingSetBoundCheck_true_of_goldilocks_operand_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_assumptions
      (cset := cset) (samples := samples) hCset hSamples hRaw hCollapse)
    hBLe

theorem samplingSetBoundCheck_true_of_goldilocks_operand_assumptions_inRange
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_assumptions_inRange
      (cset := cset) (samples := samples) hCset hSamples hRawInRange hCollapse)
    hBLe

theorem samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_fieldOp_assumptions
      (cset := cset) (samples := samples) hCset hSamples hRaw hFieldOps)
    hBLe

theorem samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions_inRange
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_fieldOp_assumptions_inRange
      (cset := cset) (samples := samples) hCset hSamples hRawInRange hFieldOps)
    hBLe

theorem samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_rawCoeff_assumptions
      (cset := cset) (samples := samples) hCset hSamples hRawCoeff hCollapse)
    hBLe

theorem samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hBLe : Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset)) :
  samplingSetBoundCheck cset samples = true := by
  apply samplingSetBoundCheck_complete
  exact Nat.le_trans
    (empiricalExpansionFactor_le_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
      (cset := cset) (samples := samples) hCset hSamples hRawCoeff hFieldOps)
    hBLe

theorem goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge
  {cset : Array Coeffs}
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  Parameters.Goldilocks.B ≤ theorem9UpperBound (maxRhoNorm cset) := by
  exact Nat.le_trans hMax (theorem9UpperBound_ge_self (maxRhoNorm cset))

theorem samplingSetBoundCheck_true_of_goldilocks_operand_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_assumptions
    hCset hSamples hRaw hCollapse
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem samplingSetBoundCheck_true_of_goldilocks_operand_assumptions_inRange_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_assumptions_inRange
    hCset hSamples hRawInRange hCollapse
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRaw : GoldilocksRawNormBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions
    hCset hSamples hRaw hFieldOps
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions_inRange_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawInRange : GoldilocksRawInRangeBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_fieldOp_assumptions_inRange
    hCset hSamples hRawInRange hFieldOps
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hCollapse : GoldilocksRawCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_assumptions
    hCset hSamples hRawCoeff hCollapse
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

theorem samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_fieldOp_assumptions_of_maxRhoNorm_ge
  {cset samples : Array Coeffs}
  (hCset : ∀ i : Fin cset.size, normInfCoeffs cset[i] ≤ Parameters.Goldilocks.B)
  (hSamples : ∀ j : Fin samples.size, normInfCoeffs samples[j] ≤ Parameters.Goldilocks.B)
  (hRawCoeff : GoldilocksRawCoeffBoundAssumption)
  (hFieldOps : GoldilocksFieldOpCollapseAssumption)
  (hMax : Parameters.Goldilocks.B ≤ maxRhoNorm cset) :
  samplingSetBoundCheck cset samples = true := by
  exact samplingSetBoundCheck_true_of_goldilocks_operand_rawCoeff_fieldOp_assumptions
    hCset hSamples hRawCoeff hFieldOps
    (goldilocksB_le_theorem9UpperBound_of_maxRhoNorm_ge hMax)

def samplingSetSanity : Bool :=
  decide (theorem9UpperBound 2 = 216)


end SuperNeo
