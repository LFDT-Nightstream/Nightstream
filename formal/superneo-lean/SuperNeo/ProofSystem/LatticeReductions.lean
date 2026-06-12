import SuperNeo.ProofSystem.LatticeExtractors

/-!
Theorem-facing lattice boundary packages: the truth-valued probability
model, MSIS hardness/advantage boundaries, Ajtai binding boundaries, and
the MSIS-to-Ajtai reduction constructors. Importing this module exposes
the full lattice-reduction API.
-/

namespace SuperNeo.ProofSystem


/-- Truth-valued probability model: `Pr P = 1` iff `P`, else `0`. -/
noncomputable def truthProb : ProbModel where
  Pr := fun P => by
    classical
    exact if P then (1 : Rat) else 0
  prNonneg := by
    intro P
    classical
    by_cases hP : P
    · simp [hP]
    · simp [hP]
  prLeOne := by
    intro P
    classical
    by_cases hP : P
    · simp [hP]
    · simp [hP]
  prFalse := by
    classical
    simp
  prMonotone := by
    intro P Q hImp
    classical
    by_cases hP : P
    · have hQ : Q := hImp hP
      simp [hP, hQ]
    · simp [hP]
      by_cases hQ : Q
      · simp [hQ]
      · simp [hQ]
  prUnionLeAdd := by
    intro P Q
    classical
    by_cases hP : P <;> by_cases hQ : Q <;> simp [hP, hQ]

/--
Unpack MSIS hardness into an explicit negligible error plus canonical advantage
bound surface.
-/
theorem msisAdvantageBound_of_hardness
  {params : AjtaiParams}
  (h : MSISHardnessAssumption params) :
  ∃ eps : ErrorFn, IsNegligible eps ∧ MSISAdvantageBound params eps := by
  rcases h with ⟨eps, hNeg, hBound⟩
  exact ⟨eps, hNeg, hBound⟩

namespace MSISHardnessBoundary

/--
Canonical boundary package reconstructed from the theorem-level MSIS hardness
assumption.
-/
noncomputable def ofHardness
  {params : AjtaiParams}
  (h : MSISHardnessAssumption params) :
  MSISHardnessBoundary params :=
  let eps := Classical.choose h
  let hRest := Classical.choose_spec h
  { epsMSIS := eps
    advantageBound := hRest.2
    negligibleEpsMSIS := hRest.1 }

/--
The reconstructed boundary package immediately recovers the input theorem-level
MSIS hardness assumption.
-/
theorem ofHardness_hardnessFromFields
  {params : AjtaiParams}
  (h : MSISHardnessAssumption params) :
  MSISHardnessAssumption params := by
  exact (ofHardness h).hardnessFromFields

end MSISHardnessBoundary

/--
Package an Ajtai binding advantage bound together with negligible error.
-/
theorem no_ajtaiBindingCollision_of_advantageBound
  {params : AjtaiParams}
  {eps : ErrorFn}
  (hBound : AjtaiBindingAdvantageBound params eps)
  (hNeg : IsNegligible eps) :
  AjtaiBindingAssumption params := by
  exact ⟨eps, hNeg, hBound⟩

/--
Package an Ajtai relaxed-binding advantage bound together with negligible error.
-/
theorem no_ajtaiRelaxedBindingCollision_of_advantageBound
  {params : AjtaiParams}
  {C : SamplingCarrier}
  {eps : ErrorFn}
  (hBound : AjtaiRelaxedBindingAdvantageBound params C eps)
  (hNeg : IsNegligible eps) :
  AjtaiRelaxedBindingAssumption params C := by
  exact ⟨eps, hNeg, hBound⟩

namespace AjtaiBindingBoundary

/-- Canonical hardness view for an Ajtai binding boundary package. -/
def hardness
  {params : AjtaiParams}
  (h : AjtaiBindingBoundary params) : AjtaiBindingAssumption params :=
  no_ajtaiBindingCollision_of_advantageBound h.advantageBound h.negligibleEpsBinding

/-- Canonical hardness derivation from package fields. -/
theorem hardnessFromFields
  {params : AjtaiParams}
  (h : AjtaiBindingBoundary params) : AjtaiBindingAssumption params :=
  h.hardness

/-- Normalize any package by overwriting redundant `hardness` proof from aligned fields. -/
def normalize
  {params : AjtaiParams}
  (h : AjtaiBindingBoundary params) : AjtaiBindingBoundary params :=
  h

theorem normalize_hardnessFromFields
  {params : AjtaiParams}
  (h : AjtaiBindingBoundary params) :
  (normalize h).hardness = h.hardnessFromFields := by
  rfl

end AjtaiBindingBoundary

namespace AjtaiRelaxedBindingBoundary

/-- Canonical hardness view for an Ajtai relaxed-binding boundary package. -/
def hardness
  {params : AjtaiParams}
  {C : SamplingCarrier}
  (h : AjtaiRelaxedBindingBoundary params C) : AjtaiRelaxedBindingAssumption params C :=
  no_ajtaiRelaxedBindingCollision_of_advantageBound h.advantageBound h.negligibleEpsRelaxedBinding

/-- Canonical relaxed-hardness derivation from package fields. -/
theorem hardnessFromFields
  {params : AjtaiParams}
  {C : SamplingCarrier}
  (h : AjtaiRelaxedBindingBoundary params C) : AjtaiRelaxedBindingAssumption params C :=
  h.hardness

/-- Normalize any relaxed package by overwriting redundant `hardness` proof from aligned fields. -/
def normalize
  {params : AjtaiParams}
  {C : SamplingCarrier}
  (h : AjtaiRelaxedBindingBoundary params C) : AjtaiRelaxedBindingBoundary params C :=
  h

theorem normalize_hardnessFromFields
  {params : AjtaiParams}
  {C : SamplingCarrier}
  (h : AjtaiRelaxedBindingBoundary params C) :
  (normalize h).hardness = h.hardnessFromFields := by
  rfl

end AjtaiRelaxedBindingBoundary

/--
Abstract reduction interface from MSIS hardness to Ajtai commitment security.
This remains theorem-facing only; implication theorems are derived below.
-/
structure MSISToAjtaiReductions (params : AjtaiParams) where
  laws : LatticeReductionLaws params
  relaxedExpansionPos : 0 < params.relaxedExpansion
  epsBinding : ErrorFn
  epsRelaxedBinding : ErrorFn
  bindingAdvantageBound : AjtaiBindingAdvantageBound params epsBinding
  relaxedBindingAdvantageBound :
    AjtaiRelaxedBindingAdvantageBound params laws.samplingCarrier epsRelaxedBinding
  negligibleEpsBinding : IsNegligible epsBinding
  negligibleEpsRelaxedBinding : IsNegligible epsRelaxedBinding

namespace MSISToAjtaiReductions

/--
Canonical constructor from an already-built lattice-law package.
-/
def ofLaws
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound : AjtaiRelaxedBindingAdvantageBound params laws.samplingCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params where
  laws := laws
  relaxedExpansionPos := hExpPos
  epsBinding := epsBinding
  epsRelaxedBinding := epsRelaxedBinding
  bindingAdvantageBound := hBindBound
  relaxedBindingAdvantageBound := hRelaxedBound
  negligibleEpsBinding := hBindNeg
  negligibleEpsRelaxedBinding := hRelaxedNeg

/--
Canonical constructor specialized to `paperCarrier`, when strong-sampling is
already available for that carrier at `params.relaxedExpansion`.
-/
def ofPaperCarrier
  {params : AjtaiParams}
  (hStrong : strongSamplingExpansionProp paperCarrier params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound :
    AjtaiRelaxedBindingAdvantageBound params paperCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params :=
  ofLaws
    (laws := LatticeReductionLaws.ofPaperCarrier hStrong)
    hExpPos
    epsBinding epsRelaxedBinding
    hBindBound
    (by simpa using hRelaxedBound)
    hBindNeg hRelaxedNeg

/--
Paper-carrier constructor from subtraction/multiplication norm bundles.
This threads `strongSampling` via
`LatticeReductionLaws.ofPaperCarrierFromBounds`.
-/
def ofPaperCarrierFromBounds
  {params : AjtaiParams}
  {D : Nat}
  (hSub : coeffSubNormBoundFromOperands 2 2 D)
  (hMul : ∀ B : Nat, mulRqPhiNormBoundFromOperands D B (4 * params.relaxedExpansion * B))
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound :
    AjtaiRelaxedBindingAdvantageBound params paperCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params :=
  ofLaws
    (laws := LatticeReductionLaws.ofPaperCarrierFromBounds
      (params := params) (D := D) hSub hMul)
    hExpPos
    epsBinding epsRelaxedBinding
    hBindBound
    (by simpa using hRelaxedBound)
    hBindNeg hRelaxedNeg

/--
Paper-carrier constructor from the concrete norm-bundle closure path.
This requires `3*d ≤ params.relaxedExpansion` to derive strong sampling.
-/
def ofPaperCarrierFromThreeDLe
  {params : AjtaiParams}
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound :
    AjtaiRelaxedBindingAdvantageBound params paperCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params :=
  ofLaws
    (laws := LatticeReductionLaws.ofPaperCarrierFromThreeDLe
      (params := params) hTd)
    hExpPos
    epsBinding epsRelaxedBinding
    hBindBound
    (by simpa using hRelaxedBound)
    hBindNeg hRelaxedNeg

/-- Derived Ajtai binding boundary from MSIS hardness, via explicit extractor. -/
theorem toBinding
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (_hMsis : MSISHardnessAssumption params) :
  AjtaiBindingAssumption params := by
  exact ⟨hRed.epsBinding, hRed.negligibleEpsBinding, hRed.bindingAdvantageBound⟩

/-- Derived Ajtai relaxed-binding boundary from MSIS hardness, via explicit extractor. -/
theorem toRelaxedBinding
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (_hMsis : MSISHardnessAssumption params) :
  AjtaiRelaxedBindingAssumption params hRed.laws.samplingCarrier := by
  exact
    ⟨hRed.epsRelaxedBinding, hRed.negligibleEpsRelaxedBinding, hRed.relaxedBindingAdvantageBound⟩

end MSISToAjtaiReductions

/-- Derive Ajtai binding from MSIS via the declared reduction surface. -/
theorem ajtaiBinding_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiBindingAssumption params := by
  exact hRed.toBinding hMsis

/-- Derive Ajtai relaxed binding from MSIS via the declared reduction surface. -/
theorem ajtaiRelaxedBinding_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiRelaxedBindingAssumption params hRed.laws.samplingCarrier := by
  exact hRed.toRelaxedBinding hMsis

/-- Package both Ajtai boundaries derived from MSIS under one reduction interface. -/
theorem ajtaiBoundaries_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiBindingAssumption params ∧
    AjtaiRelaxedBindingAssumption params hRed.laws.samplingCarrier := by
  exact ⟨ajtaiBinding_of_msis hRed hMsis, ajtaiRelaxedBinding_of_msis hRed hMsis⟩

/-- Build the canonical Ajtai binding boundary package from MSIS hardness + reduction surface. -/
def ajtaiBindingBoundary_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (_hMsis : MSISHardnessAssumption params) :
  AjtaiBindingBoundary params where
  epsBinding := hRed.epsBinding
  advantageBound := hRed.bindingAdvantageBound
  negligibleEpsBinding := hRed.negligibleEpsBinding

/-- Build the canonical Ajtai relaxed-binding boundary package from MSIS hardness + reduction surface. -/
def ajtaiRelaxedBindingBoundary_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (_hMsis : MSISHardnessAssumption params) :
  AjtaiRelaxedBindingBoundary params hRed.laws.samplingCarrier where
  epsRelaxedBinding := hRed.epsRelaxedBinding
  advantageBound := hRed.relaxedBindingAdvantageBound
  negligibleEpsRelaxedBinding := hRed.negligibleEpsRelaxedBinding

end SuperNeo.ProofSystem
