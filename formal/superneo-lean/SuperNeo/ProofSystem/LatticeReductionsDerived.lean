import SuperNeo.ProofSystem.LatticeReductions

namespace SuperNeo.ProofSystem

/--
Event-level extractor bridge: any standard Ajtai binding collision yields a
canonical homogeneous MSIS break event.
-/
theorem bindingCollisionEvent_implies_msisBreakEvent
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion) :
  Nonempty (BindingCollision params) → MSISBreakEvent params := by
  intro hColl
  rcases hColl with ⟨coll⟩
  exact msisBreakEvent_of_bindingCollision laws hExpPos coll

/--
Event-level extractor bridge: any relaxed Ajtai binding collision over the
carrier in `laws` yields a canonical homogeneous MSIS break event.
-/
theorem relaxedBindingCollisionEvent_implies_msisBreakEvent
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion) :
  Nonempty (RelaxedBindingCollision params laws.samplingCarrier) →
    MSISBreakEvent params := by
  intro hColl
  rcases hColl with ⟨coll⟩
  exact msisBreakEvent_of_relaxedBindingCollision laws hExpPos coll

/--
Derive the standard Ajtai binding advantage bound directly from MSIS hardness
and the extractor/event implication.
-/
theorem ajtaiBindingAdvantageBound_of_msisBoundary
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (hMsis : MSISHardnessBoundary params) :
  AjtaiBindingAdvantageBound params hMsis.epsMSIS := by
  intro prob n
  have hEvent :
      AjtaiBindingAdvantage prob (canonicalAjtaiBindingGame params) n ≤
        MSISAdvantage prob (canonicalMSISGame params) n := by
    exact prob.prMonotone
      (bindingCollisionEvent_implies_msisBreakEvent laws hExpPos)
  exact Rat.le_trans hEvent (hMsis.advantageBound prob n)

/--
Derive the relaxed Ajtai binding advantage bound directly from MSIS hardness
and the relaxed extractor/event implication.
-/
theorem ajtaiRelaxedBindingAdvantageBound_of_msisBoundary
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (hMsis : MSISHardnessBoundary params) :
  AjtaiRelaxedBindingAdvantageBound params laws.samplingCarrier hMsis.epsMSIS := by
  intro prob n
  have hEvent :
      AjtaiRelaxedBindingAdvantage prob
          (canonicalAjtaiRelaxedBindingGame params laws.samplingCarrier) n ≤
        MSISAdvantage prob (canonicalMSISGame params) n := by
    exact prob.prMonotone
      (relaxedBindingCollisionEvent_implies_msisBreakEvent laws hExpPos)
  exact Rat.le_trans hEvent (hMsis.advantageBound prob n)

namespace MSISToAjtaiReductions

/--
Canonical constructor from an MSIS hardness boundary alone: both Ajtai
error terms are instantiated by the same MSIS error function, and their
advantage bounds are derived from the extractor/event implications.
-/
def ofLawsAndMSISBoundary
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (hMsis : MSISHardnessBoundary params) :
  MSISToAjtaiReductions params where
  laws := laws
  relaxedExpansionPos := hExpPos
  epsBinding := hMsis.epsMSIS
  epsRelaxedBinding := hMsis.epsMSIS
  bindingAdvantageBound :=
    ajtaiBindingAdvantageBound_of_msisBoundary laws hExpPos hMsis
  relaxedBindingAdvantageBound :=
    ajtaiRelaxedBindingAdvantageBound_of_msisBoundary laws hExpPos hMsis
  negligibleEpsBinding := by
    simpa using hMsis.negligibleEpsMSIS
  negligibleEpsRelaxedBinding := by
    simpa using hMsis.negligibleEpsMSIS

/--
Paper-carrier specialization of `ofLawsAndMSISBoundary` under the concrete
`3*d ≤ relaxedExpansion` side condition already used on the final protocol path.
-/
def ofPaperCarrierFromThreeDLeAndMSISBoundary
  {params : AjtaiParams}
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (hMsis : MSISHardnessBoundary params) :
  MSISToAjtaiReductions params :=
  ofLawsAndMSISBoundary
    (laws := LatticeReductionLaws.ofPaperCarrierFromThreeDLe
      (params := params) hTd)
    hExpPos
    hMsis

end MSISToAjtaiReductions

end SuperNeo.ProofSystem
