import SuperNeo.ProofSystem.LatticeReductions
import SuperNeo.ProofSystem.LatticeReductionsDerived
import SuperNeo.SamplingSet

/-!
Contract interface for `SuperNeo.ProofSystem.LatticeReductions`.

Spec: `specs/ProofSystem/LatticeReductions.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
Anchors:
- Theorem 2 (Ajtai properties), lines 319-321.
- Definition 16 (MSIS), lines 743-744.
- Definition 18 (Ajtai commitment), lines 753-756.
-/

namespace SuperNeo
namespace ProofSystem.LatticeReductionsInterface

noncomputable section

/-! ## Reduction Boundary Package -/

abbrev MSISToAjtaiReductions := SuperNeo.ProofSystem.MSISToAjtaiReductions
abbrev LatticeReductionLaws := SuperNeo.ProofSystem.LatticeReductionLaws

def LatticeReductionLaws_ofCarrier
  {params : AjtaiParams}
  (C : SamplingCarrier)
  (hStrong : strongSamplingExpansionProp C params.relaxedExpansion) :
  LatticeReductionLaws params :=
  SuperNeo.ProofSystem.LatticeReductionLaws.ofCarrier C hStrong

def LatticeReductionLaws_ofPaperCarrier
  {params : AjtaiParams}
  (hStrong : strongSamplingExpansionProp paperCarrier params.relaxedExpansion) :
  LatticeReductionLaws params :=
  SuperNeo.ProofSystem.LatticeReductionLaws.ofPaperCarrier hStrong

theorem LatticeReductionLaws_paperStrongSampling_of_bounds
  {params : AjtaiParams}
  {D : Nat}
  (hSub : coeffSubNormBoundFromOperands 2 2 D)
  (hMul : ∀ B : Nat, mulRqPhiNormBoundFromOperands D B (4 * params.relaxedExpansion * B)) :
  strongSamplingExpansionProp paperCarrier params.relaxedExpansion :=
  SuperNeo.ProofSystem.LatticeReductionLaws.paperStrongSampling_of_bounds (params := params) (D := D) hSub hMul

theorem LatticeReductionLaws_paperStrongSampling_of_three_d_le
  {params : AjtaiParams}
  (hTd : 3 * d ≤ params.relaxedExpansion) :
  strongSamplingExpansionProp paperCarrier params.relaxedExpansion :=
  SuperNeo.ProofSystem.LatticeReductionLaws.paperStrongSampling_of_three_d_le (params := params) hTd

def LatticeReductionLaws_ofPaperCarrierFromBounds
  {params : AjtaiParams}
  {D : Nat}
  (hSub : coeffSubNormBoundFromOperands 2 2 D)
  (hMul : ∀ B : Nat, mulRqPhiNormBoundFromOperands D B (4 * params.relaxedExpansion * B)) :
  LatticeReductionLaws params :=
  SuperNeo.ProofSystem.LatticeReductionLaws.ofPaperCarrierFromBounds
    (params := params) (D := D) hSub hMul

def LatticeReductionLaws_ofPaperCarrierFromThreeDLe
  {params : AjtaiParams}
  (hTd : 3 * d ≤ params.relaxedExpansion) :
  LatticeReductionLaws params :=
  SuperNeo.ProofSystem.LatticeReductionLaws.ofPaperCarrierFromThreeDLe
    (params := params) hTd

def MSISToAjtaiReductions_ofLaws
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound : AjtaiRelaxedBindingAdvantageBound params laws.samplingCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params :=
  SuperNeo.ProofSystem.MSISToAjtaiReductions.ofLaws
    laws hExpPos epsBinding epsRelaxedBinding hBindBound hRelaxedBound hBindNeg hRelaxedNeg

def MSISToAjtaiReductions_ofPaperCarrier
  {params : AjtaiParams}
  (hStrong : strongSamplingExpansionProp paperCarrier params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound : AjtaiRelaxedBindingAdvantageBound params paperCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params :=
  SuperNeo.ProofSystem.MSISToAjtaiReductions.ofPaperCarrier
    hStrong hExpPos epsBinding epsRelaxedBinding hBindBound hRelaxedBound hBindNeg hRelaxedNeg

def MSISToAjtaiReductions_ofPaperCarrierFromBounds
  {params : AjtaiParams}
  {D : Nat}
  (hSub : coeffSubNormBoundFromOperands 2 2 D)
  (hMul : ∀ B : Nat, mulRqPhiNormBoundFromOperands D B (4 * params.relaxedExpansion * B))
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound : AjtaiRelaxedBindingAdvantageBound params paperCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params :=
  SuperNeo.ProofSystem.MSISToAjtaiReductions.ofPaperCarrierFromBounds
    (params := params) (D := D)
    hSub hMul hExpPos epsBinding epsRelaxedBinding hBindBound hRelaxedBound hBindNeg hRelaxedNeg

def MSISToAjtaiReductions_ofPaperCarrierFromThreeDLe
  {params : AjtaiParams}
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (epsBinding epsRelaxedBinding : ErrorFn)
  (hBindBound : AjtaiBindingAdvantageBound params epsBinding)
  (hRelaxedBound : AjtaiRelaxedBindingAdvantageBound params paperCarrier epsRelaxedBinding)
  (hBindNeg : IsNegligible epsBinding)
  (hRelaxedNeg : IsNegligible epsRelaxedBinding) :
  MSISToAjtaiReductions params :=
  SuperNeo.ProofSystem.MSISToAjtaiReductions.ofPaperCarrierFromThreeDLe
    (params := params)
    hTd hExpPos epsBinding epsRelaxedBinding hBindBound hRelaxedBound hBindNeg hRelaxedNeg

def MSISToAjtaiReductions_ofLawsAndMSISBoundary
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (hMsis : MSISHardnessBoundary params) :
  MSISToAjtaiReductions params :=
  SuperNeo.ProofSystem.MSISToAjtaiReductions.ofLawsAndMSISBoundary
    laws hExpPos hMsis

def MSISToAjtaiReductions_ofPaperCarrierFromThreeDLeAndMSISBoundary
  {params : AjtaiParams}
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (hMsis : MSISHardnessBoundary params) :
  MSISToAjtaiReductions params :=
  SuperNeo.ProofSystem.MSISToAjtaiReductions.ofPaperCarrierFromThreeDLeAndMSISBoundary
    (params := params) hTd hExpPos hMsis

/-! ## Extractor and Norm-Transfer Theorems -/

theorem bindingCollision_subWitness_norm_lt_msisNormBound
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (coll : BindingCollision params) :
  normInfVec (subVec params.msgLen coll.opening1.witness coll.opening2.witness) <
    params.msisNormBound :=
  SuperNeo.ProofSystem.bindingCollision_subWitness_norm_lt_msisNormBound
    (params := params) laws hExpPos coll

theorem relaxedBindingCollision_subWitness_norm_lt_msisNormBound
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (coll : RelaxedBindingCollision params laws.samplingCarrier) :
  normInfVec
      (subVec params.msgLen
        (smulVec coll.delta1 coll.opening2.witness)
        (smulVec coll.delta2 coll.opening1.witness)) <
    params.msisNormBound :=
  SuperNeo.ProofSystem.relaxedBindingCollision_subWitness_norm_lt_msisNormBound
    (params := params) laws hExpPos coll

theorem msisBreakEvent_of_bindingCollision
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (coll : BindingCollision params) :
  MSISBreakEvent params :=
  SuperNeo.ProofSystem.msisBreakEvent_of_bindingCollision (params := params) laws hExpPos coll

theorem msisBreakEvent_of_relaxedBindingCollision
  {params : AjtaiParams}
  (laws : LatticeReductionLaws params)
  (hExpPos : 0 < params.relaxedExpansion)
  (coll : RelaxedBindingCollision params laws.samplingCarrier) :
  MSISBreakEvent params :=
  SuperNeo.ProofSystem.msisBreakEvent_of_relaxedBindingCollision
    (params := params) laws hExpPos coll

/-! ## Hardness and Binding Derivations -/

abbrev truthProb := SuperNeo.ProofSystem.truthProb

theorem msisAdvantageBound_of_hardness
  {params : AjtaiParams}
  (h : MSISHardnessAssumption params) :
  ∃ eps : ErrorFn, IsNegligible eps ∧ MSISAdvantageBound params eps :=
  SuperNeo.ProofSystem.msisAdvantageBound_of_hardness (params := params) h

noncomputable def MSISHardnessBoundary_ofHardness
  {params : AjtaiParams}
  (h : MSISHardnessAssumption params) :
  MSISHardnessBoundary params :=
  SuperNeo.ProofSystem.MSISHardnessBoundary.ofHardness h

theorem MSISHardnessBoundary_ofHardness_hardnessFromFields
  {params : AjtaiParams}
  (h : MSISHardnessAssumption params) :
  MSISHardnessAssumption params :=
  SuperNeo.ProofSystem.MSISHardnessBoundary.ofHardness_hardnessFromFields h

theorem no_ajtaiBindingCollision_of_advantageBound
  {params : AjtaiParams}
  {eps : ErrorFn}
  (hBound : AjtaiBindingAdvantageBound params eps)
  (hNeg : IsNegligible eps) :
  AjtaiBindingAssumption params :=
  SuperNeo.ProofSystem.no_ajtaiBindingCollision_of_advantageBound (params := params) hBound hNeg

theorem no_ajtaiRelaxedBindingCollision_of_advantageBound
  {params : AjtaiParams}
  {C : SamplingCarrier}
  {eps : ErrorFn}
  (hBound : AjtaiRelaxedBindingAdvantageBound params C eps)
  (hNeg : IsNegligible eps) :
  AjtaiRelaxedBindingAssumption params C :=
  SuperNeo.ProofSystem.no_ajtaiRelaxedBindingCollision_of_advantageBound (params := params) hBound hNeg

def AjtaiBindingBoundary_hardness
  {params : AjtaiParams}
  (h : AjtaiBindingBoundary params) : AjtaiBindingAssumption params :=
  h.hardness

def AjtaiBindingBoundary_hardnessFromFields
  {params : AjtaiParams}
  (h : AjtaiBindingBoundary params) : AjtaiBindingAssumption params :=
  h.hardnessFromFields

def AjtaiRelaxedBindingBoundary_hardness
  {params : AjtaiParams}
  {C : SamplingCarrier}
  (h : AjtaiRelaxedBindingBoundary params C) : AjtaiRelaxedBindingAssumption params C :=
  h.hardness

def AjtaiRelaxedBindingBoundary_hardnessFromFields
  {params : AjtaiParams}
  {C : SamplingCarrier}
  (h : AjtaiRelaxedBindingBoundary params C) : AjtaiRelaxedBindingAssumption params C :=
  h.hardnessFromFields

theorem MSISToAjtaiReductions_toBinding
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiBindingAssumption params :=
  hRed.toBinding hMsis

theorem MSISToAjtaiReductions_toRelaxedBinding
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiRelaxedBindingAssumption params hRed.laws.samplingCarrier :=
  hRed.toRelaxedBinding hMsis

theorem ajtaiBinding_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiBindingAssumption params :=
  SuperNeo.ProofSystem.ajtaiBinding_of_msis hRed hMsis

theorem ajtaiRelaxedBinding_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiRelaxedBindingAssumption params hRed.laws.samplingCarrier :=
  SuperNeo.ProofSystem.ajtaiRelaxedBinding_of_msis hRed hMsis

theorem ajtaiBoundaries_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiBindingAssumption params ∧
    AjtaiRelaxedBindingAssumption params hRed.laws.samplingCarrier :=
  SuperNeo.ProofSystem.ajtaiBoundaries_of_msis hRed hMsis

def ajtaiBindingBoundary_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiBindingBoundary params :=
  SuperNeo.ProofSystem.ajtaiBindingBoundary_of_msis hRed hMsis

def ajtaiRelaxedBindingBoundary_of_msis
  {params : AjtaiParams}
  (hRed : MSISToAjtaiReductions params)
  (hMsis : MSISHardnessAssumption params) :
  AjtaiRelaxedBindingBoundary params hRed.laws.samplingCarrier :=
  SuperNeo.ProofSystem.ajtaiRelaxedBindingBoundary_of_msis hRed hMsis

end
end ProofSystem.LatticeReductionsInterface
end SuperNeo
