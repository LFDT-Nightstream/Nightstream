import SuperNeo.ProtocolTheorem
import SuperNeo.Sumcheck
import SuperNeo.InvertibilityAxioms
import SuperNeo.ProofSystem.Security
import SuperNeo.ProofSystem.Types

/-!
Paper-facing end-to-end protocol theorem surface.

This module exposes the final theorem shells through the `ProofSystem` namespace
so callers can remain on protocol-native names across constraint system,
sumcheck, folding, and final theorem composition.
-/

namespace SuperNeo.ProofSystem.Protocol

abbrev ReductionAssumptions := SuperNeo.SuperNeoReductionAssumption
abbrev CheckBundleReductionAssumptions := SuperNeo.SuperNeoCheckBundleReductionAssumption
abbrev StrongReductionAssumptions := SuperNeo.SuperNeoStrongCCSReductionAssumption
abbrev StrongCheckBundleReductionAssumptions :=
  SuperNeo.SuperNeoStrongCheckBundleReductionAssumption

/--
Explicit registry of trusted assumptions for the paper-facing final theorem.

`reduction` captures the already-threaded protocol reduction assumptions. The
other fields keep cryptographic/probabilistic boundaries explicit at the final
API surface, even when their full proofs are still external.
-/
structure FinalTheoremAssumptions where
  reduction : ReductionAssumptions
  sumcheckSoundnessBoundary : SuperNeo.SumcheckSoundnessAssumption
  sumcheckCompletenessBoundary : SuperNeo.SumcheckCompletenessAssumption
  schwartzZippelBoundary : Prop
  ajtaiBindingBoundary : Prop
  ajtaiRelaxedBindingBoundary : Prop
  lowNormInvertibilityBoundary : SuperNeo.LowNormInvertibilityAssumption
  errorModel : SuperNeo.ProofSystem.Security.ErrorModel

/--
Canonical completeness-statement shape for the final protocol theorem.
-/
def FinalCompletenessStatement (_hA : FinalTheoremAssumptions) : Prop :=
  ∀ {ctx : PSContext}
    {claimL claimR claimOut : PSClaim}
    {witL witR witOut : PSWitness}
    {instL instR : PSSumcheckInstance}
    {trL trR : PSSumcheckTranscript},
    SuperNeo.ClaimShapeValid claimL ->
    SuperNeo.ClaimShapeValid claimR ->
    SuperNeo.IsDBarMatrix ctx.bar ->
    SuperNeo.IsDVec claimL.a ->
    SuperNeo.IsDVec claimL.b ->
    SuperNeo.IsDVec claimR.a ->
    SuperNeo.IsDVec claimR.b ->
    SuperNeo.IsDVec claimOut.a ->
    SuperNeo.IsDVec claimOut.b ->
    SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b ->
    SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b ->
    SuperNeo.p10CoreProp ctx.bar claimOut.a claimOut.b ->
    PSSumcheckAccepted instL trL ->
    PSSumcheckAccepted instR trR ->
    witL.z = claimL.z ->
    witR.z = claimR.z ->
    SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound ->
    SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound ->
    PSCEValid ctx claimOut witOut

/--
Canonical knowledge-soundness boundary shape for the final protocol theorem.

This keeps the probability/error endpoint explicit while the concrete extractor
machinery is being finalized.
-/
def FinalKnowledgeSoundnessStatement
  (hA : FinalTheoremAssumptions)
  (_prob : SuperNeo.ProofSystem.Security.ProbModel) : Prop :=
  ∀ _lam : SuperNeo.ProofSystem.Security.SecurityParam,
    ∃ ε : SuperNeo.ProofSystem.Security.ErrorFn,
      SuperNeo.ProofSystem.Security.IsNegligible ε ∧ ε = hA.errorModel.ε_total

/-- Combined paper-facing final theorem shape (completeness + RoK boundary). -/
structure FinalTheoremShape
  (hA : FinalTheoremAssumptions)
  (prob : SuperNeo.ProofSystem.Security.ProbModel) : Prop where
  completeness : FinalCompletenessStatement hA
  knowledgeSoundness : FinalKnowledgeSoundnessStatement hA prob

theorem finalCompleteness_of_assumptions
  (hA : FinalTheoremAssumptions) :
  FinalCompletenessStatement hA := by
  intro ctx claimL claimR claimOut witL witR witOut instL instR trL trR
  intro hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
  intro hP10L hP10R hP10Out hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR
  exact SuperNeo.superneoProtocolTheorem_of_assumptions hA.reduction
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

theorem finalKnowledgeSoundness_boundary
  (hA : FinalTheoremAssumptions)
  (prob : SuperNeo.ProofSystem.Security.ProbModel) :
  FinalKnowledgeSoundnessStatement hA prob := by
  intro _lam
  exact ⟨hA.errorModel.ε_total, hA.errorModel.hNeg_total, rfl⟩

theorem finalTheoremShape_of_assumptions
  (hA : FinalTheoremAssumptions)
  (prob : SuperNeo.ProofSystem.Security.ProbModel) :
  FinalTheoremShape hA prob := by
  exact ⟨finalCompleteness_of_assumptions hA, finalKnowledgeSoundness_boundary hA prob⟩

/-- End-to-end CE-validity wrapper from reduction assumptions. -/
theorem ceValid_of_assumptions
  (hRed : ReductionAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : SuperNeo.ClaimShapeValid claimL)
  (hShapeR : SuperNeo.ClaimShapeValid claimR)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hAL : SuperNeo.IsDVec claimL.a)
  (hBL : SuperNeo.IsDVec claimL.b)
  (hAR : SuperNeo.IsDVec claimR.a)
  (hBR : SuperNeo.IsDVec claimR.b)
  (hAOut : SuperNeo.IsDVec claimOut.a)
  (hBOut : SuperNeo.IsDVec claimOut.b)
  (hP10L : SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : SuperNeo.p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut :=
  SuperNeo.superneoProtocolTheorem_of_assumptions hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

/-- End-to-end CE-validity wrapper from check-bundle reduction assumptions. -/
theorem ceValid_of_checkBundle_assumptions
  (hRed : CheckBundleReductionAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : SuperNeo.ClaimShapeValid claimL)
  (hShapeR : SuperNeo.ClaimShapeValid claimR)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hAL : SuperNeo.IsDVec claimL.a)
  (hBL : SuperNeo.IsDVec claimL.b)
  (hAR : SuperNeo.IsDVec claimR.a)
  (hBR : SuperNeo.IsDVec claimR.b)
  (hAOut : SuperNeo.IsDVec claimOut.a)
  (hBOut : SuperNeo.IsDVec claimOut.b)
  (hP10L : SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : SuperNeo.p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut :=
  SuperNeo.superneoProtocolTheorem_of_checkBundle_assumptions hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

/-- End-to-end wrapper with explicit invertibility witness extraction. -/
theorem ceValid_with_invertibility
  (hRed : ReductionAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : SuperNeo.ClaimShapeValid claimL)
  (hShapeR : SuperNeo.ClaimShapeValid claimR)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hAL : SuperNeo.IsDVec claimL.a)
  (hBL : SuperNeo.IsDVec claimL.b)
  (hAR : SuperNeo.IsDVec claimR.a)
  (hBR : SuperNeo.IsDVec claimR.b)
  (hAOut : SuperNeo.IsDVec claimOut.a)
  (hBOut : SuperNeo.IsDVec claimOut.b)
  (hP10L : SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : SuperNeo.p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : SuperNeo.Coeffs,
    SuperNeo.mulRq claimOut.invDelta deltaInv = SuperNeo.oneRq ∧
      PSCEValid ctx claimOut witOut :=
  SuperNeo.superneoProtocolTheorem_with_invertibility_of_assumptions hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

/-- End-to-end CE-validity wrapper from reduction assumptions with strong SumCheck acceptance. -/
theorem ceValid_of_assumptions_with_strongAccepted
  (hRed : ReductionAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : SuperNeo.ClaimShapeValid claimL)
  (hShapeR : SuperNeo.ClaimShapeValid claimR)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hAL : SuperNeo.IsDVec claimL.a)
  (hBL : SuperNeo.IsDVec claimL.b)
  (hAR : SuperNeo.IsDVec claimR.a)
  (hBR : SuperNeo.IsDVec claimR.b)
  (hAOut : SuperNeo.IsDVec claimOut.a)
  (hBOut : SuperNeo.IsDVec claimOut.b)
  (hP10L : SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : SuperNeo.p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut :=
  SuperNeo.superneoProtocolTheorem_of_assumptions_with_strongAccepted hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR hWitnessL hWitnessR hNormL hNormR

/-- End-to-end CE-validity wrapper from strong reduction assumptions. -/
theorem ceValid_of_strong_assumptions
  (hRed : StrongReductionAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : SuperNeo.ClaimShapeValid claimL)
  (hShapeR : SuperNeo.ClaimShapeValid claimR)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hAL : SuperNeo.IsDVec claimL.a)
  (hBL : SuperNeo.IsDVec claimL.b)
  (hAR : SuperNeo.IsDVec claimR.a)
  (hBR : SuperNeo.IsDVec claimR.b)
  (hAOut : SuperNeo.IsDVec claimOut.a)
  (hBOut : SuperNeo.IsDVec claimOut.b)
  (hP10L : SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : SuperNeo.p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut :=
  SuperNeo.superneoProtocolTheorem_of_strongCCS_assumptionBundle hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR hWitnessL hWitnessR hNormL hNormR

/-- End-to-end CE-validity wrapper from strong check-bundle reduction assumptions. -/
theorem ceValid_of_strong_checkBundle_assumptions
  (hRed : StrongCheckBundleReductionAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : SuperNeo.ClaimShapeValid claimL)
  (hShapeR : SuperNeo.ClaimShapeValid claimR)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hAL : SuperNeo.IsDVec claimL.a)
  (hBL : SuperNeo.IsDVec claimL.b)
  (hAR : SuperNeo.IsDVec claimR.a)
  (hBR : SuperNeo.IsDVec claimR.b)
  (hAOut : SuperNeo.IsDVec claimOut.a)
  (hBOut : SuperNeo.IsDVec claimOut.b)
  (hP10L : SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : SuperNeo.p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut :=
  SuperNeo.superneoProtocolTheorem_of_strongCheckBundle_assumptions hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR hWitnessL hWitnessR hNormL hNormR

/-- End-to-end wrapper with explicit invertibility witness from check-bundle assumptions. -/
theorem ceValid_with_invertibility_of_checkBundle_assumptions
  (hRed : CheckBundleReductionAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : SuperNeo.ClaimShapeValid claimL)
  (hShapeR : SuperNeo.ClaimShapeValid claimR)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hAL : SuperNeo.IsDVec claimL.a)
  (hBL : SuperNeo.IsDVec claimL.b)
  (hAR : SuperNeo.IsDVec claimR.a)
  (hBR : SuperNeo.IsDVec claimR.b)
  (hAOut : SuperNeo.IsDVec claimOut.a)
  (hBOut : SuperNeo.IsDVec claimOut.b)
  (hP10L : SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : SuperNeo.p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : SuperNeo.Coeffs,
    SuperNeo.mulRq claimOut.invDelta deltaInv = SuperNeo.oneRq ∧
      PSCEValid ctx claimOut witOut :=
  SuperNeo.superneoProtocolTheorem_with_invertibility_of_checkBundle_assumptions hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

/-- End-to-end wrapper with invertibility witness under strong SumCheck acceptance. -/
theorem ceValid_with_invertibility_of_assumptions_with_strongAccepted
  (hRed : ReductionAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : SuperNeo.ClaimShapeValid claimL)
  (hShapeR : SuperNeo.ClaimShapeValid claimR)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hAL : SuperNeo.IsDVec claimL.a)
  (hBL : SuperNeo.IsDVec claimL.b)
  (hAR : SuperNeo.IsDVec claimR.a)
  (hBR : SuperNeo.IsDVec claimR.b)
  (hAOut : SuperNeo.IsDVec claimOut.a)
  (hBOut : SuperNeo.IsDVec claimOut.b)
  (hP10L : SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : SuperNeo.p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : SuperNeo.Coeffs,
    SuperNeo.mulRq claimOut.invDelta deltaInv = SuperNeo.oneRq ∧
      PSCEValid ctx claimOut witOut :=
  SuperNeo.superneoProtocolTheorem_with_invertibility_of_assumptions_with_strongAccepted hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR hWitnessL hWitnessR hNormL hNormR

/-- End-to-end wrapper with invertibility witness from strong reduction assumptions. -/
theorem ceValid_with_invertibility_of_strong_assumptions
  (hRed : StrongReductionAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : SuperNeo.ClaimShapeValid claimL)
  (hShapeR : SuperNeo.ClaimShapeValid claimR)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hAL : SuperNeo.IsDVec claimL.a)
  (hBL : SuperNeo.IsDVec claimL.b)
  (hAR : SuperNeo.IsDVec claimR.a)
  (hBR : SuperNeo.IsDVec claimR.b)
  (hAOut : SuperNeo.IsDVec claimOut.a)
  (hBOut : SuperNeo.IsDVec claimOut.b)
  (hP10L : SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : SuperNeo.p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : SuperNeo.Coeffs,
    SuperNeo.mulRq claimOut.invDelta deltaInv = SuperNeo.oneRq ∧
      PSCEValid ctx claimOut witOut :=
  SuperNeo.superneoProtocolTheorem_with_invertibility_of_strongCCS_assumptionBundle hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR hWitnessL hWitnessR hNormL hNormR

/-- End-to-end wrapper with invertibility witness from strong check-bundle assumptions. -/
theorem ceValid_with_invertibility_of_strong_checkBundle_assumptions
  (hRed : StrongCheckBundleReductionAssumptions)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : SuperNeo.ClaimShapeValid claimL)
  (hShapeR : SuperNeo.ClaimShapeValid claimR)
  (hBar : SuperNeo.IsDBarMatrix ctx.bar)
  (hAL : SuperNeo.IsDVec claimL.a)
  (hBL : SuperNeo.IsDVec claimL.b)
  (hAR : SuperNeo.IsDVec claimR.a)
  (hBR : SuperNeo.IsDVec claimR.b)
  (hAOut : SuperNeo.IsDVec claimOut.a)
  (hBOut : SuperNeo.IsDVec claimOut.b)
  (hP10L : SuperNeo.p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : SuperNeo.p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : SuperNeo.p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : SuperNeo.normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : SuperNeo.normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : SuperNeo.Coeffs,
    SuperNeo.mulRq claimOut.invDelta deltaInv = SuperNeo.oneRq ∧
      PSCEValid ctx claimOut witOut :=
  SuperNeo.superneoProtocolTheorem_with_invertibility_of_strongCheckBundle_assumptions hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR hWitnessL hWitnessR hNormL hNormR

end SuperNeo.ProofSystem.Protocol
