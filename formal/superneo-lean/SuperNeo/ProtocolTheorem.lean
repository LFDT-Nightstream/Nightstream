import SuperNeo.InteractiveReductions

/-! Top-level protocol theorem shells over reduction layers. -/


namespace SuperNeo

open F

/--
End-to-end SuperNeo protocol theorem wrapper (assumption-parameterized):
compose `Π_CCS`, `Π_RLC`, and `Π_DEC` reductions into CE validity.
-/
theorem superneoProtocolTheorem_of_assumptions
  (hRed : SuperNeoReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut := by
  exact superneoReduction_ceValid_of_assumptions
    hRed.1 hRed.2.1 hRed.2.2
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedL hAcceptedR
    hWitnessL hWitnessR hNormL hNormR

theorem superneoProtocolTheorem_of_checkBundle_assumptions
  (hRed : SuperNeoCheckBundleReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut := by
  exact superneoProtocolTheorem_of_assumptions
    (superneoReductionAssumption_of_checkBundleReductionAssumption hRed)
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

theorem superneoProtocolTheorem_of_assumptions_with_strongAccepted
  (hRed : SuperNeoReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut := by
  exact superneoProtocolTheorem_of_assumptions hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    (sumcheckAcceptedStrong_implies_accepted hAcceptedStrongL)
    (sumcheckAcceptedStrong_implies_accepted hAcceptedStrongR)
    hWitnessL hWitnessR hNormL hNormR

theorem superneoProtocolTheorem_of_checkBundle_assumptions_with_strongAccepted
  (hRed : SuperNeoCheckBundleReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut := by
  exact superneoProtocolTheorem_of_assumptions_with_strongAccepted
    (superneoReductionAssumption_of_checkBundleReductionAssumption hRed)
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR
    hWitnessL hWitnessR hNormL hNormR

/--
End-to-end wrapper plus explicit invertibility witness for `invDelta`.
-/
theorem superneoProtocolTheorem_with_invertibility_of_assumptions
  (hRed : SuperNeoReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claimOut.invDelta deltaInv = oneRq ∧ PSCEValid ctx claimOut witOut := by
  have hCEValid : PSCEValid ctx claimOut witOut :=
    superneoProtocolTheorem_of_assumptions hRed
      hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
      hP10L hP10R hP10Out
      hAcceptedL hAcceptedR
      hWitnessL hWitnessR hNormL hNormR
  have hCE : PSCERelation ctx claimOut witOut := ceRelation_of_ceValid hCEValid
  rcases claimArithmetic_invertibilityWitness hCE.1.2 with ⟨deltaInv, hMul⟩
  exact ⟨deltaInv, hMul, hCEValid⟩

theorem superneoProtocolTheorem_with_invertibility_of_checkBundle_assumptions
  (hRed : SuperNeoCheckBundleReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claimOut.invDelta deltaInv = oneRq ∧ PSCEValid ctx claimOut witOut := by
  exact superneoProtocolTheorem_with_invertibility_of_assumptions
    (superneoReductionAssumption_of_checkBundleReductionAssumption hRed)
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

theorem superneoProtocolTheorem_with_invertibility_of_assumptions_with_strongAccepted
  (hRed : SuperNeoReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claimOut.invDelta deltaInv = oneRq ∧ PSCEValid ctx claimOut witOut := by
  exact superneoProtocolTheorem_with_invertibility_of_assumptions hRed
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    (sumcheckAcceptedStrong_implies_accepted hAcceptedStrongL)
    (sumcheckAcceptedStrong_implies_accepted hAcceptedStrongR)
    hWitnessL hWitnessR hNormL hNormR

/--
End-to-end theorem wrapper variant driven by strong `Π_CCS` assumptions and
round-consistent accepted transcripts.
-/
theorem superneoProtocolTheorem_of_strongCCS_assumptions
  (hCCSStrong : PiCCSStrongProtocolAssumption)
  (hRLC : PiRLCWeakRelationAssumption)
  (hDEC : PiDECUpgradeAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut := by
  have hFinal : PSDECFinalTarget ctx claimOut witOut :=
    superneoReduction_chain_of_strongCCS_assumptions
      hCCSStrong hRLC hDEC
      hShapeL hShapeR hBar hAL hBL hAR hBR hP10L hP10R
      hAcceptedStrongL hAcceptedStrongR hWitnessL hWitnessR hNormL hNormR
  have hCCSOut : PSCCSRelation ctx claimOut := ⟨hBar, hAOut, hBOut, hP10Out⟩
  exact ceValid_of_relations hCCSOut hFinal.1

theorem superneoProtocolTheorem_of_strongCCS_assumptionBundle
  (hRedStrong : SuperNeoStrongCCSReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut := by
  exact superneoProtocolTheorem_of_strongCCS_assumptions
    hRedStrong.1 hRedStrong.2.1 hRedStrong.2.2
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR
    hWitnessL hWitnessR hNormL hNormR

theorem superneoProtocolTheorem_of_strongCheckBundle_assumptions
  (hRedStrong : SuperNeoStrongCheckBundleReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut := by
  exact superneoProtocolTheorem_of_strongCCS_assumptionBundle
    (superneoStrongCCSReductionAssumption_of_strongCheckBundleReductionAssumption hRedStrong)
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR
    hWitnessL hWitnessR hNormL hNormR

theorem superneoProtocolTheorem_of_reductionAssumption_with_strongAccepted
  (hRed : SuperNeoReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut := by
  exact superneoProtocolTheorem_of_strongCCS_assumptionBundle
    (superneoStrongCCSReductionAssumption_of_reductionAssumption hRed)
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR
    hWitnessL hWitnessR hNormL hNormR

theorem superneoProtocolTheorem_with_invertibility_of_strongCCS_assumptions
  (hCCSStrong : PiCCSStrongProtocolAssumption)
  (hRLC : PiRLCWeakRelationAssumption)
  (hDEC : PiDECUpgradeAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claimOut.invDelta deltaInv = oneRq ∧ PSCEValid ctx claimOut witOut := by
  have hCEValid : PSCEValid ctx claimOut witOut :=
    superneoProtocolTheorem_of_strongCCS_assumptions
      hCCSStrong hRLC hDEC
      hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
      hP10L hP10R hP10Out
      hAcceptedStrongL hAcceptedStrongR
      hWitnessL hWitnessR hNormL hNormR
  have hCE : PSCERelation ctx claimOut witOut := ceRelation_of_ceValid hCEValid
  rcases claimArithmetic_invertibilityWitness hCE.1.2 with ⟨deltaInv, hMul⟩
  exact ⟨deltaInv, hMul, hCEValid⟩

theorem superneoProtocolTheorem_with_invertibility_of_strongCCS_assumptionBundle
  (hRedStrong : SuperNeoStrongCCSReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claimOut.invDelta deltaInv = oneRq ∧ PSCEValid ctx claimOut witOut := by
  exact superneoProtocolTheorem_with_invertibility_of_strongCCS_assumptions
    hRedStrong.1 hRedStrong.2.1 hRedStrong.2.2
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR
    hWitnessL hWitnessR hNormL hNormR

theorem superneoProtocolTheorem_with_invertibility_of_strongCheckBundle_assumptions
  (hRedStrong : SuperNeoStrongCheckBundleReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claimOut.invDelta deltaInv = oneRq ∧ PSCEValid ctx claimOut witOut := by
  exact superneoProtocolTheorem_with_invertibility_of_strongCCS_assumptionBundle
    (superneoStrongCCSReductionAssumption_of_strongCheckBundleReductionAssumption hRedStrong)
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR
    hWitnessL hWitnessR hNormL hNormR

theorem superneoProtocolTheorem_with_invertibility_of_reductionAssumption_with_strongAccepted
  (hRed : SuperNeoReductionAssumption)
  {ctx : PSContext}
  {claimL claimR claimOut : PSClaim}
  {witL witR witOut : PSWitness}
  {instL instR : PSSumcheckInstance}
  {trL trR : PSSumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hAOut : IsDVec claimOut.a)
  (hBOut : IsDVec claimOut.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hP10Out : p10CoreProp ctx.bar claimOut.a claimOut.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  ∃ deltaInv : Coeffs, mulRq claimOut.invDelta deltaInv = oneRq ∧ PSCEValid ctx claimOut witOut := by
  exact superneoProtocolTheorem_with_invertibility_of_strongCCS_assumptionBundle
    (superneoStrongCCSReductionAssumption_of_reductionAssumption hRed)
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedStrongL hAcceptedStrongR
    hWitnessL hWitnessR hNormL hNormR

end SuperNeo
