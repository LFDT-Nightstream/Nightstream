import SuperNeo.ProofSystem.Types
import SuperNeo.ProofSystem.Folding.PiCCS
import SuperNeo.ProofSystem.Folding.PiRLC
import SuperNeo.ProofSystem.Folding.PiDEC

/-! Interactive reduction composition surfaces (Pi_CCS, Pi_RLC, Pi_DEC). -/


namespace SuperNeo

open F

abbrev PSDECFinalTarget := SuperNeo.ProofSystem.Folding.PiDEC.FinalTarget

abbrev PSCCSProtocolAssumptions := SuperNeo.ProofSystem.Folding.PiCCS.ProtocolAssumptions
abbrev PSCCSStrongProtocolAssumptions := SuperNeo.ProofSystem.Folding.PiCCS.StrongProtocolAssumptions
abbrev PSCCSCheckBundleProtocolAssumptions :=
  SuperNeo.ProofSystem.Folding.PiCCS.CheckBundleProtocolAssumptions
abbrev PSCCSStrongCheckBundleProtocolAssumptions :=
  SuperNeo.ProofSystem.Folding.PiCCS.StrongCheckBundleProtocolAssumptions
abbrev PSRLCWeakAssumptions := SuperNeo.ProofSystem.Folding.PiRLC.WeakAssumptions
abbrev PSDECUpgradeAssumptions := SuperNeo.ProofSystem.Folding.PiDEC.UpgradeAssumptions

/--
Composition assumptions for the chain `Π_DEC ∘ Π_RLC ∘ Π_CCS`.
-/
def SuperNeoReductionAssumption : Prop :=
  PSCCSProtocolAssumptions ∧
  PSRLCWeakAssumptions ∧
  PSDECUpgradeAssumptions

/--
Composition assumptions for the chain when `Π_CCS` is consumed through the
strong round-consistent SumCheck interface.
-/
def SuperNeoStrongCCSReductionAssumption : Prop :=
  PSCCSStrongProtocolAssumptions ∧
  PSRLCWeakAssumptions ∧
  PSDECUpgradeAssumptions

/--
Composition assumptions where the `Π_CCS` leg is supplied through the named
check-bundle protocol surface.
-/
def SuperNeoCheckBundleReductionAssumption : Prop :=
  PSCCSCheckBundleProtocolAssumptions ∧
  PSRLCWeakAssumptions ∧
  PSDECUpgradeAssumptions

/--
Composition assumptions where the strong `Π_CCS` leg is supplied through the
strong check-bundle protocol surface.
-/
def SuperNeoStrongCheckBundleReductionAssumption : Prop :=
  PSCCSStrongCheckBundleProtocolAssumptions ∧
  PSRLCWeakAssumptions ∧
  PSDECUpgradeAssumptions

theorem superneoReductionAssumption_of_checkBundleReductionAssumption
  (hRed : SuperNeoCheckBundleReductionAssumption) :
  SuperNeoReductionAssumption := by
  exact ⟨
    piCCSProtocolAssumption_of_checkBundleProtocolAssumption hRed.1,
    hRed.2.1,
    hRed.2.2
  ⟩

theorem superneoStrongCCSReductionAssumption_of_strongCheckBundleReductionAssumption
  (hRed : SuperNeoStrongCheckBundleReductionAssumption) :
  SuperNeoStrongCCSReductionAssumption := by
  exact ⟨
    piCCSStrongProtocolAssumption_of_strongCheckBundleProtocolAssumption hRed.1,
    hRed.2.1,
    hRed.2.2
  ⟩

theorem superneoStrongCheckBundleReductionAssumption_of_checkBundleReductionAssumption
  (hRed : SuperNeoCheckBundleReductionAssumption) :
  SuperNeoStrongCheckBundleReductionAssumption := by
  exact ⟨
    piCCSStrongCheckBundleProtocolAssumption_of_checkBundleProtocolAssumption hRed.1,
    hRed.2.1,
    hRed.2.2
  ⟩

theorem superneoStrongCCSReductionAssumption_of_reductionAssumption
  (hRed : SuperNeoReductionAssumption) :
  SuperNeoStrongCCSReductionAssumption := by
  exact ⟨
    piCCSStrongProtocolAssumption_of_protocolAssumption hRed.1,
    hRed.2.1,
    hRed.2.2
  ⟩

theorem superneoStrongCCSReductionAssumption_of_checkBundleReductionAssumption
  (hRed : SuperNeoCheckBundleReductionAssumption) :
  SuperNeoStrongCCSReductionAssumption := by
  exact superneoStrongCCSReductionAssumption_of_reductionAssumption
    (superneoReductionAssumption_of_checkBundleReductionAssumption hRed)

/--
Composed reduction theorem in relation form:
accepted `Π_CCS` transcripts for two inputs imply the final `Π_DEC` target.
-/
theorem superneoReduction_chain_of_assumptions
  (hCCS : PiCCSProtocolAssumption)
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
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSDECFinalTarget ctx claimOut witOut := by
  have hRelL : PSCCSRelation ctx claimL ∧ PSCERelation ctx claimL witL :=
    piCCSStrongIR_relations_of_protocolAssumption hCCS
      hShapeL hBar hAL hBL hP10L hAcceptedL hWitnessL hNormL
  have hRelR : PSCCSRelation ctx claimR ∧ PSCERelation ctx claimR witR :=
    piCCSStrongIR_relations_of_protocolAssumption hCCS
      hShapeR hBar hAR hBR hP10R hAcceptedR hWitnessR hNormR
  exact piDEC_of_piRLC_assumptions hRLC hDEC hRelL.2 hRelR.2

theorem superneoReduction_chain_of_protocolAssumption
  (hProto : SuperNeoReductionAssumption)
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
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSDECFinalTarget ctx claimOut witOut := by
  exact superneoReduction_chain_of_assumptions
    hProto.1 hProto.2.1 hProto.2.2
    hShapeL hShapeR hBar hAL hBL hAR hBR hP10L hP10R
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

theorem superneoReduction_chain_of_checkBundle_protocolAssumption
  (hProto : SuperNeoCheckBundleReductionAssumption)
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
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSDECFinalTarget ctx claimOut witOut := by
  exact superneoReduction_chain_of_protocolAssumption
    (superneoReductionAssumption_of_checkBundleReductionAssumption hProto)
    hShapeL hShapeR hBar hAL hBL hAR hBR hP10L hP10R
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

theorem superneoReduction_chain_of_assumptions_with_strongAccepted
  (hCCS : PiCCSProtocolAssumption)
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
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSDECFinalTarget ctx claimOut witOut := by
  exact superneoReduction_chain_of_assumptions
    hCCS hRLC hDEC
    hShapeL hShapeR hBar hAL hBL hAR hBR hP10L hP10R
    (sumcheckAcceptedStrong_implies_accepted hAcceptedStrongL)
    (sumcheckAcceptedStrong_implies_accepted hAcceptedStrongR)
    hWitnessL hWitnessR hNormL hNormR

theorem superneoReduction_chain_of_protocolAssumption_with_strongAccepted
  (hProto : SuperNeoReductionAssumption)
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
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSDECFinalTarget ctx claimOut witOut := by
  exact superneoReduction_chain_of_assumptions_with_strongAccepted
    hProto.1 hProto.2.1 hProto.2.2
    hShapeL hShapeR hBar hAL hBL hAR hBR hP10L hP10R
    hAcceptedStrongL hAcceptedStrongR hWitnessL hWitnessR hNormL hNormR

/--
Composed reduction theorem variant using the stronger `Π_CCS` protocol interface
that consumes round-consistent accepted SumCheck transcripts.
-/
theorem superneoReduction_chain_of_strongCCS_assumptions
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
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSDECFinalTarget ctx claimOut witOut := by
  have hRelL : PSCCSRelation ctx claimL ∧ PSCERelation ctx claimL witL :=
    piCCSStrongIR_relations_of_strongProtocolAssumption hCCSStrong
      hShapeL hBar hAL hBL hP10L hAcceptedStrongL hWitnessL hNormL
  have hRelR : PSCCSRelation ctx claimR ∧ PSCERelation ctx claimR witR :=
    piCCSStrongIR_relations_of_strongProtocolAssumption hCCSStrong
      hShapeR hBar hAR hBR hP10R hAcceptedStrongR hWitnessR hNormR
  exact piDEC_of_piRLC_assumptions hRLC hDEC hRelL.2 hRelR.2

theorem superneoReduction_chain_of_strongCCS_protocolAssumption
  (hProto : SuperNeoStrongCCSReductionAssumption)
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
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSDECFinalTarget ctx claimOut witOut := by
  exact superneoReduction_chain_of_strongCCS_assumptions
    hProto.1 hProto.2.1 hProto.2.2
    hShapeL hShapeR hBar hAL hBL hAR hBR hP10L hP10R
    hAcceptedStrongL hAcceptedStrongR hWitnessL hWitnessR hNormL hNormR

theorem superneoReduction_chain_of_strongCheckBundle_protocolAssumption
  (hProto : SuperNeoStrongCheckBundleReductionAssumption)
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
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedStrongL : PSSumcheckAcceptedStrong instL trL)
  (hAcceptedStrongR : PSSumcheckAcceptedStrong instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSDECFinalTarget ctx claimOut witOut := by
  exact superneoReduction_chain_of_strongCCS_protocolAssumption
    (superneoStrongCCSReductionAssumption_of_strongCheckBundleReductionAssumption hProto)
    hShapeL hShapeR hBar hAL hBL hAR hBR hP10L hP10R
    hAcceptedStrongL hAcceptedStrongR hWitnessL hWitnessR hNormL hNormR

theorem superneoReduction_ceRelation_of_assumptions
  (hCCS : PiCCSProtocolAssumption)
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
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCERelation ctx claimOut witOut := by
  exact (superneoReduction_chain_of_assumptions
    hCCS hRLC hDEC
    hShapeL hShapeR hBar hAL hBL hAR hBR hP10L hP10R
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR).1

theorem superneoReduction_ceRelation_of_checkBundle_assumptions
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
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCERelation ctx claimOut witOut := by
  exact superneoReduction_ceRelation_of_assumptions
    (superneoReductionAssumption_of_checkBundleReductionAssumption hRed).1
    (superneoReductionAssumption_of_checkBundleReductionAssumption hRed).2.1
    (superneoReductionAssumption_of_checkBundleReductionAssumption hRed).2.2
    hShapeL hShapeR hBar hAL hBL hAR hBR hP10L hP10R
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

theorem superneoReduction_ceValid_of_assumptions
  (hCCS : PiCCSProtocolAssumption)
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
  (hAcceptedL : PSSumcheckAccepted instL trL)
  (hAcceptedR : PSSumcheckAccepted instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  PSCEValid ctx claimOut witOut := by
  have hCCSOut : PSCCSRelation ctx claimOut := ⟨hBar, hAOut, hBOut, hP10Out⟩
  have hCEOut : PSCERelation ctx claimOut witOut :=
    superneoReduction_ceRelation_of_assumptions
      hCCS hRLC hDEC
      hShapeL hShapeR hBar hAL hBL hAR hBR hP10L hP10R
      hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR
  exact ceValid_of_relations hCCSOut hCEOut

theorem superneoReduction_ceValid_of_checkBundle_assumptions
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
  exact superneoReduction_ceValid_of_assumptions
    (superneoReductionAssumption_of_checkBundleReductionAssumption hRed).1
    (superneoReductionAssumption_of_checkBundleReductionAssumption hRed).2.1
    (superneoReductionAssumption_of_checkBundleReductionAssumption hRed).2.2
    hShapeL hShapeR hBar hAL hBL hAR hBR hAOut hBOut
    hP10L hP10R hP10Out
    hAcceptedL hAcceptedR hWitnessL hWitnessR hNormL hNormR

end SuperNeo
