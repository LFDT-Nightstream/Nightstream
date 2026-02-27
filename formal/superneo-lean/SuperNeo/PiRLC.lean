import SuperNeo.PiCCS

/-! Pi_RLC reduction statement and bridge lemmas. -/


namespace SuperNeo

open F

/--
Weak-interactive-reduction boundary for `Π_RLC`:
two CE relations fold into one relaxed CE relation.
-/
def PiRLCWeakRelationAssumption : Prop :=
  ∀ (ctx : ProtocolCtx)
    (claimL claimR claimOut : CEClaim)
    (witL witR witOut : CEWitness),
    CERelation ctx claimL witL →
    CERelation ctx claimR witR →
    CERelationRelaxed ctx claimOut witOut

/--
Optional boundary for re-establishing the CE norm bound after weak reduction.
-/
def PiRLCNormRebindAssumption : Prop :=
  ∀ (ctx : ProtocolCtx) (claim : CEClaim) (wit : CEWitness),
    CERelationRelaxed ctx claim wit →
    normInfCoeffs wit.z < ctx.ceNormBound

/-- Compact protocol bundle for `Π_RLC`. -/
def PiRLCProtocolAssumption : Prop :=
  PiRLCWeakRelationAssumption ∧ PiRLCNormRebindAssumption

theorem piRLCWeakIR_relaxed_of_assumption
  (hWeak : PiRLCWeakRelationAssumption)
  {ctx : ProtocolCtx}
  {claimL claimR claimOut : CEClaim}
  {witL witR witOut : CEWitness}
  (hLeft : CERelation ctx claimL witL)
  (hRight : CERelation ctx claimR witR) :
  CERelationRelaxed ctx claimOut witOut := by
  exact hWeak ctx claimL claimR claimOut witL witR witOut hLeft hRight

theorem piRLCWeakIR_ce_of_assumption
  (hWeak : PiRLCWeakRelationAssumption)
  {ctx : ProtocolCtx}
  {claimL claimR claimOut : CEClaim}
  {witL witR witOut : CEWitness}
  (hLeft : CERelation ctx claimL witL)
  (hRight : CERelation ctx claimR witR)
  (hNormOut : normInfCoeffs witOut.z < ctx.ceNormBound) :
  CERelation ctx claimOut witOut := by
  exact ceRelation_of_relaxed_and_norm
    (piRLCWeakIR_relaxed_of_assumption hWeak hLeft hRight) hNormOut

theorem piRLCWeakIR_ce_of_protocolAssumption
  (hProto : PiRLCProtocolAssumption)
  {ctx : ProtocolCtx}
  {claimL claimR claimOut : CEClaim}
  {witL witR witOut : CEWitness}
  (hLeft : CERelation ctx claimL witL)
  (hRight : CERelation ctx claimR witR) :
  CERelation ctx claimOut witOut := by
  have hRelaxed : CERelationRelaxed ctx claimOut witOut :=
    piRLCWeakIR_relaxed_of_assumption hProto.1 hLeft hRight
  exact ceRelation_of_relaxed_and_norm hRelaxed (hProto.2 ctx claimOut witOut hRelaxed)

theorem piRLCWeakIR_relaxed_of_ceValid_pair
  (hWeak : PiRLCWeakRelationAssumption)
  {ctx : ProtocolCtx}
  {claimL claimR claimOut : CEClaim}
  {witL witR witOut : CEWitness}
  (hLeft : CEValid ctx claimL witL)
  (hRight : CEValid ctx claimR witR) :
  CERelationRelaxed ctx claimOut witOut := by
  exact hWeak ctx claimL claimR claimOut witL witR witOut
    (ceRelation_of_ceValid hLeft)
    (ceRelation_of_ceValid hRight)

theorem piRLCWeakIR_relaxed_of_piCCS_assumptions
  (hSumSound : SumcheckSoundnessAssumption)
  (hCCSLink : PiCCSArithmeticLinkAssumption)
  (hWeak : PiRLCWeakRelationAssumption)
  {ctx : ProtocolCtx}
  {claimL claimR claimOut : CEClaim}
  {witL witR witOut : CEWitness}
  {instL instR : SumcheckInstance}
  {trL trR : SumcheckTranscript}
  (hShapeL : ClaimShapeValid claimL)
  (hShapeR : ClaimShapeValid claimR)
  (hBar : IsDBarMatrix ctx.bar)
  (hAL : IsDVec claimL.a)
  (hBL : IsDVec claimL.b)
  (hAR : IsDVec claimR.a)
  (hBR : IsDVec claimR.b)
  (hP10L : p10CoreProp ctx.bar claimL.a claimL.b)
  (hP10R : p10CoreProp ctx.bar claimR.a claimR.b)
  (hAcceptedL : SumcheckAcceptedProp instL trL)
  (hAcceptedR : SumcheckAcceptedProp instR trR)
  (hWitnessL : witL.z = claimL.z)
  (hWitnessR : witR.z = claimR.z)
  (hNormL : normInfCoeffs witL.z < ctx.ceNormBound)
  (hNormR : normInfCoeffs witR.z < ctx.ceNormBound) :
  CERelationRelaxed ctx claimOut witOut := by
  have hRelL : CCSRelation ctx claimL ∧ CERelation ctx claimL witL :=
    piCCSStrongIR_relations_of_assumptions hSumSound hCCSLink
      hShapeL hBar hAL hBL hP10L hAcceptedL hWitnessL hNormL
  have hRelR : CCSRelation ctx claimR ∧ CERelation ctx claimR witR :=
    piCCSStrongIR_relations_of_assumptions hSumSound hCCSLink
      hShapeR hBar hAR hBR hP10R hAcceptedR hWitnessR hNormR
  exact hWeak ctx claimL claimR claimOut witL witR witOut hRelL.2 hRelR.2

end SuperNeo
