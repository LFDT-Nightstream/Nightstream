import SuperNeo.PiRLC

/-! Pi_DEC reduction statement and bridge lemmas. -/


namespace SuperNeo

open F

/--
`Π_DEC` boundary: from a relaxed CE relation, recover decomposition and norm obligations.
-/
def PiDECUpgradeAssumption : Prop :=
  ∀ (ctx : ProtocolCtx) (claim : CEClaim) (wit : CEWitness),
    CERelationRelaxed ctx claim wit →
    p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit ∧
    normInfCoeffs wit.z < ctx.ceNormBound

/--
`Π_DEC` packaging of final reduction obligations prior to CE validation.
-/
def PiDECFinalTarget (ctx : ProtocolCtx) (claim : CEClaim) (wit : CEWitness) : Prop :=
  CERelation ctx claim wit ∧
    p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit

theorem piDECFinalTarget_of_assumption
  (hDec : PiDECUpgradeAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {wit : CEWitness}
  (hRelaxed : CERelationRelaxed ctx claim wit) :
  PiDECFinalTarget ctx claim wit := by
  have hUp : p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit ∧
      normInfCoeffs wit.z < ctx.ceNormBound :=
    hDec ctx claim wit hRelaxed
  refine ⟨?_, hUp.1⟩
  exact ceRelation_of_relaxed_and_norm hRelaxed hUp.2

theorem piDEC_ceRelation_of_assumption
  (hDec : PiDECUpgradeAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {wit : CEWitness}
  (hRelaxed : CERelationRelaxed ctx claim wit) :
  CERelation ctx claim wit := by
  exact (piDECFinalTarget_of_assumption hDec hRelaxed).1

theorem piDECDecomp_of_assumption
  (hDec : PiDECUpgradeAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {wit : CEWitness}
  (hRelaxed : CERelationRelaxed ctx claim wit) :
  p20DecompProp claim.zDecomp ctx.bSplit ctx.kSplit := by
  exact (piDECFinalTarget_of_assumption hDec hRelaxed).2

theorem piDECInvertibilityWitness_of_ceRelation
  {ctx : ProtocolCtx} {claim : CEClaim} {wit : CEWitness}
  (hCE : CERelation ctx claim wit) :
  ∃ deltaInv : Coeffs, mulRq claim.invDelta deltaInv = oneRq := by
  exact claimArithmetic_invertibilityWitness hCE.1.2

theorem piDECFinal_with_invertibility_of_assumption
  (hDec : PiDECUpgradeAssumption)
  {ctx : ProtocolCtx} {claim : CEClaim} {wit : CEWitness}
  (hRelaxed : CERelationRelaxed ctx claim wit) :
  ∃ deltaInv : Coeffs,
    mulRq claim.invDelta deltaInv = oneRq ∧ PiDECFinalTarget ctx claim wit := by
  have hFinal : PiDECFinalTarget ctx claim wit := piDECFinalTarget_of_assumption hDec hRelaxed
  rcases piDECInvertibilityWitness_of_ceRelation hFinal.1 with ⟨deltaInv, hMul⟩
  exact ⟨deltaInv, hMul, hFinal⟩

/--
Composition hook: `Π_RLC` weak output upgraded by `Π_DEC`.
-/
theorem piDEC_of_piRLC_assumptions
  (hWeak : PiRLCWeakRelationAssumption)
  (hDec : PiDECUpgradeAssumption)
  {ctx : ProtocolCtx}
  {claimL claimR claimOut : CEClaim}
  {witL witR witOut : CEWitness}
  (hLeft : CERelation ctx claimL witL)
  (hRight : CERelation ctx claimR witR) :
  PiDECFinalTarget ctx claimOut witOut := by
  have hRelaxed : CERelationRelaxed ctx claimOut witOut :=
    piRLCWeakIR_relaxed_of_assumption hWeak hLeft hRight
  exact piDECFinalTarget_of_assumption hDec hRelaxed

end SuperNeo
