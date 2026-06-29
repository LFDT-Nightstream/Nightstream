import SuperNeo.FoldingProtocol.PiRLC

/-!
Reduction-of-knowledge step `Π_DEC`.
-/

namespace SuperNeo

/-- Knowledge-style `Π_DEC` target statement. -/
def piDECKnowledgeStatement (ctx : ProtocolTargetContext) : Prop :=
  ∃ deltaInv : Coeffs,
    mulRq ctx.invDelta deltaInv = oneRq ∧
    ceRelaxedRelation ctx ∧
    SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive `Π_DEC` directly from the weak `Π_RLC` statement. -/
theorem piDEC_of_weak
  {ctx : ProtocolTargetContext}
  (hWeak : piRLCWeakStatement ctx) :
  piDECKnowledgeStatement ctx := by
  have hTarget : protocolTargetProp ctx := hWeak.1
  rcases hTarget with ⟨_hThm3, _hSplit, _hEvalHom, _hVecMod, _hScalMod, _hSampling,
      _hMleSize, _hMleId, _hInterp, hInvDelta⟩
  rcases hInvDelta with ⟨deltaInv, hMul⟩
  exact ⟨deltaInv, hMul, hWeak.1, hWeak.2⟩

/-- Derive `Π_DEC` directly from the CE relation. -/
theorem piDEC_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_weak (piRLCWeak_of_ce hCE)

/-- Derive `Π_DEC` statement from weak relation and invertibility boundary. -/
theorem piDEC_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piDECKnowledgeStatement ctx := by
  exact piDEC_of_weak (piRLCWeak_of_assumptions h hWitness)

end SuperNeo
