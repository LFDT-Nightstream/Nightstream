import SuperNeo.PiRLC

/-!
Reduction-of-knowledge step `Π_DEC`.
-/

namespace SuperNeo

/-- Assumptions consumed by the `Π_DEC` step. -/
structure PiDECAssumptions (ctx : ProtocolTargetContext) where
  weak : PiRLCAssumptions ctx
  lowNormInvertibilityBoundary : lowNormInvertibilityAssumption Goldilocks.halfQ

/-- Knowledge-style `Π_DEC` target statement. -/
def piDECKnowledgeStatement (ctx : ProtocolTargetContext) : Prop :=
  ∃ deltaInv : Coeffs,
    mulRq ctx.invDelta deltaInv = oneRq ∧
    ceRelaxedRelation ctx ∧
    SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive `Π_DEC` statement from weak relation and invertibility boundary. -/
theorem piDEC_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiDECAssumptions ctx)
  (hAccepted : ∃ tr : SumCheckTranscript,
      SumCheckAccepted (sumcheckInstanceOfContext ctx) tr) :
  piDECKnowledgeStatement ctx := by
  have hWeak : piRLCWeakStatement ctx :=
    piRLCWeak_of_assumptions h.weak hAccepted
  have hWin : invertibilityWindowProp Goldilocks.halfQ ctx.invDelta :=
    h.weak.strong.relations.target.arithmetic.invertibilityWindow
  rcases invertibleRq_of_lowNormAssumption h.lowNormInvertibilityBoundary hWin with ⟨deltaInv, hMul⟩
  exact ⟨deltaInv, hMul, hWeak.1, hWeak.2⟩

end SuperNeo
