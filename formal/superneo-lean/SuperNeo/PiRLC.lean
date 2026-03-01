import SuperNeo.PiCCS

/-!
Weak interactive-reduction step `Π_RLC`.
-/

namespace SuperNeo

/-- Assumptions consumed by the `Π_RLC` reduction step. -/
structure PiRLCAssumptions (ctx : ProtocolTargetContext) where
  strong : PiCCSAssumptions ctx

/-- Weak `Π_RLC` target statement. -/
def piRLCWeakStatement (ctx : ProtocolTargetContext) : Prop :=
  ceRelaxedRelation ctx ∧
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive weak `Π_RLC` statement from strong `Π_CCS`. -/
theorem piRLCWeak_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiRLCAssumptions ctx)
  (hAccepted : ∃ tr : SumCheckTranscript,
      SumCheckAccepted (sumcheckInstanceOfContext ctx) tr) :
  piRLCWeakStatement ctx := by
  have hStrong : piCCSStrongStatement ctx :=
    piCCSStrong_of_assumptions h.strong hAccepted
  exact ⟨ceRelaxedRelation_of_ce hStrong.1, hStrong.2⟩

end SuperNeo
