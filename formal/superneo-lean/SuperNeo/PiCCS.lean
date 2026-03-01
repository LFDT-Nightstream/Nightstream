import SuperNeo.ProtocolRelations

/-!
Strong interactive-reduction step `Π_CCS`.
-/

namespace SuperNeo

/-- Assumptions consumed by the `Π_CCS` reduction step. -/
structure PiCCSAssumptions (ctx : ProtocolTargetContext) where
  relations : ProtocolRelationsAssumptions ctx

/-- Strong `Π_CCS` target statement. -/
def piCCSStrongStatement (ctx : ProtocolTargetContext) : Prop :=
  ceRelation ctx ∧
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive strong `Π_CCS` statement from relation assumptions and transcript witness. -/
theorem piCCSStrong_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : PiCCSAssumptions ctx)
  (hAccepted : ∃ tr : SumCheckTranscript,
      SumCheckAccepted (sumcheckInstanceOfContext ctx) tr) :
  piCCSStrongStatement ctx := by
  have hCE : ceRelation ctx :=
    ceRelation_of_assumptions h.relations hAccepted
  exact ⟨hCE, ceClaimTrue_of_ce h.relations hCE⟩

end SuperNeo
