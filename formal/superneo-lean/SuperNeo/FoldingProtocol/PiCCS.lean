import SuperNeo.FoldingProtocol.ProtocolRelations

/-!
Strong interactive-reduction step `Π_CCS`.
-/

namespace SuperNeo

/-- Strong `Π_CCS` target statement. -/
def piCCSStrongStatement (ctx : ProtocolTargetContext) : Prop :=
  ceRelation ctx ∧
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive the strong `Π_CCS` statement directly from the CE relation. -/
theorem piCCSStrong_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  piCCSStrongStatement ctx := by
  exact ⟨hCE, ceClaimTrue_of_ce hCE⟩

/-- Derive strong `Π_CCS` statement from relation assumptions and transcript witness. -/
theorem piCCSStrong_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piCCSStrongStatement ctx := by
  have hCCS : ccsRelation ctx := protocolTargetProp_of_assumptions h
  exact piCCSStrong_of_ce (ceRelation_of_ccsRelation hCCS hWitness)

end SuperNeo
