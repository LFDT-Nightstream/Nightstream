import SuperNeo.FoldingProtocol.PiCCS

/-!
Weak interactive-reduction step `Π_RLC`.
-/

namespace SuperNeo

/-- Weak `Π_RLC` target statement. -/
def piRLCWeakStatement (ctx : ProtocolTargetContext) : Prop :=
  ceRelaxedRelation ctx ∧
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx)

/-- Derive weak `Π_RLC` statement directly from the CE relation. -/
theorem piRLCWeak_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  piRLCWeakStatement ctx := by
  exact ⟨ceRelaxedRelation_of_ce hCE, ceClaimTrue_of_ce hCE⟩

/-- Derive weak `Π_RLC` statement from relation assumptions and transcript witness. -/
theorem piRLCWeak_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolTargetAssumptions ctx)
  (hWitness : SumCheckTransitionWitness ctx) :
  piRLCWeakStatement ctx := by
  have hCCS : ccsRelation ctx := protocolTargetProp_of_assumptions h
  exact piRLCWeak_of_ce (ceRelation_of_ccsRelation hCCS hWitness)

end SuperNeo
