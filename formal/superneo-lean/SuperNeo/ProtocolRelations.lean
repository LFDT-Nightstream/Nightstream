import SuperNeo.ProtocolTarget
import SuperNeo.SumCheck

/-!
CCS/CE relation layer.

This module defines paper-facing relation predicates on top of the protocol
context and ties them to the protocol-target and SumCheck boundaries.
-/

namespace SuperNeo

/-- Build a SumCheck instance from protocol-target context fields. -/
def sumcheckInstanceOfContext (ctx : ProtocolTargetContext) : SumCheckInstance :=
  { rounds := ctx.kSplit
    maxDegree := ctx.m.size
    domainSize := ctx.cset.size
    claimedValue := ct ctx.invDelta }

/-- CCS relation: protocol target holds. -/
def ccsRelation (ctx : ProtocolTargetContext) : Prop :=
  protocolTargetProp ctx

/-- CE relation: CCS relation plus an accepted SumCheck transcript witness. -/
def ceRelation (ctx : ProtocolTargetContext) : Prop :=
  ccsRelation ctx ∧
  ∃ tr : SumCheckTranscript,
    SumCheckAccepted (sumcheckInstanceOfContext ctx) tr

/-- Relaxed CE relation: keep only CCS relation (claim-truth may be deferred). -/
def ceRelaxedRelation (ctx : ProtocolTargetContext) : Prop :=
  ccsRelation ctx

/-- Assumptions needed to derive relation-level statements. -/
structure ProtocolRelationsAssumptions (ctx : ProtocolTargetContext) where
  target : ProtocolTargetAssumptions ctx
  sumcheckSoundness : SumcheckSoundnessAssumption
  sumcheckCompleteness : SumcheckCompletenessAssumption

/-- Derive CCS relation from target assumptions. -/
theorem ccsRelation_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsAssumptions ctx) :
  ccsRelation ctx := by
  exact protocolTargetProp_of_assumptions h.target

/-- Derive CE relation from explicit transcript acceptance witness. -/
theorem ceRelation_of_assumptions
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsAssumptions ctx)
  (hAccepted : ∃ tr : SumCheckTranscript,
      SumCheckAccepted (sumcheckInstanceOfContext ctx) tr) :
  ceRelation ctx := by
  exact ⟨ccsRelation_of_assumptions h, hAccepted⟩

/-- Derive CE relation from claim-truth via SumCheck completeness boundary. -/
theorem ceRelation_of_claimTrue
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsAssumptions ctx)
  (hClaimTrue : SumCheckClaimTrue (sumcheckInstanceOfContext ctx)) :
  ceRelation ctx := by
  refine ceRelation_of_assumptions h ?_
  exact h.sumcheckCompleteness _ hClaimTrue

/-- Soundness lift: any CE witness yields SumCheck claim truth. -/
theorem ceClaimTrue_of_ce
  {ctx : ProtocolTargetContext}
  (h : ProtocolRelationsAssumptions ctx)
  (hCE : ceRelation ctx) :
  SumCheckClaimTrue (sumcheckInstanceOfContext ctx) := by
  rcases hCE.2 with ⟨tr, hAcc⟩
  exact h.sumcheckSoundness _ _ hAcc

/-- CE implies relaxed CE. -/
theorem ceRelaxedRelation_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : ceRelation ctx) :
  ceRelaxedRelation ctx := by
  exact hCE.1

end SuperNeo
