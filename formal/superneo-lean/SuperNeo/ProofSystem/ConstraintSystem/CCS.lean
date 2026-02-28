import SuperNeo.ProtocolBase

/-!
Paper-facing constraint-system relations for SuperNeo.

This module intentionally exposes compact names for the protocol relations so
callers can import a single constraint-system surface (ArkLib-style) instead of
threading internal milestone modules.
-/

namespace SuperNeo.ProofSystem.ConstraintSystem

abbrev Context := SuperNeo.ProtocolCtx
abbrev Claim := SuperNeo.CEClaim
abbrev Witness := SuperNeo.CEWitness

/-- Paper-facing CCS relation. -/
def CCS (ctx : Context) (claim : Claim) : Prop :=
  SuperNeo.CCSRelation ctx claim

/-- Paper-facing CE relation (with witness norm bound). -/
def CE (ctx : Context) (claim : Claim) (witness : Witness) : Prop :=
  SuperNeo.CERelation ctx claim witness

/-- Relaxed CE relation used in weak-reduction paths. -/
def CERelaxed (ctx : Context) (claim : Claim) (witness : Witness) : Prop :=
  SuperNeo.CERelationRelaxed ctx claim witness

theorem ce_of_ceValid
  {ctx : Context} {claim : Claim} {witness : Witness}
  (h : SuperNeo.CEValid ctx claim witness) :
  CE ctx claim witness :=
  SuperNeo.ceRelation_of_ceValid h

theorem ccs_of_ceValid
  {ctx : Context} {claim : Claim} {witness : Witness}
  (h : SuperNeo.CEValid ctx claim witness) :
  CCS ctx claim :=
  SuperNeo.ccsRelation_of_ceValid h

theorem ceValid_of_relations
  {ctx : Context} {claim : Claim} {witness : Witness}
  (hCCS : CCS ctx claim)
  (hCE : CE ctx claim witness) :
  SuperNeo.CEValid ctx claim witness :=
  SuperNeo.ceValid_of_relations hCCS hCE

end SuperNeo.ProofSystem.ConstraintSystem
