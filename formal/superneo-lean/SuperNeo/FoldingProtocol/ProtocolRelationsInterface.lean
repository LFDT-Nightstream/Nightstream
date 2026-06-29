import SuperNeo.FoldingProtocol.ProtocolRelations

/-!
Contract interface for `SuperNeo.ProtocolRelations`.

Spec: ./formal/superneo-lean/specs/ProtocolRelations.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Definition 12 (Norm-bounded CCS), Section 7.1, lines 457-459.
- Definition 13 (Norm-bounded CCS Evaluation Relation), Section 7.1, lines 461-465.
- Section 7.1 (Relations), lines 449-465.

Theorem-valued surfaces are restated with explicit statements so that drift in
the implementation fails this module instead of silently changing the contract.
-/

namespace SuperNeo

namespace ProtocolRelationsInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `sumcheckInstanceOfContext`. -/
abbrev sumcheckInstanceOfContext := SuperNeo.sumcheckInstanceOfContext

/-- [Role: Theorem-Target] Curated re-export of `sumcheckFullFieldDenominatorAlignment`. -/
abbrev sumcheckFullFieldDenominatorAlignment :=
  SuperNeo.sumcheckFullFieldDenominatorAlignment

/-- [Role: Boundary] Named setup-side boundary for the active Goldilocks/full-field Lund route. -/
abbrev GoldilocksFullFieldLundBoundary :=
  SuperNeo.GoldilocksFullFieldLundBoundary

/-- [Role: Theorem-Target] Canonical Lund boundary constructor from `cset` cardinality. -/
def GoldilocksFullFieldLundBoundary_ofCsetCardinality
  {ctx : ProtocolTargetContext}
  (hCard : ctx.cset.size = Goldilocks.q) :
  SuperNeo.GoldilocksFullFieldLundBoundary ctx :=
  SuperNeo.GoldilocksFullFieldLundBoundary.ofCsetCardinality hCard

/-- [Role: Boundary] Accepted SumCheck transition witness for one protocol context. -/
abbrev SumCheckTransitionWitness := SuperNeo.SumCheckTransitionWitness

/-- [Role: Theorem-Target] An accepted transcript exists under a transition witness. -/
theorem SumCheckTransitionWitness_accepted_exists
  {ctx : ProtocolTargetContext}
  (h : SuperNeo.SumCheckTransitionWitness ctx) :
  ∃ tr : SumCheckTranscript,
    SumCheckAccepted (SuperNeo.sumcheckInstanceOfContext ctx) tr :=
  h.accepted_exists

/-- [Role: Definitional] Compact CCS relation predicate. -/
abbrev ccsRelation := SuperNeo.ccsRelation

/-- [Role: Definitional] Compact CE relation predicate. -/
abbrev ceRelation := SuperNeo.ceRelation

/-- [Role: Definitional] Compact relaxed CE relation predicate. -/
abbrev ceRelaxedRelation := SuperNeo.ceRelaxedRelation

/-! ## Section 7.1 Theorem-Native Owner -/

/-- [Role: Boundary] Theorem-native Section 7.1 owner: Definition-14 parameters, CCS/CE statement-witness pairs, sharing facts, and two-way relation bridges. -/
abbrev ProtocolSection71TheoremInstance :=
  SuperNeo.ProtocolSection71TheoremInstance

/-- [Role: Theorem-Target] One theorem-native Section 7.1 instance yields the compact CCS relation. -/
theorem ProtocolSection71TheoremInstance_ccsRelation
  {ctx : ProtocolTargetContext}
  (h : SuperNeo.ProtocolSection71TheoremInstance ctx) :
  SuperNeo.ccsRelation ctx :=
  h.ccsRelation

/-- [Role: Theorem-Target] One theorem-native Section 7.1 instance yields the compact CE relation. -/
theorem ProtocolSection71TheoremInstance_ceRelation
  {ctx : ProtocolTargetContext}
  (h : SuperNeo.ProtocolSection71TheoremInstance ctx) :
  SuperNeo.ceRelation ctx :=
  h.ceRelation

/-! ## Relation Bridges -/

/-- [Role: Theorem-Target] Compact CCS relation from the protocol-target proposition. -/
theorem ccsRelation_of_protocolTargetProp
  {ctx : ProtocolTargetContext}
  (hTarget : protocolTargetProp ctx) :
  SuperNeo.ccsRelation ctx :=
  SuperNeo.ccsRelation_of_protocolTargetProp hTarget

/-- [Role: Theorem-Target] The compact CCS relation is the protocol-target proposition. -/
theorem ccsRelation_iff_protocolTargetProp
  {ctx : ProtocolTargetContext} :
  SuperNeo.ccsRelation ctx ↔ protocolTargetProp ctx :=
  SuperNeo.ccsRelation_iff_protocolTargetProp

/-- [Role: Theorem-Target] CE is exactly CCS plus an accepted SumCheck transcript witness. -/
theorem ceRelation_iff
  {ctx : ProtocolTargetContext} :
  SuperNeo.ceRelation ctx ↔
    SuperNeo.ccsRelation ctx ∧
      ∃ tr : SumCheckTranscript,
        SumCheckAccepted (SuperNeo.sumcheckInstanceOfContext ctx) tr :=
  SuperNeo.ceRelation_iff

/-- [Role: Theorem-Target] Relaxed CE is definitionally CCS. -/
theorem ceRelaxedRelation_iff
  {ctx : ProtocolTargetContext} :
  SuperNeo.ceRelaxedRelation ctx ↔ SuperNeo.ccsRelation ctx :=
  SuperNeo.ceRelaxedRelation_iff

/-- [Role: Theorem-Target] CE relation from the CCS relation plus an accepted transition witness. -/
theorem ceRelation_of_ccsRelation
  {ctx : ProtocolTargetContext}
  (hCCS : SuperNeo.ccsRelation ctx)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx) :
  SuperNeo.ceRelation ctx :=
  SuperNeo.ceRelation_of_ccsRelation hCCS hWitness

/-- [Role: Theorem-Target] CE relation from the CCS relation plus claim truth. -/
theorem ceRelation_of_ccsRelation_claimTrue
  {ctx : ProtocolTargetContext}
  (hCCS : SuperNeo.ccsRelation ctx)
  (hClaimTrue : SumCheckClaimTrue (SuperNeo.sumcheckInstanceOfContext ctx)) :
  SuperNeo.ceRelation ctx :=
  SuperNeo.ceRelation_of_ccsRelation_claimTrue hCCS hClaimTrue

/-- [Role: Theorem-Target] Soundness lift: any CE witness yields SumCheck claim truth. -/
theorem ceClaimTrue_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : SuperNeo.ceRelation ctx) :
  SumCheckClaimTrue (SuperNeo.sumcheckInstanceOfContext ctx) :=
  SuperNeo.ceClaimTrue_of_ce hCE

/-- [Role: Theorem-Target] CE implies relaxed CE. -/
theorem ceRelaxedRelation_of_ce
  {ctx : ProtocolTargetContext}
  (hCE : SuperNeo.ceRelation ctx) :
  SuperNeo.ceRelaxedRelation ctx :=
  SuperNeo.ceRelaxedRelation_of_ce hCE

end ProtocolRelationsInterface

end SuperNeo
