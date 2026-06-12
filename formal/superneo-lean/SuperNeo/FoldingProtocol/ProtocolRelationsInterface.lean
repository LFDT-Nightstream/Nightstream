import SuperNeo.FoldingProtocol.ProtocolRelations

/-!
Contract interface for `SuperNeo.ProtocolRelations`.

Spec: ./formal/superneo-lean/specs/ProtocolRelations.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Definition 12 (Norm-bounded CCS), Section 7.1, lines 457-459.
- Definition 13 (Norm-bounded CCS Evaluation Relation), Section 7.1, lines 461-465.
- Section 7.1 (Relations), lines 449-465.
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
abbrev GoldilocksFullFieldLundBoundary_ofCsetCardinality
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.GoldilocksFullFieldLundBoundary.ofCsetCardinality (ctx := ctx)

/-- [Role: Boundary] Accepted SumCheck transition witness for one protocol context. -/
abbrev SumCheckTransitionWitness := SuperNeo.SumCheckTransitionWitness

/-- [Role: Theorem-Target] An accepted transcript exists under a transition witness. -/
abbrev SumCheckTransitionWitness_accepted_exists
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.SumCheckTransitionWitness.accepted_exists (ctx := ctx)

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
abbrev ProtocolSection71TheoremInstance_ccsRelation
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.ProtocolSection71TheoremInstance.ccsRelation (ctx := ctx)

/-- [Role: Theorem-Target] One theorem-native Section 7.1 instance yields the compact CE relation. -/
abbrev ProtocolSection71TheoremInstance_ceRelation
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.ProtocolSection71TheoremInstance.ceRelation (ctx := ctx)

/-! ## Relation Bridges -/

/-- [Role: Theorem-Target] Compact CCS relation from the protocol-target proposition. -/
abbrev ccsRelation_of_protocolTargetProp
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.ccsRelation_of_protocolTargetProp (ctx := ctx)

/-- [Role: Theorem-Target] The compact CCS relation is the protocol-target proposition. -/
abbrev ccsRelation_iff_protocolTargetProp
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.ccsRelation_iff_protocolTargetProp (ctx := ctx)

/-- [Role: Theorem-Target] Unfolding lemma for the CE relation. -/
abbrev ceRelation_iff
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.ceRelation_iff (ctx := ctx)

/-- [Role: Theorem-Target] Unfolding lemma for the relaxed CE relation. -/
abbrev ceRelaxedRelation_iff
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.ceRelaxedRelation_iff (ctx := ctx)

/-- [Role: Theorem-Target] CE relation from the CCS relation plus an accepted transition witness. -/
abbrev ceRelation_of_ccsRelation
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.ceRelation_of_ccsRelation (ctx := ctx)

/-- [Role: Theorem-Target] CE relation from the CCS relation plus claim truth. -/
abbrev ceRelation_of_ccsRelation_claimTrue
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.ceRelation_of_ccsRelation_claimTrue (ctx := ctx)

/-- [Role: Theorem-Target] Soundness lift: any CE witness yields SumCheck claim truth. -/
abbrev ceClaimTrue_of_ce
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.ceClaimTrue_of_ce (ctx := ctx)

/-- [Role: Theorem-Target] CE implies relaxed CE. -/
abbrev ceRelaxedRelation_of_ce
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.ceRelaxedRelation_of_ce (ctx := ctx)

end ProtocolRelationsInterface

end SuperNeo
