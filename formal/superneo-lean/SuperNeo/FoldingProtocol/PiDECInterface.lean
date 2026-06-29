import SuperNeo.FoldingProtocol.PiDEC

/-!
Contract interface for `SuperNeo.PiDEC`.

Spec: ./formal/superneo-lean/specs/PiDEC.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Section 7.5 (Π_DEC), lines 585-593.
- Theorem 7 (Π_DEC is reduction of knowledge), lines 594-596.
-/

namespace SuperNeo

namespace PiDECInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `piDECKnowledgeStatement`. -/
abbrev piDECKnowledgeStatement := SuperNeo.piDECKnowledgeStatement

/-- [Role: Theorem-Target] Derive `Π_DEC` directly from the weak `Π_RLC` statement. -/
theorem piDEC_of_weak
  {ctx : ProtocolTargetContext} :
  piRLCWeakStatement ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_weak

/-- [Role: Theorem-Target] Derive `Π_DEC` directly from the CE relation. -/
theorem piDEC_of_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_ce

/-! ## Boundary Surfaces -/

/-- [Role: Theorem-Target] Curated theorem surface `piDEC_of_assumptions`. -/
theorem piDEC_of_assumptions
  {ctx : ProtocolTargetContext} :
  ProtocolTargetAssumptions ctx →
  SumCheckTransitionWitness ctx →
  piDECKnowledgeStatement ctx :=
  SuperNeo.piDEC_of_assumptions

end PiDECInterface

end SuperNeo
