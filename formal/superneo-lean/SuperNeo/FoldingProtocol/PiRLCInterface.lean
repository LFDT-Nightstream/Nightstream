import SuperNeo.FoldingProtocol.PiRLC

/-!
Contract interface for `SuperNeo.PiRLC`.

Spec: ./formal/superneo-lean/specs/PiRLC.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Section 7.4 (Π_RLC), lines 549-583.
- Lemma 4 (Π_RLC is weak), lines 582-583.
-/

namespace SuperNeo

namespace PiRLCInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `piRLCWeakStatement`. -/
abbrev piRLCWeakStatement := SuperNeo.piRLCWeakStatement

/-- [Role: Theorem-Target] Weak `Π_RLC` follows directly from the CE relation. -/
theorem piRLCWeak_of_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_ce

/-! ## Boundary Surfaces -/

/-- [Role: Theorem-Target] Curated theorem surface `piRLCWeak_of_assumptions`. -/
theorem piRLCWeak_of_assumptions
  {ctx : ProtocolTargetContext} :
  ProtocolTargetAssumptions ctx →
  SumCheckTransitionWitness ctx →
  piRLCWeakStatement ctx :=
  SuperNeo.piRLCWeak_of_assumptions

end PiRLCInterface

end SuperNeo
