import SuperNeo.FoldingProtocol.PiCCS

/-!
Contract interface for `SuperNeo.PiCCS`.

Spec: ./formal/superneo-lean/specs/PiCCS.spec.md

Paper anchors (Source: ./formal/superneo-lean/SuperNeo.pdf.md):
- Section 7.3 (Π_CCS), lines 481-548.
- Lemma 3 (Π_CCS is strong), lines 545-546.
-/

namespace SuperNeo

namespace PiCCSInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `piCCSStrongStatement`. -/
abbrev piCCSStrongStatement := SuperNeo.piCCSStrongStatement

/-- [Role: Theorem-Target] Strong `Π_CCS` follows directly from the CE relation. -/
theorem piCCSStrong_of_ce
  {ctx : ProtocolTargetContext} :
  ceRelation ctx →
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_ce

/-! ## Boundary Surfaces -/

/-- [Role: Theorem-Target] Canonical strong `Π_CCS` constructor from assumptions and witness. -/
theorem piCCSStrong_of_assumptions
  {ctx : ProtocolTargetContext} :
  ProtocolTargetAssumptions ctx →
  SumCheckTransitionWitness ctx →
  piCCSStrongStatement ctx :=
  SuperNeo.piCCSStrong_of_assumptions

end PiCCSInterface

end SuperNeo
