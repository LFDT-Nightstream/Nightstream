import SuperNeo.ProtocolMathTarget

/-!
Contract interface for `SuperNeo.ProtocolMathTarget`.

Spec: `./formal/superneo-lean/specs/ProtocolMathTarget.spec.md`

Paper anchors (Source: `./formal/superneo-lean/SuperNeo.pdf.md`):
- Section 7 (Neo's folding scheme for CCS), lines 447–467: Relations (Definitions 11–13)
- Section 7.2–7.5, lines 467–596: Folding scheme via interactive reductions (Π_CCS, Π_RLC, Π_DEC)
-/

namespace SuperNeo

namespace ProtocolMathTargetInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `protocolMathTargetProp`. -/
abbrev protocolMathTargetProp := SuperNeo.protocolMathTargetProp

/-- [Role: Theorem-Target] Curated re-export of `protocolMathTargetWithThm3Prop`. -/
abbrev protocolMathTargetWithThm3Prop := SuperNeo.protocolMathTargetWithThm3Prop

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `protocolMathTargetProp_of_arithmeticBundle`. -/
abbrev protocolMathTargetProp_of_arithmeticBundle := SuperNeo.protocolMathTargetProp_of_arithmeticBundle

/-- [Role: Theorem-Target] Curated theorem surface `protocolMathTargetWithThm3Prop_of_p10_arithmeticBundle`. -/
abbrev protocolMathTargetWithThm3Prop_of_p10_arithmeticBundle := SuperNeo.protocolMathTargetWithThm3Prop_of_p10_arithmeticBundle

/-- [Role: Theorem-Target] Curated theorem surface `protocolMathTargetWithThm3Prop_of_thm3_preconditions`. -/
abbrev protocolMathTargetWithThm3Prop_of_thm3_preconditions := SuperNeo.protocolMathTargetWithThm3Prop_of_thm3_preconditions

/-- [Role: Theorem-Target] Curated theorem surface `protocolMathTargetProp_of_checks`. -/
abbrev protocolMathTargetProp_of_checks := SuperNeo.protocolMathTargetProp_of_checks

/-- [Role: Theorem-Target] Curated theorem surface `protocolMathTargetWithThm3Prop_of_checks`. -/
abbrev protocolMathTargetWithThm3Prop_of_checks := SuperNeo.protocolMathTargetWithThm3Prop_of_checks

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `protocolMathTargetWithThm3Prop_of_thm3_assumption` requiring closure. -/
abbrev protocolMathTargetWithThm3Prop_of_thm3_assumption := SuperNeo.protocolMathTargetWithThm3Prop_of_thm3_assumption

end ProtocolMathTargetInterface

end SuperNeo
