import SuperNeo.ProtocolReduction

/-!
Contract interface for `SuperNeo.ProtocolReduction`.

Spec: `./formal/superneo-lean/specs/ProtocolReduction.spec.md`

Paper anchors (Source: `./formal/superneo-lean/SuperNeo.pdf.md`):
- Section 7 (Neo's folding scheme for CCS), lines 447–596: Relations, reduction steps (Π_CCS, Π_RLC, Π_DEC)
- Section 7.2–7.5, lines 467–596: Folding scheme via interactive reductions
-/

namespace SuperNeo

namespace ProtocolReductionInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `p10ForClaim`. -/
abbrev p10ForClaim := SuperNeo.p10ForClaim

/-- [Role: Theorem-Target] Curated re-export of `arithmeticBundleForClaim`. -/
abbrev arithmeticBundleForClaim := SuperNeo.arithmeticBundleForClaim

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `superneoMathProtocolSkeleton_of_props`. -/
abbrev superneoMathProtocolSkeleton_of_props := SuperNeo.superneoMathProtocolSkeleton_of_props

/-- [Role: Theorem-Target] Curated theorem surface `superneoMathProtocolSkeleton_of_checks`. -/
abbrev superneoMathProtocolSkeleton_of_checks := SuperNeo.superneoMathProtocolSkeleton_of_checks

/-- [Role: Theorem-Target] Curated theorem surface `smoke_checks_imply_props`. -/
abbrev smoke_checks_imply_props := SuperNeo.smoke_checks_imply_props

/-- [Role: Theorem-Target] Curated theorem surface `smoke_props_imply_check_subset`. -/
abbrev smoke_props_imply_check_subset := SuperNeo.smoke_props_imply_check_subset

/-- [Role: Theorem-Target] Curated theorem surface `smoke_protocolMathTarget_compose`. -/
abbrev smoke_protocolMathTarget_compose := SuperNeo.smoke_protocolMathTarget_compose

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `superneoMathProtocolSkeleton_of_thm3_assumption` requiring closure. -/
abbrev superneoMathProtocolSkeleton_of_thm3_assumption := SuperNeo.superneoMathProtocolSkeleton_of_thm3_assumption

end ProtocolReductionInterface

end SuperNeo
