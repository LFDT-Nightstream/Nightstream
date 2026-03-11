import SuperNeo.Goldilocks

/-!
Contract interface for `SuperNeo.Goldilocks`.

Spec: `specs/Goldilocks.spec.md`

Paper anchors:
- Definition 1 (Fields, Rings, Dimensions), Section 4, lines 275-282: field modulus `q`.
- Appendix B.2, lines 709-727: concrete `q = 2^64 - 2^32 + 1`.
-/

namespace SuperNeo

namespace GoldilocksInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `Goldilocks.q`. -/
abbrev Goldilocks_q := SuperNeo.Goldilocks.q

/-- [Role: Theorem-Target] Curated re-export of `Goldilocks.halfQ`. -/
abbrev Goldilocks_halfQ := SuperNeo.Goldilocks.halfQ

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `Goldilocks.q_pos`. -/
abbrev Goldilocks_q_pos := SuperNeo.Goldilocks.q_pos

/-- [Role: Theorem-Target] Curated theorem surface `Goldilocks.q_ne_zero`. -/
abbrev Goldilocks_q_ne_zero := SuperNeo.Goldilocks.q_ne_zero

/-- [Role: Theorem-Target] Curated theorem surface `Goldilocks.q_gt_one`. -/
abbrev Goldilocks_q_gt_one := SuperNeo.Goldilocks.q_gt_one

/-- [Role: Theorem-Target] Curated theorem surface `Goldilocks.halfQ_lt_q`. -/
abbrev Goldilocks_halfQ_lt_q := SuperNeo.Goldilocks.halfQ_lt_q

/-- [Role: Theorem-Target] Curated theorem surface `Goldilocks.halfQ_le_q`. -/
abbrev Goldilocks_halfQ_le_q := SuperNeo.Goldilocks.halfQ_le_q

/-- [Role: Theorem-Target] Curated theorem surface `Goldilocks.one_le_halfQ`. -/
abbrev Goldilocks_one_le_halfQ := SuperNeo.Goldilocks.one_le_halfQ

end GoldilocksInterface

end SuperNeo
