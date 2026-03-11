import SuperNeo.Interp

/-!
Contract interface for `SuperNeo.Interp`.

Spec: `specs/Interp.spec.md`

Paper anchors:
- Infrastructure module supporting evaluation checks in Sections 7.3-7.4.
-/

namespace SuperNeo

namespace InterpInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Distinct interpolation nodes. -/
abbrev interpolationNodesDistinct := SuperNeo.interpolationNodesDistinct

/-- [Role: Theorem-Target] Polynomial evaluation from coefficient arrays. -/
abbrev polyEval := SuperNeo.polyEval

/-- [Role: Theorem-Target] Constructive interpolation coefficients. -/
abbrev interpolateCoeffs := SuperNeo.interpolateCoeffs

/-- [Role: Theorem-Target] Pointwise interpolation relation on the sample set. -/
abbrev interpolatesOn := SuperNeo.interpolatesOn

/-- [Role: Theorem-Target] Curated re-export of `interpolationProp`. -/
abbrev interpolationProp := SuperNeo.interpolationProp

/-- [Role: Theorem-Target] Curated re-export of `interpolationCase`. -/
abbrev interpolationCase := SuperNeo.interpolationCase

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `interpolationCase_sound`. -/
abbrev interpolationCase_sound := SuperNeo.interpolationCase_sound

/-- [Role: Theorem-Target] Curated theorem surface `interpolationCase_complete`. -/
abbrev interpolationCase_complete := SuperNeo.interpolationCase_complete

/-- [Role: Theorem-Target] Curated theorem surface `interpolationCase_eq_true_iff`. -/
abbrev interpolationCase_eq_true_iff := SuperNeo.interpolationCase_eq_true_iff

/-- [Role: Theorem-Target] Curated theorem surface `interpolationProp_sizes`. -/
abbrev interpolationProp_sizes := SuperNeo.interpolationProp_sizes

/-- [Role: Theorem-Target] Curated theorem surface `interpolationProp_eval_eq`. -/
abbrev interpolationProp_eval_eq := SuperNeo.interpolationProp_eval_eq

/-- [Role: Theorem-Target] Constructive interpolation is correct on the sample nodes. -/
abbrev interpolateCoeffs_interpolatesOn := SuperNeo.interpolateCoeffs_interpolatesOn

/-- [Role: Theorem-Target] Constructive interpolation coefficients are unique. -/
abbrev interpolateCoeffs_unique := SuperNeo.interpolateCoeffs_unique

/-- [Role: Theorem-Target] Constructive interpolation yields the theorem-facing proposition. -/
abbrev interpolateCoeffs_interpolationProp := SuperNeo.interpolateCoeffs_interpolationProp

/-- [Role: Theorem-Target] Curated theorem surface `not_interpolationAssumption`. -/
abbrev not_interpolationAssumption := SuperNeo.not_interpolationAssumption

/-! ## Boundary Surfaces -/

/-- [Role: Legacy Boundary / Refuted] Legacy surface `interpolationAssumption`; do not use it as a real closure target. -/
abbrev interpolationAssumption := SuperNeo.interpolationAssumption

end InterpInterface

end SuperNeo
