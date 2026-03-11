import SuperNeo.CoeffMaps

/-!
Contract interface for `SuperNeo.CoeffMaps`.

Spec: `specs/CoeffMaps.spec.md`

Paper anchors:
- Definition 2, Section 4, lines 284-288: `cf : R_q → F_q^d` and `cf⁻¹ : F_q^d → R_q`.
-/

namespace SuperNeo

namespace CoeffMapsInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `cf`. -/
abbrev cf := SuperNeo.cf

/-- [Role: Theorem-Target] Curated re-export of `cfInv`. -/
abbrev cfInv := SuperNeo.cfInv

/-- [Role: Theorem-Target] Curated re-export of `coeffMapRoundTripProp`. -/
abbrev coeffMapRoundTripProp := SuperNeo.coeffMapRoundTripProp

/-- [Role: Theorem-Target] Curated re-export of `coeffMapRoundTrip`. -/
abbrev coeffMapRoundTrip := SuperNeo.coeffMapRoundTrip

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `cfInv_cf`. -/
abbrev cfInv_cf := SuperNeo.cfInv_cf

/-- [Role: Theorem-Target] Curated theorem surface `cf_cfInv`. -/
abbrev cf_cfInv := SuperNeo.cf_cfInv

/-- [Role: Theorem-Target] Curated theorem surface `cf_size`. -/
abbrev cf_size := SuperNeo.cf_size

/-- [Role: Theorem-Target] Curated theorem surface `cfInv_size`. -/
abbrev cfInv_size := SuperNeo.cfInv_size

/-- [Role: Theorem-Target] Curated theorem surface `ct_cf`. -/
abbrev ct_cf := SuperNeo.ct_cf

/-- [Role: Theorem-Target] Curated theorem surface `ct_cfInv`. -/
abbrev ct_cfInv := SuperNeo.ct_cfInv

/-- [Role: Theorem-Target] Curated theorem surface `coeffMapRoundTrip_complete`. -/
abbrev coeffMapRoundTrip_complete := SuperNeo.coeffMapRoundTrip_complete

/-- [Role: Theorem-Target] Curated theorem surface `coeffMapRoundTrip_eq_true_iff`. -/
abbrev coeffMapRoundTrip_eq_true_iff := SuperNeo.coeffMapRoundTrip_eq_true_iff

/-- [Role: Theorem-Target] Curated theorem surface `ringMulShapeProp_cfInv_iff`. -/
abbrev ringMulShapeProp_cfInv_iff := SuperNeo.ringMulShapeProp_cfInv_iff

end CoeffMapsInterface

end SuperNeo
