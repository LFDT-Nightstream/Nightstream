import SuperNeo.EvalLink

/-!
Contract interface for `SuperNeo.EvalLink`.

Spec: `specs/EvalLink.spec.md`

Paper anchors:
- Remark 2 (Matrix-vector Product Evaluation), Section 5, lines 388-389.
-/

namespace SuperNeo

namespace EvalLinkInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `evalLinkIdentity`. -/
abbrev evalLinkIdentity := SuperNeo.evalLinkIdentity

/-- [Role: Theorem-Target] Curated re-export of `evalLinkIdentityProp`. -/
abbrev evalLinkIdentityProp := SuperNeo.evalLinkIdentityProp

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `evalLinkIdentity_sound`. -/
abbrev evalLinkIdentity_sound := @SuperNeo.evalLinkIdentity_sound

/-- [Role: Theorem-Target] Curated theorem surface `evalLinkIdentity_complete`. -/
abbrev evalLinkIdentity_complete := @SuperNeo.evalLinkIdentity_complete

/-- [Role: Theorem-Target] Curated theorem surface `evalLinkIdentity_iff_prop`. -/
abbrev evalLinkIdentity_iff_prop := @SuperNeo.evalLinkIdentity_iff_prop

/-! ## Boundary Surfaces -/

/-- [Role: Theorem-Target] Theorem-facing eval-link boundary surface. -/
abbrev evalLinkAssumption := SuperNeo.evalLinkAssumption

/-- [Role: Theorem-Target] Check-facing eval-link boundary surface. -/
abbrev evalLinkCheckAssumption := SuperNeo.evalLinkCheckAssumption

/-- [Role: Theorem-Target] Conversion from check-facing to theorem-facing eval-link boundary. -/
theorem evalLinkAssumption_of_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hCheck : evalLinkCheckAssumption bar m) :
  evalLinkAssumption bar m :=
  SuperNeo.evalLinkAssumption_of_checkAssumption hCheck

/-- [Role: Theorem-Target] Conversion from theorem-facing to check-facing eval-link boundary. -/
theorem evalLinkCheckAssumption_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hAssm : evalLinkAssumption bar m) :
  evalLinkCheckAssumption bar m :=
  SuperNeo.evalLinkCheckAssumption_of_assumption hAssm

/-- [Role: Theorem-Target] Equivalence between theorem and check eval-link boundaries. -/
theorem evalLinkAssumption_iff_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)} :
  evalLinkAssumption bar m ↔ evalLinkCheckAssumption bar m :=
  SuperNeo.evalLinkAssumption_iff_checkAssumption

/-- [Role: Theorem-Target] Theorem-native eval-link boundary constructor from P12. -/
theorem evalLinkAssumption_of_matrixTransformAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hMatrix : matrixTransformAssumption bar m) :
  evalLinkAssumption bar m :=
  SuperNeo.evalLinkAssumption_of_matrixTransformAssumption hMatrix

/-- [Role: Theorem-Target] Theorem-native eval-link boundary constructor from P10 via P12. -/
theorem evalLinkAssumption_of_thm3CoreAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar) :
  evalLinkAssumption bar m :=
  SuperNeo.evalLinkAssumption_of_thm3CoreAssumption hThm3

/-- [Role: Theorem-Target] Theorem-native eval-link boundary constructor from `P10` only. -/
theorem evalLinkAssumption_of_p10
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar) :
  evalLinkAssumption bar m :=
  SuperNeo.evalLinkAssumption_of_p10 hThm3

/-- [Role: Theorem-Target] Theorem-native eval-link boundary constructor from `(P10 + P11)`. -/
theorem evalLinkAssumption_of_p10_p11
  {bar : Array (Array F)} {m : Array (Array F)}
  (hThm3 : thm3CoreAssumption bar)
  (hLift : barLiftLinearityAssumption bar) :
  evalLinkAssumption bar m :=
  SuperNeo.evalLinkAssumption_of_p10_p11 hThm3 hLift

end EvalLinkInterface

end SuperNeo
