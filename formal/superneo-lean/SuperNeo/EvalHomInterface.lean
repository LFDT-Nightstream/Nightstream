import SuperNeo.EvalHom

/-!
Contract interface for `SuperNeo.EvalHom`.

Spec: `specs/EvalHom.spec.md`

Paper anchors:
- Theorem 5 (Evaluation Homomorphism), Section 5, lines 390-400.
-/

namespace SuperNeo

namespace EvalHomInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `evalBarMzAt`. -/
abbrev evalBarMzAt := SuperNeo.evalBarMzAt

/-- [Role: Theorem-Target] Curated re-export of `evalHom2Prop`. -/
abbrev evalHom2Prop := SuperNeo.evalHom2Prop

/-- [Role: Theorem-Target] Curated re-export of `evalHom2`. -/
abbrev evalHom2 := SuperNeo.evalHom2

/-! ## Key Theorems -/

/-- [Role: Theorem-Target] Curated theorem surface `evalHom2_sound`. -/
abbrev evalHom2_sound := @SuperNeo.evalHom2_sound

/-- [Role: Theorem-Target] Curated theorem surface `evalHom2_complete`. -/
abbrev evalHom2_complete := @SuperNeo.evalHom2_complete

/-- [Role: Theorem-Target] Curated theorem surface `evalHom2_iff_prop`. -/
abbrev evalHom2_iff_prop := @SuperNeo.evalHom2_iff_prop

/-! ## Boundary Surfaces -/

/-- [Role: Theorem-Target] Theorem-facing eval-hom boundary surface. -/
abbrev evalHomAssumption := SuperNeo.evalHomAssumption

/-- [Role: Theorem-Target] Check-facing eval-hom boundary surface. -/
abbrev evalHomCheckAssumption := SuperNeo.evalHomCheckAssumption

/-- [Role: Theorem-Target] Native closure of theorem-facing eval-hom boundary. -/
theorem evalHomAssumption_constructive
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F} :
  evalHomAssumption bar m r ρ1 ρ2 :=
  SuperNeo.evalHomAssumption_constructive

/-- [Role: Theorem-Target] Backward-compatible alias for constructive eval-hom closure. -/
theorem evalHomAssumption_native
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F} :
  evalHomAssumption bar m r ρ1 ρ2 :=
  SuperNeo.evalHomAssumption_native

/-- [Role: Theorem-Target] Conversion from check-facing to theorem-facing eval-hom boundary. -/
theorem evalHomAssumption_of_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  (hCheck : evalHomCheckAssumption bar m r ρ1 ρ2) :
  evalHomAssumption bar m r ρ1 ρ2 :=
  SuperNeo.evalHomAssumption_of_checkAssumption hCheck

/-- [Role: Theorem-Target] Conversion from theorem-facing to check-facing eval-hom boundary. -/
theorem evalHomCheckAssumption_of_assumption
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  (hAssm : evalHomAssumption bar m r ρ1 ρ2) :
  evalHomCheckAssumption bar m r ρ1 ρ2 :=
  SuperNeo.evalHomCheckAssumption_of_assumption hAssm

/-- [Role: Theorem-Target] Equivalence between theorem and check eval-hom boundaries. -/
theorem evalHomAssumption_iff_checkAssumption
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F} :
  evalHomAssumption bar m r ρ1 ρ2 ↔
    evalHomCheckAssumption bar m r ρ1 ρ2 :=
  SuperNeo.evalHomAssumption_iff_checkAssumption

/-- [Role: Theorem-Target] Theorem-native eval-hom boundary constructor from eval-link + module-hom. -/
theorem evalHomAssumption_of_evalLink_and_moduleAssumptions
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  {hVec : VecModuleHom} {hScal : ScalarModuleHom}
  (hEvalLink : evalLinkAssumption bar m)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal) :
  evalHomAssumption bar m r ρ1 ρ2 :=
  SuperNeo.evalHomAssumption_of_evalLink_and_moduleAssumptions hEvalLink hVecAssm hScalAssm

/-- [Role: Theorem-Target] Theorem-native eval-hom boundary constructor from P10 + module-hom. -/
theorem evalHomAssumption_of_thm3_and_moduleAssumptions
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  {hVec : VecModuleHom} {hScal : ScalarModuleHom}
  (hThm3 : thm3CoreAssumption bar)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal) :
  evalHomAssumption bar m r ρ1 ρ2 :=
  SuperNeo.evalHomAssumption_of_thm3_and_moduleAssumptions hThm3 hVecAssm hScalAssm

/-- [Role: Theorem-Target] Theorem-native eval-hom boundary constructor from `P10` + module-hom. -/
theorem evalHomAssumption_of_p10_and_moduleAssumptions
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  {hVec : VecModuleHom} {hScal : ScalarModuleHom}
  (hThm3 : thm3CoreAssumption bar)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal) :
  evalHomAssumption bar m r ρ1 ρ2 :=
  SuperNeo.evalHomAssumption_of_p10_and_moduleAssumptions hThm3 hVecAssm hScalAssm

/-- [Role: Theorem-Target] Theorem-native eval-hom boundary constructor from `(P10 + P11)` + module-hom. -/
theorem evalHomAssumption_of_p10_p11_and_moduleAssumptions
  {bar : Array (Array F)} {m : Array (Array F)}
  {r : Array F} {ρ1 ρ2 : F}
  {hVec : VecModuleHom} {hScal : ScalarModuleHom}
  (hThm3 : thm3CoreAssumption bar)
  (hLift : barLiftLinearityAssumption bar)
  (hVecAssm : vecModuleAssumption hVec)
  (hScalAssm : scalarModuleAssumption hScal) :
  evalHomAssumption bar m r ρ1 ρ2 :=
  SuperNeo.evalHomAssumption_of_p10_p11_and_moduleAssumptions hThm3 hLift hVecAssm hScalAssm

end EvalHomInterface

end SuperNeo
