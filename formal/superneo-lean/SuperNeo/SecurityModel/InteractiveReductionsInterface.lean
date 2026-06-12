import SuperNeo.SecurityModel.InteractiveReductions

/-!
Contract interface for `SuperNeo.InteractiveReductions`.

Spec: `specs/InteractiveReductions.spec.md`

Paper anchors:
- Theorem 6 (Strong-Weak Composition), Section 6, lines 438-447.
- Definition 9 (Weak Interactive Reductions), lines 404-416.
- Definition 10 (Strong Interactive Reductions), lines 418-436.
-/

namespace SuperNeo

namespace InteractiveReductionsInterface

/-! ## Core Surfaces -/

/-- [Role: Theorem-Target] Curated re-export of `strongCompositionStatement`. -/
abbrev strongCompositionStatement := SuperNeo.strongCompositionStatement

/-- [Role: Theorem-Target] Curated re-export of `weakCompositionStatement`. -/
abbrev weakCompositionStatement := SuperNeo.weakCompositionStatement

/-! ## Boundary Surfaces -/

/-- [Role: Boundary] Boundary surface `InteractiveReductionAssumptions` requiring closure. -/
abbrev InteractiveReductionAssumptions := SuperNeo.InteractiveReductionAssumptions

/-- [Role: Theorem-Target] Curated theorem surface `strongComposition_of_assumptions`. -/
abbrev strongComposition_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.strongComposition_of_assumptions (ctx := ctx)

/-- [Role: Theorem-Target] Curated theorem surface `weakComposition_of_assumptions`. -/
abbrev weakComposition_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.weakComposition_of_assumptions (ctx := ctx)

/--
[Role: Theorem-Target] Witness-level SumCheck failure-advantage bound from
interactive-reduction assumptions.
-/
abbrev sumcheckFailureAdvantageBound_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext} :=
  SuperNeo.sumcheckFailureAdvantageBound_of_assumptions (ctx := ctx)

end InteractiveReductionsInterface

end SuperNeo
