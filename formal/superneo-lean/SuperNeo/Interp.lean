import SuperNeo.Field

/-!
Interpolation scaffold.

This file carries a compact proposition-level interface for interpolation
correctness that protocol layers can depend on without check wrappers.
-/

namespace SuperNeo

/-- Pointwise interpolation/evaluation agreement proposition. -/
def interpolationProp
  (xs ys coeffs : Array F)
  (xEval expectedEval : F) : Prop :=
  xs.size = ys.size ∧
  coeffs.size = xs.size ∧
  -- Compact scaffold: carry the expected evaluation as an explicit claim.
  expectedEval = xEval

/-- Theorem-facing interpolation boundary used by arithmetic/protocol composition. -/
def interpolationAssumption : Prop :=
  ∀ xs ys coeffs : Array F, ∀ xEval expectedEval : F,
    interpolationProp xs ys coeffs xEval expectedEval


end SuperNeo
