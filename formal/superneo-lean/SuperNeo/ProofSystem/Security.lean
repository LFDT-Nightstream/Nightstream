/-!
Security-model surfaces for paper-faithful protocol theorems.

This file is intentionally lightweight: it provides theorem statement interfaces
for probability and negligible error accounting without forcing a specific
probability encoding yet.
-/

namespace SuperNeo.ProofSystem.Security

/-- Security parameter (typically denoted `λ`). -/
abbrev SecurityParam := Nat

/-- Error function indexed by the security parameter. -/
abbrev ErrorFn := SecurityParam → Rat

/-- Event over an outcome space. -/
abbrev Event (α : Type) := α → Prop

/--
Minimal probability-model interface used by theorem statements.
Concrete measure-theoretic laws are intentionally deferred.
-/
structure ProbModel where
  Dist : Type → Type
  Pr : {α : Type} → Dist α → Event α → Rat

/--
Standard negligible-function shape over `Nat -> Rat`.
-/
def IsNegligible (f : ErrorFn) : Prop :=
  ∀ c : Nat, ∃ N : Nat, ∀ n : Nat, n ≥ N → f n ≤ (1 : Rat) / (((n + 1) ^ c : Nat) : Rat)

/--
Protocol error budget registry used by paper-facing final theorem statements.

Each component is tracked separately plus an explicit total error function.
-/
structure ErrorModel where
  ε_sumcheck : ErrorFn
  ε_schwartzZippel : ErrorFn
  ε_binding : ErrorFn
  ε_relaxedBinding : ErrorFn
  ε_total : ErrorFn
  hNeg_sumcheck : IsNegligible ε_sumcheck
  hNeg_schwartzZippel : IsNegligible ε_schwartzZippel
  hNeg_binding : IsNegligible ε_binding
  hNeg_relaxedBinding : IsNegligible ε_relaxedBinding
  hNeg_total : IsNegligible ε_total

end SuperNeo.ProofSystem.Security

