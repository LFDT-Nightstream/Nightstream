import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Semantics.RingAction

/-!
Owns: the exact parent equation for one fixed fifteen-input Pi_RLC ring
combination.

Does not own: transcript derivation of rho, source-claim reconstruction,
one-point projection security, or R1CS materialization.

Emits constraints: no.

Authority boundary: the equation assumes the supplied inputs and rho values
are already authoritative; it binds only the supplied parent.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `DirectRingCombination` | `identities.*` | `parent = sum_i ringAction rho_i input_i` for fifteen inputs | Authoritative inputs and transcript-derived rho values | No — Rust refinement open |

No R1CS temporaries appear in this semantic statement.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

open scoped BigOperators

/-- The irreducible paper equation for one fixed Pi_RLC ring combination. -/
def DirectRingCombination
    (rhos inputs : Fin inputCount → RingCoefficients)
    (parent : RingCoefficients) : Prop :=
  parent = ∑ inputIndex, ringAction (rhos inputIndex) (inputs inputIndex)

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
