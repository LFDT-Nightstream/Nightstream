import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Parameters
import SuperNeo.Primitives.Ring

/-!
Owns: concrete coefficient-level ring action used by every Pi_RLC algebra
claim.

Does not own: multi-input combination, projection security, or emitted rows.

Emits constraints: no.

Authority boundary: both ring operands are supplied values; this module only
defines their exact quotient-ring product.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `RingCoefficients`, `ringAction` | `identities.*` | Exact degree-54 product modulo `X^54 + X^27 + 1` | Supplied canonical ring operands | No — Rust refinement open |
| `ringAction_apply` | `identities.*` | Exposes the exact coefficient computed by `SuperNeo.mulRq` | Semantic carrier above | No — Rust refinement open |

This file defines semantics only. Rust row emission remains owned by
`engine/r1cs_circuit/ring_action.rs`.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

/-- Extensional fixed-degree view used at the recursive-verifier boundary. -/
abbrev RingCoefficients := Fin SuperNeo.d → SuperNeo.F

/-- Convert the extensional view to the paper ring's canonical array carrier. -/
def toRingArray (value : RingCoefficients) : SuperNeo.Coeffs :=
  Array.ofFn value

/-- Exact Phi_81 ring multiplication used by the Rust Toom-3 emitter. -/
def ringAction (rho input : RingCoefficients) : RingCoefficients :=
  fun coefficient =>
    SuperNeo.coeffAt
      (SuperNeo.mulRq (toRingArray rho) (toRingArray input))
      coefficient.1

theorem ringAction_apply
    (rho input : RingCoefficients) (coefficient : Fin SuperNeo.d) :
    ringAction rho input coefficient =
      SuperNeo.coeffAt
        (SuperNeo.mulRq (toRingArray rho) (toRingArray input))
        coefficient.1 := rfl

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
