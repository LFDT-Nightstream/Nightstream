import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Claims.Padding

/-!
Owns: the protocol-named exact padded-K combination for the single `y_zcol`
claim.

Does not own: the shared padded carrier, transcript binding, one-point
projection security, or row-level quotient advice.

Emits constraints: no.

Authority boundary: `YZcolCombination` itself explicitly binds every input and
parent zero tail at `padding.y_zcol`; Pi_DEC owns no `y_zcol` recomposition or
zero-tail authority. Input/rho authority is an upstream premise.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `YZcolCombination` | `identities.y_zcol`, `padding.y_zcol` | One padded-K combination under the protocol claim name | Fixed padded shape | No — Rust refinement open |
| `yZcolCombinationWithIntermediates_iff_direct` | `identities.y_zcol`, `padding.y_zcol` | Exact intermediates substitute for this claim | Exact coefficient relation | No — Rust refinement open |

The shared `PaddedKCombination` carrier does not transfer Pi_DEC's downstream
`y_ring` validation to this independent claim.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

abbrev YZcolValue := PaddedKVector

def YZcolCombination
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → YZcolValue)
    (parent : YZcolValue) : Prop :=
  PaddedKCombination rhos inputs parent

def YZcolCombinationWithIntermediates
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → YZcolValue)
    (parent : YZcolValue) : Prop :=
  PaddedKCombinationWithIntermediates rhos inputs parent

theorem yZcolCombinationWithIntermediates_iff_direct
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → YZcolValue)
    (parent : YZcolValue) :
    YZcolCombinationWithIntermediates rhos inputs parent ↔
      YZcolCombination rhos inputs parent :=
  paddedKCombinationWithIntermediates_iff_direct rhos inputs parent

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
