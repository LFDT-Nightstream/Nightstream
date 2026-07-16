import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Claims.Padding

/-!
Owns: exact padded-K combination semantics for the three production `y_ring`
rows.

Does not own: the shared padded carrier, transcript binding, one-point
projection security, or row-level quotient advice.

Emits constraints: no.

Authority boundary: `YRingCombination` itself explicitly binds every input and
parent zero tail at `padding.y_ring`. Pi_DEC separately validates the parent
`y_ring` branch downstream; input/rho authority is an upstream premise.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `YRingCombination` | `identities.y_ring`, `padding.y_ring` | Three independent padded-K combinations | Fixed `yRingRows = 3` | No — Rust refinement open |
| `yRingCombinationWithIntermediates_iff_direct` | `identities.y_ring`, `padding.y_ring` | Exact intermediates substitute row-by-row | Exact coefficient relation | No — Rust refinement open |

The Pi_DEC statement above is specific to `y_ring`; it must not be transferred
to the structurally similar `y_zcol` carrier.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

abbrev YRingValue := Fin yRingRows → PaddedKVector

def YRingCombination
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → YRingValue)
    (parent : YRingValue) : Prop :=
  ∀ row,
    PaddedKCombination rhos
      (fun inputIndex => inputs inputIndex row) (parent row)

def YRingCombinationWithIntermediates
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → YRingValue)
    (parent : YRingValue) : Prop :=
  ∀ row,
    PaddedKCombinationWithIntermediates rhos
      (fun inputIndex => inputs inputIndex row) (parent row)

theorem yRingCombinationWithIntermediates_iff_direct
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → YRingValue)
    (parent : YRingValue) :
    YRingCombinationWithIntermediates rhos inputs parent ↔
      YRingCombination rhos inputs parent := by
  simp only [YRingCombinationWithIntermediates, YRingCombination,
    paddedKCombinationWithIntermediates_iff_direct]

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
