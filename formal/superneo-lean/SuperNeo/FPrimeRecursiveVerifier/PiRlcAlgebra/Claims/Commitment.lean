import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ExactMaterialization

/-!
Owns: exact combination semantics for the fixed commitment lanes.

Does not own: optional Nebula advice coordinates, transcript binding,
one-point projection security, or low-level ring multiplication.

Emits constraints: no.

Authority boundary: the parent commitment is bound by the exact combination;
claim and rho authority are upstream premises.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `CommitmentCombination` | `identities.commitment` | Eighteen degree-54 combinations over fifteen inputs | Authoritative inputs and rho values | No — Rust refinement open |
| `CommitmentCombinationWithIntermediates` | `identities.commitment` | States the source relation with explicit ring products | Exact coefficient relation | No — Rust refinement open |
| `commitmentCombinationWithIntermediates_iff_direct` | `identities.commitment` | Exact products may be substituted lane-by-lane | Semantic carriers above | No — Rust refinement open |

The Rust owner is `pi_rlc_circuit/commitment.rs`; actual multiplication rows
are emitted by the ring-action leaf.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

abbrev CommitmentValue := Fin commitmentLanes → RingCoefficients

/-- Direct paper equation for all fixed commitment lanes. -/
def CommitmentCombination
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → CommitmentValue)
    (parent : CommitmentValue) : Prop :=
  ∀ lane,
    DirectRingCombination rhos (fun inputIndex => inputs inputIndex lane) (parent lane)

/-- Source-R1CS shape with intermediate products on every lane. -/
def CommitmentCombinationWithIntermediates
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → CommitmentValue)
    (parent : CommitmentValue) : Prop :=
  ∀ lane,
    IntermediateRingCombination rhos (fun inputIndex => inputs inputIndex lane) (parent lane)

theorem commitmentCombinationWithIntermediates_iff_direct
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → CommitmentValue)
    (parent : CommitmentValue) :
    CommitmentCombinationWithIntermediates rhos inputs parent ↔
      CommitmentCombination rhos inputs parent := by
  simp only [CommitmentCombinationWithIntermediates, CommitmentCombination,
    intermediateRingCombination_iff_direct]

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
