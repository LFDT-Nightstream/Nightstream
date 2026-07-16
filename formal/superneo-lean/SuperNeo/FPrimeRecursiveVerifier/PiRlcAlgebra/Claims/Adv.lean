import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Claims.Commitment

/-!
Owns: exact combination semantics for the present-case Nebula advice
commitment's `ops`, `is`, and `fs` coordinates.

Does not own: optional presence, coordinate shape validation, transcript
binding, one-point projection security, or commitment-lane arithmetic.

Emits constraints: no.

Authority boundary: when advice is present for every input and the parent,
each parent coordinate is bound independently. Presence parity and shape are
separate composition obligations.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `AdvCombination` | `identities.adv` | Exact combination for all three coordinates | Advice present with valid shape | No — Rust refinement open |
| `advCombinationWithIntermediates_iff_direct` | `identities.adv` | Exact intermediates may be substituted coordinate-wise | Exact coefficient relation | No — Rust refinement open |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

/-- The three commitment coordinates of one present Nebula advice value. -/
structure AdvValue where
  ops : CommitmentValue
  isCoordinate : CommitmentValue
  fs : CommitmentValue

/-- Direct exact combination for every present advice coordinate. -/
def AdvCombination
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → AdvValue)
    (parent : AdvValue) : Prop :=
  CommitmentCombination rhos (fun inputIndex => (inputs inputIndex).ops) parent.ops ∧
    CommitmentCombination rhos
      (fun inputIndex => (inputs inputIndex).isCoordinate) parent.isCoordinate ∧
    CommitmentCombination rhos (fun inputIndex => (inputs inputIndex).fs) parent.fs

/-- Source exact-combination shape with per-coordinate intermediates. -/
def AdvCombinationWithIntermediates
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → AdvValue)
    (parent : AdvValue) : Prop :=
  CommitmentCombinationWithIntermediates rhos
      (fun inputIndex => (inputs inputIndex).ops) parent.ops ∧
    CommitmentCombinationWithIntermediates rhos
      (fun inputIndex => (inputs inputIndex).isCoordinate) parent.isCoordinate ∧
    CommitmentCombinationWithIntermediates rhos
      (fun inputIndex => (inputs inputIndex).fs) parent.fs

theorem advCombinationWithIntermediates_iff_direct
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → AdvValue)
    (parent : AdvValue) :
    AdvCombinationWithIntermediates rhos inputs parent ↔
      AdvCombination rhos inputs parent := by
  simp only [AdvCombinationWithIntermediates, AdvCombination,
    commitmentCombinationWithIntermediates_iff_direct]

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
