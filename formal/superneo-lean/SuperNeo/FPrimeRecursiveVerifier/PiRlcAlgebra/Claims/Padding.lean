import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Refinement.ExactMaterialization

/-!
Owns: the shared padded-K carrier, active-limb extraction, exact limbwise
combination, and canonical zero-tail predicate used by `y_ring` and `y_zcol`.

Does not own: the number of `y_ring` rows, the `y_zcol` protocol identity,
transcript binding, or one-point projection security.

Emits constraints: no.

Authority boundary: active equations bind the parent active lanes; generic
`PaddingZero` premises bind whichever input and parent values a claim-specific
caller supplies. This file assigns no Pi_CCS or Pi_DEC authority by itself.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `activeRing`, `PaddedKActiveCombination` | `identities.y_ring`, `identities.y_zcol` | Reads and combines only lanes zero through 53 | Fixed padded carrier | No — Rust refinement open |
| `PaddingZero` | `padding.y_ring`, `padding.y_zcol` | Lanes 54 through 63 are zero | Claim-specific supplied value | No — Rust refinement open |
| `PaddedKCombination` | `identities.*`, `padding.*` | Active equations plus every input and parent zero tail | Authoritative claim values and rho values | No — Rust refinement open |
| `paddedKCombinationWithIntermediates_iff_direct` | `identities.*`, `padding.*` | Exact intermediates preserve the padded relation | Exact coefficient relation | No — Rust refinement open |
| `paddedKActiveCombination_congr` | `identities.*` | Active equality permits claim substitution | `ActiveEqual` for inputs and parent | No — Rust refinement open |

`Claims.YRing` and `Claims.YZcol` assign their zero-tail authorities
separately. In particular, this shared carrier does not imply that Pi_DEC owns
or validates the `y_zcol` tail.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

abbrev PaddedKVector := Fin extensionLimbs → Fin paddedDegree → SuperNeo.F

/-- The active ring element in one base-field limb. -/
def activeRing (value : PaddedKVector) (limb : Fin extensionLimbs) : RingCoefficients :=
  fun coefficient =>
    value limb ⟨coefficient.1, by
      exact Nat.lt_trans coefficient.2 ringDegree_lt_paddedDegree⟩

/-- Canonical zero padding outside the degree-54 ring. -/
def PaddingZero (value : PaddedKVector) : Prop :=
  ∀ limb lane, SuperNeo.d ≤ lane.1 → value limb lane = 0

/-- Only the active limbwise ring equations. -/
def PaddedKActiveCombination
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → PaddedKVector)
    (parent : PaddedKVector) : Prop :=
  ∀ limb,
    DirectRingCombination rhos
      (fun inputIndex => activeRing (inputs inputIndex) limb)
      (activeRing parent limb)

/-- Active equations plus the current explicit input and parent padding rows. -/
def PaddedKCombination
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → PaddedKVector)
    (parent : PaddedKVector) : Prop :=
  PaddedKActiveCombination rhos inputs parent ∧
    (∀ inputIndex, PaddingZero (inputs inputIndex)) ∧
    PaddingZero parent

def PaddedKCombinationWithIntermediates
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → PaddedKVector)
    (parent : PaddedKVector) : Prop :=
  (∀ limb,
    IntermediateRingCombination rhos
      (fun inputIndex => activeRing (inputs inputIndex) limb)
      (activeRing parent limb)) ∧
    (∀ inputIndex, PaddingZero (inputs inputIndex)) ∧
    PaddingZero parent

theorem paddedKCombinationWithIntermediates_iff_direct
    (rhos : Fin inputCount → RingCoefficients)
    (inputs : Fin inputCount → PaddedKVector)
    (parent : PaddedKVector) :
    PaddedKCombinationWithIntermediates rhos inputs parent ↔
      PaddedKCombination rhos inputs parent := by
  simp only [PaddedKCombinationWithIntermediates, PaddedKCombination,
    PaddedKActiveCombination, intermediateRingCombination_iff_direct]

/-- Equality on active lanes is exactly equality of the active projections. -/
def ActiveEqual (left right : PaddedKVector) : Prop :=
  ∀ limb coefficient,
    activeRing left limb coefficient = activeRing right limb coefficient

theorem activeRing_eq_of_activeEqual
    {left right : PaddedKVector}
    (hEqual : ActiveEqual left right)
    (limb : Fin extensionLimbs) :
    activeRing left limb = activeRing right limb := by
  funext coefficient
  exact hEqual limb coefficient

/-- The active ring relation cannot observe any padding-lane change. -/
theorem paddedKActiveCombination_congr
    (rhos : Fin inputCount → RingCoefficients)
    {leftInputs rightInputs : Fin inputCount → PaddedKVector}
    {leftParent rightParent : PaddedKVector}
    (hInputs : ∀ inputIndex, ActiveEqual (leftInputs inputIndex) (rightInputs inputIndex))
    (hParent : ActiveEqual leftParent rightParent) :
    PaddedKActiveCombination rhos leftInputs leftParent ↔
      PaddedKActiveCombination rhos rightInputs rightParent := by
  have hInputProjection :
      ∀ inputIndex limb,
        activeRing (leftInputs inputIndex) limb =
          activeRing (rightInputs inputIndex) limb := by
    intro inputIndex limb
    exact activeRing_eq_of_activeEqual (hInputs inputIndex) limb
  have hParentProjection :
      ∀ limb, activeRing leftParent limb = activeRing rightParent limb := by
    intro limb
    exact activeRing_eq_of_activeEqual hParent limb
  constructor
  · intro hLeft limb
    simpa [hInputProjection, hParentProjection] using hLeft limb
  · intro hRight limb
    simpa [hInputProjection, hParentProjection] using hRight limb

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
