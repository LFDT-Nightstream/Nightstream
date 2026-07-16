import SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra.Semantics.Combination

/-!
Owns: model-level equivalence between exact ring-output materialization and
direct substitution.

Does not own: production one-point projection, concrete scalar SSA traces,
retained-column decoding, or Rust trace validation.

Emits constraints: no. It proves equivalence between exact semantic
representations.

Authority boundary: this theorem applies only after exact coefficient-wise
ring equality is established; production projection has a different
exact-or-bad-root boundary.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `IntermediateRingCombination` | `identities.*` | States explicit ring products and their parent sum | Exact coefficient equality | No — Rust refinement open |
| `intermediateRingCombination_iff_direct` | `identities.*` | Explicit products are equivalent to direct substitution | Exact coefficient relation | No — production projection/Rust refinement open |
| `intermediateProducts_unique`, `reconstructedProducts_satisfy_definitions` | `identities.*` | Every model product has one canonical reconstruction | Supplied semantic operands | No — Rust refinement open |

This compares two representations of the paper's exact coefficient relation.
Production Pi_RLC rows use a transcript-bound one-point projection, so this
theorem must not be used as their deterministic soundness argument.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

open scoped BigOperators

/-- Current source shape with one explicit ring output per input. -/
def IntermediateRingCombination
    (rhos inputs : Fin inputCount → RingCoefficients)
    (parent : RingCoefficients) : Prop :=
  ∃ products : Fin inputCount → RingCoefficients,
    (∀ inputIndex,
      products inputIndex = ringAction (rhos inputIndex) (inputs inputIndex)) ∧
      parent = ∑ inputIndex, products inputIndex

/-- Canonical reconstruction for every projected source product. -/
def reconstructedProducts
    (rhos inputs : Fin inputCount → RingCoefficients) :
    Fin inputCount → RingCoefficients :=
  fun inputIndex => ringAction (rhos inputIndex) (inputs inputIndex)

/-- Soundness and completeness of eliminating the intermediate products. -/
theorem intermediateRingCombination_iff_direct
    (rhos inputs : Fin inputCount → RingCoefficients)
    (parent : RingCoefficients) :
    IntermediateRingCombination rhos inputs parent ↔
      DirectRingCombination rhos inputs parent := by
  classical
  constructor
  · rintro ⟨products, hProducts, hParent⟩
    unfold DirectRingCombination
    calc
      parent = ∑ inputIndex, products inputIndex := hParent
      _ = ∑ inputIndex, ringAction (rhos inputIndex) (inputs inputIndex) := by
        apply Finset.sum_congr rfl
        intro inputIndex _
        exact hProducts inputIndex
  · intro hDirect
    refine ⟨reconstructedProducts rhos inputs, ?_, ?_⟩
    · intro inputIndex
      rfl
    · exact hDirect

/-- The source product columns have exactly one reconstruction. -/
theorem intermediateProducts_unique
    (rhos inputs : Fin inputCount → RingCoefficients)
    {products : Fin inputCount → RingCoefficients}
    (hProducts : ∀ inputIndex,
      products inputIndex = ringAction (rhos inputIndex) (inputs inputIndex)) :
    products = reconstructedProducts rhos inputs := by
  funext inputIndex
  exact hProducts inputIndex

/-- The canonical reconstruction always satisfies every source definition. -/
theorem reconstructedProducts_satisfy_definitions
    (rhos inputs : Fin inputCount → RingCoefficients) :
    ∀ inputIndex,
      reconstructedProducts rhos inputs inputIndex =
        ringAction (rhos inputIndex) (inputs inputIndex) := by
  intro inputIndex
  rfl

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
