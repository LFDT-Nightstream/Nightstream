import OpeningConvergence.BatchEvalReductionInterface

/-!
# Module 2: BatchEvalReduction — Implementation

Proves that Phase 1 (point unification via eta/gamma-linearization +
rho-batched sumcheck + u_i* scalar outputs) is a sound batch evaluation
reduction.

## Theorems implemented here
- Theorem 3: ClaimedSumCorrectness
- Theorem 4: CoefficientLinearization
- Theorem 5: GammaLinearization
- Theorem 6: Phase1Soundness (Core + FailureBound + composition)
- Bridge: SameObjectPayloadUniqueness

## Spec
See `specs/BatchEvalReduction.spec.md`
-/

namespace OpeningConvergence.BatchEvalReduction

-- The theorem-facing proofs currently live in the Interface file.
-- Keep this file as the implementation owner if later extracted proofs,
-- specializations, or executable refinements need a separate home.

end OpeningConvergence.BatchEvalReduction
