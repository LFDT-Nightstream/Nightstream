import OpeningConvergence.ConvergenceSoundnessInterface

/-!
# Module 4: ConvergenceSoundness — Implementation

Composes the full reduction pipeline (Modules 1-3) into the end-to-end
v1 verifier soundness theorem.

## Theorems implemented here
- Theorem 9: FinalOpeningAdequacy
- Theorem 10: V1ConvergenceSoundness
- S1: AccumulatorDedupSound
- S2: BucketPartitionDeterministic
- S3: CollapseGroupingDeterministic

## Spec
See `specs/ConvergenceSoundness.spec.md`
-/

namespace OpeningConvergence.ConvergenceSoundness

-- The theorem-facing proofs currently live in the Interface file.
-- Keep this file as the implementation owner if later extracted proofs,
-- specializations, or executable refinements need a separate home.

end OpeningConvergence.ConvergenceSoundness
