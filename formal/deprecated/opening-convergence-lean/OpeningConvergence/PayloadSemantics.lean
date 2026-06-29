import OpeningConvergence.PayloadSemanticsInterface

/-!
# Module 1: PayloadSemantics — Implementation

Proves that what the Ajtai PCS opens is faithfully represented in
`FamilyEvalPayload`, and that UNPACK recovers the correct field-element view.

## Theorems implemented here
- Theorem 1: UnpackLinearity
- Theorem 2: PayloadPcsConsistency (already proved in Interface by `exact`)

## Spec
See `specs/PayloadSemantics.spec.md`
-/

namespace OpeningConvergence.PayloadSemantics

-- The theorem-facing proofs currently live in the Interface file.
-- Keep this file as the implementation owner if later extracted proofs,
-- specializations, or executable refinements need a separate home.

end OpeningConvergence.PayloadSemantics
