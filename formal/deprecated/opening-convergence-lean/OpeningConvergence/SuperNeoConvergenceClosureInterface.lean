import OpeningConvergence.ConvergenceSoundnessInterface
import OpeningConvergence.SuperNeoExtensionBridgeInterface

/-!
# Module 7: SuperNeoConvergenceClosure — Interface

Owns the concrete repo-facing end-to-end theorem obtained by instantiating the
generic convergence package at the actual extension-field SuperNeo boundary
`boundaryK`.

This is the theorem-facing closure step from the local abstract package to the
concrete Nightstream claim boundary.

## Spec
See `specs/SuperNeoConvergenceClosure.spec.md`
-/

namespace OpeningConvergence.SuperNeoConvergenceClosure

abbrev K := SuperNeo.ExtensionFieldInterface.KExt
abbrev Registry := OpeningConvergence.SuperNeoExtensionBridge.Registry

/-- Concrete claim-satisfaction predicate at the real extension-field SuperNeo
boundary. -/
abbrev ClaimSatisfiedK
    (registry : Registry)
    {ell : Nat}
    (claim : FamilyEvalClaim K ell) : Prop :=
  (OpeningConvergence.SuperNeoExtensionBridge.boundaryK registry).verify
    claim.openedObject
    claim.point
    claim.payload

/-- Concrete list-level claim satisfaction at the real extension-field SuperNeo
boundary. -/
abbrev ClaimsSatisfiedK
    (registry : Registry)
    {ell : Nat}
    (claims : List (FamilyEvalClaim K ell)) : Prop :=
  OpeningConvergence.ConvergenceSoundness.ClaimsSatisfied
    (OpeningConvergence.SuperNeoExtensionBridge.boundaryK registry)
    claims

/-- Concrete end-to-end closure theorem: accepted v1 convergence over the real
extension-field SuperNeo boundary implies that all original claims satisfy that
boundary, with the frozen total error bound. -/
theorem v1ConvergenceSoundness_boundaryK
    (registry : Registry)
    {ell numBuckets : Nat}
    (verifier :
      OpeningConvergence.ConvergenceSoundness.V1VerifierAccepts
        K
        (OpeningConvergence.SuperNeoExtensionBridge.boundaryK registry)
        ell
        numBuckets)
    (hCard : Fintype.card K ≥ MIN_FIELD_CARD)
    :
    ClaimsSatisfiedK registry verifier.pipeline.originalClaims ∧
      OpeningConvergence.ConvergenceSoundness.v1FailureProbability
        (OpeningConvergence.SuperNeoExtensionBridge.boundaryK registry)
        verifier
      ≤
      OpeningConvergence.ConvergenceSoundness.v1TotalErrorBound
        K
        (OpeningConvergence.ConvergenceSoundness.pipelineBucketParams
          (OpeningConvergence.SuperNeoExtensionBridge.boundaryK registry)
          verifier.pipeline) := by
  exact OpeningConvergence.ConvergenceSoundness.v1ConvergenceSoundness
    (pcs := OpeningConvergence.SuperNeoExtensionBridge.boundaryK registry)
    verifier
    hCard

end OpeningConvergence.SuperNeoConvergenceClosure
