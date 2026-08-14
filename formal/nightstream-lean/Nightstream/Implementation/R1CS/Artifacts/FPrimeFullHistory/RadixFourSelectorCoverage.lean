import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourSelectorCoverage

/-!
Stable facade for complete selector coverage of the production-width
radix-four candidate.

Owns: one handwritten import boundary over the generated candidate profile,
coalesced owner/gate runs, and exact selective polynomial.

Does not own: artifact decoding, arithmetic-family identity, recursive or
terminal semantics, security reduction, profile selection, or row removal.

Emits constraints: no.

| Child | Artifact ownership | Semantic owner |
|---|---|---|
| generated candidate | exact Rust-emitted profile and coverage data | `RadixFourSelectorCoverageArtifact` |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourSelectorCoverage

export Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourSelectorCoverage
  (profileId normBase decompositionExponent normBound kappa effectiveLambda
    batchSize scanSteps sourceOwnerRunCount sourceNonemptyOwnerRunCount
    coalescedRunCount rawCoverage)

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourSelectorCoverage
