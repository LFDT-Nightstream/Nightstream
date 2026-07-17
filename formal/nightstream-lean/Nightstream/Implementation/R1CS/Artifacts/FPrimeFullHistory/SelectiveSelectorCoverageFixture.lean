import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveSelectorCoverageFixture

/-!
Stable facade for the run-compressed three-arm selector-coverage fixture.

Owns: one handwritten import boundary over the generated fixture value.

Does not own: decoding, semantic refinement, a production F-prime relation,
complete family coverage, constraint necessity, or row removal.

Emits constraints: no.

| Child | Artifact ownership | Semantic owner |
|---|---|---|
| generated fixture | exact Rust-emitted runs and polynomial terms | SelectorComposition.SelectorCoverageArtifact |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveSelectorCoverageFixture

export Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveSelectorCoverageFixture
  (rawCoverage)

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveSelectorCoverageFixture
