import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveGroupedProductRewriteFixture

/-!
Stable facade for one exact grouped-product rewrite fixture.

Owns: one handwritten import boundary over the generated source recurrence,
its exact low-norm slots, and its six final materialized rows.

Does not own: decoding, source-to-final assignment refinement, production
coverage, recursive or terminal conformance, constraint necessity, or row
removal.

Emits constraints: no.

| Child | Artifact ownership | Semantic owner |
|---|---|---|
| generated steps | exact Rust executable provenance | grouped-product decoder |
| generated slots | exact source and accumulator images | source-image interpreter |
| generated rows | exact final selective matrices | evaluation-row refinement |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteFixture

export Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveGroupedProductRewriteFixture
  (sourceRowCount sourceColumnCount finalColumnCount arm rawSourceSlots
    rawSourceDefinitions rawDerivedSlots rawSourceRows rawStep00 rawStep01 rawStep02
    rawStep03 rawStep04 rawStep05 rawSteps rawRow00 rawRow01 rawRow02
    rawRow03 rawRow04 rawRow05 rawRows)

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteFixture
