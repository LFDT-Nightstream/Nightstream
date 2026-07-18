import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveSelectorCoverageSchema

/-! Generated file: run-compressed selector coverage for one deterministic
three-arm selective-compiler fixture.

Owns: the complete exclusive owner ledger and the complete final-matrix
general/evaluation selector support after Rust has reconciled them exactly.

Does not own: a production F-prime relation, branch semantics, source-row
refinement, constraint necessity, a trusted production count, or row removal.

Emits constraints: no. Empty owner runs remain visible; selector support is
split at owner boundaries and never expanded to one record per row.

| Artifact branch | Exact Rust source | Semantic status |
|---|---|---|
| owner runs | exclusive emitted-row ledger | provenance, reconciled in Lean |
| gate runs | final selector-port CSC matrices | physical support, reconciled in Lean |
| coefficient | checked final CSC value | must decode as field one |
| polynomial | final ordered sparse terms | compared to independent Lean syntax |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveSelectorCoverageFixture

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.Wire

def rawCoverage : RawCoverage where
  schemaVersion := 1
  rows := 753
  columns := 1458
  selectorColumns := [54, 55, 56]
  polynomialArity := 13
  polynomialTerms := [
    { coefficient := 1, exponents := [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0] }
  , { coefficient := 1, exponents := [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1] }
  , { coefficient := 9223372034707292161, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0] }
  , { coefficient := 13835058052060938241, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0] }
  , { coefficient := 13835058052060938241, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0] }
  , { coefficient := 4611686017353646080, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0] }
  , { coefficient := 4611686017353646081, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0] }
  , { coefficient := 9223372034707292159, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 2, 0] }
  ]
  ownerRuns := [
    { start := 0, stop := 3, family := .selectorDomain, arm := none }
  , { start := 3, stop := 3, family := .sharedDomain, arm := none }
  , { start := 3, stop := 67, family := .armDomain, arm := some 0 }
  , { start := 67, stop := 131, family := .armDomain, arm := some 1 }
  , { start := 131, stop := 195, family := .armDomain, arm := some 2 }
  , { start := 195, stop := 196, family := .oneHot, arm := none }
  , { start := 196, stop := 248, family := .publicPadding, arm := none }
  , { start := 248, stop := 299, family := .privatePadding, arm := none }
  , { start := 299, stop := 301, family := .retained, arm := some 0 }
  , { start := 301, stop := 440, family := .retained, arm := some 0 }
  , { start := 440, stop := 446, family := .productSum, arm := some 0 }
  , { start := 446, stop := 448, family := .retained, arm := some 1 }
  , { start := 448, stop := 587, family := .retained, arm := some 1 }
  , { start := 587, stop := 593, family := .productSum, arm := some 1 }
  , { start := 593, stop := 595, family := .retained, arm := some 2 }
  , { start := 595, stop := 734, family := .retained, arm := some 2 }
  , { start := 734, stop := 740, family := .productSum, arm := some 2 }
  , { start := 740, stop := 753, family := .ringPadding, arm := none }
  ]
  gateRuns := [
    { start := 0, stop := 3, port := .general, column := 0, coefficient := 1 }
  , { start := 3, stop := 67, port := .general, column := 54, coefficient := 1 }
  , { start := 67, stop := 131, port := .general, column := 55, coefficient := 1 }
  , { start := 131, stop := 195, port := .general, column := 56, coefficient := 1 }
  , { start := 195, stop := 196, port := .general, column := 0, coefficient := 1 }
  , { start := 196, stop := 248, port := .general, column := 0, coefficient := 1 }
  , { start := 248, stop := 299, port := .general, column := 0, coefficient := 1 }
  , { start := 299, stop := 301, port := .general, column := 54, coefficient := 1 }
  , { start := 301, stop := 440, port := .general, column := 54, coefficient := 1 }
  , { start := 440, stop := 446, port := .evaluation, column := 54, coefficient := 1 }
  , { start := 446, stop := 448, port := .general, column := 55, coefficient := 1 }
  , { start := 448, stop := 587, port := .general, column := 55, coefficient := 1 }
  , { start := 587, stop := 593, port := .evaluation, column := 55, coefficient := 1 }
  , { start := 593, stop := 595, port := .general, column := 56, coefficient := 1 }
  , { start := 595, stop := 734, port := .general, column := 56, coefficient := 1 }
  , { start := 734, stop := 740, port := .evaluation, column := 56, coefficient := 1 }
  , { start := 740, stop := 753, port := .general, column := 0, coefficient := 1 }
  ]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveSelectorCoverageFixture
