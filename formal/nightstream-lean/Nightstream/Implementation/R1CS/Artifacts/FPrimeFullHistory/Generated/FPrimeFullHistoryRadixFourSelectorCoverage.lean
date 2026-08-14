import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveSelectorCoverageSchema

/-! Generated file: complete run-compressed selector coverage for the exact
production-width WASM Nebula radix-four candidate.

Owns: every exclusive compiler owner interval, every final general/evaluation
selector-port interval, and the ordered selective polynomial read from the
final Rust relation.

Does not own: arithmetic-family identity, source-to-final assignment
refinement, recursive or terminal relation semantics, constraint necessity,
security reduction, or permission to remove rows.

Emits constraints: no. Rust emits this file only after it reconciles the
complete selector CSC ports with the exclusive owner ledger.

| Artifact branch | Exact Rust source | Scope |
|---|---|---|
| owner runs | production selective compiler ledger | all rows |
| gate runs | final selector-port CSC matrices | all rows |
| polynomial | final ordered sparse terms | all 74 terms |
| profile constants | radix-four candidate parameters | provenance |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourSelectorCoverage

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.Wire

def profileId : String := "wasm-nebula-radix-four-candidate-v1"
def normBase : Nat := 4
def decompositionExponent : Nat := 7
def normBound : Nat := 16384
def kappa : Nat := 18
def effectiveLambda : Nat := 114
def batchSize : Nat := 3
def scanSteps : Nat := 1088
def sourceOwnerRunCount : Nat := 185526
def sourceNonemptyOwnerRunCount : Nat := 180665
def coalescedRunCount : Nat := 14

def rawCoverage : RawCoverage where
  schemaVersion := 3
  rows := 8102331
  columns := 12288726
  selectorColumns := [2430, 2431]
  polynomialArity := 13
  polynomialTerms := [
    { coefficient := 1, exponents := [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [0, 1, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584319, exponents := [0, 1, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584314, exponents := [0, 1, 6, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 14, exponents := [0, 1, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584314, exponents := [0, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0] }
  , { coefficient := 1, exponents := [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1] }
  , { coefficient := 18446744069414584320, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0] }
  , { coefficient := 13835058052060938241, exponents := [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0] }
  , { coefficient := 4611686017353646080, exponents := [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0] }
  , { coefficient := 4611686017353646080, exponents := [0, 1, 2, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0] }
  , { coefficient := 13835058052060938241, exponents := [1, 1, 2, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0] }
  , { coefficient := 4611686017353646080, exponents := [0, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0] }
  , { coefficient := 13835058052060938241, exponents := [1, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0] }
  , { coefficient := 13835058052060938241, exponents := [0, 1, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0] }
  , { coefficient := 4611686017353646080, exponents := [1, 1, 2, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 9223372034707292161, exponents := [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 9223372034707292161, exponents := [0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 9223372034707292160, exponents := [1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 13835058052060938241, exponents := [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 4611686017353646080, exponents := [0, 1, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 13835058052060938241, exponents := [0, 1, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 9223372034707292160, exponents := [1, 1, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 4611686017353646080, exponents := [0, 1, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 9223372034707292161, exponents := [1, 1, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }
  , { coefficient := 9223372034707292161, exponents := [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] }
  , { coefficient := 13835058052060938241, exponents := [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0] }
  , { coefficient := 4611686017353646080, exponents := [1, 1, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0] }
  , { coefficient := 13835058052060938241, exponents := [1, 1, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0] }
  , { coefficient := 4611686017353646080, exponents := [1, 1, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0] }
  , { coefficient := 18446744069414584320, exponents := [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 9223372034707292161, exponents := [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 9223372034707292161, exponents := [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 9223372034707292161, exponents := [0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 9223372034707292160, exponents := [1, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 9223372034707292161, exponents := [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 9223372034707292160, exponents := [1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 9223372034707292161, exponents := [1, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0] }
  , { coefficient := 1, exponents := [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1] }
  , { coefficient := 18446744069414584320, exponents := [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }
  , { coefficient := 1, exponents := [1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1] }
  , { coefficient := 9223372034707292161, exponents := [0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1] }
  , { coefficient := 9223372034707292160, exponents := [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1] }
  , { coefficient := 1, exponents := [1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1] }
  , { coefficient := 9223372034707292161, exponents := [0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1] }
  , { coefficient := 18446744069414584320, exponents := [1, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1] }
  ]
  ownerRuns := [
    { start := 0, stop := 2, family := .selectorDomain, arm := none }
  , { start := 2, stop := 23013, family := .armDomain, arm := some 0 }
  , { start := 23013, stop := 4982069, family := .armDomain, arm := some 1 }
  , { start := 4982069, stop := 4982070, family := .oneHot, arm := none }
  , { start := 4982070, stop := 4982074, family := .publicPadding, arm := none }
  , { start := 4982074, stop := 4982123, family := .privatePadding, arm := none }
  , { start := 4982123, stop := 5107005, family := .retained, arm := some 0 }
  , { start := 5107005, stop := 5114835, family := .poseidon2, arm := some 0 }
  , { start := 5114835, stop := 5372845, family := .retained, arm := some 1 }
  , { start := 5372845, stop := 5474711, family := .poseidon2, arm := some 1 }
  , { start := 5474711, stop := 8030453, family := .shiftedTernaryCanonical, arm := some 1 }
  , { start := 8030453, stop := 8062529, family := .polynomialEvaluation, arm := some 1 }
  , { start := 8062529, stop := 8102309, family := .productSum, arm := some 1 }
  , { start := 8102309, stop := 8102331, family := .ringPadding, arm := none }
  ]
  gateRuns := [
    { start := 0, stop := 2, port := .general, column := 0, coefficient := 1 }
  , { start := 2, stop := 23013, port := .generalEvaluation, column := 2430, coefficient := 1 }
  , { start := 23013, stop := 4982069, port := .generalEvaluation, column := 2431, coefficient := 1 }
  , { start := 4982069, stop := 4982070, port := .general, column := 0, coefficient := 1 }
  , { start := 4982070, stop := 4982074, port := .general, column := 0, coefficient := 1 }
  , { start := 4982074, stop := 4982123, port := .general, column := 0, coefficient := 1 }
  , { start := 4982123, stop := 5107005, port := .general, column := 2430, coefficient := 1 }
  , { start := 5107005, stop := 5114835, port := .general, column := 2430, coefficient := 1 }
  , { start := 5114835, stop := 5372845, port := .general, column := 2431, coefficient := 1 }
  , { start := 5372845, stop := 5474711, port := .general, column := 2431, coefficient := 1 }
  , { start := 5474711, stop := 8030453, port := .general, column := 2431, coefficient := 1 }
  , { start := 8030453, stop := 8062529, port := .evaluation, column := 2431, coefficient := 1 }
  , { start := 8062529, stop := 8102309, port := .evaluation, column := 2431, coefficient := 1 }
  , { start := 8102309, stop := 8102331, port := .general, column := 0, coefficient := 1 }
  ]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourSelectorCoverage
