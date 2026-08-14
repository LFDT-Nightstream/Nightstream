import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.SelectorCoverageArtifact
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition.RadixFourSelectorCoverageArtifact

/-!
Executable regression checks for the compact selector-coverage decoder.

| Mutation | Decoder obligation exercised |
|---|---|
| schema version | explicit wire-version boundary |
| duplicate selector | selector-column separation |
| selector column zero | separation from constant-one authority |
| reversed empty owner | cursor-anchored owner partition |
| non-unit gate | literal final-matrix coefficient |
| wrong gate port | ledger-to-gate reconciliation |
| changed polynomial | independent 74-term syntax equality |

Two accepted mutations pin the decoder's deliberate limit: selector support
cannot distinguish arithmetic-family labels within one gate class, and it
cannot recover an expected inventory of zero-length organizational nodes.
-/

namespace Tests.FPrimeFullHistorySelectiveCcsSelectorCoverage

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.SelectorCoverageArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.RadixFourSelectorCoverageArtifact

private abbrev fixtureRaw :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveSelectorCoverageFixture.rawCoverage

def wrongSchema : RawCoverage :=
  { fixtureRaw with schemaVersion := supportedSchemaVersion + 1 }

def duplicateSelector : RawCoverage :=
  { fixtureRaw with selectorColumns := [54, 54, 56] }

def zeroSelector : RawCoverage :=
  { fixtureRaw with selectorColumns := [0, 55, 56] }

def reversedEmptyOwner : RawCoverage :=
  { fixtureRaw with
    ownerRuns :=
      match fixtureRaw.ownerRuns with
      | first :: second :: rest =>
          first :: { second with start := second.stop + 1 } :: rest
      | runs => runs }

def nonUnitGate : RawCoverage :=
  { fixtureRaw with
    gateRuns :=
      match fixtureRaw.gateRuns with
      | first :: rest => { first with coefficient := 2 } :: rest
      | [] => [] }

def wrongGatePort : RawCoverage :=
  { fixtureRaw with
    gateRuns :=
      match fixtureRaw.gateRuns with
      | first :: rest => { first with port := .evaluation } :: rest
      | [] => [] }

def changedPolynomial : RawCoverage :=
  { fixtureRaw with
    polynomialTerms :=
      match fixtureRaw.polynomialTerms with
      | first :: rest => { first with coefficient := first.coefficient + 1 } :: rest
      | [] => [] }

def relabelFirstRetained : List RawOwnerRun → List RawOwnerRun
  | [] => []
  | owner :: owners =>
      if owner.family = .retained then
        { owner with family := .poseidon2 } :: owners
      else
        owner :: relabelFirstRetained owners

def sameGateFamilyRelabel : RawCoverage :=
  { fixtureRaw with ownerRuns := relabelFirstRetained fixtureRaw.ownerRuns }

def withoutEmptySharedNode : RawCoverage :=
  { fixtureRaw with
    ownerRuns := fixtureRaw.ownerRuns.filter fun owner =>
      ¬ (owner.family = .sharedDomain && owner.start = owner.stop) }

example : CoverageValid fixtureRaw :=
  fixture_coverage_valid

example : (decodeCoverage fixtureRaw).isSome = true :=
  fixture_coverage_decodes

example : ¬ CoverageValid wrongSchema := by
  native_decide

example : ¬ CoverageValid duplicateSelector := by
  native_decide

example : ¬ CoverageValid zeroSelector := by
  native_decide

example : ¬ CoverageValid reversedEmptyOwner := by
  native_decide

example : ¬ CoverageValid nonUnitGate := by
  native_decide

example : ¬ CoverageValid wrongGatePort := by
  native_decide

example : ¬ CoverageValid changedPolynomial := by
  native_decide

example : CoverageValid sameGateFamilyRelabel := by
  native_decide

example : sameGateFamilyRelabel ≠ fixtureRaw := by
  native_decide

example : CoverageValid withoutEmptySharedNode := by
  native_decide

example : withoutEmptySharedNode ≠ fixtureRaw := by
  native_decide

#check fixture_nonempty_owner_has_exact_gate
#check fixture_every_row_reconciles
#check fixture_polynomial_exact

private abbrev candidateRaw :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourSelectorCoverage.rawCoverage

example : CoverageValid candidateRaw :=
  candidate_coverage_valid

example :
    candidateRaw.rows = 8102331 ∧
    candidateRaw.columns = 12288726 ∧
    candidateRaw.ownerRuns.length = 14 := by
  decide

#check candidate_nonempty_owner_has_exact_gate
#check candidate_every_row_reconciles
#check candidate_polynomial_exact

end Tests.FPrimeFullHistorySelectiveCcsSelectorCoverage
