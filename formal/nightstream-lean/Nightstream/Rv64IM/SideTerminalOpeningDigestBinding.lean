import Nightstream.Rv64IM.Generated.SideTerminalOpeningDigestBindingCorpus

/-!
Owns the RV64IM side-terminal opening-digest theorem boundary. This file keeps
two explicit views:

- the current native side-terminal checker shape, which only requires the claim
  witness, opening witness, and local digest chain to agree, and
- the corrected theorem boundary, which also requires the carried
  `opening_artifact_digest` to equal the canonical digest determined by the
  theorem-owned opening artifact.
-/

namespace Nightstream.Rv64IM

open Nightstream.Rv64IM.Generated

def currentSideTerminalCheck
    (c : SideTerminalOpeningDigestBindingCase) : Bool :=
  c.localDigestChainConsistent &&
    c.claimWitnessAccepted &&
    c.openingWitnessAccepted

def canonicalOpeningDigestBound
    (c : SideTerminalOpeningDigestBindingCase) : Bool :=
  c.statementOpeningDigest = c.canonicalOpeningDigest

def fixedSideTerminalCheck
    (c : SideTerminalOpeningDigestBindingCase) : Bool :=
  currentSideTerminalCheck c && canonicalOpeningDigestBound c

def rustRefinesFixedSideTerminalCheck
    (c : SideTerminalOpeningDigestBindingCase) : Bool :=
  c.rustObservedAccepted == fixedSideTerminalCheck c

def sideTerminalOpeningDigestCounterexample
    (c : SideTerminalOpeningDigestBindingCase) : Bool :=
  currentSideTerminalCheck c &&
    !canonicalOpeningDigestBound c &&
    !fixedSideTerminalCheck c

structure Rv64imSideTerminalOpeningDigestBindingReport where
  name : String
  currentAccepted : Bool
  canonicalBindingSatisfied : Bool
  fixedAccepted : Bool
  rustObservedAccepted : Bool
  rustRefinesFixedBoundary : Bool
  currentCounterexample : Bool
  blockers : List String
deriving Repr

def sideTerminalOpeningDigestBindingBlockers
    (c : SideTerminalOpeningDigestBindingCase) : List String :=
  if rustRefinesFixedSideTerminalCheck c then
    []
  else
    let missingCanonical :=
      if canonicalOpeningDigestBound c then
        []
      else
        [ "side_terminal_native_statement_carries_noncanonical_opening_artifact_digest" ]
    let staleAcceptedArtifactBoundary :=
      if c.rustObservedAccepted && !canonicalOpeningDigestBound c then
        [ "accepted_artifact_side_terminal_builder_accepts_caller_supplied_opening_artifact_digest" ]
      else
        []
    missingCanonical ++ staleAcceptedArtifactBoundary

def sideTerminalOpeningDigestBindingReport
    (c : SideTerminalOpeningDigestBindingCase) :
    Rv64imSideTerminalOpeningDigestBindingReport :=
  { name := c.name
  , currentAccepted := currentSideTerminalCheck c
  , canonicalBindingSatisfied := canonicalOpeningDigestBound c
  , fixedAccepted := fixedSideTerminalCheck c
  , rustObservedAccepted := c.rustObservedAccepted
  , rustRefinesFixedBoundary := rustRefinesFixedSideTerminalCheck c
  , currentCounterexample := sideTerminalOpeningDigestCounterexample c
  , blockers := sideTerminalOpeningDigestBindingBlockers c
  }

def rv64imSideTerminalOpeningDigestBindingReports :
    List Rv64imSideTerminalOpeningDigestBindingReport :=
  Generated.SideTerminalOpeningDigestBindingCases.cases.map
    sideTerminalOpeningDigestBindingReport

def sideTerminalOpeningDigestBindingCounterexamples :
    List String :=
  rv64imSideTerminalOpeningDigestBindingReports.filterMap fun report =>
    if report.currentCounterexample then some report.name else none

private def appendUniqueStrings
    (acc : List String)
    (xs : List String) : List String :=
  xs.foldl (fun acc x => if x ∈ acc then acc else acc ++ [x]) acc

def uniqueSideTerminalOpeningDigestBindingBlockers : List String :=
  rv64imSideTerminalOpeningDigestBindingReports.foldl
    (fun acc report => appendUniqueStrings acc report.blockers)
    []

def validGeneratedRv64imSideTerminalOpeningDigestBindingCases : Bool :=
  Generated.SideTerminalOpeningDigestBindingCases.cases.all
    rustRefinesFixedSideTerminalCheck

theorem honest_case_current_accepts :
    currentSideTerminalCheck
        Generated.SideTerminalOpeningDigestBindingCases.honestCase = true := by
  native_decide

theorem honest_case_fixed_accepts :
    fixedSideTerminalCheck
        Generated.SideTerminalOpeningDigestBindingCases.honestCase = true := by
  native_decide

theorem tampered_case_current_accepts :
    currentSideTerminalCheck
        Generated.SideTerminalOpeningDigestBindingCases.tamperedCase = true := by
  native_decide

theorem tampered_case_breaks_canonical_binding :
    canonicalOpeningDigestBound
        Generated.SideTerminalOpeningDigestBindingCases.tamperedCase = false := by
  native_decide

theorem tampered_case_fixed_rejects :
    fixedSideTerminalCheck
        Generated.SideTerminalOpeningDigestBindingCases.tamperedCase = false := by
  native_decide

theorem tampered_case_exposes_missing_opening_digest_binding :
    sideTerminalOpeningDigestCounterexample
        Generated.SideTerminalOpeningDigestBindingCases.tamperedCase = true := by
  native_decide

end Nightstream.Rv64IM
