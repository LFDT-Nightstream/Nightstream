import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourSelectorCoverage
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.SelectorCoverage

/-!
Contract: kernel-checked consumption of complete selector coverage for the
production-width radix-four candidate.

Assurance tier: Rust-conformant for property
`FPRIME-R4-SELECTOR-COVERAGE`. The Rust generator reads the final selector CSC
ports and the exclusive owner ledger before it emits the artifact. This tier
does not extend to other matrix ports or full relation semantics.

Owns: fail-closed decoding, exact candidate dimensions and parameters, exact
polynomial syntax, and expected selector-gate reconciliation for every one of
the 8,102,331 candidate rows.

Does not own: arithmetic-family identity, source-to-final assignment
refinement, recursive or terminal relation soundness or completeness,
constraint necessity, security reduction, production selection, or row
removal.

Emits constraints: no.

| Obligation | Evidence | Result |
|---|---|---|
| profile and dimensions | Rust-generated constants | `candidate_profile_exact` |
| compact wire validity | handwritten executable decoder | `candidate_coverage_valid` |
| all-row selector coverage | 14 maximal owner/gate intervals | `candidate_every_row_reconciles` |
| polynomial identity | exact Rust terms versus independent Lean syntax | `candidate_polynomial_exact` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.RadixFourSelectorCoverageArtifact

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.Wire
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.RadixFourSelectorCoverage

private abbrev candidateRaw := rawCoverage

theorem candidate_profile_exact :
    profileId = "wasm-nebula-radix-four-candidate-v1" ∧
    normBase = 4 ∧ decompositionExponent = 7 ∧ normBound = 16384 ∧
    kappa = 18 ∧ effectiveLambda = 114 ∧ batchSize = 3 ∧
    scanSteps = 1088 := by
  decide

theorem candidate_shape_exact :
    candidateRaw.rows = 8102331 ∧
    candidateRaw.columns = 12288726 ∧
    candidateRaw.selectorColumns = [2430, 2431] := by
  decide

theorem candidate_run_census_exact :
    sourceOwnerRunCount = 185526 ∧
    sourceNonemptyOwnerRunCount = 180665 ∧
    coalescedRunCount = 14 ∧
    candidateRaw.ownerRuns.length = coalescedRunCount ∧
    candidateRaw.gateRuns.length = coalescedRunCount := by
  decide

theorem candidate_coverage_valid : CoverageValid candidateRaw := by
  decide

def candidateCoverage : ValidatedCoverage :=
  ⟨candidateRaw, candidate_coverage_valid⟩

theorem candidate_coverage_decodes :
    (decodeCoverage candidateRaw).isSome = true :=
  (decodeCoverage_isSome_iff candidateRaw).2 candidate_coverage_valid

theorem candidate_nonempty_owner_has_exact_gate
    {owner : RawOwnerRun}
    (member : owner ∈ candidateCoverage.raw.ownerRuns)
    (nonempty : owner.start < owner.stop) :
    ∃ gate,
      gate ∈ candidateCoverage.raw.gateRuns ∧
      gate.start = owner.start ∧
      gate.stop = owner.stop ∧
      gate.coefficient = 1 ∧
      expectedGate candidateCoverage.raw.selectorColumns owner =
        some (gate.port, gate.column) :=
  candidateCoverage.nonemptyOwner_has_exactGate member nonempty

theorem candidate_every_row_reconciles
    (row : Fin candidateCoverage.raw.rows) :
    ∃ owner gate,
      owner ∈ candidateCoverage.raw.ownerRuns ∧
      gate ∈ candidateCoverage.raw.gateRuns ∧
      ownerCovers owner row.val ∧
      gateCovers gate row.val ∧
      expectedGate candidateCoverage.raw.selectorColumns owner =
        some (gate.port, gate.column) :=
  candidateCoverage.row_reconciles row

theorem candidate_polynomial_exact :
    candidateCoverage.raw.polynomialTerms = expectedPolynomialTerms :=
  candidateCoverage.polynomial_exact

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.RadixFourSelectorCoverageArtifact
