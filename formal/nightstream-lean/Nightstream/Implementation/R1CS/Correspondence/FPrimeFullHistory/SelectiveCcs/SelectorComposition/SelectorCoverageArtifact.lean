import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveSelectorCoverageFixture
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.SelectorCoverage

/-!
Contract: kernel-checked consumption of one compact Rust selector-coverage
fixture.

Owns: decoding the fixture, exact ledger-to-gate interval compatibility for
every fixture row, and equality of the exported 74 polynomial terms with the
independent Lean polynomial syntax.

Does not own: a production F-prime relation, complete family coverage, matrix
action at either selector port, arithmetic-family identity, an expected
zero-cost-node inventory, branch residual semantics, constraint necessity,
trusted row counts, or row removal.

Emits constraints: no.

Authority boundary: the handwritten decoder checks only that each generated
family label belongs to the gate class exhibited by final selector CSC
support. Labels within one gate class are not distinguished. A later row-shape
classifier must establish arithmetic-family identity, and a later theorem
must connect the selector matrices to the interpreted row point before a
physical residual-gating theorem applies.

| Fixture obligation | Evidence | Result |
|---|---|---|
| compact wire validity | executable handwritten decoder | `fixture_coverage_valid` |
| all-row interval coverage | ledger and gate interval partitions | `fixture_every_row_reconciles` |
| gate-class compatibility | exhaustive family-label-to-port map | `fixture_nonempty_owner_has_exact_gate` |
| polynomial identity | Rust `Structure.f` versus independent Lean terms | `fixture_polynomial_exact` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.SelectorCoverageArtifact

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.Wire

private abbrev fixtureRaw :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveSelectorCoverageFixture.rawCoverage

theorem fixture_coverage_valid : CoverageValid fixtureRaw := by
  decide

def fixtureCoverage : ValidatedCoverage :=
  ⟨fixtureRaw, fixture_coverage_valid⟩

theorem fixture_coverage_decodes :
    (decodeCoverage fixtureRaw).isSome = true :=
  (decodeCoverage_isSome_iff fixtureRaw).2 fixture_coverage_valid

theorem fixture_nonempty_owner_has_exact_gate
    {owner : RawOwnerRun}
    (member : owner ∈ fixtureCoverage.raw.ownerRuns)
    (nonempty : owner.start < owner.stop) :
    ∃ gate,
      gate ∈ fixtureCoverage.raw.gateRuns ∧
      gate.start = owner.start ∧
      gate.stop = owner.stop ∧
      gate.coefficient = 1 ∧
      expectedGate fixtureCoverage.raw.selectorColumns owner =
        some (gate.port, gate.column) :=
  fixtureCoverage.nonemptyOwner_has_exactGate member nonempty

theorem fixture_every_row_reconciles
    (row : Fin fixtureCoverage.raw.rows) :
    ∃ owner gate,
      owner ∈ fixtureCoverage.raw.ownerRuns ∧
      gate ∈ fixtureCoverage.raw.gateRuns ∧
      ownerCovers owner row.val ∧
      gateCovers gate row.val ∧
      expectedGate fixtureCoverage.raw.selectorColumns owner =
        some (gate.port, gate.column) :=
  fixtureCoverage.row_reconciles row

theorem fixture_polynomial_exact :
    fixtureCoverage.raw.polynomialTerms = expectedPolynomialTerms :=
  fixtureCoverage.polynomial_exact

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.SelectorCoverageArtifact
