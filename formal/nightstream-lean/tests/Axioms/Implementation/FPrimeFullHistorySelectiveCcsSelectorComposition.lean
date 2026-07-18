import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.SelectorComposition
import tests.Axioms.Support

/-!
Fail-closed dependency gate for selector-composition soundness, completeness,
and inclusion-minimality.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics.goldilocks_noZeroProducts' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics.goldilocks_noZeroProducts

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics.exists_accepts_iff_selectedBranch' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Semantics.exists_accepts_iff_selectedBranch

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Complement.exists_complementAccepts_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Complement.exists_complementAccepts_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating.evaluate_general_gated' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating.evaluate_general_gated

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating.evaluate_evaluation_gated' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.PolynomialGating.evaluate_evaluation_gated

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.RowPointGating.residualAt_general_gated' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.RowPointGating.residualAt_general_gated

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.RowPointGating.residualAt_evaluation_gated' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.RowPointGating.residualAt_evaluation_gated

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Necessity.eachBranchGate_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Necessity.eachBranchGate_necessary

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Necessity.inactiveAdviceZero_not_required' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.Necessity.inactiveAdviceZero_not_required

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorGap_eq_zero_iff_total' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.SelectorComposition.selectorGap_eq_zero_iff_total

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ArtifactRefinement.generated_selector_rows_shape' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ArtifactRefinement.generated_selector_rows_shape

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ArtifactRefinement.generated_total_row_iff_selectorTotal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ArtifactRefinement.generated_total_row_iff_selectorTotal

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ArtifactRefinement.generated_gated_row_residual' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.ArtifactRefinement.generated_gated_row_residual

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.ValidatedCoverage.row_reconciles' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.ValidatedCoverage.row_reconciles

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.SelectorCoverageArtifact.fixture_coverage_valid' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.SelectorCoverageArtifact.fixture_coverage_valid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.SelectorCoverageArtifact.fixture_every_row_reconciles' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.SelectorComposition.SelectorCoverageArtifact.fixture_every_row_reconciles
