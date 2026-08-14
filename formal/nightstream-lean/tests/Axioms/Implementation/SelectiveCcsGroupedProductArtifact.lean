import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceRowRefinement
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.FinalRowsReconstructSource
import tests.Axioms.Support

/-! Fail-closed dependency gate for the grouped-product Rust fixture. -/

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.decodedSteps_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.decodedSteps_length

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.decodedRows_length' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.decodedRows_length

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.generated_steps_and_rows_join' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.generated_steps_and_rows_join

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.generated_row_zero_iff_fiveProduct' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.generated_row_zero_iff_fiveProduct

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage.evaluate_sourceLinearForm' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage.evaluate_sourceLinearForm

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.generated_step_images_match' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.generated_step_images_match

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.generated_c_action_eq_source_image' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.generated_c_action_eq_source_image

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.generated_factor_actions_eq_source_images' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement.generated_factor_actions_eq_source_images

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowRefinement.q_polynomial_exact' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowRefinement.q_polynomial_exact

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowRefinement.sourceRows_imply_all_steps_hold' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowRefinement.sourceRows_imply_all_steps_hold

/-- info: 'Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalRowsReconstructSource.active_final_rows_have_source_witness' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalRowsReconstructSource.active_final_rows_have_source_witness
