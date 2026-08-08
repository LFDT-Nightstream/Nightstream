import Nightstream.SuperNeo
import tests.Axioms.Support

/-!
Fail-closed paper-model axioms gate. Every expectation is checked when this
module is built; the aggregate entrypoint imports all ownership groups.
-/

/-! Generic finite-check plan calculus. -/

/-- info: 'Nightstream.SuperNeo.CheckPlan.exact_without_of_redundant' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.CheckPlan.exact_without_of_redundant

/-- info: 'Nightstream.SuperNeo.CheckPlan.not_sound_without_of_necessary' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.CheckPlan.not_sound_without_of_necessary

/-- info: 'Nightstream.SuperNeo.CheckPlan.inclusionMinimalSound_of_witnesses' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.CheckPlan.inclusionMinimalSound_of_witnesses

/-- info: 'Nightstream.SuperNeo.Concrete.ccsMembership_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.ccsMembership_iff

/-- info: 'Nightstream.SuperNeo.Concrete.ceMembership_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.ceMembership_iff

/-- info: 'Nightstream.SuperNeo.Concrete.canonicalCCS_holds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.canonicalCCS_holds

/-- info: 'Nightstream.SuperNeo.Concrete.canonicalCE_holds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.canonicalCE_holds

/-- info: 'Nightstream.SuperNeo.GlobalParams.rlc_bound_for' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.GlobalParams.rlc_bound_for

/-- info: 'Nightstream.SuperNeo.SumCheck.false_acceptance_implies_bad_challenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.SumCheck.false_acceptance_implies_bad_challenge

/-- info: 'Nightstream.SuperNeo.SumCheck.check_eq_true_iff_accepted' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.SumCheck.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.BatchArity.total_le' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.BatchArity.total_le

/-! Batch-invariant typed Phi81 relation. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Shape.publicWidth_ne_257' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Shape.publicWidth_ne_257

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Structure.matrixSource_kernel_eq' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Structure.matrixSource_kernel_eq

/-! Base-field evaluation homomorphism used by `Pi_DEC` recomposition. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear.matrixEvaluation_combine' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear.matrixEvaluation_combine

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear.evaluations_combine' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear.evaluations_combine

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC.relation_evaluations_hom' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC.relation_evaluations_hom

/-! Typed Phi81 CCS/CE membership and complete evaluation authority. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.ccsMembership_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.ccsMembership_iff

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.canonicalCCS_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.canonicalCCS_holds

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.canonicalCE_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.canonicalCE_holds

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.evaluationsBound_iff_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.evaluationsBound_iff_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.ceMembership_iff_evaluationsBound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.ceMembership_iff_evaluationsBound

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.ce_evaluations_size_of_holds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.ce_evaluations_size_of_holds

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.combinedOutput_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.combinedOutput_holds

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.complete

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.same_phi_extractions_unique_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiRLC.same_phi_extractions_unique_or_collision

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.reduce_knowledge' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.reduce_knowledge

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.accepted_parent_eq_recompose_or_bindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.accepted_parent_eq_recompose_or_bindingCollision

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.complete' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.complete

/-! `Pi_DEC` child-authorization necessity. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization.signed_low_norm_base2_not_unique' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization.signed_low_norm_base2_not_unique

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization.binary_recomposition_not_unique_without_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization.binary_recomposition_not_unique_without_length

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization.fixed_length_binary_mod_recomposition_not_unique_without_range' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.Necessity.DigitAuthorization.fixed_length_binary_mod_recomposition_not_unique_without_range

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization.aggregate_digit_sum_not_functional_for_fixed_child_count' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization.aggregate_digit_sum_not_functional_for_fixed_child_count

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization.aggregate_norm_sum_not_functional_for_fixed_child_count' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization.aggregate_norm_sum_not_functional_for_fixed_child_count

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization.aggregate_only_validation_can_feed_different_next_inputs' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.Necessity.AggregateAuthorization.aggregate_only_validation_can_feed_different_next_inputs


/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.coefficient_has_accepted_preimage' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.coefficient_has_accepted_preimage

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.available_excludes_shortfall' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.available_excludes_shortfall

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.available_or_exists_shortfall' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.available_or_exists_shortfall

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.responseRefinesAt_implies_reference_within' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.responseRefinesAt_implies_reference_within

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.responseRefinesAt_valid' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.responseRefinesAt_valid

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.shortfall_excludes_responseRefinesAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.shortfall_excludes_responseRefinesAt

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.centeredValue_bounds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.centeredValue_bounds

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.acceptedFactorization' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.acceptedFactorization

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.sample54of64_eq_some_iff_reference_within' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.sample54of64_eq_some_iff_reference_within

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.successful_cursor_after_sixth_digest' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.successful_cursor_after_sixth_digest

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.source_nextState_eq_fixedBlockState' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.source_nextState_eq_fixedBlockState

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.stateAt_succ_eq_fixedBlockState' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.stateAt_succ_eq_fixedBlockState

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.coefficientDifference_bounds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.coefficientDifference_bounds

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.coefficientStrongPrecondition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.coefficientStrongPrecondition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.sampledChallenge_valid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.sampledChallenge_valid

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.expansionFactor_value' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet.expansionFactor_value

/-- info: 'Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.Accepted.parent_eq_of_children_eq' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiDEC.Accepted.parent_eq_of_children_eq
