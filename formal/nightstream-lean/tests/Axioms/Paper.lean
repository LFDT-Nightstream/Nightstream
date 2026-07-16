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

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.strong_extract_or_bad_event' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.strong_extract_or_bad_event

/-- info: 'Nightstream.SuperNeo.Folding.BatchArity.total_le' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.BatchArity.total_le

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.product_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.product_complete

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.complete

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority.accepted_does_not_determine_output_evaluations' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority.accepted_does_not_determine_output_evaluations

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority.accepted_replaceEvaluations_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority.accepted_replaceEvaluations_iff

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority.accepted_does_not_determine_common_output_point' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputEvaluationAuthority.accepted_does_not_determine_common_output_point

/-! Independent Phi81 output-claim semantics. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.freshYZcolTerm_tail_zero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.freshYZcolTerm_tail_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.boundToSources_iff_eq_canonicalClaims' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.boundToSources_iff_eq_canonicalClaims

/-! Source-derived `yZcol` base-field and production-PiDEC transport. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.BaseLinear.yZcolEvaluation_combine' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.BaseLinear.yZcolEvaluation_combine

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.BaseLinear.canonicalYZcol_product_piDec_transport' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.BaseLinear.canonicalYZcol_product_piDec_transport

/-! Combined-parent `PiDEC` opening authority for `yZcol`. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiDECParentOpening.parentAssignment_eq_recompose_or_bindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiDECParentOpening.parentAssignment_eq_recompose_or_bindingCollision

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiDECParentOpening.parentYZcol_transport_or_bindingCollision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.PiDECParentOpening.parentYZcol_transport_or_bindingCollision

/-! Batch-invariant typed Phi81 relation and output adapter. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Shape.publicWidth_ne_257' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Shape.publicWidth_ne_257

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Structure.matrixSource_kernel_eq' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Structure.matrixSource_kernel_eq

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.matrixEvaluation_apply_ofSourceData' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.matrixEvaluation_apply_ofSourceData

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.evaluations_get_ofSourceData' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.evaluations_get_ofSourceData

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

/-! Inclusion-necessity of typed Phi81 relation families. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.commitment_check_is_necessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.commitment_check_is_necessary

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.public_input_check_is_necessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.public_input_check_is_necessary

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.norm_check_is_necessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.norm_check_is_necessary

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.ccs_relation_check_is_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.ccs_relation_check_is_necessary

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.evaluation_size_check_is_necessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.evaluation_size_check_is_necessary

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.evaluation_lane_check_is_necessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.evaluation_lane_check_is_necessary

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.no_invalid_typed_point' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Necessity.no_invalid_typed_point

/-! Exactness and inclusion-minimality of the concrete typed Phi81 CE plan. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Minimality.cePlan_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Minimality.cePlan_exact

/-- info: 'Nightstream.SuperNeo.Concrete.Phi81Relation.Minimality.cePlan_inclusionMinimalSound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Phi81Relation.Minimality.cePlan_inclusionMinimalSound

/-! Current concrete CE versus independent Phi81 output semantics. -/

/-- info: 'Nightstream.SuperNeo.Concrete.Necessity.Phi81OutputMismatch.currentConcrete_ne_canonicalPhi81' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Concrete.Necessity.Phi81OutputMismatch.currentConcrete_ne_canonicalPhi81

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

/-- info: 'Nightstream.SuperNeo.Folding.Composition.fold_extraction_or_bad_event' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Composition.fold_extraction_or_bad_event

/-- info: 'Nightstream.SuperNeo.Folding.Composition.fold_knowledge_or_bad_event' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Composition.fold_knowledge_or_bad_event

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.accepted_inputsValid_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.accepted_inputsValid_or_badEvent

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.complete

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.paperNifsTransition_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.paperNifsTransition_complete

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.paperNifsTransition_of_accepted' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.paperNifsTransition_of_accepted

/-! Shared-carrier normalization of the two internal NIFS boundaries. -/

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.SharedAttempt.wiring' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.SharedAttempt.wiring

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.SharedAccepted.toAccepted' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.SharedAccepted.toAccepted

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.normalize_toAttempt_eq' depends on axioms: [Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.normalize_toAttempt_eq

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.accepted_normalize' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.accepted_normalize

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.sharedPaperNifsTransition_iff_paperNifsTransition' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.sharedPaperNifsTransition_iff_paperNifsTransition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsAlpha_is_coverageGap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsAlpha_is_coverageGap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsGamma_is_coverageGap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsGamma_is_coverageGap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.sumCheckPolynomialEncoding_is_coverageGap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.sumCheckPolynomialEncoding_is_coverageGap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.sumCheckChallengePointLink_is_coverageGap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.sumCheckChallengePointLink_is_coverageGap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piRlcBoundedSampler_is_coverageGap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piRlcBoundedSampler_is_coverageGap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.coverageStatus_eq_incomplete' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.coverageStatus_eq_incomplete

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.accepts_materializeFrom' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.accepts_materializeFrom

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.accepts_materializedSchedule' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.accepts_materializedSchedule

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.accepts_of_carrierAgreement' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.accepts_of_carrierAgreement

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.sumCheck_after_message_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.sumCheck_after_message_eq

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.piRlc_head_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.piRlc_head_eq

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.materializeFrom_append' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.materializeFrom_append

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.acceptsFrom_suffix' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.acceptsFrom_suffix

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.piRlc_list_eq' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.piRlc_list_eq

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.coefficient_has_accepted_preimage' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.coefficient_has_accepted_preimage

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.available_excludes_shortfall' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.available_excludes_shortfall

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

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.successful_cursor_in_fourth_digest_window' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.successful_cursor_in_fourth_digest_window

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.successful_execution_uses_four_blocks' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.successful_execution_uses_four_blocks

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.stateAt_succ_eq_referenceBlockState' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule.stateAt_succ_eq_referenceBlockState

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

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ReplayBridge.acceptsCanonical_challenges_eq_sampled' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ReplayBridge.acceptsCanonical_challenges_eq_sampled

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ReplayBridge.acceptsCanonical_challenges_valid' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ReplayBridge.acceptsCanonical_challenges_valid

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ReplayBridge.shortfall_excludes_replayResponseRefines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ReplayBridge.shortfall_excludes_replayResponseRefines

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.replayBoundExecution_implies_candidateNifsTransition' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.replayBoundExecution_implies_candidateNifsTransition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.replayBoundExecution_of_core_and_carrierAgreement' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.replayBoundExecution_of_core_and_carrierAgreement

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsNcTerminalSidecar_is_coverageGap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsNcTerminalSidecar_is_coverageGap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsSplitCoins_is_coverageGap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsSplitCoins_is_coverageGap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.eraseResponses_canonicalEvents' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.eraseResponses_canonicalEvents

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.materializeFrom_eraseResponses_eq_of_acceptsFrom' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.materializeFrom_eraseResponses_eq_of_acceptsFrom

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.acceptsCanonical_iff_carrierAgreement' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay.acceptsCanonical_iff_carrierAgreement

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.replayBoundAttempt_implies_candidateNifsTransition' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.replayBoundAttempt_implies_candidateNifsTransition

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.replayBoundAttempt_of_core_and_carrierAgreement' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.replayBoundAttempt_of_core_and_carrierAgreement

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsJointQSplitRefinement_is_coverageGap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsJointQSplitRefinement_is_coverageGap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsOutputProjectionSufficiency_is_coverageGap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.piCcsOutputProjectionSufficiency_is_coverageGap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.concreteTranscriptEncoding_is_coverageGap' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.concreteTranscriptEncoding_is_coverageGap

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.canonicalEvents_replaceFeEnvelope' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.canonicalEvents_replaceFeEnvelope

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.canonicalEvents_replaceNcEnvelope' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.canonicalEvents_replaceNcEnvelope

/-- info: 'Nightstream.SuperNeo.Folding.Nifs.NonInteractive.canonicalEvents_replacePiCcsOutputPoints' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.Folding.Nifs.NonInteractive.canonicalEvents_replacePiCcsOutputPoints

/-- info: 'Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot
