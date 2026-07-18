import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc
import tests.Axioms.Support

/-! Fail-closed dependency gate for SplitNc executable-source refinement. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement.directDiagonal_orderedAssignment' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement.directDiagonal_orderedAssignment

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement.directDiagonal_at_coveredCoordinates' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement.directDiagonal_at_coveredCoordinates

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement.truth_iff_inputsNormBoundedTwo' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement.truth_iff_inputsNormBoundedTwo

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement.assignmentsFitColumnDomain_of_covers' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement.assignmentsFitColumnDomain_of_covers

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement.truth_implies_trueInitial_eq_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.SourceRefinement.truth_implies_trueInitial_eq_zero

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.production_observation_collision_changes_nc_truth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.production_observation_collision_changes_nc_truth

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.artifact_checked_pi_ccs_acceptance_changes_nc_truth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.artifact_checked_pi_ccs_acceptance_changes_nc_truth

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.TerminalEqualityNecessity.terminalEquality_without_yZcolBound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.TerminalEqualityNecessity.terminalEquality_without_yZcolBound

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial.rangeProductB2_embed' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial.rangeProductB2_embed

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial.rangeProductB2_embed_eq_zero_iff_centered' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial.rangeProductB2_embed_eq_zero_iff_centered

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial.rangeProductB2_embed_eq_zero_iff_normTwo' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial.rangeProductB2_embed_eq_zero_iff_normTwo

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial.assignment_rangeProductB2_zero_iff_normBoundedTwo' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.RangePolynomial.assignment_rangeProductB2_zero_iff_normBoundedTwo

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial.directDiagonal_rangeProduct_eq_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial.directDiagonal_rangeProduct_eq_zero

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial.trueInitial_eq_zero_of_normBounded' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial.trueInitial_eq_zero_of_normBounded

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Terminal.dotChi_eq_zTilde_of_yZcolBound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Terminal.dotChi_eq_zTilde_of_yZcolBound

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Terminal.terminalRhs_eq_qNc_of_yZcolBound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Terminal.terminalRhs_eq_qNc_of_yZcolBound

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Terminal.not_terminalMismatch_of_yZcolBound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Terminal.not_terminalMismatch_of_yZcolBound

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial.qNc_cubePoint_eq_qNcOnCube' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial.qNc_cubePoint_eq_qNcOnCube

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial.trueInitial_eq_sum_qNc' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial.trueInitial_eq_sum_qNc

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.ProjectionNecessity.authoritativeLane_nonzero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.ProjectionNecessity.authoritativeLane_nonzero

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.ProjectionNecessity.erasedOutputs_not_yZcolBound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.ProjectionNecessity.erasedOutputs_not_yZcolBound

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.ProjectionNecessity.erasedOutputs_terminalMismatch' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.ProjectionNecessity.erasedOutputs_terminalMismatch

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.authoritativeYZcol_eq_radixWeightedChildProjection' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.authoritativeYZcol_eq_radixWeightedChildProjection

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.radixWeightedChildProjection_cubePoint' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.radixWeightedChildProjection_cubePoint

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.radixWeightedChildProjection_eq_weightedAuthoritativeYZcol' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.radixWeightedChildProjection_eq_weightedAuthoritativeYZcol

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.delayedProjectionStep_transfer' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.delayedProjectionStep_transfer

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.exactLane_of_delayedParentProjectionBound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.exactLane_of_delayedParentProjectionBound

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.exactWeightedAuthoritativeYZcolLane_of_bound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.exactWeightedAuthoritativeYZcolLane_of_bound

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.not_delayedParentProjectionMismatch_of_step' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection.not_delayedParentProjectionMismatch_of_step

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.betaPowerSelector_cubePoint' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.betaPowerSelector_cubePoint

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.delayedResidualCubeSum_eq_weightedCompactOldProjection' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.delayedResidualCubeSum_eq_weightedCompactOldProjection

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.acceptedProjectionIdentity_implies_exact_or_badRoot' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.acceptedProjectionIdentity_implies_exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.projectionIdentity_accepted_of_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.projectionIdentity_accepted_of_exact

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.acceptedProjectionIdentity_implies_cubeSum_eq_claimed_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.acceptedProjectionIdentity_implies_cubeSum_eq_claimed_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.compactOldPointEvaluation_eq_childLimbEvaluations' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.compactOldPointEvaluation_eq_childLimbEvaluations

/-- info: 'Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.paddedRawChildProjectionCoefficients_drop_active' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.paddedRawChildProjectionCoefficients_drop_active
