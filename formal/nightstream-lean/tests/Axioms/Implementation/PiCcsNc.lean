import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.SourceRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.CarrierCoverageRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.TerminalEqualityNecessity
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
