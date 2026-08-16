import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayExecution
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBinding
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingSetup
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingOpeningRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingOutputRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingCompleteRows
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingClaimSchedule
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingProductionSetup
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsStatementBinding
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsStatementBindingState
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayPiCcsBinding
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayDigest
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinateAccumulator
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinateOverlay
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayCoordinateSequence
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayPhase
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayPublic
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayReduction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplayPiCcsStart
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingClaimReplaySequence
import tests.Axioms.Support

/-! Fail-closed axiom guard for streaming claim replay. -/

namespace NightstreamTests.Axioms.Implementation.FPrimeFullHistoryStreamingClaimReplay

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateAccumulator
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateSequence
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPhase
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPublic
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayReduction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsStart
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplaySequence
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachineDuplex
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBindingState
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsBinding

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.artifact_valid' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms artifact_valid

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.exact_shape' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms exact_shape

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.exact_public_word_layout' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms exact_public_word_layout

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.exact_state_word_layout' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms exact_state_word_layout

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.poseidon2_width_attribution_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms poseidon2_width_attribution_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.canonical_call_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms canonical_call_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.poseidon2_call_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms poseidon2_call_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.coordinate_call_holds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinate_call_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact.glue_row_holds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms glue_row_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.full_execution' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms full_execution

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.final_execution' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms final_execution

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.full_execution_refines' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms full_execution_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.final_execution_refines' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms final_execution_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.full_rows_refine_declared_runtime' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms full_rows_refine_declared_runtime

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayExecution.final_rows_refine_declared_runtime' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms final_rows_refine_declared_runtime

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain.domain_certificate_partition_exact' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms domain_certificate_partition_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain.domain_framing_words_exact' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms domain_framing_words_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain.certified_permutation_lane' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms certified_permutation_lane

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain.checkpoint1_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms checkpoint1_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain.checkpoint2_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms checkpoint2_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain.checkpoint3_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms checkpoint3_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain.checkpoint4_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms checkpoint4_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain.domain_initial_state_state_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms domain_initial_state_state_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain.domain_initial_state_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms domain_initial_state_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigestDomain.state_fields_label_exact' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms state_fields_label_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigest.exact_digest_operation_shape' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms exact_digest_operation_shape

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigest.digest_execution' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms digest_execution

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigest.digest_execution_refines' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms digest_execution_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigest.state_digest_refines' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms state_digest_refines

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayDigest.shared_public_words_refine' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms shared_public_words_refine

/-! The coordinate boundary uses checked finite artifact certificates. The
axiom audit exposes each native certificate only as `Lean.trustCompiler`. -/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinate.fullCoordinateCall_block_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fullCoordinateCall_block_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinate.fullCoordinateCall_rows_exact' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms fullCoordinateCall_rows_exact

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinate.full_partial_commitment' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms full_partial_commitment

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinate.full_before_zero' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms full_before_zero

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinate.full_commitment_transition' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms full_commitment_transition

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinate.final_commitment_carry' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms final_commitment_carry

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateAccumulator.partialCoordinate_sum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms partialCoordinate_sum

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateAccumulator.accumulated_succ' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accumulated_succ

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateAccumulator.AcceptedRun.state_eq_accumulated' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedRun.state_eq_accumulated

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateAccumulator.AcceptedRun.final_eq_direct' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedRun.final_eq_direct

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay.activeRows_imply_step' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms activeRows_imply_step

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay.activeRows_chunkZero_initial' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms activeRows_chunkZero_initial

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay.carryRows_imply_step' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms carryRows_imply_step

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateSequence.ActiveLinkedRows.step' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ActiveLinkedRows.step

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateSequence.CarryLinkedRows.step' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CarryLinkedRows.step

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateSequence.PhaseRowsAt.step' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PhaseRowsAt.step

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateSequence.AcceptedLinkedRun.final_eq_direct' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedLinkedRun.final_eq_direct

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPublic.exact_public_carrier_layout' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exact_public_carrier_layout

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPublic.carrier_getD_bit' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms carrier_getD_bit

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPublic.carrier_getD_padding' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms carrier_getD_padding

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPublic.carrier_binary' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms carrier_binary

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayReduction.private_decomposition_redundant' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms private_decomposition_redundant

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayReduction.exact_reduction_census' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms exact_reduction_census

/-- info: 'Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachineDuplex.semanticExecuteSlice_external_toDuplex' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms semanticExecuteSlice_external_toDuplex

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPhase.rows_imply_relation' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_imply_relation

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplaySequence.publicLinked_state_eq_or_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms publicLinked_state_eq_or_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplaySequence.AcceptedRunFrom.runtime_replay_of_no_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedRunFrom.runtime_replay_of_no_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplaySequence.accepted_run_recovers_frame_or_named_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepted_run_recovers_frame_or_named_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsStart.selected_public_state_eq_bindingState' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms selected_public_state_eq_bindingState

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsStart.acceptedRun_implies_piCcsStartRelation_of_no_state_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms acceptedRun_implies_piCcsStartRelation_of_no_state_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsStart.acceptedRun_initializes_piCcs_or_named_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms acceptedRun_initializes_piCcs_or_named_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBinding.selectedAuthoritativeFields_exactVerifierInput' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms selectedAuthoritativeFields_exactVerifierInput

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBindingState.authoritativeState_eq_exactVerifierInput' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms authoritativeState_eq_exactVerifierInput

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsStatementBindingState.accepted_fields_match_exactVerifierInput_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms accepted_fields_match_exactVerifierInput_or_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsBinding.acceptedRun_selects_authoritativeFields_or_named_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms acceptedRun_selects_authoritativeFields_or_named_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsBinding.acceptedRun_selectedState_eq_authoritative_or_named_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms acceptedRun_selectedState_eq_authoritative_or_named_collision

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayPiCcsBinding.acceptedRun_matches_exactPiCcsFields_or_named_collision' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms acceptedRun_matches_exactPiCcsFields_or_named_collision

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding.exact_geometry' does not depend on any axioms -/
#guard_msgs in
#audit_axioms exact_geometry

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding.flatIndex_messagePosition' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms flatIndex_messagePosition

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding.coordinateWitness_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateWitness_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding.coordinateWitness_unit_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateWitness_unit_bound

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding.equal_binding_recovers_fields_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms equal_binding_recovers_fields_or_failure

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding.maskedWitness_partition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms maskedWitness_partition

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding.commit_mask_add_complement' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms commit_mask_add_complement

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding.exact_production_chunk_source_shape' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms exact_production_chunk_source_shape

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.ExecutablePhi81.mul_coefficients' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ExecutablePhi81.mul_coefficients

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.integerResidue_zero_val' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms integerResidue_zero_val

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.integerResidue_signedDigit' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms integerResidue_signedDigit

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.seededMatrix_coefficients' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms seededMatrix_coefficients

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.exact_rust_identity' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exact_rust_identity

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.exact_chunk_geometry' does not depend on any axioms -/
#guard_msgs in
#audit_axioms exact_chunk_geometry

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.flattenCommitment_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms flattenCommitment_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.equal_concrete_binding_recovers_fields_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms equal_concrete_binding_recovers_fields_or_failure

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup.exact_output_width' does not depend on any axioms -/
#guard_msgs in
#audit_axioms exact_output_width

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingProductionSetup

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingProductionSetup.exact_identity' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms exact_identity

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows.Layout.wordStarts_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Layout.wordStarts_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows.coordinateBlock_exact_geometry' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateBlock_exact_geometry

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows.coordinateBlock_baseRotations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateBlock_baseRotations

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows.coordinateBlock_bitColumn' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateBlock_bitColumn

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows.coordinateBlock_tail_bitColumn_none' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateBlock_tail_bitColumn_none

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows.selected_word_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms selected_word_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows.selected_coordinate_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms selected_coordinate_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows.coordinateBlock_coefficient_residue' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateBlock_coefficient_residue

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows.coordinateBlock_inputValue_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateBlock_inputValue_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows.exact_tail_width' does not depend on any axioms -/
#guard_msgs in
#audit_axioms exact_tail_width

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows.zeroRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms zeroRows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows.openingBlockRows_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms openingBlockRows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows.openingRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms openingRows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows.sourceRows_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms sourceRows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows.zero_word_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms zero_word_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows.opening_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms opening_of_rows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows.active_digit_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms active_digit_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOpeningRows.sourceColumnsExact_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms sourceColumnsExact_of_rows

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows.linearValue_residue' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms linearValue_residue

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows.coordinateBlock_linearValue_eq_ring_products' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateBlock_linearValue_eq_ring_products

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows.maskedCommitment_coordinate_eq_linearValue' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms maskedCommitment_coordinate_eq_linearValue

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows.compact_output_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms compact_output_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows.compact_output_exact_of_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms compact_output_exact_of_rows

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows.shapeRows_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms shapeRows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows.shape_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms shape_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows.coordinateRows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateRows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows.rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows.production_rows_length' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms production_rows_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingCompleteRows.rows_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_sound

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.claimFramePosition_recompose' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms claimFramePosition_recompose

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.point_chunk_geometry' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms point_chunk_geometry

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.evaluation_chunk_geometry' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms evaluation_chunk_geometry

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.activeFields_nodup' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms activeFields_nodup

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.layout_selected_eq_chunkMask' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms layout_selected_eq_chunkMask

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.rows_imply_claimChunkCommitment' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms rows_imply_claimChunkCommitment

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.maskedWitness_sum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms maskedWitness_sum

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.commitments_sum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms commitments_sum

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.concrete_commitments_sum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms concrete_commitments_sum

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.claimChunk_active_range' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms claimChunk_active_range

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule.active_phase_row_census' does not depend on any axioms -/
#guard_msgs in
#audit_axioms active_phase_row_census

end NightstreamTests.Axioms.Implementation.FPrimeFullHistoryStreamingClaimReplay
