import Nightstream.Protocol.NebulaV2
import tests.Axioms.Support

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CommitmentBundle
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
open Nightstream.Protocol.NebulaV2.Fingerprint
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.IdealAcceptance
open Nightstream.Protocol.NebulaV2.IdealCompleteness
open Nightstream.Protocol.NebulaV2.IdealFingerprint
open Nightstream.Protocol.NebulaV2.IdealSequence
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.Memory
open Nightstream.Protocol.NebulaV2.MemoryWireGeometry
open Nightstream.Protocol.NebulaV2.Ports
open Nightstream.Protocol.NebulaV2.SequenceBinding
open Nightstream.Protocol.NebulaV2.Soundness
open Nightstream.Protocol.NebulaV2.StatementAuthority
open Nightstream.Protocol.NebulaV2.ShiftedTernary41V1
open Nightstream.Protocol.NebulaV2.WasmState
open Nightstream.Protocol.NebulaV2.WasmStateEncoding
open Nightstream.Protocol.NebulaV2.WasmStatement
open Nightstream.Protocol.NebulaV2.WasmIdealAcceptance
open Nightstream.Protocol.NebulaV2.WasmIdealCompleteness

/-- info: 'Nightstream.Protocol.NebulaV2.Soundness.StatementIdentity.encode_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms StatementIdentity.encode_injective

/-- info: 'Nightstream.Protocol.NebulaV2.StatementAuthority.Opens.aggregate_key_is_recomputed' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Opens.aggregate_key_is_recomputed

/-- info: 'Nightstream.Protocol.NebulaV2.StatementAuthority.Opens.initial_snapshot_is_verifier_owned' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Opens.initial_snapshot_is_verifier_owned

/-- info: 'Nightstream.Protocol.NebulaV2.Memory.balanced_implies_executes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms balanced_implies_executes

/-- info: 'Nightstream.Protocol.NebulaV2.ValidSegment.finalValid_of_balance' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ValidSegment.finalValid_of_balance

/-- info: 'Nightstream.Protocol.NebulaV2.ValidChain.executes' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#audit_axioms ValidChain.executes

/-- info: 'Nightstream.Protocol.NebulaV2.Fingerprint.boundedDifference_ne_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms boundedDifference_ne_zero

/-- info: 'Nightstream.Protocol.NebulaV2.Fingerprint.difference_totalDegree_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms difference_totalDegree_le

/-- info: 'Nightstream.Protocol.NebulaV2.Lifecycle.terminal_consumes_trailing' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms terminal_consumes_trailing

/-- info: 'Nightstream.Protocol.NebulaV2.Lifecycle.completeSchedule' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms completeSchedule

/-- info: 'Nightstream.Protocol.NebulaV2.CommitmentBundle.productMap_linear_combination' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms productMap_linear_combination

/-- info: 'Nightstream.Protocol.NebulaV2.CommitmentBundle.assignment_eq_or_bindingFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms assignment_eq_or_bindingFailure

/-- info: 'Nightstream.Protocol.NebulaV2.SequenceBinding.close_binds_exact_sequence_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms close_binds_exact_sequence_or_collision

/-- info: 'Nightstream.Protocol.NebulaV2.FPrime.close_preserves_global_timestamp_and_final_root' does not depend on any axioms -/
#guard_msgs in
#audit_axioms close_preserves_global_timestamp_and_final_root

/-- info: 'Nightstream.Protocol.NebulaV2.FPrime.MatchesActive.binds_challenge_and_segment_bounds' does not depend on any axioms -/
#guard_msgs in
#audit_axioms MatchesActive.binds_challenge_and_segment_bounds

/-- info: 'Nightstream.Protocol.NebulaV2.Completion.valid_trace_has_exact_capacity' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Completion.valid_trace_has_exact_capacity

/-- info: 'Nightstream.Protocol.NebulaV2.FPrime.VerifiedRun.full_segment_has_exact_claim_count' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms VerifiedRun.full_segment_has_exact_claim_count

/-- info: 'Nightstream.Protocol.NebulaV2.FPrime.openSegment_products_are_one' does not depend on any axioms -/
#guard_msgs in
#audit_axioms openSegment_products_are_one

/-- info: 'Nightstream.Protocol.NebulaV2.Ports.decodeAt_some_preserves_position' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms decodeAt_some_preserves_position

/-- info: 'Nightstream.Protocol.NebulaV2.IdealFingerprint.balance_or_evaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms balance_or_evaluationFailure

/-- info: 'Nightstream.Protocol.NebulaV2.IdealSequence.Checks.exact_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Checks.exact_or_collision

/-- info: 'Nightstream.Protocol.NebulaV2.FPrime.VerifiedRun.mono' does not depend on any axioms -/
#guard_msgs in
#audit_axioms VerifiedRun.mono

/-- info: 'Nightstream.Protocol.NebulaV2.FPrime.VerifiedRun.to_closed_has_balanced_products' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms VerifiedRun.to_closed_has_balanced_products

/-- info: 'Nightstream.Protocol.NebulaV2.GlobalFPrime.SegmentRun.finalProductsBalanced' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms GlobalFPrime.SegmentRun.finalProductsBalanced

/-- info: 'Nightstream.Protocol.NebulaV2.ProductState.accumulate_one_eq_expected' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ProductState.accumulate_one_eq_expected

/-- info: 'Nightstream.Protocol.NebulaV2.IdealAcceptance.FPrimeEvidence.fingerprintAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FPrimeEvidence.fingerprintAccepted

/-- info: 'Nightstream.Protocol.NebulaV2.IdealAcceptance.SegmentCheck.valid_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheck.valid_or_failure

/-- info: 'Nightstream.Protocol.NebulaV2.IdealAcceptance.SegmentCheck.dPre_binds_authoritative_lanes' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheck.dPre_binds_authoritative_lanes

/-- info: 'Nightstream.Protocol.NebulaV2.IdealAcceptance.SegmentCheck.fingerprint_challenges_bind_authoritative_lanes' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheck.fingerprint_challenges_bind_authoritative_lanes

/-- info: 'Nightstream.Protocol.NebulaV2.IdealAcceptance.CheckedChain.valid_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedChain.valid_or_failure

/-- info: 'Nightstream.Protocol.NebulaV2.IdealAcceptance.CheckedChain.globalClaimCount' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedChain.globalClaimCount

/-- info: 'Nightstream.Protocol.NebulaV2.IdealAcceptance.ideal_acceptance_implies_execution_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ideal_acceptance_implies_execution_or_failure

/-- info: 'Nightstream.Protocol.NebulaV2.WasmIdealAcceptance.production_acceptance_implies_execution_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms production_acceptance_implies_execution_or_failure

/-- info: 'Nightstream.Protocol.NebulaV2.FullClaim.VerifiedRun.initialActiveWellFormed' does not depend on any axioms -/
#guard_msgs in
#audit_axioms FullClaim.VerifiedRun.initialActiveWellFormed

/-- info: 'Nightstream.Protocol.NebulaV2.IdealCompleteness.HonestSegment.opened' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms HonestSegment.opened

/-- info: 'Nightstream.Protocol.NebulaV2.IdealCompleteness.HonestChain.toGlobalFPrime' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms HonestChain.toGlobalFPrime

/-- info: 'Nightstream.Protocol.NebulaV2.IdealCompleteness.CompletenessInput.globalFPrime' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompletenessInput.globalFPrime

/-- info: 'Nightstream.Protocol.NebulaV2.IdealCompleteness.valid_execution_with_honest_artifacts_is_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms valid_execution_with_honest_artifacts_is_accepted

/-- info: 'Nightstream.Protocol.NebulaV2.WasmIdealCompleteness.valid_fixed_wasm_execution_with_honest_artifacts_is_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms valid_fixed_wasm_execution_with_honest_artifacts_is_accepted

/-- info: 'Nightstream.Protocol.NebulaV2.WasmIdealCompleteness.completeness_input_final_state_terminal' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms completeness_input_final_state_terminal

/-- info: 'Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.decode_encode' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ShiftedTernary41V1.decode_encode

/-- info: 'Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.trits_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms trits_injective

/-- info: 'Nightstream.Protocol.NebulaV2.CompactCommit.encodeFields_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactCommit.encodeFields_injective

/-- info: 'Nightstream.Protocol.NebulaV2.CompactCommit.token_collision_implies_primary_or_short_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactCommit.token_collision_implies_primary_or_short_failure

/-- info: 'Nightstream.Protocol.NebulaV2.CommitmentBundle.bindingFailure_implies_component_failure' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CommitmentBundle.bindingFailure_implies_component_failure

/-- info: 'Nightstream.Protocol.NebulaV2.CompactChain.chainRoot_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactChain.chainRoot_injective

/-- info: 'Nightstream.Protocol.NebulaV2.CompactChain.root_collision_implies_hash_or_ajtai_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactChain.root_collision_implies_hash_or_ajtai_failure

/-- info: 'Nightstream.Protocol.NebulaV2.IdealFingerprint.ChallengePair.point_ofPoint' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms IdealFingerprint.ChallengePair.point_ofPoint

/-- info: 'Nightstream.Protocol.NebulaV2.Transcript.coordinateIndex_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Transcript.coordinateIndex_injective

/-- info: 'Nightstream.Protocol.NebulaV2.CanonicalFieldBits.decode_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalFieldBits.decode_injective

/-- info: 'Nightstream.Protocol.NebulaV2.CanonicalFieldBits.modulusWord_not_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalFieldBits.modulusWord_not_canonical

/-- info: 'Nightstream.Protocol.NebulaV2.CanonicalFieldBits.zero_and_modulus_are_modulo_aliases' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalFieldBits.zero_and_modulus_are_modulo_aliases

/-- info: 'Nightstream.Protocol.NebulaV2.WasmState.AppStateVector.Terminal.trapped_exit_code_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms AppStateVector.Terminal.trapped_exit_code_exact

/-- info: 'Nightstream.Protocol.NebulaV2.WasmState.Machine.active_does_not_complete' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Machine.active_does_not_complete

/-- info: 'Nightstream.Protocol.NebulaV2.WasmState.Machine.halted_row_is_event_drain' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Machine.halted_row_is_event_drain

/-- info: 'Nightstream.Protocol.NebulaV2.WasmStateEncoding.encode_injective' does not depend on any axioms -/
#guard_msgs in
#audit_axioms WasmStateEncoding.encode_injective

/-- info: 'Nightstream.Protocol.NebulaV2.WasmStateEncoding.Image.fieldValue_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Image.fieldValue_injective

/-- info: 'Nightstream.Protocol.NebulaV2.WasmStatement.terminal_output_fields_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms terminal_output_fields_exact

/-- info: 'Nightstream.Protocol.NebulaV2.WasmStatement.ResultImage.Decodes.mode_exit_and_flags_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ResultImage.Decodes.mode_exit_and_flags_exact

/-- info: 'Nightstream.Protocol.NebulaV2.WasmStatement.completed_execution_derives_terminal' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms completed_execution_derives_terminal

/-- info: 'Nightstream.Protocol.NebulaV2.Digest.lanes_injective' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Digest.lanes_injective

/-- info: 'Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry.blockWidth_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms blockWidth_exact

/-- info: 'Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry.aligned_add' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms aligned_add

/-- info: 'Nightstream.Protocol.NebulaV2.MemoryWireGeometry.stepPublicBits_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms stepPublicBits_exact

/-- info: 'Nightstream.Protocol.NebulaV2.MemoryWireGeometry.mandatoryBundleBits_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms mandatoryBundleBits_exact
