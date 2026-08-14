import Nightstream.Protocol.Nebula
import tests.Axioms.Support

open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.Protocol.Nebula.ConcreteLaneGeometry
open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.IdealAcceptance
open Nightstream.Protocol.Nebula.IdealCompleteness
open Nightstream.Protocol.Nebula.IdealFingerprint
open Nightstream.Protocol.Nebula.IdealSequence
open Nightstream.Protocol.Nebula.Lifecycle
open Nightstream.Protocol.Nebula.Memory
open Nightstream.Protocol.Nebula.MemoryWireGeometry
open Nightstream.Protocol.Nebula.Ports
open Nightstream.Protocol.Nebula.SequenceBinding
open Nightstream.Protocol.Nebula.Soundness
open Nightstream.Protocol.Nebula.StatementAuthority
open Nightstream.Protocol.Nebula.ShiftedTernary41V1
open Nightstream.Protocol.Nebula.WasmState
open Nightstream.Protocol.Nebula.WasmStateEncoding
open Nightstream.Protocol.Nebula.WasmStatement
open Nightstream.Protocol.Nebula.WasmIdealAcceptance
open Nightstream.Protocol.Nebula.WasmIdealCompleteness

/-- info: 'Nightstream.Protocol.Nebula.Soundness.StatementIdentity.encode_injective' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms StatementIdentity.encode_injective

/-- info: 'Nightstream.Protocol.Nebula.StatementAuthority.Opens.aggregate_key_is_recomputed' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Opens.aggregate_key_is_recomputed

/-- info: 'Nightstream.Protocol.Nebula.StatementAuthority.Opens.initial_snapshot_is_verifier_owned' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Opens.initial_snapshot_is_verifier_owned

/-- info: 'Nightstream.Protocol.Nebula.Memory.balanced_implies_executes' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms balanced_implies_executes

/-- info: 'Nightstream.Protocol.Nebula.ValidSegment.finalValid_of_balance' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ValidSegment.finalValid_of_balance

/-- info: 'Nightstream.Protocol.Nebula.ValidChain.executes' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#audit_axioms ValidChain.executes

/-- info: 'Nightstream.Protocol.Nebula.Fingerprint.boundedDifference_ne_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms boundedDifference_ne_zero

/-- info: 'Nightstream.Protocol.Nebula.Fingerprint.difference_totalDegree_le' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms difference_totalDegree_le

/-- info: 'Nightstream.Protocol.Nebula.Lifecycle.terminal_consumes_trailing' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms terminal_consumes_trailing

/-- info: 'Nightstream.Protocol.Nebula.Lifecycle.completeSchedule' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms completeSchedule

/-- info: 'Nightstream.Protocol.Nebula.CommitmentBundle.productMap_linear_combination' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms productMap_linear_combination

/-- info: 'Nightstream.Protocol.Nebula.CommitmentBundle.assignment_eq_or_bindingFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms assignment_eq_or_bindingFailure

/-- info: 'Nightstream.Protocol.Nebula.SequenceBinding.close_binds_exact_sequence_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms close_binds_exact_sequence_or_collision

/-- info: 'Nightstream.Protocol.Nebula.FPrime.close_preserves_global_timestamp_and_final_root' does not depend on any axioms -/
#guard_msgs in
#audit_axioms close_preserves_global_timestamp_and_final_root

/-- info: 'Nightstream.Protocol.Nebula.FPrime.MatchesActive.binds_challenge_and_segment_bounds' does not depend on any axioms -/
#guard_msgs in
#audit_axioms MatchesActive.binds_challenge_and_segment_bounds

/-- info: 'Nightstream.Protocol.Nebula.Completion.valid_trace_has_exact_capacity' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Completion.valid_trace_has_exact_capacity

/-- info: 'Nightstream.Protocol.Nebula.FPrime.VerifiedRun.full_segment_has_exact_claim_count' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms VerifiedRun.full_segment_has_exact_claim_count

/-- info: 'Nightstream.Protocol.Nebula.FPrime.openSegment_products_are_one' does not depend on any axioms -/
#guard_msgs in
#audit_axioms openSegment_products_are_one

/-- info: 'Nightstream.Protocol.Nebula.Ports.decodeAt_some_preserves_position' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms decodeAt_some_preserves_position

/-- info: 'Nightstream.Protocol.Nebula.IdealFingerprint.balance_or_evaluationFailure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms balance_or_evaluationFailure

/-- info: 'Nightstream.Protocol.Nebula.IdealSequence.Checks.exact_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Checks.exact_or_collision

/-- info: 'Nightstream.Protocol.Nebula.FPrime.VerifiedRun.mono' does not depend on any axioms -/
#guard_msgs in
#audit_axioms VerifiedRun.mono

/-- info: 'Nightstream.Protocol.Nebula.FPrime.VerifiedRun.to_closed_has_balanced_products' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms VerifiedRun.to_closed_has_balanced_products

/-- info: 'Nightstream.Protocol.Nebula.GlobalFPrime.SegmentRun.finalProductsBalanced' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms GlobalFPrime.SegmentRun.finalProductsBalanced

/-- info: 'Nightstream.Protocol.Nebula.ProductState.accumulate_one_eq_expected' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ProductState.accumulate_one_eq_expected

/-- info: 'Nightstream.Protocol.Nebula.IdealAcceptance.FPrimeEvidence.fingerprintAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FPrimeEvidence.fingerprintAccepted

/-- info: 'Nightstream.Protocol.Nebula.IdealAcceptance.SegmentCheck.valid_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheck.valid_or_failure

/-- info: 'Nightstream.Protocol.Nebula.IdealAcceptance.SegmentCheck.dPre_binds_authoritative_lanes' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheck.dPre_binds_authoritative_lanes

/-- info: 'Nightstream.Protocol.Nebula.IdealAcceptance.SegmentCheck.fingerprint_challenges_bind_authoritative_lanes' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SegmentCheck.fingerprint_challenges_bind_authoritative_lanes

/-- info: 'Nightstream.Protocol.Nebula.IdealAcceptance.CheckedChain.valid_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedChain.valid_or_failure

/-- info: 'Nightstream.Protocol.Nebula.IdealAcceptance.CheckedChain.globalClaimCount' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms CheckedChain.globalClaimCount

/-- info: 'Nightstream.Protocol.Nebula.IdealAcceptance.ideal_acceptance_implies_execution_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ideal_acceptance_implies_execution_or_failure

/-- info: 'Nightstream.Protocol.Nebula.WasmIdealAcceptance.production_acceptance_implies_execution_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms production_acceptance_implies_execution_or_failure

/-- info: 'Nightstream.Protocol.Nebula.FullClaim.VerifiedRun.initialActiveWellFormed' does not depend on any axioms -/
#guard_msgs in
#audit_axioms FullClaim.VerifiedRun.initialActiveWellFormed

/-- info: 'Nightstream.Protocol.Nebula.IdealCompleteness.HonestSegment.opened' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms HonestSegment.opened

/-- info: 'Nightstream.Protocol.Nebula.IdealCompleteness.HonestChain.toGlobalFPrime' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms HonestChain.toGlobalFPrime

/-- info: 'Nightstream.Protocol.Nebula.IdealCompleteness.CompletenessInput.globalFPrime' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompletenessInput.globalFPrime

/-- info: 'Nightstream.Protocol.Nebula.IdealCompleteness.valid_execution_with_honest_artifacts_is_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms valid_execution_with_honest_artifacts_is_accepted

/-- info: 'Nightstream.Protocol.Nebula.WasmIdealCompleteness.valid_fixed_wasm_execution_with_honest_artifacts_is_accepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms valid_fixed_wasm_execution_with_honest_artifacts_is_accepted

/-- info: 'Nightstream.Protocol.Nebula.WasmIdealCompleteness.completeness_input_final_state_terminal' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms completeness_input_final_state_terminal

/-- info: 'Nightstream.Protocol.Nebula.ShiftedTernary41V1.decode_encode' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ShiftedTernary41V1.decode_encode

/-- info: 'Nightstream.Protocol.Nebula.ShiftedTernary41V1.trits_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms trits_injective

/-- info: 'Nightstream.Protocol.Nebula.CompactCommit.encodeFields_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactCommit.encodeFields_injective

/-- info: 'Nightstream.Protocol.Nebula.CompactCommit.token_collision_implies_primary_or_short_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactCommit.token_collision_implies_primary_or_short_failure

/-- info: 'Nightstream.Protocol.Nebula.CommitmentBundle.bindingFailure_implies_component_failure' does not depend on any axioms -/
#guard_msgs in
#audit_axioms CommitmentBundle.bindingFailure_implies_component_failure

/-- info: 'Nightstream.Protocol.Nebula.CompactChain.chainRoot_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactChain.chainRoot_injective

/-- info: 'Nightstream.Protocol.Nebula.CompactChain.root_collision_implies_hash_or_ajtai_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CompactChain.root_collision_implies_hash_or_ajtai_failure

/-- info: 'Nightstream.Protocol.Nebula.IdealFingerprint.ChallengePair.point_ofPoint' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms IdealFingerprint.ChallengePair.point_ofPoint

/-- info: 'Nightstream.Protocol.Nebula.Transcript.coordinateIndex_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Transcript.coordinateIndex_injective

/-- info: 'Nightstream.Protocol.Nebula.CanonicalFieldBits.decode_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalFieldBits.decode_injective

/-- info: 'Nightstream.Protocol.Nebula.CanonicalFieldBits.modulusWord_not_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalFieldBits.modulusWord_not_canonical

/-- info: 'Nightstream.Protocol.Nebula.CanonicalFieldBits.zero_and_modulus_are_modulo_aliases' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CanonicalFieldBits.zero_and_modulus_are_modulo_aliases

/-- info: 'Nightstream.Protocol.Nebula.WasmState.AppStateVector.Terminal.trapped_exit_code_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms AppStateVector.Terminal.trapped_exit_code_exact

/-- info: 'Nightstream.Protocol.Nebula.WasmState.Machine.active_does_not_complete' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Machine.active_does_not_complete

/-- info: 'Nightstream.Protocol.Nebula.WasmState.Machine.halted_row_is_event_drain' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Machine.halted_row_is_event_drain

/-- info: 'Nightstream.Protocol.Nebula.WasmStateEncoding.encode_injective' does not depend on any axioms -/
#guard_msgs in
#audit_axioms WasmStateEncoding.encode_injective

/-- info: 'Nightstream.Protocol.Nebula.WasmStateEncoding.Image.fieldValue_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Image.fieldValue_injective

/-- info: 'Nightstream.Protocol.Nebula.WasmStatement.terminal_output_fields_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms terminal_output_fields_exact

/-- info: 'Nightstream.Protocol.Nebula.WasmStatement.ResultImage.Decodes.mode_exit_and_flags_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ResultImage.Decodes.mode_exit_and_flags_exact

/-- info: 'Nightstream.Protocol.Nebula.WasmStatement.completed_execution_derives_terminal' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms completed_execution_derives_terminal

/-- info: 'Nightstream.Protocol.Nebula.Digest.lanes_injective' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Digest.lanes_injective

/-- info: 'Nightstream.Protocol.Nebula.ConcreteLaneGeometry.blockWidth_exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms blockWidth_exact

/-- info: 'Nightstream.Protocol.Nebula.ConcreteLaneGeometry.aligned_add' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms aligned_add

/-- info: 'Nightstream.Protocol.Nebula.MemoryWireGeometry.stepPublicBits_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms stepPublicBits_exact

/-- info: 'Nightstream.Protocol.Nebula.MemoryWireGeometry.mandatoryBundleBits_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms mandatoryBundleBits_exact
