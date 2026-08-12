import Nightstream.Assurance.NebulaV2
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2.ConcreteField
open Nightstream.Assurance.NebulaV2.FingerprintSecurity
open Nightstream.Assurance.NebulaV2.SecurityBudget

/-- info: 'Nightstream.Assurance.NebulaV2.AjtaiBinding.signed_unit_collision_to_kernel' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.AjtaiBinding.signed_unit_collision_to_kernel

/-- info: 'Nightstream.Assurance.NebulaV2.AjtaiBinding.primary_failure_to_kernel' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.AjtaiBinding.primary_failure_to_kernel

/-- info: 'Nightstream.Assurance.NebulaV2.AjtaiBinding.exact_primary_failure_to_kernel' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.AjtaiBinding.exact_primary_failure_to_kernel

/-- info: 'Nightstream.Assurance.NebulaV2.AjtaiBinding.exact_short_failure_to_kernel' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.AjtaiBinding.exact_short_failure_to_kernel

/-- info: 'Nightstream.Assurance.NebulaV2.AjtaiBinding.bundle_failure_to_full_kernel' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.AjtaiBinding.bundle_failure_to_full_kernel

/-- info: 'Nightstream.Assurance.NebulaV2.SeededSetupSecurity.HybridAssumption.total_lt_post_union' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.SeededSetupSecurity.HybridAssumption.total_lt_post_union

/-- info: 'Nightstream.Implementation.NebulaV2.ConcreteField.seven_not_square' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms seven_not_square

/-- info: 'Nightstream.Implementation.NebulaV2.ConcreteField.encode_injective_below_goldilocks' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms encode_injective_below_goldilocks

/-- info: 'Nightstream.Implementation.NebulaV2.ConcreteField.challengeField_cardinality' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms challengeField_cardinality

/-- info: 'Nightstream.Implementation.NebulaV2.ConcreteField.superNeoEquiv_mul' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms superNeoEquiv_mul

/-- info: 'Nightstream.Assurance.NebulaV2.FingerprintSecurity.unbalanced_check_probability_le_profile' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms unbalanced_check_probability_le_profile

/-- info: 'Nightstream.Assurance.NebulaV2.FingerprintSecurity.planning_fingerprint_bits_at_least_186' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms planning_fingerprint_bits_at_least_186

/-- info: 'Nightstream.Assurance.NebulaV2.FingerprintSecurity.planning_fingerprint_bits_not_187' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms planning_fingerprint_bits_not_187

/-- info: 'Nightstream.Assurance.NebulaV2.IdealTranscript.derive_repeatedPoint_eq_tableEquiv_at' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.IdealTranscript.derive_repeatedPoint_eq_tableEquiv_at

/-- info: 'Nightstream.Assurance.NebulaV2.TranscriptSecurity.uniformTableProbability_accepts_eq_repeatedProbability' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.TranscriptSecurity.uniformTableProbability_accepts_eq_repeatedProbability

/-- info: 'Nightstream.Assurance.NebulaV2.TranscriptSecurity.actual_event_probability_le_uniform_add_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.TranscriptSecurity.actual_event_probability_le_uniform_add_failure

/-- info: 'Nightstream.Assurance.NebulaV2.TranscriptSecurity.actual_fingerprint_probability_le_profile_add_transcript' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.TranscriptSecurity.actual_fingerprint_probability_le_profile_add_transcript

/-- info: 'Nightstream.Assurance.NebulaV2.PrechallengeKnowledge.close_binds_extracted_sequence_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.PrechallengeKnowledge.close_binds_extracted_sequence_or_named_failure

/-- info: 'Nightstream.Assurance.NebulaV2.CompactSequenceSecurity.classify_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.CompactSequenceSecurity.classify_failure

/-- info: 'Nightstream.Assurance.NebulaV2.CompactSequenceSecurity.compact_acceptance_implies_execution_or_release_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.CompactSequenceSecurity.compact_acceptance_implies_execution_or_release_failure

/-- info: 'Nightstream.Assurance.NebulaV2.ReleasePipeline.deployed_acceptance_implies_execution_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ReleasePipeline.deployed_acceptance_implies_execution_or_named_failure

/-- info: 'Nightstream.Assurance.NebulaV2.WasmReleasePipeline.deployed_acceptance_implies_fixed_wasm_execution_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.WasmReleasePipeline.deployed_acceptance_implies_fixed_wasm_execution_or_named_failure

/-- info: 'Nightstream.Assurance.NebulaV2.SecurityBudget.total_lt_target96' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms total_lt_target96
