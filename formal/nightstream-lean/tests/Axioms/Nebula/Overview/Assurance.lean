import Nightstream.Assurance.Nebula
import tests.Axioms.Support

open Nightstream.Implementation.Nebula.ConcreteField
open Nightstream.Assurance.Nebula.FingerprintSecurity
open Nightstream.Assurance.Nebula.SecurityBudget

/-- info: 'Nightstream.Assurance.Nebula.AjtaiBinding.signed_unit_collision_to_kernel' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.AjtaiBinding.signed_unit_collision_to_kernel

/-- info: 'Nightstream.Assurance.Nebula.AjtaiBinding.primary_failure_to_kernel' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.AjtaiBinding.primary_failure_to_kernel

/-- info: 'Nightstream.Assurance.Nebula.AjtaiBinding.exact_primary_failure_to_kernel' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.AjtaiBinding.exact_primary_failure_to_kernel

/-- info: 'Nightstream.Assurance.Nebula.AjtaiBinding.exact_short_failure_to_kernel' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.AjtaiBinding.exact_short_failure_to_kernel

/-- info: 'Nightstream.Assurance.Nebula.AjtaiBinding.bundle_failure_to_full_kernel' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.AjtaiBinding.bundle_failure_to_full_kernel

/-- info: 'Nightstream.Assurance.Nebula.SeededSetupSecurity.HybridAssumption.total_lt_post_union' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.SeededSetupSecurity.HybridAssumption.total_lt_post_union

/-- info: 'Nightstream.Implementation.Nebula.ConcreteField.seven_not_square' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms seven_not_square

/-- info: 'Nightstream.Implementation.Nebula.ConcreteField.encode_injective_below_goldilocks' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms encode_injective_below_goldilocks

/-- info: 'Nightstream.Implementation.Nebula.ConcreteField.challengeField_cardinality' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms challengeField_cardinality

/-- info: 'Nightstream.Implementation.Nebula.ConcreteField.superNeoEquiv_mul' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms superNeoEquiv_mul

/-- info: 'Nightstream.Assurance.Nebula.FingerprintSecurity.unbalanced_check_probability_le_profile' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms unbalanced_check_probability_le_profile

/-- info: 'Nightstream.Assurance.Nebula.FingerprintSecurity.planning_fingerprint_bits_at_least_186' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms planning_fingerprint_bits_at_least_186

/-- info: 'Nightstream.Assurance.Nebula.FingerprintSecurity.planning_fingerprint_bits_not_187' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms planning_fingerprint_bits_not_187

/-- info: 'Nightstream.Assurance.Nebula.IdealTranscript.derive_repeatedPoint_eq_tableEquiv_at' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.IdealTranscript.derive_repeatedPoint_eq_tableEquiv_at

/-- info: 'Nightstream.Assurance.Nebula.TranscriptSecurity.uniformTableProbability_accepts_eq_repeatedProbability' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.TranscriptSecurity.uniformTableProbability_accepts_eq_repeatedProbability

/-- info: 'Nightstream.Assurance.Nebula.TranscriptSecurity.actual_event_probability_le_uniform_add_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.TranscriptSecurity.actual_event_probability_le_uniform_add_failure

/-- info: 'Nightstream.Assurance.Nebula.TranscriptSecurity.actual_fingerprint_probability_le_profile_add_transcript' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.TranscriptSecurity.actual_fingerprint_probability_le_profile_add_transcript

/-- info: 'Nightstream.Assurance.Nebula.PrechallengeKnowledge.close_binds_extracted_sequence_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.PrechallengeKnowledge.close_binds_extracted_sequence_or_named_failure

/-- info: 'Nightstream.Assurance.Nebula.CompactSequenceSecurity.classify_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.CompactSequenceSecurity.classify_failure

/-- info: 'Nightstream.Assurance.Nebula.CompactSequenceSecurity.compact_acceptance_implies_execution_or_release_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.CompactSequenceSecurity.compact_acceptance_implies_execution_or_release_failure

/-- info: 'Nightstream.Assurance.Nebula.ReleasePipeline.deployed_acceptance_implies_execution_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ReleasePipeline.deployed_acceptance_implies_execution_or_named_failure

/-- info: 'Nightstream.Assurance.Nebula.WasmReleasePipeline.deployed_acceptance_implies_fixed_wasm_execution_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.WasmReleasePipeline.deployed_acceptance_implies_fixed_wasm_execution_or_named_failure

/-- info: 'Nightstream.Assurance.Nebula.SecurityBudget.total_lt_target96' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms total_lt_target96
