import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.Conformance
import tests.Axioms.Support

/-! Fail-closed dependency guard for native-step differential conformance. -/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.nativeVerifyStep_eq_ok_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.nativeVerifyStep_eq_ok_iff

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.controlFlowAndCallConservationCheck_eq_true_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.controlFlowAndCallConservationCheck_eq_true_iff

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.nativeAccepted_with_boundaries_iff_localHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.nativeAccepted_with_boundaries_iff_localHolds

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.generated_all_check' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.generated_all_check

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.oracleReplayConforms_of_mem_generated' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.oracleReplayConforms_of_mem_generated

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.controlFlowAndCallConservation_of_mem_generated' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.controlFlowAndCallConservation_of_mem_generated

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.nativeOutcome_eq_recorded_of_mem_generated' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.nativeOutcome_eq_recorded_of_mem_generated
