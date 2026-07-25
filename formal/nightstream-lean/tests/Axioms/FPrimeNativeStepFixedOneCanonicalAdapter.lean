import tests.FPrimeNativeStepFixedOneCanonicalAdapter
import tests.Axioms.Support

/-!
Fail-closed guards for the universal native-step to frozen fixed-one adapter.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.application_eq_some_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.application_eq_some_iff

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.priorHash_eq_computed' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.priorHash_eq_computed

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.nextHash_eq_computed' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.nextHash_eq_computed

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.active_freshPublic_eq_encode_prior_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.active_freshPublic_eq_encode_prior_iff

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.transition_iff_holds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.transition_iff_holds

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.canonicalAccepts_iff_holds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.canonicalAccepts_iff_holds

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.nativeAccepted_with_boundaries_and_outgoing_iff_canonicalAccepts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.nativeAccepted_with_boundaries_and_outgoing_iff_canonicalAccepts

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.checkedRecorded_with_boundaries_and_outgoing_iff_canonicalAccepts' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter.checkedRecorded_with_boundaries_and_outgoing_iff_canonicalAccepts
