import tests.FPrimeNativeStepCanonicalPlainCarrierSerialization
import tests.Axioms.Support

/-!
Fail-closed guards for lossless production plain-carrier serialization.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.Carrier.coordinates_getD_one' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.Carrier.coordinates_getD_one

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.Carrier.coordinates_getD_body' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.Carrier.coordinates_getD_body

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.Carrier.coordinates_getD_padding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.Carrier.coordinates_getD_padding

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.Carrier.coordinates_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.Carrier.coordinates_injective

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.serializeClaim_encodeClaim' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.serializeClaim_encodeClaim

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.serializeClaim_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.serializeClaim_injective

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.sourceCheck_serializeClaim_iff_check' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization.sourceCheck_serializeClaim_iff_check
