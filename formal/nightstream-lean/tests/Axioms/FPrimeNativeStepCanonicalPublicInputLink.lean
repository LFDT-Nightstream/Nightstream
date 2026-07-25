import tests.FPrimeNativeStepCanonicalPublicInputLink
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guard for the typed canonical fresh-public link.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLink.equalityFactorization' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLink.equalityFactorization
