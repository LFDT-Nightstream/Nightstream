import tests.FPrimeNativeStepCanonicalPlainCarrierLink
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guard for the typed plain carrier link.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink.equalityFactorization' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink.equalityFactorization

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink.rawCheck_reduces_to_typedCarrier' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink.rawCheck_reduces_to_typedCarrier

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink.check_reduces_to_logicalPaperLink' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierLink.check_reduces_to_logicalPaperLink
