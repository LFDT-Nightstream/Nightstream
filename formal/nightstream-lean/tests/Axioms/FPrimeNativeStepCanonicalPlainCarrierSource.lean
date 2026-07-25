import tests.FPrimeNativeStepCanonicalPlainCarrierSource
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guard for the source-shaped plain link.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSource.sourceCheck_reduces_to_logicalPaperLink' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSource.sourceCheck_reduces_to_logicalPaperLink

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSource.sourceBatchCheck_reduces_to_logicalPaperLink' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSource.sourceBatchCheck_reduces_to_logicalPaperLink
