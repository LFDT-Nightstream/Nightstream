import tests.FPrimeProductionFreshPublicSingletonBridge
import tests.Axioms.Support

/-!
Fail-closed guards for the paper-singleton production fresh-public bridge.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ProductionFreshPublicSingletonBridge.freshPublic_eq_encodeInstance_iff_sourceCheck' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ProductionFreshPublicSingletonBridge.freshPublic_eq_encodeInstance_iff_sourceCheck

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ProductionFreshPublicSingletonBridge.freshPublic_eq_encodeInstance_iff_program' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ProductionFreshPublicSingletonBridge.freshPublic_eq_encodeInstance_iff_program

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ProductionFreshPublicSingletonBridge.freshPublic_eq_encodeInstance_reduces_to_logicalPaperLink' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ProductionFreshPublicSingletonBridge.freshPublic_eq_encodeInstance_reduces_to_logicalPaperLink
