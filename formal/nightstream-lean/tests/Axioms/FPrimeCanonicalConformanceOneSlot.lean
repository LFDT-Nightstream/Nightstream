import Nightstream.Implementation.Rust.CanonicalConformance.OneSlot
import tests.Axioms.Support

/-! Fail-closed dependency guard for the one-slot canonical conformance schema. -/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.stepAgrees_eq_true_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.stepAgrees_eq_true_iff

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.terminalAgrees_eq_true_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.terminalAgrees_eq_true_iff
