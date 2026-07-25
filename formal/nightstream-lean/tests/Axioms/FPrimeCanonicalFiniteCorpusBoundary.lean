import Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.FiniteCorpusBoundary
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guards for the finite-corpus refinement
obstructions.
-/

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.FiniteCorpusBoundary.Step.not_attemptedUniversalBridge' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.FiniteCorpusBoundary.Step.not_attemptedUniversalBridge

/-- info: 'Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.FiniteCorpusBoundary.Terminal.not_attemptedUniversalBridge' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.CanonicalConformance.OneSlot.FiniteCorpusBoundary.Terminal.not_attemptedUniversalBridge
