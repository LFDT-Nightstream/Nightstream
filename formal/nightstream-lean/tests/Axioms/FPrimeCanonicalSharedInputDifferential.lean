import tests.FPrimeCanonicalSharedInputDifferential
import tests.Axioms.Support

/-! Fail-closed dependency guard for the generated shared-input differential. -/

/-- info: 'Nightstream.Tests.FPrimeCanonicalSharedInputDifferential.generated_all_agree' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Tests.FPrimeCanonicalSharedInputDifferential.generated_all_agree

/-- info: 'Nightstream.Tests.FPrimeCanonicalSharedTerminalDifferential.generated_all_agree' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Tests.FPrimeCanonicalSharedTerminalDifferential.generated_all_agree
