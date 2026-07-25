import tests.FPrimeEncodingCanonicalBits
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guard for canonical output-bit recovery.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeEncodingCanonicalBits.publicBit_eq_encodedBit' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeEncodingCanonicalBits.publicBit_eq_encodedBit
