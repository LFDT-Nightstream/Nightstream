import tests.FPrimeTerminalLink
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guards for the complete 270-row terminal link.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.fPrimeTerminalLink_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.fPrimeTerminalLink_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.fPrimeTerminalLink_complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.fPrimeTerminalLink_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.satisfies_iff_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.satisfies_iff_holds
