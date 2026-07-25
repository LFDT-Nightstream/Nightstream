import tests.FPrimeTerminalLinkCanonicalRefinement
import tests.Axioms.Support

/-!
Fail-closed kernel-dependency guards for the terminal-link paper refinement.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement.check_of_satisfies' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement.check_of_satisfies

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement.satisfies_of_check' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement.satisfies_of_check

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement.satisfies_iff_logicalPaperLink' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement.satisfies_iff_logicalPaperLink

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement.producerAligned_of_encodingRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement.producerAligned_of_encodingRows

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement.satisfies_iff_logicalPaperLink_of_encodingRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkCanonicalRefinement.satisfies_iff_logicalPaperLink_of_encodingRows
