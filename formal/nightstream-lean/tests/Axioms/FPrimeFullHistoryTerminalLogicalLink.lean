import tests.FPrimeFullHistoryTerminalLogicalLink
import tests.Axioms.Support

/-!
Fail-closed guards for the full-history terminal logical-prefix refinement.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLogicalLinkSound.logicalCheck_of_holds' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLogicalLinkSound.logicalCheck_of_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLogicalLinkSound.logicalCheck_of_rows' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLogicalLinkSound.logicalCheck_of_rows
