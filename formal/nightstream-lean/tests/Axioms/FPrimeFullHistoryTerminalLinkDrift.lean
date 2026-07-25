import tests.FPrimeFullHistoryTerminalLinkDrift
import tests.Axioms.Support

/-!
Fail-closed guards for the full-history terminal-link drift obstruction.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift.generatedSnapshot_rowCount_eq_logicalWidth' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift.generatedSnapshot_rowCount_eq_logicalWidth

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift.currentPlainOwner_rowCount_eq_logicalPlusPadding' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift.currentPlainOwner_rowCount_eq_logicalPlusPadding

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift.generatedSnapshot_ne_currentPlainOwner' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift.generatedSnapshot_ne_currentPlainOwner

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift.generatedSnapshot_missingPlainPaddingRows' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift.generatedSnapshot_missingPlainPaddingRows

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift.generatedSnapshotRows_ne_currentPlainOwnerRows' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkDrift.generatedSnapshotRows_ne_currentPlainOwnerRows
