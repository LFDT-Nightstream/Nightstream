import tests.FPrimeFullHistoryCurrentTerminalLinkCompletion
import tests.Axioms.Support

/-!
Fail-closed guards for constructive completion of the current plain
terminal-link owner from the captured full-history snapshot.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkCompletion.currentRows_of_snapshotLinkRows' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkCompletion.currentRows_of_snapshotLinkRows

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkCompletion.completedAssignment_producerAligned' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkCompletion.completedAssignment_producerAligned

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkCompletion.output_and_snapshot_rows_construct_currentPlainOwner' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkCompletion.output_and_snapshot_rows_construct_currentPlainOwner
