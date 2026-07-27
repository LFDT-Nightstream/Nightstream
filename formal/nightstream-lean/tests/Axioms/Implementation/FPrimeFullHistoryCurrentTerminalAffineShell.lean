import tests.FPrimeFullHistoryCurrentTerminalAffineShell
import tests.Axioms.Support

/-!
Fail-closed guards for the coefficient-exact bounded current terminal affine
shell.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalAffineShellSound.Captured.rows_iff_holds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalAffineShellSound.Captured.rows_iff_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalAffineShellSound.Captured.priorLatest_iff_currentPlacementRows' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalAffineShellSound.Captured.priorLatest_iff_currentPlacementRows
