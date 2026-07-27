import tests.Axioms.Support
import tests.FPrimeProductionOwnerProgramBoundary

/-!
Fail-closed guards for the current production owner-program stop boundary.
-/

/-- info: 'Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.AlignmentOpacity.not_attemptedOwnerAlignmentBridge' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.AlignmentOpacity.not_attemptedOwnerAlignmentBridge

/-- info: 'Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.CurrentArtifact.currentTerminalLink_starts_after_historicalProgram' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.CurrentArtifact.currentTerminalLink_starts_after_historicalProgram

/-- info: 'Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.CurrentArtifact.currentTerminalLink_not_in_historicalProgram' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.CurrentArtifact.currentTerminalLink_not_in_historicalProgram

/-- info: 'Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.CurrentProfile.diagnosticMatrixCount_ne_activeProduction' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.CurrentProfile.diagnosticMatrixCount_ne_activeProduction

/-- info: 'Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.CurrentProfile.diagnosticMatrixCount_eq_three' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.CurrentProfile.diagnosticMatrixCount_eq_three

/-- info: 'Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.CurrentProfile.activeProductionMatrixCount_eq_thirteen' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.CurrentProfile.activeProductionMatrixCount_eq_thirteen

/-- info: 'Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.TerminalSelectionOpacity.acceptingSelection_accepts' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.TerminalSelectionOpacity.acceptingSelection_accepts

/-- info: 'Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.TerminalSelectionOpacity.rejectingSelection_rejects' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.TerminalSelectionOpacity.rejectingSelection_rejects

/-- info: 'Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.TerminalSelectionOpacity.terminalFacts_do_not_select_relationChecks' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeProductionOwnerProgramBoundary.TerminalSelectionOpacity.terminalFacts_do_not_select_relationChecks
