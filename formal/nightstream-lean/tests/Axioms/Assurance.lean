import Nightstream.Assurance
import tests.Axioms.Support

/-!
Fail-closed composed assurance axioms gate. Every expectation is checked when this
module is built; the aggregate entrypoint imports all ownership groups.
-/

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.output_binding' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.output_binding

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.local_sound_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.local_sound_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.step_holds_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.step_holds_or_badRoot

/-- info: 'Nightstream.Assurance.FPrimeTrace.accepted_trace_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeTrace.accepted_trace_sound

/-- info: 'Nightstream.Assurance.FPrimeTrace.accepted_trace_valid_execution' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeTrace.accepted_trace_valid_execution

/-- info: 'Nightstream.Assurance.FPrimeCircuit.split_check_eq_true_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeCircuit.split_check_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursive_artifact_sound_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.recursive_artifact_sound_or_badRoot

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.terminal_artifact_sound_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.terminal_artifact_sound_or_badRoot

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursiveCheck_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.recursiveCheck_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.terminalCheck_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.terminalCheck_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursiveNativeCheck_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.recursiveNativeCheck_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.terminalNativeCheck_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.terminalNativeCheck_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics_nifsVerify' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics_nifsVerify

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursive_verify_sound_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.recursive_verify_sound_or_badRoot

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursive_rows_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.recursive_rows_complete

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.terminal_rows_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.terminal_rows_complete

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursiveContextCheck_eq_true_iff' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.recursiveContextCheck_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.RecursiveSemanticAccepted.parentShapeAgrees' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.RecursiveSemanticAccepted.parentShapeAgrees

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.TerminalSemanticAccepted.parentShapeAgrees' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.TerminalSemanticAccepted.parentShapeAgrees

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.RecursiveSemanticAccepted.parentSerializes' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.RecursiveSemanticAccepted.parentSerializes

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.TerminalSemanticAccepted.parentSerializes' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.TerminalSemanticAccepted.parentSerializes

/-- info: 'Nightstream.Assurance.FPrimeFullHistorySemantics.recursiveCoreLaws_of_start' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeFullHistorySemantics.recursiveCoreLaws_of_start

/-- info: 'Nightstream.Assurance.FPrimeFullHistorySemantics.recursiveCoreLaws' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeFullHistorySemantics.recursiveCoreLaws

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursive_rows_nifsVerify_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeConcreteNifs.recursive_rows_nifsVerify_or_badRoot

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_sound_or_bad' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_sound_or_bad

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_complete

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_execution_sound_or_bad' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_execution_sound_or_bad
