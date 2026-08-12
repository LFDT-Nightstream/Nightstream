import Nightstream.HyperNova.Construction2.PaperTrace
import tests.Axioms.Support

/-! Fail-closed dependency gate for the exact global Construction-2 trace. -/

/-- info: 'Nightstream.HyperNova.Construction2.PaperTrace.Invocation.classified' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.PaperTrace.Invocation.classified

/-- info: 'Nightstream.HyperNova.Construction2.PaperTrace.Adjacent.hashPreimage_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.PaperTrace.Adjacent.hashPreimage_eq

/-- info: 'Nightstream.HyperNova.Construction2.PaperTrace.Adjacent.iff_hashPreimage_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.PaperTrace.Adjacent.iff_hashPreimage_eq

/-- info: 'Nightstream.HyperNova.Construction2.PaperTrace.Adjacent.nextFreshPublic_eq_currentOutput' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.PaperTrace.Adjacent.nextFreshPublic_eq_currentOutput

/-- info: 'Nightstream.HyperNova.Construction2.PaperTrace.Run.exactBranchSchedule' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.PaperTrace.Run.exactBranchSchedule

/-- info: 'Nightstream.HyperNova.Construction2.PaperTrace.Run.last_iteration_add_one' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.PaperTrace.Run.last_iteration_add_one

/-- info: 'Nightstream.HyperNova.Construction2.PaperTrace.ClosedRun.trailingFreshPublic_eq_lastOutput' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.PaperTrace.ClosedRun.trailingFreshPublic_eq_lastOutput

/-- info: 'Nightstream.HyperNova.Construction2.PaperTrace.TerminalEndpoint.iff_hashPreimage_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.PaperTrace.TerminalEndpoint.iff_hashPreimage_eq

/-- info: 'Nightstream.HyperNova.Construction2.PaperTrace.ClosedRun.bottom_rejected' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.PaperTrace.ClosedRun.bottom_rejected

/-- info: 'Nightstream.HyperNova.Construction2.PaperTrace.HonestTerminalData.terminalTransition' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.PaperTrace.HonestTerminalData.terminalTransition

/-- info: 'Nightstream.HyperNova.Construction2.PaperTrace.HonestTerminalData.close' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.PaperTrace.HonestTerminalData.close
