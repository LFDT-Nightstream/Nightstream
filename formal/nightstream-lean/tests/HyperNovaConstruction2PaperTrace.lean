import Nightstream.HyperNova.Construction2.PaperTrace

/-! Regression surface for the exact global Construction-2 paper trace. -/

set_option autoImplicit false

namespace tests.HyperNovaConstruction2PaperTrace

open Nightstream.HyperNova.Construction2.PaperTrace

#check Invocation.classified
#check Invocation.isBase_of_iteration_zero
#check Invocation.isRecursive_of_iteration_positive
#check Adjacent.hashPreimage_eq
#check Adjacent.iff_hashPreimage_eq
#check Adjacent.nextFreshPublic_eq_currentOutput
#check Run.exactBranchSchedule
#check Run.last_iteration_add_one
#check ClosedRun.trailingFreshPublic_eq_lastOutput
#check TerminalEndpoint.iff_hashPreimage_eq
#check ClosedRun.terminal_iteration_eq_invocationCount
#check ClosedRun.bottom_rejected
#check HonestTerminalData.terminalTransition
#check HonestTerminalData.close

end tests.HyperNovaConstruction2PaperTrace
