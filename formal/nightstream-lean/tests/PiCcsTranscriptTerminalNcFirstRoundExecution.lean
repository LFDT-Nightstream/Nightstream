import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Execution

/-! Focused checks for terminal-NC round-zero execution. -/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc

#check FirstRound.Execution.MessageBound
#check FirstRound.Execution.IncomingBound
#check FirstRound.Execution.PermutationBound
#check FirstRound.Execution.firstBoundary_eq_callInput
#check FirstRound.Execution.secondBoundary_eq_callInput
#check FirstRound.Execution.thirdBoundary_eq_callInput
#check FirstRound.Execution.squeezeBoundary_eq_callInput
#check FirstRound.Execution.permutationBound_of_runRound
#check FirstRound.Execution.successor_eq_callOutputState
#check FirstRound.Execution.incomingBound_of_prologue
