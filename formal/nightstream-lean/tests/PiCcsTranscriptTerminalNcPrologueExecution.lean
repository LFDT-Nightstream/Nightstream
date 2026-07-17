import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Prologue.Execution

/-! Focused checks for terminal-NC prologue execution. -/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc

#check Prologue.Execution.AfterFeBound
#check Prologue.Execution.firstBoundary_eq_callInput
#check Prologue.Execution.secondBoundary_eq_callInput
#check Prologue.Execution.run_eq_roundTagState
