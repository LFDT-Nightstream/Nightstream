import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.RoundExecution

/-! Focused checks for shared terminal FE/NC round serialization. -/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal

#check RoundExecution.absorbAll_four_of_cursorZero
#check RoundExecution.absorbAll_cons_of_full
#check RoundExecution.absorbAll_three_of_cursorOne
#check RoundExecution.appendRaw_ten_of_cursorZero
#check RoundExecution.appendRaw_singletons_of_cursorZero
#check RoundExecution.appendRaw_pair_then_singleton_of_cursorZero
#check RoundExecution.appendRaw_ten_of_cursorOne
#check RoundExecution.fieldAt_eq_wordField
