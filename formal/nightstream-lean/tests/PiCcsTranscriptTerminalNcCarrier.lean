import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.Carrier

/-! Focused checks for the indexed terminal-NC certificate decoder. -/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc

#check Carrier.domain_roundCount
#check Carrier.coefficientColumns_eq
#check Carrier.decodedRound
#check Carrier.RoundBound
#check Carrier.Bound
#check Carrier.typedRound_eq_decodedRound
#check Carrier.rawRounds_eq_typed
#check Carrier.rawRounds_eq_decoded
