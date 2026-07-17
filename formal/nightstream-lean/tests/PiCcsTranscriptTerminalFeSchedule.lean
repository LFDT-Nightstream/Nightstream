import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Fe.Schedule

/-! Focused checks for the legacy terminal FE protocol tree. -/

open Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Fe

#check Schedule.phaseIndices_eq_ownerRange
#check Schedule.familyCounts
#check Schedule.prologueCall_payload
#check Schedule.firstMessageCall_payload
#check Schedule.laterMessageCall_payload
