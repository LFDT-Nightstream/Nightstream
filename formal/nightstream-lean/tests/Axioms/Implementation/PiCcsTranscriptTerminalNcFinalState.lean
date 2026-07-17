import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FinalState
import tests.Axioms.Support

/-! Fail-closed dependency gate for final terminal-NC state refinement. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalState.finalSqueeze_outputColumn' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalState.finalSqueeze_outputColumn

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalState.retainedLanes_of_accepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalState.retainedLanes_of_accepted

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalState.boundary_of_accepted' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FinalState.boundary_of_accepted
