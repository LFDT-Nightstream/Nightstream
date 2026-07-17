import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Connectivity
import tests.Axioms.Support

/-! Fail-closed dependency gate for round-zero to round-one connectivity. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Connectivity.squeezeOutputBase_eq_laterColumnBase' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Connectivity.squeezeOutputBase_eq_laterColumnBase

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Connectivity.incomingBound_of_callOutputState' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Connectivity.incomingBound_of_callOutputState

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Connectivity.nextIncoming_of_prologueAndCarrier' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.FirstRound.Connectivity.nextIncoming_of_prologueAndCarrier
