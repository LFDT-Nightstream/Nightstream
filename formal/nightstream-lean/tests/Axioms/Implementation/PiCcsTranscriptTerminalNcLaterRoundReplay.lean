import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay
import tests.Axioms.Support

/-! Fail-closed dependency gate for complete typed later-round replay. -/

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay.stateBefore_next' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay.stateBefore_next

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay.incomingBound_all' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay.incomingBound_all

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay.messages_eq_prefix_append_final' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay.messages_eq_prefix_append_final

/-- info: 'Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay.run_eq_finalCallOutput' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiCcsTranscript.Refinement.Terminal.Nc.LaterRound.Replay.run_eq_finalCallOutput
