import Nightstream.Implementation.Nebula.Production.Carrier.StreamingStateBinding
import tests.Axioms.Support

/-! Dependency audit for bounded-chunk full-claim state binding. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.ChunkSchedule.exact_cover_and_bound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.ChunkSchedule.exact_cover_and_bound

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.ChunkSchedule.production_chunk_count_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.ChunkSchedule.production_chunk_count_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.scheduled_streamedBindingState_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.scheduled_streamedBindingState_eq

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.ready_replay_recovers_frame_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.ready_replay_recovers_frame_or_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.scheduledReady_squeeze_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.scheduledReady_squeeze_eq

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.equal_streamed_states_recovers_claim_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.equal_streamed_states_recovers_claim_or_named_failure

/-- info: 'Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.authoritativeFrame_length_r26' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionFullClaimStreaming.authoritativeFrame_length_r26
