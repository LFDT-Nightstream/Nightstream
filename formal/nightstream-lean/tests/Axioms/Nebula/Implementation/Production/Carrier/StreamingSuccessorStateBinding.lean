import Nightstream.Implementation.Nebula.Production.Carrier.StreamingSuccessorStateBinding
import tests.Axioms.Support

/-! Dependency audit for bounded-chunk successor-state binding. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming.scheduled_preCarryState_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming.scheduled_preCarryState_eq

/-- info: 'Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming.scheduled_outputState_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming.scheduled_outputState_eq

/-- info: 'Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming.production_prefix_chunk_count_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming.production_prefix_chunk_count_exact

/-- info: 'Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming.production_state_chunk_count_exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionSuccessorStateStreaming.production_state_chunk_count_exact
