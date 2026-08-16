import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFusedPass
import tests.Axioms.Support

/-! Dependency audit for one fused state-binding and algebra pass. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingFusedPass.run_schedule_exact' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingFusedPass.run_schedule_exact

/-! The carried-state theorem has the same trusted base as the initial-state specialization. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingFusedPass.accepted_run_recovers_fold_or_collision_at' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingFusedPass.accepted_run_recovers_fold_or_collision_at

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingFusedPass.accepted_run_recovers_fold_or_collision' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingFusedPass.accepted_run_recovers_fold_or_collision
