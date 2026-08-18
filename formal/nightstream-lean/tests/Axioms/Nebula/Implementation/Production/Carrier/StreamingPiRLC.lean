import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLC
import tests.Axioms.Support

/-! Dependency audit for bounded-family PiRLC refinement. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.sourceSchedule_nodup' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.sourceSchedule_nodup

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.inputFrame_length' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.inputFrame_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.run_familySchedule_binding' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.run_familySchedule_binding

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.local_rows_imply_combineOne' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.local_rows_imply_combineOne

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.typedOutput_exact' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlc.typedOutput_exact

/-! The exact field and work counts are closed arithmetic. -/
