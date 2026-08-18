import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCAuthority
import tests.Axioms.Support

/-! Dependency audit for authoritative production PiRLC replay. -/

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.complete_family_run_eq_parent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.complete_family_run_eq_parent

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.inputFrame_eq_canonical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.inputFrame_eq_canonical

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.fused_phase_recovers_inputs_or_collision' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.fused_phase_recovers_inputs_or_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.familyStart_of_transition' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.familyStart_of_transition

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.familyStateFields_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.familyStateFields_length

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.familyPhase_uses_authoritative_challenges' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.familyPhase_uses_authoritative_challenges

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.familyPhase_recovers_authoritative_inputs_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.familyPhase_recovers_authoritative_inputs_or_collision

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.local_rows_imply_concrete_phase' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.local_rows_imply_concrete_phase

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.local_rows_imply_concrete_phase_from_input_rows' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.local_rows_imply_concrete_phase_from_input_rows

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.sampler_rows_imply_authoritative_start' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority.sampler_rows_imply_authoritative_start
