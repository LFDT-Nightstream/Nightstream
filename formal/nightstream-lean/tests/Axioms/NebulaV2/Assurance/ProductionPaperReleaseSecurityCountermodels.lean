import Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurityCountermodels
import tests.Axioms.Support

set_option autoImplicit false

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurityCountermodels.every_deterministic_event_requires_closure' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurityCountermodels.every_deterministic_event_requires_closure

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurityCountermodels.every_computational_event_is_covered' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurityCountermodels.every_computational_event_is_covered

/-- info: 'Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurityCountermodels.every_computational_event_can_have_probability_one' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.NebulaV2.ProductionPaperReleaseSecurityCountermodels.every_computational_event_can_have_probability_one
