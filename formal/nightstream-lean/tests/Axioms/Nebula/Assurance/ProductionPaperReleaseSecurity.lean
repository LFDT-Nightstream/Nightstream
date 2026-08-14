import Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity
import tests.Axioms.Support

set_option autoImplicit false

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.anyBad_implies_computationalAny' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.anyBad_implies_computationalAny

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.computationalAny_implies_anyBad' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.computationalAny_implies_anyBad

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.computationalAny_probability_le_total' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.computationalAny_probability_le_total

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.falseAcceptance_implies_computationalAny' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.falseAcceptance_implies_computationalAny

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.falseAcceptance_probability_lt_target96' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.falseAcceptance_probability_lt_target96

/-- info: 'Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.generated_falseAcceptance_probability_lt_target96' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Assurance.Nebula.ProductionPaperReleaseSecurity.generated_falseAcceptance_probability_lt_target96
