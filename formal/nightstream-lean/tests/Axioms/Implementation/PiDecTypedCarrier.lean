import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecTypedCarrier
import tests.Axioms.Support

/-! Fail-closed dependency gate for the generic active `PiDEC` carrier. -/

/-- info: 'Nightstream.Implementation.R1CS.PiDecTypedCarrier.accepted_refines_paper' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecTypedCarrier.accepted_refines_paper

/-- info: 'Nightstream.Implementation.R1CS.PiDecTypedCarrier.Active.matrixCount_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecTypedCarrier.Active.matrixCount_exact

/-- info: 'Nightstream.Implementation.R1CS.PiDecTypedCarrier.Active.rowVariables_exact' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecTypedCarrier.Active.rowVariables_exact
