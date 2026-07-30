import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionBridge
import tests.Axioms.Support

set_option autoImplicit false

namespace NightstreamTests.Axioms.ConcreteNifsPiRlcActionBridge

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RingKBaseActionCoordinates.combineEvaluations_getD_low' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RingKBaseActionCoordinates.combineEvaluations_getD_low

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.RingKBaseActionCoordinates.combineEvaluations_getD_high' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RingKBaseActionCoordinates.combineEvaluations_getD_high

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionBridge.commitment_eq_derived' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcActionBridge.commitment_eq_derived

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionBridge.publicInput_eq_derived' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcActionBridge.publicInput_eq_derived

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionBridge.evaluation_getD_eq_derived' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcActionBridge.evaluation_getD_eq_derived

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionBridge.evaluations_eq_derived' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcActionBridge.evaluations_eq_derived

/-- info: 'Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionBridge.equations_of_result' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteNifsPiRlcActionBridge.equations_of_result

end NightstreamTests.Axioms.ConcreteNifsPiRlcActionBridge
