import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBinding
import tests.Axioms.Support

/-! Dependency audit for the complete production PiRLC input binding. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.exact_geometry' does not depend on any axioms -/
#guard_msgs in
#audit_axioms exact_geometry

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.familyAtOrdinal_familyOrdinal' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms familyAtOrdinal_familyOrdinal

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.inputVector_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms inputVector_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.coordinateWitness_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateWitness_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.coordinateWitness_unit_bound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms coordinateWitness_unit_bound

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.equal_input_binding_recovers_inputs_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms equal_input_binding_recovers_inputs_or_failure

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding.exact_source_geometry' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms exact_source_geometry
