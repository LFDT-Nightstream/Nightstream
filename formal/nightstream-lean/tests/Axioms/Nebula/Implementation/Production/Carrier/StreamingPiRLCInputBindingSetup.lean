import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBindingSetup
import tests.Axioms.Support

/-! Dependency audit for the concrete production PiRLC input binding. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.exact_output_width' does not depend on any axioms -/
#guard_msgs in
#audit_axioms exact_output_width

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.flattenCommitment_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms flattenCommitment_injective

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.equal_concrete_binding_recovers_inputs_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms equal_concrete_binding_recovers_inputs_or_failure

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concretePhaseBindings_sum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms concretePhaseBindings_sum

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.honest_concreteCompleteResidualRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms honest_concreteCompleteResidualRun

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBindingSetup.concrete_complete_zero_recovers_inputs_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms concrete_complete_zero_recovers_inputs_or_failure
