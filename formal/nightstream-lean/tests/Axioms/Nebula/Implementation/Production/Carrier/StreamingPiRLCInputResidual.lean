import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputResidual
import tests.Axioms.Support

/-! Dependency audit for the production PiRLC input residual. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.familyMaskedWitness_sum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms familyMaskedWitness_sum

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.familyBindings_sum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms familyBindings_sum

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.phaseWitness_eq_familyMaskedWitness' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms phaseWitness_eq_familyMaskedWitness

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.phaseBindings_sum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms phaseBindings_sum

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.honest_completeResidualRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms honest_completeResidualRun

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual.complete_zero_residual_recovers_inputs_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms complete_zero_residual_recovers_inputs_or_failure
