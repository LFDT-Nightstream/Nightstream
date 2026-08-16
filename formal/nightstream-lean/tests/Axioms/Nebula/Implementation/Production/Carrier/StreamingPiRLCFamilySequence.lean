import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilySequence
import tests.Axioms.Support

/-! Dependency audit for the complete production PiRLC family sequence. -/

open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySequence

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySequence.AcceptedRun.concreteCompleteResidualRun' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedRun.concreteCompleteResidualRun

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySequence.AcceptedRun.start_finish_recovers_inputs_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedRun.start_finish_recovers_inputs_or_failure

/-- info: 'Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySequence.AcceptedRun.outputs_exact_or_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms AcceptedRun.outputs_exact_or_failure
