import Nightstream.Implementation.NebulaV2.Production.FPrime.Recursive.SuccessorStateBinding
import tests.Axioms.Support

/-! Dependency audit for the field-native production successor binding. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding.successorFrame_injective' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding.successorFrame_injective

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding.successorFrame_fields_canonical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding.successorFrame_fields_canonical

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding.outputState_replays_stateFrame' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding.outputState_replays_stateFrame

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding.equal_outputDigest_recovers_state_or_named_failure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding.equal_outputDigest_recovers_state_or_named_failure

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding.stateFrames_ne_of_candidate_ne' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionSuccessorStateBinding.stateFrames_ne_of_candidate_ne
