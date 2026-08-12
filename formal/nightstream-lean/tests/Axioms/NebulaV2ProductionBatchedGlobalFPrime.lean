import Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime
import tests.Axioms.Support

/-! Dependency audit for the candidate-specific global delayed chain. -/

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime.SegmentRun.exactClaimCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime.SegmentRun.exactClaimCount

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime.Chain.exactSuffixCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime.Chain.exactSuffixCount

/-- info: 'Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime.Chain.completeDelayedSchedule' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.NebulaV2.ProductionBatchedGlobalFPrime.Chain.completeDelayedSchedule
