import Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime
import tests.Axioms.Support

/-! Dependency audit for the candidate-specific global delayed chain. -/

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime.SegmentRun.exactClaimCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime.SegmentRun.exactClaimCount

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime.Chain.exactSuffixCount' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime.Chain.exactSuffixCount

/-- info: 'Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime.Chain.completeDelayedSchedule' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.Nebula.ProductionBatchedGlobalFPrime.Chain.completeDelayedSchedule
