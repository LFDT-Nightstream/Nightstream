import Nightstream.Implementation.NebulaV2.Production.Memory.BatchCarrierBridge
import tests.Axioms.Support

/-! Dependency audit for the production full-claim memory carrier bridge. -/

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCarrierBridge.claim_eq_claimAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCarrierBridge.claim_eq_claimAt

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCarrierBridge.suffixBatch_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCarrierBridge.suffixBatch_eq

/-- info: 'Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCarrierBridge.rows_bind_and_consume_full_claim_memory' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.NebulaV2.ProductionMemoryBatchCarrierBridge.rows_bind_and_consume_full_claim_memory
