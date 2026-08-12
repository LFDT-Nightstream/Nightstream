import Nightstream.Implementation.NebulaV2.Memory.Segment.SourceRows
import Nightstream.Implementation.NebulaV2.FPrime.Manifest.RecursiveSchema
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryRecordFactorRows.update_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryRecordFactorRows.update_sound

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductChainRows.final_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductChainRows.final_sound

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows.rows_length_exact' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductUpdateRows.rows_length_exact

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows.operationChain_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductUpdateRows.operationChain_sound

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductUpdateRows.snapshotChain_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductUpdateRows.snapshotChain_sound

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductSemanticBridge.final_eq_foldOptionsK' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductSemanticBridge.final_eq_foldOptionsK

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductSemanticBridge.superNeoEquiv_recordFactorK' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductSemanticBridge.superNeoEquiv_recordFactorK

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductSemanticBridge.superNeoEquiv_foldOptionsK' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductSemanticBridge.superNeoEquiv_foldOptionsK

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductClaimBridge.operation_claim_product_update' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductClaimBridge.operation_claim_product_update

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryProductClaimBridge.snapshot_claim_product_update' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryProductClaimBridge.snapshot_claim_product_update

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate.claim_product_update' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryClaimProductUpdate.claim_product_update

/-- info: 'Nightstream.Implementation.NebulaV2.OperationPrefixRows.operation_source_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms OperationPrefixRows.operation_source_refines

/-- info: 'Nightstream.Implementation.NebulaV2.SnapshotSlotRows.sound' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#audit_axioms SnapshotSlotRows.sound

/-- info: 'Nightstream.Implementation.NebulaV2.SnapshotChunkRows.snapshot_source_refines' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SnapshotChunkRows.snapshot_source_refines

/-- info: 'Nightstream.Implementation.NebulaV2.MemorySourceRows.checked_step_product_update' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemorySourceRows.checked_step_product_update

/-- info: 'Nightstream.Implementation.NebulaV2.RecursiveManifestSchema.Artifact.memoryCheckedStep_product_update' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms RecursiveManifestSchema.Artifact.memoryCheckedStep_product_update
