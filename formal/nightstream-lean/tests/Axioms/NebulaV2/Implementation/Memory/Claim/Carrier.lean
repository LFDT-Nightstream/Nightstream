import Nightstream.Implementation.NebulaV2.Memory.Claim.BoundCcsPublic
import Nightstream.Implementation.NebulaV2.Memory.Claim.HashFrameRows
import Nightstream.Implementation.NebulaV2.Memory.Claim.PoseidonRows
import Nightstream.Implementation.NebulaV2.FPrime.State.PriorLinkRows
import tests.Axioms.Support

open Nightstream.Implementation.NebulaV2

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryBoundCcsPublic.encode_get_stateDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryBoundCcsPublic.encode_get_stateDigest

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryBoundCcsPublic.encode_get_memoryDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryBoundCcsPublic.encode_get_memoryDigest

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryBoundCcsPublic.authority_eq_or_memory_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryBoundCcsPublic.authority_eq_or_memory_collision

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimHashFrameRows.input_column_values' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryClaimHashFrameRows.input_column_values

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimHashFrameRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryClaimHashFrameRows.rows_complete

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimPoseidonRows.output_columns_eq_digest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryClaimPoseidonRows.output_columns_eq_digest

/-- info: 'Nightstream.Implementation.NebulaV2.MemoryClaimPoseidonRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryClaimPoseidonRows.rows_complete

/-- info: 'Nightstream.Implementation.NebulaV2.PriorStateLinkRows.claimCcsPublicExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PriorStateLinkRows.claimCcsPublicExact
