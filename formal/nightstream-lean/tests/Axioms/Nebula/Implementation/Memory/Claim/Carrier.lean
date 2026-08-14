import Nightstream.Implementation.Nebula.Memory.Claim.BoundCcsPublic
import Nightstream.Implementation.Nebula.Memory.Claim.HashFrameRows
import Nightstream.Implementation.Nebula.Memory.Claim.PoseidonRows
import Nightstream.Implementation.Nebula.FPrime.State.PriorLinkRows
import tests.Axioms.Support

open Nightstream.Implementation.Nebula

/-- info: 'Nightstream.Implementation.Nebula.MemoryBoundCcsPublic.encode_get_stateDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryBoundCcsPublic.encode_get_stateDigest

/-- info: 'Nightstream.Implementation.Nebula.MemoryBoundCcsPublic.encode_get_memoryDigest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryBoundCcsPublic.encode_get_memoryDigest

/-- info: 'Nightstream.Implementation.Nebula.MemoryBoundCcsPublic.authority_eq_or_memory_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryBoundCcsPublic.authority_eq_or_memory_collision

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimHashFrameRows.input_column_values' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryClaimHashFrameRows.input_column_values

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimHashFrameRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryClaimHashFrameRows.rows_complete

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimPoseidonRows.output_columns_eq_digest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryClaimPoseidonRows.output_columns_eq_digest

/-- info: 'Nightstream.Implementation.Nebula.MemoryClaimPoseidonRows.rows_complete' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MemoryClaimPoseidonRows.rows_complete

/-- info: 'Nightstream.Implementation.Nebula.PriorStateLinkRows.claimCcsPublicExact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PriorStateLinkRows.claimCcsPublicExact
