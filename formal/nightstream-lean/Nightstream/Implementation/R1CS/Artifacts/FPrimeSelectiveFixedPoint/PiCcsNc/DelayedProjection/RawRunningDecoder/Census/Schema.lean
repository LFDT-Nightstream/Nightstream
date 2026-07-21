import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Data

/-!
Bounded executable certificate schema for one generated raw-running decoder
shard.

`ExactChunk` checks one 252-element proof-free `List AllocationRecord` in
place. It indexes the list by `Fin 252` and computes the expected coordinate
and packed allocation formula per offset; it never constructs or compares a
second record list.

Assurance tier: checked artifact schema. Concrete generated facts live in the
`Census.Chunk*` leaves.

Owns: the bounded exactness predicate for one generated 252-record,
proof-free raw-running decoder shard and its scalar metadata.

Does not own: any concrete shard certificate, global record flattening,
assignment semantics, R1CS satisfaction, protocol acceptance, transcript
scheduling, or commitment authority.

Emits constraints: none; proof-only certificate schema.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.artifact.census.schema` | specify one exact bounded 252-record shard certificate | checked schema |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

/-- Exact bounded record payload for one generated shard.

The allocation equations retain the actual lane-major source layout instead
of assuming logical-column monotonicity. Both base columns are themselves
read from generated coordinate zero for the owning child. -/
def ExactRecords
    (chunk : Chunk) (finalColumnCount : Nat)
    (records : List AllocationRecord) : Prop :=
  records.length = chunkLength /\
    forall offset : ChunkOffset,
      let ordinal := chunkOrdinal chunk offset
      let child := ordinalChild ordinal
      let column := ordinalColumn ordinal
      let record := records.getD offset.val default
      record.child = child.val /\
        record.logicalColumn = column.val /\
        record.sourceArmColumn =
          Generated.sourceArmBase child + Generated.packedOffset column /\
        record.finalColumn =
          Generated.finalBase child + Generated.packedOffset column /\
        record.finalColumn < finalColumnCount

/-- Exact bounded certificate for one generated shard, including common
profile metadata. -/
def ExactChunk
    (chunk : Chunk)
    (schemaVersion sourceArm artifactChildCount artifactLogicalColumnCount
      finalColumnCount : Nat)
    (records : List AllocationRecord) : Prop :=
  schemaVersion = 1 /\
    sourceArm = 2 /\
    artifactChildCount = childCount /\
    artifactLogicalColumnCount = logicalColumnCount /\
    finalColumnCount = Generated.Chunk0.finalColumnCount /\
    ExactRecords chunk finalColumnCount records

namespace ExactChunk

theorem length
    {chunk : Chunk}
    {schemaVersion sourceArm artifactChildCount artifactLogicalColumnCount
      finalColumnCount : Nat}
    {records : List AllocationRecord}
    (exact : ExactChunk chunk schemaVersion sourceArm artifactChildCount
      artifactLogicalColumnCount finalColumnCount records) :
    records.length = chunkLength :=
  exact.2.2.2.2.2.1

theorem records
    {chunk : Chunk}
    {schemaVersion sourceArm artifactChildCount artifactLogicalColumnCount
      finalColumnCount : Nat}
    {records : List AllocationRecord}
    (exact : ExactChunk chunk schemaVersion sourceArm artifactChildCount
      artifactLogicalColumnCount finalColumnCount records) :
    ExactRecords chunk finalColumnCount records :=
  exact.2.2.2.2.2

theorem recordAt
    {chunk : Chunk}
    {schemaVersion sourceArm artifactChildCount artifactLogicalColumnCount
      finalColumnCount : Nat}
    {records : List AllocationRecord}
    (exact : ExactChunk chunk schemaVersion sourceArm artifactChildCount
      artifactLogicalColumnCount finalColumnCount records)
    (offset : ChunkOffset) :
    let ordinal := chunkOrdinal chunk offset
    let child := ordinalChild ordinal
    let column := ordinalColumn ordinal
    let record := records.getD offset.val default
    record.child = child.val /\
      record.logicalColumn = column.val /\
      record.sourceArmColumn =
        Generated.sourceArmBase child + Generated.packedOffset column /\
      record.finalColumn =
        Generated.finalBase child + Generated.packedOffset column /\
      record.finalColumn < finalColumnCount :=
  exact.records.2 offset

end ExactChunk

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census
