import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk1
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk2
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk3
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk4
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk5
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk6
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk7
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk8
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk9
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk10
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk11
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk12
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk13
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Generated.Chunk14

/-!
Executable view of the generated raw-running decoder shards.

Assurance tier: generated artifact data only.

Owns: bounded selection of one of the fifteen generated 252-record shards,
ordinal lookup without flattening the complete artifact, the exact
block/lane storage offset, and the generated final-column map consumed by the
generic correspondence contract.

Does not own: truth of the generated records, source semantics, R1CS
satisfaction, protocol acceptance, transcript scheduling, or commitment
authority. Those facts are established in `Census` and later refinement
leaves.

Emits constraints: none; generated-data view only.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.artifact.data` | expose bounded shard lookup and generated physical-column maps | computed artifact |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

namespace Generated

/-- Fifteen references to generated chunks. This is not a flattened
3,780-record list. -/
def chunks : Array (List AllocationRecord) := #[
  Chunk0.allocationRecords,
  Chunk1.allocationRecords,
  Chunk2.allocationRecords,
  Chunk3.allocationRecords,
  Chunk4.allocationRecords,
  Chunk5.allocationRecords,
  Chunk6.allocationRecords,
  Chunk7.allocationRecords,
  Chunk8.allocationRecords,
  Chunk9.allocationRecords,
  Chunk10.allocationRecords,
  Chunk11.allocationRecords,
  Chunk12.allocationRecords,
  Chunk13.allocationRecords,
  Chunk14.allocationRecords
]

@[simp] theorem chunks_size : chunks.size = chunkCount := by
  decide

/-- Select one bounded generated shard. -/
def chunkRecords (chunk : Chunk) : List AllocationRecord :=
  chunks[chunk.val]'(by simpa using chunk.isLt)

/-- One generated allocation record, selected through the structural shard
bijection. `getD` is made exact by the per-shard census. -/
def allocationAtOrdinal (ordinal : Ordinal) : AllocationRecord :=
  (chunkRecords (ordinalChunk ordinal)).getD
    (ordinalOffset ordinal).val default

/-- Generated record at one child-major logical coordinate. -/
def allocationAt (child : Child) (column : LogicalColumn) : AllocationRecord :=
  allocationAtOrdinal (coordinateOrdinal child column)

/-- Exact lane-major `Mat` offset within one `54 × 5` packed assignment. -/
def packedOffset (column : LogicalColumn) : Nat :=
  column.val % packedLaneCount * liveBlockCount +
    column.val / packedLaneCount

/-- Source-arm base allocation for one child, derived from generated data. -/
def sourceArmBase (child : Child) : Nat :=
  (allocationAt child ⟨0, by decide⟩).sourceArmColumn

/-- Final selective base allocation for one child, derived from generated
data. -/
def finalBase (child : Child) : Nat :=
  (allocationAt child ⟨0, by decide⟩).finalColumn

/-- Concrete generated final-column map. -/
def sourceColumnMap : SourceColumnMap where
  sourceColumn child column := (allocationAt child column).finalColumn

@[simp] theorem sourceColumnMap_apply (child : Child)
    (column : LogicalColumn) :
    sourceColumnMap.sourceColumn child column =
      (allocationAt child column).finalColumn := by
  rfl

end Generated

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
