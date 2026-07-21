import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census

/-!
Kernel composition of the bounded raw-running decoder certificates.

Assurance tier: artifact-checked for the generated fixed profile.

Owns: exact composition of the fifteen independently checked 252-record
shards; unique shard ownership for every one of the `14 * 270` logical
coordinates; exact generated source-arm and final-column formulas; bounds;
and injective physical-column ownership.

Does not own: assignment values, source-row satisfaction, semantic raw-child
authority, combined-NC acceptance, transcript scheduling, commitment binding,
or permission to remove rows.

Emits constraints: none; proof-only artifact composition.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.artifact.exact` | compose shard exactness and prove unique physical-column ownership | checked artifact |

The only executable proof below checks interval separation for `14 * 14 =
196` child pairs. Its data are the fourteen proof-free coordinate-zero base
records already covered by the bounded shard census. It does not traverse a
global 3,780-record list.
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

namespace Exact

/-- Every generated shard satisfies the same exact record predicate. The
case split is structural over `Fin 15`; no artifact data are recomputed. -/
theorem generatedChunkExact (chunk : Chunk) :
    Census.ExactRecords chunk Generated.Chunk0.finalColumnCount
      (Generated.chunkRecords chunk) := by
  rcases chunk with ⟨value, bound⟩
  have concreteBound : value < 15 := by
    simpa [chunkCount] using bound
  have cases :
      value = 0 ∨ value = 1 ∨ value = 2 ∨ value = 3 ∨ value = 4 ∨
      value = 5 ∨ value = 6 ∨ value = 7 ∨ value = 8 ∨ value = 9 ∨
      value = 10 ∨ value = 11 ∨ value = 12 ∨ value = 13 ∨ value = 14 := by
    omega
  rcases cases with h | h | h | h | h | h | h | h | h | h | h | h | h | h | h
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount] using Census.Chunk0.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk1.finalColumnCount]
      using Census.Chunk1.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk2.finalColumnCount]
      using Census.Chunk2.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk3.finalColumnCount]
      using Census.Chunk3.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk4.finalColumnCount]
      using Census.Chunk4.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk5.finalColumnCount]
      using Census.Chunk5.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk6.finalColumnCount]
      using Census.Chunk6.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk7.finalColumnCount]
      using Census.Chunk7.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk8.finalColumnCount]
      using Census.Chunk8.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk9.finalColumnCount]
      using Census.Chunk9.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk10.finalColumnCount]
      using Census.Chunk10.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk11.finalColumnCount]
      using Census.Chunk11.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk12.finalColumnCount]
      using Census.Chunk12.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk13.finalColumnCount]
      using Census.Chunk13.exact.records
  · subst value
    simpa [Generated.chunkRecords, Generated.chunks,
      Generated.Chunk0.finalColumnCount, Generated.Chunk14.finalColumnCount]
      using Census.Chunk14.exact.records

/-- Exact generated record at an arbitrary global ordinal. -/
theorem allocationAtOrdinal_exact (ordinal : Ordinal) :
    let record := Generated.allocationAtOrdinal ordinal
    record.child = (ordinalChild ordinal).val /\
      record.logicalColumn = (ordinalColumn ordinal).val /\
      record.sourceArmColumn =
        Generated.sourceArmBase (ordinalChild ordinal) +
          Generated.packedOffset (ordinalColumn ordinal) /\
      record.finalColumn =
        Generated.finalBase (ordinalChild ordinal) +
          Generated.packedOffset (ordinalColumn ordinal) /\
      record.finalColumn < Generated.Chunk0.finalColumnCount := by
  have row := (generatedChunkExact (ordinalChunk ordinal)).2
    (ordinalOffset ordinal)
  simpa [Generated.allocationAtOrdinal] using row

/-- Exact generated record at one typed child/logical coordinate. -/
theorem allocationAt_exact (child : Child) (column : LogicalColumn) :
    let record := Generated.allocationAt child column
    record.child = child.val /\
      record.logicalColumn = column.val /\
      record.sourceArmColumn =
        Generated.sourceArmBase child + Generated.packedOffset column /\
      record.finalColumn =
        Generated.finalBase child + Generated.packedOffset column /\
      record.finalColumn < Generated.Chunk0.finalColumnCount := by
  simpa [Generated.allocationAt] using
    allocationAtOrdinal_exact (coordinateOrdinal child column)

@[simp] theorem allocationAt_child (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).child = child.val :=
  (allocationAt_exact child column).1

@[simp] theorem allocationAt_logicalColumn
    (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).logicalColumn = column.val :=
  (allocationAt_exact child column).2.1

@[simp] theorem allocationAt_sourceArmColumn
    (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).sourceArmColumn =
      Generated.sourceArmBase child + Generated.packedOffset column :=
  (allocationAt_exact child column).2.2.1

@[simp] theorem allocationAt_finalColumn
    (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).finalColumn =
      Generated.finalBase child + Generated.packedOffset column :=
  (allocationAt_exact child column).2.2.2.1

theorem allocationAt_finalColumn_lt
    (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).finalColumn <
      Generated.Chunk0.finalColumnCount :=
  (allocationAt_exact child column).2.2.2.2

/-- The structural shard address of a logical coordinate exists uniquely.
Together with `generatedChunkExact`, this is exact record ownership rather
than merely interval counting. -/
theorem coordinate_unique_chunkOwner
    (child : Child) (column : LogicalColumn) :
    ∃ address : Chunk × ChunkOffset,
      chunkOrdinal address.1 address.2 = coordinateOrdinal child column /\
        ∀ other : Chunk × ChunkOffset,
          chunkOrdinal other.1 other.2 = coordinateOrdinal child column ->
            other = address := by
  let owner : Chunk × ChunkOffset :=
    (ordinalChunk (coordinateOrdinal child column),
      ordinalOffset (coordinateOrdinal child column))
  refine ⟨owner, chunkOrdinal_ordinalChunk_ordinalOffset _, ?_⟩
  intro other otherOwns
  apply chunkOrdinal_injective
  exact otherOwns.trans (chunkOrdinal_ordinalChunk_ordinalOffset _).symm

/-- Lane-major physical offsets stay inside the exact 270-coordinate block. -/
theorem packedOffset_lt (column : LogicalColumn) :
    Generated.packedOffset column < logicalColumnCount := by
  have bound := column.isLt
  simp only [Generated.packedOffset, packedLaneCount, liveBlockCount,
    logicalColumnCount] at *
  omega

/-- Rust's lane-major storage permutation is injective on all 270 logical
coordinates. -/
theorem packedOffset_injective : Function.Injective Generated.packedOffset := by
  intro left right equal
  apply Fin.ext
  have leftBound := left.isLt
  have rightBound := right.isLt
  simp only [Generated.packedOffset, packedLaneCount, liveBlockCount,
    logicalColumnCount] at equal leftBound rightBound ⊢
  omega

/-- Concrete base intervals for distinct children are disjoint in both the
source-arm and final selectively lowered assignments.

The executable input is exactly 196 proof-free child pairs over fourteen
coordinate-zero base records. -/
def BaseIntervalsSeparated : Prop :=
  forall left right : Child, left.val < right.val ->
    Generated.sourceArmBase left + logicalColumnCount <=
        Generated.sourceArmBase right /\
      Generated.finalBase left + logicalColumnCount <=
        Generated.finalBase right

theorem baseIntervalsSeparated : BaseIntervalsSeparated := by
  unfold BaseIntervalsSeparated
  native_decide

private theorem basePlusPackedOffset_injective
    (base : Child -> Nat)
    (separated : forall left right : Child, left.val < right.val ->
      base left + logicalColumnCount <= base right) :
    Function.Injective fun address : Child × LogicalColumn =>
      base address.1 + Generated.packedOffset address.2 := by
  intro left right equal
  change base left.1 + Generated.packedOffset left.2 =
    base right.1 + Generated.packedOffset right.2 at equal
  by_cases childEqual : left.1 = right.1
  · rw [childEqual] at equal
    exact Prod.ext childEqual (packedOffset_injective (Nat.add_left_cancel equal))
  · have childValueNotEqual : left.1.val ≠ right.1.val := by
      intro valuesEqual
      exact childEqual (Fin.ext valuesEqual)
    rcases Nat.lt_or_gt_of_ne childValueNotEqual with leftBefore | rightBefore
    · have intervals := separated left.1 right.1 leftBefore
      have leftOffset := packedOffset_lt left.2
      omega
    · have intervals := separated right.1 left.1 rightBefore
      have rightOffset := packedOffset_lt right.2
      omega

theorem sourceArmFormula_injective :
    Function.Injective fun address : Child × LogicalColumn =>
      Generated.sourceArmBase address.1 +
        Generated.packedOffset address.2 :=
  basePlusPackedOffset_injective Generated.sourceArmBase
    (fun left right before =>
      (baseIntervalsSeparated left right before).1)

theorem finalFormula_injective :
    Function.Injective fun address : Child × LogicalColumn =>
      Generated.finalBase address.1 + Generated.packedOffset address.2 :=
  basePlusPackedOffset_injective Generated.finalBase
    (fun left right before =>
      (baseIntervalsSeparated left right before).2)

/-- Every generated source-arm physical column has exactly one logical owner. -/
theorem sourceArmColumn_injective :
    Function.Injective fun address : Child × LogicalColumn =>
      (Generated.allocationAt address.1 address.2).sourceArmColumn := by
  intro left right equal
  apply sourceArmFormula_injective
  simpa only [allocationAt_sourceArmColumn] using equal

/-- Every generated final physical column has exactly one logical owner. -/
theorem finalColumn_injective :
    Function.Injective fun address : Child × LogicalColumn =>
      (Generated.allocationAt address.1 address.2).finalColumn := by
  intro left right equal
  apply finalFormula_injective
  simpa only [allocationAt_finalColumn] using equal

/-- The generated allocation record projects exactly to the canonical
correspondence record for the generated final-column map. -/
theorem sourceRecord_eq_recordAt (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).sourceRecord =
      Generated.sourceColumnMap.recordAt child column := by
  simp [AllocationRecord.sourceRecord, SourceColumnMap.recordAt,
    Generated.sourceColumnMap]

/-- The concrete final-column map itself has unique coordinate ownership. -/
theorem sourceColumnMap_injective :
    Function.Injective fun address : Child × LogicalColumn =>
      Generated.sourceColumnMap.sourceColumn address.1 address.2 := by
  simpa only [Generated.sourceColumnMap_apply] using finalColumn_injective

end Exact

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
