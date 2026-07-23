import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Census

/-!
Kernel composition of the bounded raw-running decoder certificates.

Assurance tier: artifact-checked for the generated fixed profile.

Owns: exact composition of the fifteen independently checked 252-record
shards; unique shard ownership for every one of the `14 * 270` logical
coordinates; exact generated source-arm and encoded-scalar formulas; complete
interval bounds; and nonoverlapping physical allocation ownership.

Does not own: assignment values, source-row satisfaction, semantic raw-child
authority, combined-NC acceptance, transcript scheduling, commitment binding,
or permission to remove rows.

Emits constraints: none; proof-only artifact composition.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.artifact.exact` | compose shard exactness and prove unique physical-allocation ownership | checked artifact |

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
      record.finalStart =
        Generated.finalStartBase (ordinalChild ordinal) +
          balancedTernaryWidth *
            Generated.packedOffset (ordinalColumn ordinal) /\
      record.width = balancedTernaryWidth /\
      record.encoding = .balancedTernary /\
      record.finalStart + record.width <=
        Generated.Chunk0.finalColumnCount := by
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
      record.finalStart =
        Generated.finalStartBase child +
          balancedTernaryWidth * Generated.packedOffset column /\
      record.width = balancedTernaryWidth /\
      record.encoding = .balancedTernary /\
      record.finalStart + record.width <=
        Generated.Chunk0.finalColumnCount := by
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

@[simp] theorem allocationAt_finalStart
    (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).finalStart =
      Generated.finalStartBase child +
        balancedTernaryWidth * Generated.packedOffset column :=
  (allocationAt_exact child column).2.2.2.1

@[simp] theorem allocationAt_width
    (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).width = balancedTernaryWidth :=
  (allocationAt_exact child column).2.2.2.2.1

@[simp] theorem allocationAt_encoding
    (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).encoding = .balancedTernary :=
  (allocationAt_exact child column).2.2.2.2.2.1

theorem allocationAt_interval_le
    (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).finalStart +
        (Generated.allocationAt child column).width <=
      Generated.Chunk0.finalColumnCount :=
  (allocationAt_exact child column).2.2.2.2.2.2

/-- Every generated final interval has the exact shape required by its
decoder and lies completely inside the final assignment. -/
theorem allocationAt_wellFormed
    (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).sourceRecord.allocation.WellFormed
      Generated.Chunk0.finalColumnCount := by
  constructor
  · simp [Encoding.ValidWidth]
  · exact allocationAt_interval_le child column

theorem allocationAt_sourceRecord_allocation
    (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).sourceRecord.allocation = {
      start := Generated.finalStartBase child +
        balancedTernaryWidth * Generated.packedOffset column
      width := balancedTernaryWidth
      encoding := .balancedTernary
    } := by
  simp [AllocationRecord.sourceRecord]

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
      Generated.finalStartBase left +
          balancedTernaryWidth * logicalColumnCount <=
        Generated.finalStartBase right

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

/-- Every generated source-arm physical column has exactly one logical owner. -/
theorem sourceArmColumn_injective :
    Function.Injective fun address : Child × LogicalColumn =>
      (Generated.allocationAt address.1 address.2).sourceArmColumn := by
  intro left right equal
  apply sourceArmFormula_injective
  simpa only [allocationAt_sourceArmColumn] using equal

/-- Distinct logical coordinates own disjoint complete final-assignment
intervals. This is stronger than uniqueness of the first digit column. -/
theorem finalIntervals_nonoverlap
    (left right : Child × LogicalColumn) (different : left ≠ right) :
    let leftRecord := Generated.allocationAt left.1 left.2
    let rightRecord := Generated.allocationAt right.1 right.2
    leftRecord.finalStart + leftRecord.width <= rightRecord.finalStart \/
      rightRecord.finalStart + rightRecord.width <= leftRecord.finalStart := by
  simp only [allocationAt_finalStart, allocationAt_width,
    balancedTernaryWidth]
  by_cases childEqual : left.1 = right.1
  · have columnDifferent : left.2 ≠ right.2 := by
      intro columnEqual
      exact different (Prod.ext childEqual columnEqual)
    have offsetDifferent :
        Generated.packedOffset left.2 ≠ Generated.packedOffset right.2 := by
      intro offsetEqual
      exact columnDifferent (packedOffset_injective offsetEqual)
    rw [childEqual]
    omega
  · have childValueDifferent : left.1.val ≠ right.1.val := by
      intro valuesEqual
      exact childEqual (Fin.ext valuesEqual)
    rcases Nat.lt_or_gt_of_ne childValueDifferent with leftBefore | rightBefore
    · left
      have separated := (baseIntervalsSeparated left.1 right.1 leftBefore).2
      simp only [balancedTernaryWidth] at separated
      have leftOffset := packedOffset_lt left.2
      omega
    · right
      have separated := (baseIntervalsSeparated right.1 left.1 rightBefore).2
      simp only [balancedTernaryWidth] at separated
      have rightOffset := packedOffset_lt right.2
      omega

/-- Complete final allocation starts have unique logical owners. -/
theorem finalStart_injective :
    Function.Injective fun address : Child × LogicalColumn =>
      (Generated.allocationAt address.1 address.2).finalStart := by
  intro left right equal
  by_cases same : left = right
  · exact same
  · have separated := finalIntervals_nonoverlap left right same
    simp only [allocationAt_width, balancedTernaryWidth] at separated
    change (Generated.allocationAt left.1 left.2).finalStart =
      (Generated.allocationAt right.1 right.2).finalStart at equal
    rw [equal] at separated
    omega

/-- The generated allocation record projects exactly to the canonical
correspondence record for the generated encoded-scalar map. -/
theorem sourceRecord_eq_recordAt (child : Child) (column : LogicalColumn) :
    (Generated.allocationAt child column).sourceRecord =
      Generated.sourceAllocationMap.recordAt child column := by
  simp [AllocationRecord.sourceRecord, SourceAllocationMap.recordAt,
    Generated.sourceAllocationMap]

/-- The concrete complete-allocation map itself has unique coordinate
ownership. -/
theorem sourceAllocationMap_injective :
    Function.Injective fun address : Child × LogicalColumn =>
      Generated.sourceAllocationMap.allocation address.1 address.2 := by
  intro left right equal
  apply finalStart_injective
  exact congrArg EncodedScalar.start equal

end Exact

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
