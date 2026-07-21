import Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder.Schema

/-!
Bounded shard schema for the production raw-running decoder artifact.

Assurance tier: checked artifact schema; this file contains no generated
columns and makes no semantic claim about them.

Owns: the exact `3,780 = 14 * 270 = 15 * 252` record domain, the child-major
ordinal map, and a bijection between ordinals and fifteen uniformly bounded
252-record shards.

Does not own: generated column values, compiler provenance, assignment
decoding, R1CS satisfaction, protocol acceptance, or commitment authority.
No theorem normalizes a global record list and this module uses no
`native_decide`.

Emits constraints: none; proof-only artifact schema.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_decoder.artifact.ordinal` | child-major coordinates cover exactly 3,780 ordinals | derived |
| `nifs.pi_ccs.nc.delayed.raw_decoder.artifact.shards` | fifteen 252-record shards cover that domain without overlap | derived |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder

/-- Proof-free allocation provenance for one authoritative raw-running
coordinate. `sourceArmColumn` is the normalized steady-arm field column and
`finalColumn` is its exact direct centered width-one slot in the selectively
lowered assignment. -/
structure AllocationRecord where
  child : Nat
  logicalColumn : Nat
  sourceArmColumn : Nat
  finalColumn : Nat
deriving DecidableEq, Repr, Inhabited

namespace AllocationRecord

/-- Semantic decoder view: correspondence clients read the final selective
assignment column, while the artifact retains the intermediate source-arm
owner for auditability. -/
def sourceRecord (record : AllocationRecord) : SourceColumnRecord where
  child := record.child
  logicalColumn := record.logicalColumn
  sourceColumn := record.finalColumn

@[simp] theorem sourceRecord_child (record : AllocationRecord) :
    record.sourceRecord.child = record.child := by
  rfl

@[simp] theorem sourceRecord_logicalColumn (record : AllocationRecord) :
    record.sourceRecord.logicalColumn = record.logicalColumn := by
  rfl

@[simp] theorem sourceRecord_sourceColumn (record : AllocationRecord) :
    record.sourceRecord.sourceColumn = record.finalColumn := by
  rfl

end AllocationRecord

/-- Exact generated record count. -/
def recordCount : Nat := childCount * logicalColumnCount

/-- Uniform generated shard count. -/
def chunkCount : Nat := 15

/-- Every generated shard contains exactly this many proof-free records. -/
def chunkLength : Nat := 252

abbrev Ordinal := Fin recordCount
abbrev Chunk := Fin chunkCount
abbrev ChunkOffset := Fin chunkLength

theorem profile_counts :
    recordCount = 3780 /\
      chunkCount = 15 /\
      chunkLength = 252 /\
      chunkCount * chunkLength = recordCount /\
      chunkLength <= 256 := by
  decide

/-- Child-major ordinal of one exact logical source coordinate. -/
def coordinateOrdinal (child : Child) (column : LogicalColumn) : Ordinal :=
  ⟨child.val * logicalColumnCount + column.val, by
    have childNext : child.val + 1 <= childCount :=
      Nat.succ_le_of_lt child.isLt
    have scaled :
        (child.val + 1) * logicalColumnCount <=
          childCount * logicalColumnCount :=
      Nat.mul_le_mul_right logicalColumnCount childNext
    have belowNext :
        child.val * logicalColumnCount + column.val <
          (child.val + 1) * logicalColumnCount := by
      simpa [Nat.add_mul] using
        Nat.add_lt_add_left column.isLt
          (child.val * logicalColumnCount)
    exact Nat.lt_of_lt_of_le belowNext scaled⟩

/-- Recover the child index from a child-major ordinal. -/
def ordinalChild (ordinal : Ordinal) : Child :=
  ⟨ordinal.val / logicalColumnCount, by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0 < logicalColumnCount)).2
    simpa [recordCount] using ordinal.isLt⟩

/-- Recover the logical coordinate from a child-major ordinal. -/
def ordinalColumn (ordinal : Ordinal) : LogicalColumn :=
  ⟨ordinal.val % logicalColumnCount,
    Nat.mod_lt _ (by decide : 0 < logicalColumnCount)⟩

@[simp] theorem coordinateOrdinal_ordinalChild_ordinalColumn
    (ordinal : Ordinal) :
    coordinateOrdinal (ordinalChild ordinal) (ordinalColumn ordinal) =
      ordinal := by
  apply Fin.ext
  change ordinal.val / logicalColumnCount * logicalColumnCount +
      ordinal.val % logicalColumnCount = ordinal.val
  simpa [Nat.mul_comm] using
    Nat.div_add_mod ordinal.val logicalColumnCount

@[simp] theorem ordinalChild_coordinateOrdinal
    (child : Child) (column : LogicalColumn) :
    ordinalChild (coordinateOrdinal child column) = child := by
  apply Fin.ext
  change
    (child.val * logicalColumnCount + column.val) /
        logicalColumnCount = child.val
  rw [Nat.mul_comm child.val logicalColumnCount,
    Nat.mul_add_div (by decide : 0 < logicalColumnCount),
    Nat.div_eq_of_lt column.isLt, Nat.add_zero]

@[simp] theorem ordinalColumn_coordinateOrdinal
    (child : Child) (column : LogicalColumn) :
    ordinalColumn (coordinateOrdinal child column) = column := by
  apply Fin.ext
  change
    (child.val * logicalColumnCount + column.val) %
        logicalColumnCount = column.val
  simpa [Nat.mod_eq_of_lt column.isLt] using
    Nat.mul_add_mod_self_right child.val logicalColumnCount column.val

/-- Exact ordinal owned by one generated shard and offset. -/
def chunkOrdinal (chunk : Chunk) (offset : ChunkOffset) : Ordinal :=
  ⟨chunk.val * chunkLength + offset.val, by
    have chunkNext : chunk.val + 1 <= chunkCount :=
      Nat.succ_le_of_lt chunk.isLt
    have scaled :
        (chunk.val + 1) * chunkLength <= chunkCount * chunkLength :=
      Nat.mul_le_mul_right chunkLength chunkNext
    have belowNext :
        chunk.val * chunkLength + offset.val <
          (chunk.val + 1) * chunkLength := by
      simpa [Nat.add_mul] using
        Nat.add_lt_add_left offset.isLt (chunk.val * chunkLength)
    simpa [profile_counts.2.2.2.1] using
      Nat.lt_of_lt_of_le belowNext scaled⟩

/-- Shard owner of one exact artifact ordinal. -/
def ordinalChunk (ordinal : Ordinal) : Chunk :=
  ⟨ordinal.val / chunkLength, by
    apply (Nat.div_lt_iff_lt_mul (by decide : 0 < chunkLength)).2
    simpa [profile_counts.2.2.2.1] using ordinal.isLt⟩

/-- In-shard offset of one exact artifact ordinal. -/
def ordinalOffset (ordinal : Ordinal) : ChunkOffset :=
  ⟨ordinal.val % chunkLength,
    Nat.mod_lt _ (by decide : 0 < chunkLength)⟩

@[simp] theorem chunkOrdinal_ordinalChunk_ordinalOffset
    (ordinal : Ordinal) :
    chunkOrdinal (ordinalChunk ordinal) (ordinalOffset ordinal) = ordinal := by
  apply Fin.ext
  change ordinal.val / chunkLength * chunkLength +
      ordinal.val % chunkLength = ordinal.val
  simpa [Nat.mul_comm] using Nat.div_add_mod ordinal.val chunkLength

@[simp] theorem ordinalChunk_chunkOrdinal
    (chunk : Chunk) (offset : ChunkOffset) :
    ordinalChunk (chunkOrdinal chunk offset) = chunk := by
  apply Fin.ext
  change
    (chunk.val * chunkLength + offset.val) / chunkLength = chunk.val
  rw [Nat.mul_comm chunk.val chunkLength,
    Nat.mul_add_div (by decide : 0 < chunkLength),
    Nat.div_eq_of_lt offset.isLt, Nat.add_zero]

@[simp] theorem ordinalOffset_chunkOrdinal
    (chunk : Chunk) (offset : ChunkOffset) :
    ordinalOffset (chunkOrdinal chunk offset) = offset := by
  apply Fin.ext
  change
    (chunk.val * chunkLength + offset.val) % chunkLength = offset.val
  simpa [Nat.mod_eq_of_lt offset.isLt] using
    Nat.mul_add_mod_self_right chunk.val chunkLength offset.val

theorem chunkOrdinal_injective :
    Function.Injective fun address : Chunk × ChunkOffset =>
      chunkOrdinal address.1 address.2 := by
  intro left right equal
  have chunks := congrArg ordinalChunk equal
  have offsets := congrArg ordinalOffset equal
  exact Prod.ext (by simpa using chunks) (by simpa using offsets)

theorem chunkOrdinal_surjective :
    Function.Surjective fun address : Chunk × ChunkOffset =>
      chunkOrdinal address.1 address.2 := by
  intro ordinal
  exact ⟨(ordinalChunk ordinal, ordinalOffset ordinal),
    chunkOrdinal_ordinalChunk_ordinalOffset ordinal⟩

/-- Structural coverage and no-overlap theorem for all generated shards. -/
theorem chunkOrdinal_bijective :
    Function.Injective (fun address : Chunk × ChunkOffset =>
      chunkOrdinal address.1 address.2) /\
      Function.Surjective (fun address : Chunk × ChunkOffset =>
        chunkOrdinal address.1 address.2) :=
  ⟨chunkOrdinal_injective, chunkOrdinal_surjective⟩

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.RawRunningDecoder
