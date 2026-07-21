import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Generated.Chunk0
import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder.Generated.Chunk1

/-!
Exact bounded certificate for the steady-recursive fresh public-`X` decoder.

Assurance tier: artifact-checked for column and disposition provenance only.

Owns: two generated proof-free shards of exactly 256 and 14 records; their
complete ordered coordinate partition; the exact consecutive normalized
source-column schedule; unique logical ownership; and final-range validity of
every selective disposition.

Does not own: source-column field values, binding rows, a complete fresh
witness `Z`, commitment authority, `InputBound`, or row removal.

Emits constraints: none.

Every `native_decide` below consumes only one compact projection of a single
generated shard: at most 256 `(Nat × Nat)` pairs or 256 `Bool` flags. No proof
field and no 270-record joined list is evaluated.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `pi_ccs.nc.fresh_x.columns` | two shards cover coordinates `0..270` in exact source-column order | artifact-checked | `chunk0_coordinateColumns_exact`, `chunk1_coordinateColumns_exact` |
| `pi_ccs.nc.fresh_x.ownership` | each normalized source column has one logical owner | derived | `sourceColumn_has_uniqueLogicalOwner` |
| `pi_ccs.nc.fresh_x.disposition` | every selective disposition is structurally in range | artifact-checked | `records_all_wellFormed` |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder

namespace Exact

/-- Both generated shards describe the same exact artifact profile. -/
theorem metadata_exact :
    Generated.Chunk0.schemaVersion = schemaVersion /\
      Generated.Chunk1.schemaVersion = schemaVersion /\
      Generated.Chunk0.sourceArm = sourceArm /\
      Generated.Chunk1.sourceArm = sourceArm /\
      Generated.Chunk0.sourceCount = sourceCount /\
      Generated.Chunk1.sourceCount = sourceCount /\
      Generated.Chunk0.logicalColumnCount = logicalColumnCount /\
      Generated.Chunk1.logicalColumnCount = logicalColumnCount /\
      Generated.Chunk0.finalColumnCount = Generated.Chunk1.finalColumnCount := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

@[simp] theorem finalColumnCount_eq :
    Generated.Chunk0.finalColumnCount = Generated.Chunk1.finalColumnCount := by
  rfl

/-- Compact coordinate/source-column projection of the 256-record shard. -/
def chunk0CoordinateColumns : List (Nat × Nat) :=
  Generated.Chunk0.records.map fun record =>
    (record.logicalColumn, record.sourceArmColumn)

/-- Compact coordinate/source-column projection of the 14-record shard. -/
def chunk1CoordinateColumns : List (Nat × Nat) :=
  Generated.Chunk1.records.map fun record =>
    (record.logicalColumn, record.sourceArmColumn)

/-- The source-arm base is read from the first generated record rather than
duplicated as handwritten artifact data. -/
def sourceArmBase : Nat :=
  (Generated.Chunk0.records.map SourceColumnRecord.sourceArmColumn).head?.getD 0

def expectedChunk0CoordinateColumns : List (Nat × Nat) :=
  (List.range 256).map fun offset =>
    (offset, sourceArmBase + offset)

def expectedChunk1CoordinateColumns : List (Nat × Nat) :=
  (List.range 14).map fun offset =>
    (256 + offset, sourceArmBase + 256 + offset)

/-- `native_decide` compares exactly 256 proof-free `(Nat × Nat)` records. -/
theorem chunk0_coordinateColumns_exact :
    chunk0CoordinateColumns = expectedChunk0CoordinateColumns := by
  native_decide

/-- `native_decide` compares exactly 14 proof-free `(Nat × Nat)` records. -/
theorem chunk1_coordinateColumns_exact :
    chunk1CoordinateColumns = expectedChunk1CoordinateColumns := by
  native_decide

/-- The two exact shard sizes cover all 270 public coordinates. -/
theorem chunk_lengths_and_coverage :
    Generated.Chunk0.records.length = 256 /\
      Generated.Chunk1.records.length = 14 /\
      256 + 14 = logicalColumnCount := by
  constructor
  · have lengths := congrArg List.length chunk0_coordinateColumns_exact
    simpa [chunk0CoordinateColumns, expectedChunk0CoordinateColumns] using
      lengths
  constructor
  · have lengths := congrArg List.length chunk1_coordinateColumns_exact
    simpa [chunk1CoordinateColumns, expectedChunk1CoordinateColumns] using
      lengths
  · rfl

/-- Compact range-validity flags for the 256-record shard. -/
def chunk0RangeValidity : List Bool :=
  Generated.Chunk0.records.map fun record =>
    decide (record.resolution.RangeValid Generated.Chunk0.finalColumnCount)

/-- Compact range-validity flags for the 14-record shard. -/
def chunk1RangeValidity : List Bool :=
  Generated.Chunk1.records.map fun record =>
    decide (record.resolution.RangeValid Generated.Chunk1.finalColumnCount)

/-- `native_decide` evaluates exactly 256 proof-free `Bool` flags. -/
theorem chunk0_rangeValidity_checked :
    chunk0RangeValidity.all (fun valid => valid) = true := by
  native_decide

/-- `native_decide` evaluates exactly 14 proof-free `Bool` flags. -/
theorem chunk1_rangeValidity_checked :
    chunk1RangeValidity.all (fun valid => valid) = true := by
  native_decide

private theorem rangeValid_of_checked
    (records : List SourceColumnRecord)
    (finalColumnCount : Nat)
    (checked :
      (records.map fun record =>
        decide (record.resolution.RangeValid finalColumnCount)).all
          (fun valid => valid) = true) :
    forall record, record ∈ records ->
      record.resolution.RangeValid finalColumnCount := by
  intro record member
  have flagMember :
      decide (record.resolution.RangeValid finalColumnCount) ∈
        records.map fun current =>
          decide (current.resolution.RangeValid finalColumnCount) :=
    List.mem_map.mpr ⟨record, member, rfl⟩
  have flagTrue := (List.all_eq_true.mp checked) _ flagMember
  exact of_decide_eq_true flagTrue

/-- Every disposition in the first generated shard has a valid final range. -/
theorem chunk0_all_rangeValid :
    forall record, record ∈ Generated.Chunk0.records ->
      record.resolution.RangeValid Generated.Chunk0.finalColumnCount := by
  exact rangeValid_of_checked Generated.Chunk0.records Generated.Chunk0.finalColumnCount
    chunk0_rangeValidity_checked

/-- Every disposition in the second generated shard has a valid final range. -/
theorem chunk1_all_rangeValid :
    forall record, record ∈ Generated.Chunk1.records ->
      record.resolution.RangeValid Generated.Chunk1.finalColumnCount := by
  exact rangeValid_of_checked Generated.Chunk1.records Generated.Chunk1.finalColumnCount
    chunk1_rangeValidity_checked

private theorem chunk0_record_formula
    (record : SourceColumnRecord)
    (member : record ∈ Generated.Chunk0.records) :
    record.sourceArmColumn = sourceArmBase + record.logicalColumn /\
      record.logicalColumn < 256 := by
  have pairMember :
      (record.logicalColumn, record.sourceArmColumn) ∈
        chunk0CoordinateColumns :=
    List.mem_map.mpr ⟨record, member, rfl⟩
  rw [chunk0_coordinateColumns_exact] at pairMember
  rcases List.mem_map.mp pairMember with ⟨offset, offsetMember, equal⟩
  have offsetBound := List.mem_range.mp offsetMember
  have logicalEqual : offset = record.logicalColumn :=
    congrArg Prod.fst equal
  have sourceEqual : sourceArmBase + offset = record.sourceArmColumn :=
    congrArg Prod.snd equal
  omega

private theorem chunk1_record_formula
    (record : SourceColumnRecord)
    (member : record ∈ Generated.Chunk1.records) :
    record.sourceArmColumn = sourceArmBase + record.logicalColumn /\
      256 <= record.logicalColumn /\
      record.logicalColumn < logicalColumnCount := by
  have pairMember :
      (record.logicalColumn, record.sourceArmColumn) ∈
        chunk1CoordinateColumns :=
    List.mem_map.mpr ⟨record, member, rfl⟩
  rw [chunk1_coordinateColumns_exact] at pairMember
  rcases List.mem_map.mp pairMember with ⟨offset, offsetMember, equal⟩
  have offsetBound := List.mem_range.mp offsetMember
  have logicalEqual : 256 + offset = record.logicalColumn :=
    congrArg Prod.fst equal
  have sourceEqual : sourceArmBase + 256 + offset =
      record.sourceArmColumn :=
    congrArg Prod.snd equal
  simp only [logicalColumnCount] at ⊢
  omega

/-- Every generated record is in exact coordinate order and uses the source
column `sourceArmBase + logicalColumn`. -/
theorem record_formula
    (record : SourceColumnRecord)
    (member : record ∈ Generated.Chunk0.records ++ Generated.Chunk1.records) :
    record.sourceArmColumn = sourceArmBase + record.logicalColumn /\
      record.logicalColumn < logicalColumnCount := by
  rcases List.mem_append.mp member with first | second
  · have formula := chunk0_record_formula record first
    exact ⟨formula.1, Nat.lt_trans formula.2 (by decide)⟩
  · have formula := chunk1_record_formula record second
    exact ⟨formula.1, formula.2.2⟩

/-- All generated normalized source columns have one logical owner. -/
theorem sourceColumn_has_uniqueLogicalOwner
    (left right : SourceColumnRecord)
    (leftMember : left ∈ Generated.Chunk0.records ++ Generated.Chunk1.records)
    (rightMember : right ∈ Generated.Chunk0.records ++ Generated.Chunk1.records)
    (sameSource : left.sourceArmColumn = right.sourceArmColumn) :
    left.logicalColumn = right.logicalColumn := by
  have leftFormula := (record_formula left leftMember).1
  have rightFormula := (record_formula right rightMember).1
  omega

/-- Typed source-column lookup exposed to correspondence clients. -/
def sourceColumn (column : LogicalColumn) : Nat :=
  sourceArmBase + column.val

/-- Exact unique coordinate ownership of the typed lookup. -/
theorem sourceColumn_injective : Function.Injective sourceColumn := by
  intro left right equal
  apply Fin.ext
  exact Nat.add_left_cancel equal

/-- Every generated record is structurally well formed. This combines its
ordered coordinate certificate with its independently checked disposition
range. -/
theorem records_all_wellFormed :
    forall record, record ∈ Generated.Chunk0.records ++ Generated.Chunk1.records ->
      record.WellFormed Generated.Chunk0.finalColumnCount := by
  intro record member
  refine ⟨(record_formula record member).2, ?_⟩
  rcases List.mem_append.mp member with first | second
  · exact chunk0_all_rangeValid record first
  · rw [finalColumnCount_eq]
    exact chunk1_all_rangeValid record second

end Exact

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.FreshSourceDecoder
