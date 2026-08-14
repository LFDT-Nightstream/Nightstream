import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX

/-!
Exact compiler check for the production strict-PiDEC canonical-X artifact.

Assurance tier: Rust-conformant for this generated row leaf.

Owns: an independent Lean compiler for the binary public-X recomposition,
shared-sign, and child-digit rows; exact row, owner, relative-index, and
physical-index comparison for all 4,590 records exported from Rust.

Does not own: the remaining strict-PiDEC rows, final selective lowering,
whole-recursive conformance, witness satisfaction, or cryptographic
soundness.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.CanonicalX

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX

private abbrev logicalCoordinates : Nat := Generated.Metadata.logicalCoordinates
private abbrev childCount : Nat := Generated.Metadata.childCount
private abbrev rowsPerCoordinate : Nat := childCount + 2

/-- Exact little-endian binary recomposition emitted by the Rust compiler. -/
def recompositionRow (coordinate : CoordinateColumns) : Row :=
  let powers := (List.range coordinate.children.length).map (fun exponent => 2 ^ exponent)
  ⟨(coordinate.parent, 1) ::
      (coordinate.children.zip powers).map
        (fun pair => (pair.1, goldilocksP - pair.2)),
    [(0, 1)], []⟩

/-- The two rows that constrain one shared sign to `{-1, 0, 1}`. -/
def signProductRow (coordinate : CoordinateColumns) : Row :=
  ⟨[(0, 1), (coordinate.sign, 1)],
    [(coordinate.sign, 1)], [(coordinate.product, 1)]⟩

def signZeroRow (coordinate : CoordinateColumns) : Row :=
  ⟨[(coordinate.product, 1)],
    [(0, goldilocksP - 1), (coordinate.sign, 1)], []⟩

/-- One binary child digit is zero or the shared sign. -/
def childDigitRow (coordinate : CoordinateColumns) (child : Nat) : Option Row := do
  let digit ← coordinate.children[child]?
  pure ⟨[(digit, 1)],
    [(digit, 1), (coordinate.sign, goldilocksP - 1)], []⟩

def expectedRow : RowOwner → Option Row
  | .recomposition activeIndex =>
      (coordinates[activeIndex]?).map recompositionRow
  | .signProduct activeIndex =>
      (coordinates[activeIndex]?).map signProductRow
  | .signZero activeIndex =>
      (coordinates[activeIndex]?).map signZeroRow
  | .childDigit activeIndex child => do
      let coordinate ← coordinates[activeIndex]?
      childDigitRow coordinate child
  | .radixFourLimb .. | .radixFourReconstruction .. => none

def expectedRelativeIndex : RowOwner → Nat
  | .recomposition activeIndex => activeIndex
  | .signProduct activeIndex =>
      logicalCoordinates + activeIndex * rowsPerCoordinate
  | .signZero activeIndex =>
      logicalCoordinates + activeIndex * rowsPerCoordinate + 1
  | .childDigit activeIndex child =>
      logicalCoordinates + activeIndex * rowsPerCoordinate + 2 + child
  | .radixFourLimb .. | .radixFourReconstruction .. => 0

def expectedPhysicalIndex (owner : RowOwner) : Nat :=
  match owner with
  | .recomposition activeIndex =>
      Generated.Metadata.recompositionRowStart + activeIndex
  | _ =>
      Generated.Metadata.canonicalityRowStart +
        (expectedRelativeIndex owner - logicalCoordinates)

/-- Fail-closed comparison of one generated record with the independent
compiler and the two Rust receipt schedules. -/
def recordMatches (record : PhysicalRow) : Bool :=
  match expectedRow record.owner with
  | none => false
  | some row =>
      decide (record.relativeIndex = expectedRelativeIndex record.owner) &&
      decide (record.physicalIndex = expectedPhysicalIndex record.owner) &&
      decide (record.row = row)

/-- Compare a shard in exact aggregate order. -/
def recordsMatchFrom : Nat → List PhysicalRow → Bool
  | _, [] => true
  | expectedIndex, record :: tail =>
      decide (record.relativeIndex = expectedIndex) &&
        recordMatches record &&
        recordsMatchFrom (expectedIndex + 1) tail

def ChunkMatches (start count : Nat) (records : List PhysicalRow) : Prop :=
  records.length = count ∧ recordsMatchFrom start records = true

/-- Bounded kernel package for every generated shard. Each native decision
examines at most 240 proof-free records. The starts and lengths partition
`0 .. 4,590` without a gap or overlap. -/
structure GeneratedRowsMatch : Prop where
  chunk0 : ChunkMatches 0 100 Generated.Rows.Chunk0.values
  chunk1 : ChunkMatches 100 100 Generated.Rows.Chunk1.values
  chunk2 : ChunkMatches 200 70 Generated.Rows.Chunk2.values
  chunk3 : ChunkMatches 270 240 Generated.Rows.Chunk3.values
  chunk4 : ChunkMatches 510 240 Generated.Rows.Chunk4.values
  chunk5 : ChunkMatches 750 240 Generated.Rows.Chunk5.values
  chunk6 : ChunkMatches 990 240 Generated.Rows.Chunk6.values
  chunk7 : ChunkMatches 1230 240 Generated.Rows.Chunk7.values
  chunk8 : ChunkMatches 1470 240 Generated.Rows.Chunk8.values
  chunk9 : ChunkMatches 1710 240 Generated.Rows.Chunk9.values
  chunk10 : ChunkMatches 1950 240 Generated.Rows.Chunk10.values
  chunk11 : ChunkMatches 2190 240 Generated.Rows.Chunk11.values
  chunk12 : ChunkMatches 2430 240 Generated.Rows.Chunk12.values
  chunk13 : ChunkMatches 2670 240 Generated.Rows.Chunk13.values
  chunk14 : ChunkMatches 2910 240 Generated.Rows.Chunk14.values
  chunk15 : ChunkMatches 3150 240 Generated.Rows.Chunk15.values
  chunk16 : ChunkMatches 3390 240 Generated.Rows.Chunk16.values
  chunk17 : ChunkMatches 3630 240 Generated.Rows.Chunk17.values
  chunk18 : ChunkMatches 3870 240 Generated.Rows.Chunk18.values
  chunk19 : ChunkMatches 4110 240 Generated.Rows.Chunk19.values
  chunk20 : ChunkMatches 4350 240 Generated.Rows.Chunk20.values

/-- Every exact Rust-exported canonical-X row equals the independent Lean
compiler output at the same relative and physical position. -/
theorem generated_rows_match_independent_compiler : GeneratedRowsMatch := by
  constructor <;> (unfold ChunkMatches; native_decide)

/-- The checked shard partition covers the complete generated row count. -/
theorem checked_partition_count :
    100 + 100 + 70 + 18 * 240 = Generated.Metadata.rowCount := by
  decide

end Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.CanonicalX
