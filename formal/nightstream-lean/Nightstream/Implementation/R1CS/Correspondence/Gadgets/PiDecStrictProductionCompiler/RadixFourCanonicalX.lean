import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.RadixFourCandidate
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecRadix4Candidate
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4

/-!
Exact compiler check for the radix-four strict-PiDEC canonical-X artifact.

Assurance tier: Rust-conformant for this generated row leaf.

Owns: an independent Lean compiler for base-four recomposition, the shared
centered sign, two signed limbs per child, and `child = low + 2 * high`;
exact comparison of all 6,480 Rust rows in shards of at most 240 records;
the refinement from each generated coordinate's rows to the canonical
radix-four split model.

Does not own: the remaining strict-PiDEC rows, selective-lowering
substitution, whole-recursive conformance, or cryptographic soundness.

Emits constraints: no.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.RadixFourCanonicalX

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDecCanonicalX.RadixFourCandidate

private abbrev logicalCoordinates : Nat := Generated.Metadata.logicalCoordinates
private abbrev childCount : Nat := Generated.Metadata.childCount
private abbrev rowsPerCoordinate : Nat := 2 + 3 * childCount

def recompositionRow (coordinate : CoordinateColumns) : Row :=
  let powers := (List.range coordinate.children.length).map (fun exponent => 4 ^ exponent)
  ⟨(coordinate.parent, 1) ::
      (coordinate.children.zip powers).map
        (fun pair => (pair.1, goldilocksP - pair.2)),
    [(0, 1)], []⟩

def signProductRow (coordinate : CoordinateColumns) : Row :=
  ⟨[(0, 1), (coordinate.sign, 1)],
    [(coordinate.sign, 1)], [(coordinate.product, 1)]⟩

def signZeroRow (coordinate : CoordinateColumns) : Row :=
  ⟨[(coordinate.product, 1)],
    [(0, goldilocksP - 1), (coordinate.sign, 1)], []⟩

def limbColumn (coordinate : CoordinateColumns)
    (child limb : Nat) : Option Nat := do
  let pair ← coordinate.limbs[child]?
  pair[limb]?

def limbRow (coordinate : CoordinateColumns)
    (child limb : Nat) : Option Row := do
  let digit ← limbColumn coordinate child limb
  pure ⟨[(digit, 1)],
    [(coordinate.sign, goldilocksP - 1), (digit, 1)], []⟩

def reconstructionRow (coordinate : CoordinateColumns)
    (child : Nat) : Option Row := do
  let digit ← coordinate.children[child]?
  let low ← limbColumn coordinate child 0
  let high ← limbColumn coordinate child 1
  pure ⟨[(digit, 1), (low, goldilocksP - 1),
    (high, goldilocksP - 2)], [(0, 1)], []⟩

def expectedRow : RowOwner → Option Row
  | .recomposition activeIndex =>
      (coordinates[activeIndex]?).map recompositionRow
  | .signProduct activeIndex =>
      (coordinates[activeIndex]?).map signProductRow
  | .signZero activeIndex =>
      (coordinates[activeIndex]?).map signZeroRow
  | .radixFourLimb activeIndex child limb => do
      let coordinate ← coordinates[activeIndex]?
      limbRow coordinate child limb
  | .radixFourReconstruction activeIndex child => do
      let coordinate ← coordinates[activeIndex]?
      reconstructionRow coordinate child
  | .childDigit .. => none

def expectedRelativeIndex : RowOwner → Nat
  | .recomposition activeIndex => activeIndex
  | .signProduct activeIndex =>
      logicalCoordinates + activeIndex * rowsPerCoordinate
  | .signZero activeIndex =>
      logicalCoordinates + activeIndex * rowsPerCoordinate + 1
  | .radixFourLimb activeIndex child limb =>
      logicalCoordinates + activeIndex * rowsPerCoordinate + 2 + 3 * child + limb
  | .radixFourReconstruction activeIndex child =>
      logicalCoordinates + activeIndex * rowsPerCoordinate + 2 + 3 * child + 2
  | .childDigit .. => 0

def expectedPhysicalIndex (owner : RowOwner) : Nat :=
  match owner with
  | .recomposition activeIndex =>
      Generated.Metadata.recompositionRowStart + activeIndex
  | _ =>
      Generated.Metadata.canonicalityRowStart +
        (expectedRelativeIndex owner - logicalCoordinates)

def recordMatches (record : PhysicalRow) : Bool :=
  match expectedRow record.owner with
  | none => false
  | some row =>
      decide (record.relativeIndex = expectedRelativeIndex record.owner) &&
      decide (record.physicalIndex = expectedPhysicalIndex record.owner) &&
      decide (record.row = row)

def recordsMatchFrom : Nat → List PhysicalRow → Bool
  | _, [] => true
  | expectedIndex, record :: tail =>
      decide (record.relativeIndex = expectedIndex) &&
        recordMatches record &&
        recordsMatchFrom (expectedIndex + 1) tail

def ChunkMatches (start count : Nat) (records : List PhysicalRow) : Prop :=
  records.length = count ∧ recordsMatchFrom start records = true

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
  chunk21 : ChunkMatches 4590 240 Generated.Rows.Chunk21.values
  chunk22 : ChunkMatches 4830 240 Generated.Rows.Chunk22.values
  chunk23 : ChunkMatches 5070 240 Generated.Rows.Chunk23.values
  chunk24 : ChunkMatches 5310 240 Generated.Rows.Chunk24.values
  chunk25 : ChunkMatches 5550 240 Generated.Rows.Chunk25.values
  chunk26 : ChunkMatches 5790 240 Generated.Rows.Chunk26.values
  chunk27 : ChunkMatches 6030 240 Generated.Rows.Chunk27.values
  chunk28 : ChunkMatches 6270 210 Generated.Rows.Chunk28.values

theorem generated_rows_match_independent_compiler : GeneratedRowsMatch := by
  constructor <;> (unfold ChunkMatches; native_decide)

theorem checked_partition_count :
    100 + 100 + 70 + 25 * 240 + 210 = Generated.Metadata.rowCount := by
  decide

theorem candidate_geometry_matches_model :
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.params.b =
        Generated.Metadata.radix ∧
      Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.params.k =
        Generated.Metadata.childCount ∧
      Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.combinedBound =
        16384 := by
  simpa using
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix4.parameter_values

theorem generated_coordinate_count :
    coordinates.length = logicalCoordinates := by
  native_decide

theorem generated_row_count :
    Generated.rows.length = Generated.Metadata.rowCount := by
  native_decide

/-- One generated coordinate selected by its bounded logical index. -/
def coordinateAt (activeIndex : Fin logicalCoordinates) : CoordinateColumns :=
  coordinates.get ⟨activeIndex.val, by
    rw [generated_coordinate_count]
    exact activeIndex.isLt⟩

/-- The generated dynamic coordinate map as the fixed seven-child model map. -/
def modelColumns (activeIndex : Fin logicalCoordinates) :
    PiDecRadix4Candidate.CoordinateColumns :=
  let coordinate := coordinateAt activeIndex
  { parent := coordinate.parent
    children := fun child => coordinate.children.getD child.val 0
    sign := coordinate.sign
    product := coordinate.product
    low := fun child => (coordinate.limbs.getD child.val []).getD 0 0
    high := fun child => (coordinate.limbs.getD child.val []).getD 1 0 }

private def recompositionIndex (activeIndex : Fin logicalCoordinates) :
    Fin Generated.rows.length :=
  ⟨activeIndex.val, by
    rw [generated_row_count]
    have activeBound := activeIndex.isLt
    change activeIndex.val < 270 at activeBound
    change activeIndex.val < 6480
    omega⟩

private def canonicalIndex (activeIndex : Fin logicalCoordinates)
    (offset : Fin rowsPerCoordinate) : Fin Generated.rows.length :=
  ⟨logicalCoordinates + activeIndex.val * rowsPerCoordinate + offset.val, by
    rw [generated_row_count]
    have activeBound := activeIndex.isLt
    have offsetBound := offset.isLt
    change activeIndex.val < 270 at activeBound
    change offset.val < 23 at offsetBound
    change 270 + activeIndex.val * 23 + offset.val < 6480
    omega⟩

private def childRowIndices (activeIndex : Fin logicalCoordinates)
    (child : Fin childCount) : List (Fin Generated.rows.length) :=
  [canonicalIndex activeIndex ⟨2 + 3 * child.val, by
      have bound := child.isLt
      simp only [rowsPerCoordinate, childCount] at bound ⊢
      omega⟩,
   canonicalIndex activeIndex ⟨2 + 3 * child.val + 1, by
      have bound := child.isLt
      simp only [rowsPerCoordinate, childCount] at bound ⊢
      omega⟩,
   canonicalIndex activeIndex ⟨2 + 3 * child.val + 2, by
      have bound := child.isLt
      simp only [rowsPerCoordinate, childCount] at bound ⊢
      omega⟩]

/-- The 24 generated row indices for one logical coordinate, in model order. -/
def coordinateRowIndices (activeIndex : Fin logicalCoordinates) :
    List (Fin Generated.rows.length) :=
  recompositionIndex activeIndex ::
    canonicalIndex activeIndex ⟨0, by
      decide⟩ ::
    canonicalIndex activeIndex ⟨1, by
      decide⟩ ::
    (List.ofFn (childRowIndices activeIndex)).flatten

/-- Exact generated rows for one coordinate, reordered only by their typed
indices into the local model order. -/
def coordinateArtifactRows (activeIndex : Fin logicalCoordinates) : List Row :=
  (coordinateRowIndices activeIndex).map
    (fun index => (Generated.rows.get index).row)

/-- Every generated coordinate has exactly the 24 rows consumed by the local
canonical-split refinement theorem. -/
theorem generated_coordinate_rows_match_model :
    ∀ activeIndex : Fin logicalCoordinates,
      coordinateArtifactRows activeIndex =
        PiDecRadix4Candidate.rows (modelColumns activeIndex) := by
  native_decide

private theorem coordinate_artifact_rows_satisfy
    {assignment : Nat → Nat}
    (satisfies : Satisfies (Generated.rows.map PhysicalRow.row) assignment)
    (activeIndex : Fin logicalCoordinates) :
    Satisfies (coordinateArtifactRows activeIndex) assignment := by
  intro row member
  rcases List.mem_map.mp member with ⟨index, _, rfl⟩
  apply satisfies
  exact List.mem_map.mpr
    ⟨Generated.rows.get index, List.get_mem Generated.rows index, rfl⟩

/-- Satisfaction of the exact generated Rust row artifact forces the unique
canonical seven-child split at every public-X coordinate. -/
theorem generated_rows_force_canonical_split
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (Generated.rows.map PhysicalRow.row) assignment) :
    ∀ activeIndex : Fin logicalCoordinates,
      ∀ child,
        PiDecRadix4Candidate.fieldAt assignment
            ((modelColumns activeIndex).children child) =
          PiDecRadix4Candidate.radix4SplitScalar
            (PiDecRadix4Candidate.fieldAt assignment
              (modelColumns activeIndex).parent) child := by
  intro activeIndex
  apply PiDecRadix4Candidate.rows_force_canonical_split canonical one
  rw [← generated_coordinate_rows_match_model activeIndex]
  exact coordinate_artifact_rows_satisfy satisfies activeIndex

end Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.RadixFourCanonicalX
