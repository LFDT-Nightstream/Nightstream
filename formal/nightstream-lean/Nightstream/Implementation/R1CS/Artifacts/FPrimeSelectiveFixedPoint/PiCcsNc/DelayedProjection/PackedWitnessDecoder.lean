import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder.Generated.Layout

/-!
Exact compact certificate for the production full-`Z` decoder.

Assurance tier: artifact-checked for the generated fixed profile.

Owns: exact production dimensions; the bijection between all generated
logical coordinates and the `211,797 × 54` packed matrix cells; its
fourteen-child lift; the exact 54-live/10-computed-zero lane partition; the
bounded fixture's 108 one-hot constructor/commitment probes; and the exact
bounded/production commitment-data flattening dimensions.

Does not own: witness values, commitment binding, combined-NC acceptance,
delayed-projection rows, transcript scheduling, costs, or row removal.

Emits constraints: none; generated direct-dataflow certificate only.

The closed computations inspect either 64 proof-free `LaneSourceRecord`s or
108 proof-free `Nat` probe indices. No witness cell, key coefficient, or
proof-bearing structure is enumerated.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `pi_ccs_nc.full_z_decoder.artifact.dimensions` | exact fourteen-child `54 × 217,139` geometry | checked artifact |
| `pi_ccs_nc.full_z_decoder.artifact.lanes` | lanes 0–53 read matching witness rows and lanes 54–63 are virtual | checked artifact |
| `pi_ccs_nc.full_z_decoder.artifact.coordinates` | affine packed-cell/logical-column map is bijective | derived from checked dimensions |
| `pi_ccs_nc.full_z_decoder.artifact.commitment_probe` | every coordinate of two complete blocks passed through the real constructor, commitment recomputation, and commitment-data indexing | bounded checked artifact |
| `pi_ccs_nc.full_z_decoder.artifact.commitment_layout` | bounded κ=4 and production κ=18 commitments use 54 coefficient lanes per commitment row | generated dimensions / derived indexing |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder

namespace GeneratedLayout

export Generated.Layout
  (schemaVersion relationRows logicalWidth childCount matrixRows matrixColumns
    booleanLaneCount fixtureCommitmentWidth fixtureCommitmentDataLength
    productionCommitmentWidth productionCommitmentDataLength
    commitmentProbeBlocks laneSources commitmentProbeColumns)

end GeneratedLayout

abbrev Child := Fin GeneratedLayout.childCount
abbrev LiveLane := Fin GeneratedLayout.matrixRows
abbrev Block := Fin GeneratedLayout.matrixColumns
abbrev LogicalColumn := Fin GeneratedLayout.logicalWidth
abbrev BooleanLane := Fin GeneratedLayout.booleanLaneCount
abbrev PaddingLane := Fin (GeneratedLayout.booleanLaneCount - GeneratedLayout.matrixRows)
abbrev FixtureCommitmentRow := Fin GeneratedLayout.fixtureCommitmentWidth
abbrev FixtureCommitmentData := Fin GeneratedLayout.fixtureCommitmentDataLength
abbrev ProductionCommitmentRow := Fin GeneratedLayout.productionCommitmentWidth
abbrev ProductionCommitmentData := Fin GeneratedLayout.productionCommitmentDataLength
abbrev CommitmentProbeColumn :=
  Fin (GeneratedLayout.commitmentProbeBlocks * GeneratedLayout.matrixRows)

theorem dimensions_exact :
      GeneratedLayout.schemaVersion = 2 /\
      GeneratedLayout.relationRows = 14944219 /\
      GeneratedLayout.logicalWidth = 11437038 /\
      GeneratedLayout.childCount = 14 /\
      GeneratedLayout.matrixRows = 54 /\
      GeneratedLayout.matrixColumns = 211797 /\
      GeneratedLayout.booleanLaneCount = 64 /\
      GeneratedLayout.matrixColumns * GeneratedLayout.matrixRows =
        GeneratedLayout.logicalWidth /\
      GeneratedLayout.booleanLaneCount - GeneratedLayout.matrixRows = 10 /\
      GeneratedLayout.fixtureCommitmentWidth = 4 /\
      GeneratedLayout.fixtureCommitmentDataLength = 216 /\
      GeneratedLayout.productionCommitmentWidth = 18 /\
      GeneratedLayout.productionCommitmentDataLength = 972 /\
      GeneratedLayout.commitmentProbeBlocks = 2 /\
      GeneratedLayout.fixtureCommitmentWidth * GeneratedLayout.matrixRows =
        GeneratedLayout.fixtureCommitmentDataLength /\
      GeneratedLayout.productionCommitmentWidth * GeneratedLayout.matrixRows =
        GeneratedLayout.productionCommitmentDataLength := by
  decide

theorem laneSources_length :
    GeneratedLayout.laneSources.length = GeneratedLayout.booleanLaneCount := by
  decide

theorem commitmentProbeColumns_length :
    GeneratedLayout.commitmentProbeColumns.length =
      GeneratedLayout.commitmentProbeBlocks * GeneratedLayout.matrixRows := by
  decide

/-- The generator exercised this exact logical coordinate through the real
bounded constructor, packed matrix, Ajtai recomputation, and flattened public
commitment path. The closed input is exactly 108 proof-free `Nat` values. -/
def commitmentProbeColumnAt (column : CommitmentProbeColumn) : Nat :=
  GeneratedLayout.commitmentProbeColumns.get
    ⟨column.val, by
      rw [commitmentProbeColumns_length]
      exact column.isLt⟩

theorem commitmentProbeColumnAt_exact (column : CommitmentProbeColumn) :
    commitmentProbeColumnAt column = column.val := by
  native_decide +revert

/-- Generated owner record for one Boolean lane. -/
def laneSourceAt (lane : BooleanLane) : LaneSourceRecord :=
  GeneratedLayout.laneSources.get
    ⟨lane.val, by rw [laneSources_length]; exact lane.isLt⟩

/-- The generated 64-record certificate is exact at every lane. The reverted
closed input is precisely 64 proof-free records. -/
theorem laneSourceAt_exact (lane : BooleanLane) :
    (laneSourceAt lane).booleanLane = lane.val /\
      (laneSourceAt lane).witnessLane =
        if lane.val < GeneratedLayout.matrixRows then some lane.val else none := by
  native_decide +revert

/-- Embed a live packed-witness lane into the generated Boolean-lane cube. -/
def booleanLaneOfLive (lane : LiveLane) : BooleanLane :=
  ⟨lane.val, by
    have rowsLt : GeneratedLayout.matrixRows <
        GeneratedLayout.booleanLaneCount := by decide
    exact Nat.lt_trans lane.isLt rowsLt⟩

/-- Embed one of the ten virtual lanes after the 54 live lanes. -/
def booleanLaneOfPadding (lane : PaddingLane) : BooleanLane :=
  ⟨GeneratedLayout.matrixRows + lane.val, by
    have laneBound := lane.isLt
    simp only [GeneratedLayout.booleanLaneCount, GeneratedLayout.matrixRows] at *
    omega⟩

@[simp] theorem liveLane_source (lane : LiveLane) :
    (laneSourceAt (booleanLaneOfLive lane)).witnessLane = some lane.val := by
  rw [(laneSourceAt_exact (booleanLaneOfLive lane)).2]
  simp [booleanLaneOfLive, lane.isLt]

@[simp] theorem paddingLane_source (lane : PaddingLane) :
    (laneSourceAt (booleanLaneOfPadding lane)).witnessLane = none := by
  rw [(laneSourceAt_exact (booleanLaneOfPadding lane)).2]
  have laneBound := lane.isLt
  simp only [booleanLaneOfPadding, GeneratedLayout.booleanLaneCount,
    GeneratedLayout.matrixRows] at *
  simp [show ¬54 + lane.val < 54 by omega]

/-- Full logical coordinate belonging to one generated matrix cell. -/
def logicalColumnAt (address : Block × LiveLane) : LogicalColumn :=
  ⟨address.1.val * GeneratedLayout.matrixRows + address.2.val, by
    have blockBound := address.1.isLt
    have laneBound := address.2.isLt
    simp only [GeneratedLayout.matrixColumns, GeneratedLayout.matrixRows,
      GeneratedLayout.logicalWidth] at *
    omega⟩

/-- Inverse generated decoder address. -/
def packedAddress (column : LogicalColumn) : Block × LiveLane :=
  (⟨column.val / GeneratedLayout.matrixRows, by
      have columnBound := column.isLt
      simp only [GeneratedLayout.matrixRows, GeneratedLayout.matrixColumns,
        GeneratedLayout.logicalWidth] at *
      omega⟩,
    ⟨column.val % GeneratedLayout.matrixRows,
      Nat.mod_lt _ (by decide)⟩)

@[simp] theorem logicalColumnAt_packedAddress (column : LogicalColumn) :
    logicalColumnAt (packedAddress column) = column := by
  apply Fin.ext
  change column.val / GeneratedLayout.matrixRows *
      GeneratedLayout.matrixRows + column.val % GeneratedLayout.matrixRows =
        column.val
  simpa [Nat.mul_comm] using Nat.div_add_mod column.val
    GeneratedLayout.matrixRows

@[simp] theorem packedAddress_logicalColumnAt (address : Block × LiveLane) :
    packedAddress (logicalColumnAt address) = address := by
  apply Prod.ext
  · apply Fin.ext
    change
      (address.1.val * GeneratedLayout.matrixRows + address.2.val) /
          GeneratedLayout.matrixRows = address.1.val
    rw [Nat.mul_comm address.1.val GeneratedLayout.matrixRows,
      Nat.mul_add_div (by decide), Nat.div_eq_of_lt address.2.isLt,
      Nat.add_zero]
  · apply Fin.ext
    change
      (address.1.val * GeneratedLayout.matrixRows + address.2.val) %
          GeneratedLayout.matrixRows = address.2.val
    simpa [Nat.mod_eq_of_lt address.2.isLt] using
      Nat.mul_add_mod_self_right address.1.val
        GeneratedLayout.matrixRows address.2.val

theorem logicalColumnAt_bijective :
    Function.Injective logicalColumnAt /\
      Function.Surjective logicalColumnAt := by
  constructor
  · intro left right equal
    calc
      left = packedAddress (logicalColumnAt left) :=
        (packedAddress_logicalColumnAt left).symm
      _ = packedAddress (logicalColumnAt right) := by rw [equal]
      _ = right := packedAddress_logicalColumnAt right
  · intro column
    exact ⟨packedAddress column, logicalColumnAt_packedAddress column⟩

/-- Child-major lift: every physical cell of every one of the fourteen raw
children has exactly one logical assignment coordinate, and conversely. -/
def childLogicalColumnAt
    (address : Child × (Block × LiveLane)) : Child × LogicalColumn :=
  (address.1, logicalColumnAt address.2)

theorem childLogicalColumnAt_bijective :
    Function.Injective childLogicalColumnAt /\
      Function.Surjective childLogicalColumnAt := by
  constructor
  · intro left right equal
    have children : left.1 = right.1 :=
      congrArg (fun output : Child × LogicalColumn => output.1) equal
    have columns : logicalColumnAt left.2 = logicalColumnAt right.2 :=
      congrArg (fun output : Child × LogicalColumn => output.2) equal
    apply Prod.ext
    · exact children
    · exact logicalColumnAt_bijective.1 columns
  · intro target
    obtain ⟨address, equality⟩ := logicalColumnAt_bijective.2 target.2
    refine ⟨(target.1, address), ?_⟩
    apply Prod.ext
    · rfl
    · exact equality

/-- Rust `Commitment.data` stores each commitment row as one consecutive
54-coefficient column: `data[row * 54 + lane]`. -/
def commitmentDataIndex (width : Nat)
    (address : Fin width × LiveLane) :
    Fin (width * GeneratedLayout.matrixRows) :=
  ⟨address.1.val * GeneratedLayout.matrixRows + address.2.val, by
    have rowNext : address.1.val + 1 <= width :=
      Nat.succ_le_of_lt address.1.isLt
    have scaled :
        (address.1.val + 1) * GeneratedLayout.matrixRows <=
          width * GeneratedLayout.matrixRows :=
      Nat.mul_le_mul_right GeneratedLayout.matrixRows rowNext
    have belowNext :
        address.1.val * GeneratedLayout.matrixRows + address.2.val <
          (address.1.val + 1) * GeneratedLayout.matrixRows := by
      simpa [Nat.add_mul] using
        Nat.add_lt_add_left address.2.isLt
          (address.1.val * GeneratedLayout.matrixRows)
    exact Nat.lt_of_lt_of_le belowNext scaled⟩

/-- Inverse of the Rust flattened commitment-data index. -/
def commitmentDataAddress (width : Nat)
    (index : Fin (width * GeneratedLayout.matrixRows)) :
    Fin width × LiveLane :=
  (⟨index.val / GeneratedLayout.matrixRows, by
      have indexBound := index.isLt
      simp only [GeneratedLayout.matrixRows] at *
      omega⟩,
    ⟨index.val % GeneratedLayout.matrixRows,
      Nat.mod_lt _ (by decide)⟩)

@[simp] theorem commitmentDataIndex_address (width : Nat)
    (index : Fin (width * GeneratedLayout.matrixRows)) :
    commitmentDataIndex width (commitmentDataAddress width index) = index := by
  apply Fin.ext
  change index.val / GeneratedLayout.matrixRows *
      GeneratedLayout.matrixRows + index.val % GeneratedLayout.matrixRows =
        index.val
  simpa [Nat.mul_comm] using Nat.div_add_mod index.val
    GeneratedLayout.matrixRows

@[simp] theorem commitmentDataAddress_index (width : Nat)
    (address : Fin width × LiveLane) :
    commitmentDataAddress width (commitmentDataIndex width address) =
      address := by
  apply Prod.ext
  · apply Fin.ext
    change
      (address.1.val * GeneratedLayout.matrixRows + address.2.val) /
          GeneratedLayout.matrixRows = address.1.val
    rw [Nat.mul_comm address.1.val GeneratedLayout.matrixRows,
      Nat.mul_add_div (by decide), Nat.div_eq_of_lt address.2.isLt,
      Nat.add_zero]
  · apply Fin.ext
    change
      (address.1.val * GeneratedLayout.matrixRows + address.2.val) %
          GeneratedLayout.matrixRows = address.2.val
    simpa [Nat.mod_eq_of_lt address.2.isLt] using
      Nat.mul_add_mod_self_right address.1.val
        GeneratedLayout.matrixRows address.2.val

theorem commitmentDataIndex_bijective (width : Nat) :
    Function.Injective (commitmentDataIndex width) /\
      Function.Surjective (commitmentDataIndex width) := by
  constructor
  · intro left right equal
    calc
      left = commitmentDataAddress width (commitmentDataIndex width left) :=
        (commitmentDataAddress_index width left).symm
      _ = commitmentDataAddress width (commitmentDataIndex width right) := by
        rw [equal]
      _ = right := commitmentDataAddress_index width right
  · intro index
    exact ⟨commitmentDataAddress width index,
      commitmentDataIndex_address width index⟩

theorem fixtureCommitmentDataIndex_bijective :
    Function.Injective
        (commitmentDataIndex GeneratedLayout.fixtureCommitmentWidth) /\
      Function.Surjective
        (commitmentDataIndex GeneratedLayout.fixtureCommitmentWidth) :=
  commitmentDataIndex_bijective _

theorem productionCommitmentDataIndex_bijective :
    Function.Injective
        (commitmentDataIndex GeneratedLayout.productionCommitmentWidth) /\
      Function.Surjective
        (commitmentDataIndex GeneratedLayout.productionCommitmentWidth) :=
  commitmentDataIndex_bijective _

end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessDecoder
