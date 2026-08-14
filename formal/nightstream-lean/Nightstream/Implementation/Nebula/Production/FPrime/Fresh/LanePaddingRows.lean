import Nightstream.Implementation.Nebula.Production.FPrime.Fresh.LaneAuthority
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Contract: generated zero rows for every alignment-only coordinate in the
reference V2 memory lanes.

The operations lane has 18 padding bits. Each snapshot lane has 44 padding
bits. These 106 bits are inside the authority-bearing product commitment and
must have one canonical value. Satisfaction of this row block derives
`ProductionFreshLaneAuthority.PaddingZero`; it is not a prover premise.

This file owns only the padding rows and their soundness and completeness.
It does not own record-field rows, lane projections, product updates,
commitment binding, NIFS extraction, or Rust refinement.

Assurance tier: implementation model.

Emits constraints: exactly 106 constant rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionFreshLanePaddingRows

open Nightstream.Implementation.Nebula.ProductionFreshLaneAuthority
open Nightstream.Implementation.Nebula.ProductionFreshLinearSubstitution
open Nightstream.Implementation.Nebula.ProductionFreshLowNormEncoding
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.ConcreteLaneGeometry
open Nightstream.SuperNeo.Concrete

/-- Alignment-only offsets in the operations lane. -/
def operationsOffsets : List Nat :=
  (List.range operationAlignmentPadding).map fun index =>
    operationPayloadWidth + index

/-- Alignment-only offsets in one snapshot lane. -/
def snapshotOffsets : List Nat :=
  (List.range snapshotAlignmentPadding).map fun index =>
    snapshotPayloadWidth + index

/-- Exact source-column pins for all three lane tails. -/
def pins : List (Nat × Nat) :=
  operationsOffsets.map (fun offset => (publicWidth + offset, 0)) ++
  snapshotOffsets.map (fun offset =>
    (publicWidth + operationsLaneWidth + offset, 0)) ++
  snapshotOffsets.map (fun offset =>
    (publicWidth + operationsLaneWidth + snapshotLaneWidth + offset, 0))

def rows : List Row := ConstantPins.rows pins

@[simp] theorem operationsOffsets_length : operationsOffsets.length = 18 := by
  simp [operationsOffsets, operationAlignmentPadding_exact]

@[simp] theorem snapshotOffsets_length : snapshotOffsets.length = 44 := by
  simp [snapshotOffsets, snapshotAlignmentPadding_exact]

@[simp] theorem pins_length : pins.length = 106 := by
  simp [pins]

@[simp] theorem rows_length_exact : rows.length = 106 := by
  simp [rows, ConstantPins.rows]

theorem pins_valuesCanonical : ConstantPins.ValuesCanonical pins := by
  intro pin member
  simp only [pins, List.mem_append] at member
  rcases member with operationsOrInitial | final
  · rcases operationsOrInitial with operations | initial
    · rcases List.mem_map.mp operations with ⟨offset, _member, rfl⟩
      norm_num [goldilocksP]
    · rcases List.mem_map.mp initial with ⟨offset, _member, rfl⟩
      norm_num [goldilocksP]
  · rcases List.mem_map.mp final with ⟨offset, _member, rfl⟩
    norm_num [goldilocksP]

private theorem selfIncluded : rowsIncluded rows rows = true := by
  rw [rowsIncluded, List.all_eq_true]
  intro row member
  exact decide_eq_true member

private theorem sourceNat_canonical
    {privateWidth : Nat} (source : SourceAssignment privateWidth) :
    forall column, sourceNat source column < goldilocksP := by
  intro column
  by_cases within : column < sourceWidth privateWidth
  · simp only [sourceNat, dif_pos within]
    simpa only [goldilocksP,
      Nightstream.SuperNeo.Concrete.goldilocksModulus] using
      (source ⟨column, within⟩).isLt
  · simp [sourceNat, within, goldilocksP]

private theorem source_value_zero_of_nat
    {privateWidth : Nat} (source : SourceAssignment privateWidth)
    (column : Fin (sourceWidth privateWidth))
    (zero : sourceNat source column.val = 0) :
    source column = 0 := by
  apply Fin.ext
  simpa only [sourceNat_sourceColumn] using zero

/-- Satisfying the generated padding rows forces every committed padding
coordinate to canonical field zero. -/
theorem sound
    {privateWidth : Nat} {source : SourceAssignment privateWidth}
    (sourceOne : sourceNat source 0 = 1)
    (holds : Satisfies rows (sourceNat source)) :
    PaddingZero source := by
  have facts := ConstantPins.sound pins_valuesCanonical selfIncluded
    (sourceNat_canonical source) sourceOne holds
  constructor
  · intro offset lower upper
    let column : Fin (sourceWidth privateWidth) := laneSourceColumn
      ⟨offset, by
        have within : operationsLaneWidth <= laneWidth := by
          norm_num [operationsLaneWidth_exact, laneWidth]
        exact Nat.lt_of_lt_of_le upper within⟩
    apply source_value_zero_of_nat source column
    have pinMember : (publicWidth + offset, 0) ∈ pins := by
      apply List.mem_append_left
      apply List.mem_append_left
      apply List.mem_map.mpr
      refine ⟨offset, ?_, rfl⟩
      apply List.mem_map.mpr
      refine ⟨offset - operationPayloadWidth, ?_, by omega⟩
      apply List.mem_range.mpr
      rw [operationAlignmentPadding_exact]
      rw [operationsLaneWidth_exact] at upper
      rw [operationPayloadWidth_exact] at lower ⊢
      omega
    have zero := facts (publicWidth + offset, 0) pinMember
    simpa [column, laneSourceColumn, directSourceColumn] using zero
  · intro offset lower upper
    let laneOffset : Fin laneWidth :=
      ⟨operationsLaneWidth + offset, by
        norm_num [operationsLaneWidth_exact, snapshotLaneWidth_exact,
          laneWidth] at upper ⊢
        omega⟩
    let column : Fin (sourceWidth privateWidth) :=
      laneSourceColumn laneOffset
    apply source_value_zero_of_nat source column
    have pinMember :
        (publicWidth + operationsLaneWidth + offset, 0) ∈ pins := by
      apply List.mem_append_left
      apply List.mem_append_right
      apply List.mem_map.mpr
      refine ⟨offset, ?_, rfl⟩
      apply List.mem_map.mpr
      refine ⟨offset - snapshotPayloadWidth, ?_, by omega⟩
      apply List.mem_range.mpr
      rw [snapshotAlignmentPadding_exact]
      rw [snapshotLaneWidth_exact] at upper
      rw [snapshotPayloadWidth_exact] at lower ⊢
      omega
    have zero := facts
      (publicWidth + operationsLaneWidth + offset, 0) pinMember
    simpa [column, laneOffset, laneSourceColumn, directSourceColumn,
      Nat.add_assoc] using zero
  · intro offset lower upper
    let laneOffset : Fin laneWidth :=
      ⟨operationsLaneWidth + snapshotLaneWidth + offset, by
        norm_num [operationsLaneWidth_exact, snapshotLaneWidth_exact,
          laneWidth] at upper ⊢
        omega⟩
    let column : Fin (sourceWidth privateWidth) :=
      laneSourceColumn laneOffset
    apply source_value_zero_of_nat source column
    have pinMember :
        (publicWidth + operationsLaneWidth + snapshotLaneWidth + offset, 0) ∈
          pins := by
      apply List.mem_append_right
      apply List.mem_map.mpr
      refine ⟨offset, ?_, rfl⟩
      apply List.mem_map.mpr
      refine ⟨offset - snapshotPayloadWidth, ?_, by omega⟩
      apply List.mem_range.mpr
      rw [snapshotAlignmentPadding_exact]
      rw [snapshotLaneWidth_exact] at upper
      rw [snapshotPayloadWidth_exact] at lower ⊢
      omega
    have zero := facts
      (publicWidth + operationsLaneWidth + snapshotLaneWidth + offset, 0)
      pinMember
    simpa [column, laneOffset, laneSourceColumn, directSourceColumn,
      Nat.add_assoc] using zero

/-- Canonical zero padding satisfies the exact generated row block. -/
theorem complete
    {privateWidth : Nat} {source : SourceAssignment privateWidth}
    (sourceOne : sourceNat source 0 = 1)
    (padding : PaddingZero source) :
    Satisfies rows (sourceNat source) := by
  apply ConstantPins.complete pins_valuesCanonical sourceOne
  intro pin member
  simp only [pins, List.mem_append] at member
  rcases member with operationsOrInitial | final
  · rcases operationsOrInitial with operations | initial
    · rcases List.mem_map.mp operations with ⟨offset, offsetMember, rfl⟩
      rcases List.mem_map.mp offsetMember with ⟨index, indexMember, rfl⟩
      have indexBound := List.mem_range.mp indexMember
      have value := padding.operations (operationPayloadWidth + index)
        (by omega) (by
          norm_num [operationAlignmentPadding_exact,
            operationsLaneWidth_exact, operationPayloadWidth_exact] at *
          omega)
      have valueNat : sourceNat source
          (laneSourceColumn (privateWidth := privateWidth)
            ⟨operationPayloadWidth + index, by
            norm_num [operationAlignmentPadding_exact,
              operationPayloadWidth_exact, laneWidth] at *
            omega⟩).val = 0 := by
        rw [sourceNat_sourceColumn]
        exact congrArg Fin.val value
      simpa [laneSourceColumn, directSourceColumn] using valueNat
    · rcases List.mem_map.mp initial with ⟨offset, offsetMember, rfl⟩
      rcases List.mem_map.mp offsetMember with ⟨index, indexMember, rfl⟩
      have indexBound := List.mem_range.mp indexMember
      have value := padding.initialSnapshot
        (snapshotPayloadWidth + index) (by omega) (by
          norm_num [snapshotAlignmentPadding_exact,
            snapshotLaneWidth_exact, snapshotPayloadWidth_exact] at *
          omega)
      have valueNat : sourceNat source
          (laneSourceColumn (privateWidth := privateWidth)
            ⟨operationsLaneWidth + (snapshotPayloadWidth + index), by
              norm_num [snapshotAlignmentPadding_exact,
                operationsLaneWidth_exact, snapshotPayloadWidth_exact,
                laneWidth] at *
              omega⟩).val = 0 := by
        rw [sourceNat_sourceColumn]
        exact congrArg Fin.val value
      simpa [laneSourceColumn, directSourceColumn, Nat.add_assoc] using valueNat
  · rcases List.mem_map.mp final with ⟨offset, offsetMember, rfl⟩
    rcases List.mem_map.mp offsetMember with ⟨index, indexMember, rfl⟩
    have indexBound := List.mem_range.mp indexMember
    have value := padding.finalSnapshot
      (snapshotPayloadWidth + index) (by omega) (by
        norm_num [snapshotAlignmentPadding_exact,
          snapshotLaneWidth_exact, snapshotPayloadWidth_exact] at *
        omega)
    have valueNat : sourceNat source
        (laneSourceColumn (privateWidth := privateWidth)
          ⟨operationsLaneWidth + snapshotLaneWidth +
              (snapshotPayloadWidth + index), by
            norm_num [snapshotAlignmentPadding_exact,
              operationsLaneWidth_exact, snapshotLaneWidth_exact,
              snapshotPayloadWidth_exact, laneWidth] at *
            omega⟩).val = 0 := by
      rw [sourceNat_sourceColumn]
      exact congrArg Fin.val value
    simpa [laneSourceColumn, directSourceColumn, Nat.add_assoc] using valueNat

end Nightstream.Implementation.Nebula.ProductionFreshLanePaddingRows
