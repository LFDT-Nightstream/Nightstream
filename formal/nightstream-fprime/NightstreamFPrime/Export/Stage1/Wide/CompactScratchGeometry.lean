import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportScratch
import NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource
import NightstreamFPrime.Export.Stage1.Wide.PhysicalCompactRelabel

/-! Geometry of the existing compact ring templates after wide relocation.
All ranges come from the checked emitted column map. -/

namespace NightstreamFPrime.Export.Stage1.Wide.CompactScratchGeometry

open NightstreamFPrime.Spec NightstreamFPrime.Layout NightstreamFPrime.Export.Package
open Layout.MatrixProgram AssignmentTransportScratch

theorem columnRanges_eq : PhysicalRelabel.columnRanges =
    [⟨0, 0, 19512839⟩, ⟨19776407, 19568242, 52326⟩, ⟨20572364, 19646884, 8212655⟩] ++
      (List.range 17).map (fun source => ⟨19528289 + source * 15504, 19567324 + source * 54, 54⟩) := by
  have split : PhysicalRelabel.columnRanges = PhysicalRelabel.columnRanges.take 3 ++
      (List.range 17).map (fun source =>
        ⟨Layout.Stage1.Spartan.sourceToSpartan (Layout.Stage1.PiRLCStarts.challengeWordStart source),
          Layout.Stage1.Wide.SourceOrder.column (Layout.Stage1.Wide.PiRLCStarts.challengeWordStart source), 54⟩) := rfl
  rw [split, PhysicalMatrixSource.ordinary_ranges_eq]
  apply congrArg (List.append _)
  apply List.map_congr_left
  intro source member
  have bounded : source < 17 := List.mem_range.mp member
  have old : Layout.Stage1.Spartan.sourceToSpartan (Layout.Stage1.PiRLCStarts.challengeWordStart source) =
      19528289 + source * 15504 := by
    rw [Layout.Stage1.PiRLCStarts.challengeWordStart_eq, Layout.Stage1.PiRLCStarts.phaseLogicalStart_eq]
    unfold Layout.Stage1.Spartan.sourceToSpartan
    rw [if_neg (by change ¬19513117 + source * 15504 + 15450 < 14722512; omega),
      if_neg (by change ¬19513117 + source * 15504 + 15450 < 14722516; omega),
      if_neg (by change ¬19513117 + source * 15504 + 15450 < 14751804; omega)]
    change 14751526 + (19513117 + source * 15504 + 15450 - 14751804) = _
    omega
  have current : Layout.Stage1.Wide.SourceOrder.column (Layout.Stage1.Wide.PiRLCStarts.challengeWordStart source) =
      19567324 + source * 54 := by
    change Layout.Stage1.Wide.SourceOrder.column (19513117 + 54485 + source * 54) = _
    rw [Layout.Stage1.Wide.SourceOrder.column_late _ (by omega) (by change _ < 27859538; omega)]
    omega
  rw [old, current]

private def Position (source target : Nat) : Prop :=
  (source = target ∧ source < 19512839) ∨
  (19776407 ≤ source ∧ source < 19828733 ∧ target = 19568242 + (source - 19776407)) ∨
  (20572364 ≤ source ∧ source < 28785019 ∧ target = 19646884 + (source - 20572364)) ∨
  (19512839 ≤ source ∧ source < 19776407 ∧ 19567324 ≤ target ∧ target < 19568242)

private theorem column_position (source target : Nat)
    (mapped : PhysicalRelabel.column source = .ok target) : Position source target := by
  have lookup : PhysicalRelabel.projection.column? source = some target := by
    cases selected : PhysicalRelabel.projection.column? source with
    | none => simp [PhysicalRelabel.column, selected] at mapped
    | some value =>
      rw [PhysicalRelabel.column, selected] at mapped
      have equal := Except.ok.inj mapped
      subst value
      rfl
  obtain ⟨range, member, selected⟩ := PhysicalMatrixSource.selected_range PhysicalRelabel.columnRanges source target lookup
  have position := PhysicalMatrixSource.range_position range source target selected
  rw [columnRanges_eq] at member
  simp only [List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with (rfl | rfl | rfl) | member
  · left
    change 0 ≤ source ∧ source - 0 < 19512839 ∧ target = 0 + (source - 0) at position
    omega
  · right; left
    change 19776407 ≤ source ∧ source - 19776407 < 52326 ∧
      target = 19568242 + (source - 19776407) at position
    omega
  · right; right; left
    change 20572364 ≤ source ∧ source - 20572364 < 8212655 ∧
      target = 19646884 + (source - 20572364) at position
    omega
  · obtain ⟨scalar, scalarMember, rfl⟩ := List.mem_map.mp member
    have scalarBound : scalar < 17 := List.mem_range.mp scalarMember
    change 19528289 + scalar * 15504 ≤ source ∧
      source - (19528289 + scalar * 15504) < 54 ∧
      target = 19567324 + scalar * 54 + (source - (19528289 + scalar * 15504)) at position
    right; right; right
    omega

theorem column_zero : PhysicalRelabel.column 0 = .ok 0 := by
  norm_num [PhysicalRelabel.column, PhysicalRelabel.projection, columnRanges_eq,
    SourceProjection.column?, SourceProjectionRange.column?, List.range_succ,
    List.filterMap_cons, List.filterMap_nil]

theorem column_outside (source target : Nat) (mapped : PhysicalRelabel.column source = .ok target)
    (outside : source < PiRLCCombinationScratchGeometry.scratchStart ∨
      PiRLCCombinationScratchGeometry.scratchEnd ≤ source) : Outside target := by
  have position := column_position source target mapped
  simp only [PiRLCCombinationScratchGeometry.scratchStart_eq,
    PiRLCCombinationScratchGeometry.scratchEnd_eq] at outside
  simp only [Outside, scratchStart_eq, scratchEnd_eq]
  rcases position with first | output | suffix | digits <;> omega

theorem column_scratch (source target count : Nat) (mapped : PhysicalRelabel.column source = .ok target)
    (contained : PiRLCCombinationScratchGeometry.scratchStart ≤ source ∧
      source + count ≤ PiRLCCombinationScratchGeometry.scratchEnd) :
    scratchStart ≤ target ∧ target + count ≤ scratchEnd := by
  have position := column_position source target mapped
  simp only [PiRLCCombinationScratchGeometry.scratchStart_eq,
    PiRLCCombinationScratchGeometry.scratchEnd_eq] at contained
  rw [scratchStart_eq, scratchEnd_eq]
  rcases position with first | output | suffix | digits <;> omega

theorem column_product (source target : Nat) (mapped : PhysicalRelabel.column source = .ok target)
    (inside : 19776407 ≤ source ∧ source < 19828733) :
    19568242 ≤ target ∧ target < 19620568 := by
  have position := column_position source target mapped
  rcases position with first | output | suffix | digits <;> omega

theorem product_injective (left right target : Nat)
    (leftMapped : PhysicalRelabel.column left = .ok target)
    (rightMapped : PhysicalRelabel.column right = .ok target)
    (inside : 19568242 ≤ target ∧ target < 19620568) : left = right := by
  have leftPosition := column_position left target leftMapped
  have rightPosition := column_position right target rightMapped
  rcases leftPosition with first | output | suffix | digits <;>
    rcases rightPosition with first | output | suffix | digits <;> omega

theorem output_range (descriptor : PiRLCProductSchedule.Descriptor) :
    19776407 ≤ PiRLCCombinationScratchGeometry.inputColumn descriptor 109 ∧
      PiRLCCombinationScratchGeometry.inputColumn descriptor 109 < 19828733 := by
  have bounded : descriptor.invocation.val < 52326 := descriptor.invocation.isLt
  rw [PiRLCCombinationScratchGeometry.inputColumn_eq descriptor 109 (by decide)]
  simp only [PiRLCCombinationScratchGeometry.sourceColumn, show ¬109 < 54 by decide,
    show ¬109 < 108 by decide, show ¬109 = 108 by decide, if_false, if_true]
  rw [PiRLCTransportValues.reference_output_column]
  change 19776407 ≤ Layout.Stage1.Spartan.sourceToSpartan (19776685 + descriptor.invocation.val) ∧
    Layout.Stage1.Spartan.sourceToSpartan (19776685 + descriptor.invocation.val) < 19828733
  unfold Layout.Stage1.Spartan.sourceToSpartan
  rw [if_neg (by change ¬19776685 + descriptor.invocation.val < 14722512; omega),
    if_neg (by change ¬19776685 + descriptor.invocation.val < 14722516; omega),
    if_neg (by change ¬19776685 + descriptor.invocation.val < 14751804; omega)]
  change 19776407 ≤ 14751526 + (19776685 + descriptor.invocation.val - 14751804) ∧
    14751526 + (19776685 + descriptor.invocation.val - 14751804) < 19828733
  omega

/-- The application insertion is injective and preserves the private scratch
interval and all columns outside it. -/
theorem inserted_outside (count column : Nat) (outside : Outside column) :
    Outside (if column < Layout.Stage1.Wide.SourceOrder.privateColumns then column else column + count) := by
  split
  · exact outside
  · right
    rw [scratchEnd_eq]
    rename_i after
    rw [Layout.Stage1.Wide.SourceOrder.privateColumns_eq] at after
    omega

theorem inserted_injective (count left right : Nat)
    (same : (if left < Layout.Stage1.Wide.SourceOrder.privateColumns then left else left + count) =
      (if right < Layout.Stage1.Wide.SourceOrder.privateColumns then right else right + count)) : left = right := by
  split_ifs at same <;> omega

end NightstreamFPrime.Export.Stage1.Wide.CompactScratchGeometry
