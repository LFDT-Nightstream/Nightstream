import NightstreamFPrime.Export.Stage1.PiRLCSamplerCandidateWiring
import NightstreamFPrime.Export.MatrixProgram.SourceGridSupport

/-!
Owns the compact source grids for digest-lane values. The two decoded symbols
and reject bits per lane reuse First54 slots; all other values keep their
existing coordinates. The partition follows the two 17-word candidate gadgets
after the 66-word canonical field decomposition.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerLogicalGrids

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open PiRLCSamplerOrdinaryRetainedBlocks
open PiRLCSamplerCandidateWiring
open PiRLCSamplerOrdinaryRetainedGeometry (piRlcGeometry)

abbrev Program := Lifecycle.Stage1.Application.Program

def sourceStart : Nat :=
  Spartan.sourceToSpartan (PiRLCStarts.samplerLogicalStart + 592)

def start (segment : Fin 8) : Nat :=
  match segment.val with
  | 0 => 0 | 1 => 67 | 2 => 68 | 3 => 82
  | 4 => 83 | 5 => 84 | 6 => 85 | _ => 99

def count (segment : Fin 8) : Nat :=
  match segment.val with
  | 0 => 67 | 2 | 6 => 14 | _ => 1

def selected (position : Fin logicalCountPerLane) : Fin 8 :=
  if position.val < 67 then 0 else if position.val = 67 then 1
  else if position.val < 82 then 2 else if position.val = 82 then 3
  else if position.val = 83 then 4 else if position.val = 84 then 5
  else if position.val < 99 then 6 else 7

private def isLocal (segment : Fin 8) : Bool :=
  segment.val == 0 || segment.val == 2 || segment.val == 4 || segment.val == 6

def block (program : Program) (segment : Fin 8) : LowNormBlock.Block (sourceWidth program) :=
  if isLocal segment then logicalBlock program
  else if segment.val == 1 || segment.val == 5 then PiRLCFirst54RetainedBlocks.symbolBlock program
  else PiRLCFirst54RetainedBlocks.rejectBlock program

def retainedStart (program : Program) (segment : Fin 8) : Nat :=
  if isLocal segment then PiRLCSamplerOrdinaryRetainedGeometry.logicalStart program
  else if segment.val == 1 || segment.val == 5 then PiRLCRetainedGeometry.symbolStart program
  else PiRLCRetainedGeometry.rejectStart program

def slotStart (lane : Fin laneCount) (segment : Fin 8) : Nat :=
  if isLocal segment then lane.val * 100 + start segment
  else lane.val * 2 + if segment.val < 4 then 0 else 1

def majorSlotStride (segment : Fin 8) : Nat := if isLocal segment then 3200 else 64
def minorSlotStride (segment : Fin 8) : Nat := if isLocal segment then 400 else 8

def grid (program : Program) (lane : Fin laneCount) (segment : Fin 8) : SourceGrid :=
  SourceGrid.ofSemantic (block program segment) (retainedStart program segment)
    (sourceStart + lane.val * 100 + start segment)
    17 15504 8 992 (count segment) (slotStart lane segment)
    (majorSlotStride segment) (minorSlotStride segment)

def gridAt (program : Program) (index : Fin (laneCount * 8)) : SourceGrid :=
  grid program (Fin.decodeProd index).1 (Fin.decodeProd index).2

def grids (program : Program) : List SourceGrid :=
  (List.finRange (laneCount * 8)).map (gridAt program)

theorem segment_bounds (segment : Fin 8) :
    0 < count segment ∧ start segment + count segment ≤ logicalCountPerLane := by
  fin_cases segment <;> decide

theorem selected_bounds (position : Fin logicalCountPerLane) :
    start (selected position) ≤ position.val ∧
      position.val < start (selected position) + count (selected position) := by
  have bound : position.val < 100 := position.isLt
  unfold selected
  split_ifs <;> simp only [start, count] <;> omega

theorem selected_eq (position : Fin logicalCountPerLane) (segment : Fin 8)
    (lower : start segment ≤ position.val)
    (upper : position.val < start segment + count segment) :
    selected position = segment := by
  fin_cases segment <;> simp only [start, count] at lower upper
  all_goals simp only [selected]; split_ifs <;> first | rfl | omega

theorem block_fits {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (segment : Fin 8) :
    retainedStart program segment + (block program segment).coordinateCount ≤ logicalWidth := by
  by_cases own : isLocal segment = true
  · simp only [retainedStart, block, if_pos own]
    exact PiRLCSamplerOrdinaryRetainedGeometry.logicalFits geometry
  · simp only [retainedStart, block, if_neg own]
    by_cases symbol : (segment.val == 1 || segment.val == 5) = true
    · simp only [if_pos symbol]
      exact PiRLCRetainedGeometry.symbolFits (piRlcGeometry geometry)
    · simp only [if_neg symbol]
      exact PiRLCRetainedGeometry.rejectFits (piRlcGeometry geometry)

private theorem slot_bound (program : Program) (descriptor : Lane) (segment : Fin 8)
    (offset : Fin (count segment)) :
    slotStart descriptor.lane segment + descriptor.source.val * majorSlotStride segment +
      descriptor.round.val * minorSlotStride segment + offset.val < (block program segment).slotCount := by
  have sourceLt : descriptor.source.val < 17 := descriptor.source.isLt
  have roundLt : descriptor.round.val < 8 := descriptor.round.isLt
  have laneLt : descriptor.lane.val < 4 := descriptor.lane.isLt
  have offsetLt := offset.isLt
  fin_cases segment <;>
    norm_num [slotStart, start, count, isLocal, majorSlotStride, minorSlotStride, block,
      logicalBlock_slotCount, PiRLCFirst54RetainedBlocks.symbolBlock,
      PiRLCFirst54RetainedBlocks.rejectBlock] at offsetLt ⊢ <;> omega

theorem grid_form? {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (segment : Fin 8) (offset : Fin (count segment)) :
    (grid program descriptor.lane segment).form? logicalWidth
        (sourceStart + descriptor.source.val * 15504 + descriptor.round.val * 992 +
          descriptor.lane.val * 100 + start segment + offset.val) =
      some ((block program segment).form (retainedStart program segment)
        (block_fits geometry segment)
        ⟨slotStart descriptor.lane segment + descriptor.source.val * majorSlotStride segment +
          descriptor.round.val * minorSlotStride segment + offset.val,
          slot_bound program descriptor segment offset⟩) := by
  have roundLt : descriptor.round.val < 8 := descriptor.round.isLt
  have offsetLt : offset.val < 100 := lt_of_lt_of_le offset.isLt (by
    have bounds := segment_bounds segment
    change count segment ≤ 100
    change 0 < count segment ∧ start segment + count segment ≤ 100 at bounds
    omega)
  convert SourceGrid.form?_ofSemantic (block program segment) (retainedStart program segment)
    (sourceStart + descriptor.lane.val * 100 + start segment)
    17 15504 8 992 (count segment) (slotStart descriptor.lane segment)
    (majorSlotStride segment) (minorSlotStride segment)
    (block_fits geometry segment) (by decide) (by decide)
    descriptor.source descriptor.round offset (by omega) (by omega)
    (slot_bound program descriptor segment offset) using 1 <;> congr 1 <;> omega

def position (segment : Fin 8) (offset : Fin (count segment)) : Fin logicalCountPerLane :=
  ⟨start segment + offset.val, lt_of_lt_of_le (Nat.add_lt_add_left offset.isLt _)
    (segment_bounds segment).2⟩

def mappedForm {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (segment : Fin 8) (offset : Fin (count segment)) : SparseForm logicalWidth :=
  (block program segment).form (retainedStart program segment) (block_fits geometry segment)
    ⟨slotStart descriptor.lane segment + descriptor.source.val * majorSlotStride segment +
      descriptor.round.val * minorSlotStride segment + offset.val,
      slot_bound program descriptor segment offset⟩

private theorem mappedForm_local {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (segment : Fin 8) (offset : Fin (count segment))
    (own : isLocal segment = true) :
    mappedForm geometry descriptor segment offset =
      localForm geometry descriptor (position segment offset) := by
  fin_cases segment
  all_goals norm_num [isLocal] at own
  all_goals
    apply congrArg ((logicalBlock program).form
      (PiRLCSamplerOrdinaryRetainedGeometry.logicalStart program)
      (PiRLCSamplerOrdinaryRetainedGeometry.logicalFits geometry))
    apply Fin.ext
    norm_num [slotStart, majorSlotStride, minorSlotStride, isLocal, start,
      logicalSlot, laneIndex, Fin.encodeProd, position, laneCount,
      logicalCountPerLane, roundCount, PiRLCSamplerOrdinaryRows.digestRoundCount]
    omega

theorem mappedForm_eq_logicalForm {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (segment : Fin 8) (offset : Fin (count segment)) :
    mappedForm geometry descriptor segment offset =
      logicalForm geometry descriptor (position segment offset) := by
  have offsetLt := offset.isLt
  by_cases own : isLocal segment = true
  · have outside : (position segment offset).val ≠ 82 ∧
        (position segment offset).val ≠ 99 ∧ (position segment offset).val ≠ 67 ∧
        (position segment offset).val ≠ 84 := by
      fin_cases segment
      all_goals norm_num [isLocal] at own
      all_goals norm_num [position, start, count] at offsetLt ⊢ <;> omega
    rw [mappedForm_local geometry descriptor segment offset own]
    simp only [logicalForm, if_neg outside.1, if_neg outside.2.1,
      if_neg outside.2.2.1, if_neg outside.2.2.2]
  · fin_cases segment <;> norm_num [isLocal] at own
    all_goals
      have zero : offset.val = 0 := by simpa [count] using Nat.eq_zero_of_le_zero (by omega : offset.val ≤ 0)
      simp only [mappedForm, block, retainedStart, slotStart, majorSlotStride,
        minorSlotStride, isLocal, position, start, zero, logicalForm,
        PiRLCRetainedInputs.first54Inputs]
      norm_num
      congr 1
      apply Fin.ext
      simp [PiRLCFirst54DirectSchedule.candidateIndex, candidate, Fin.encodeProd,
        PiRLCFirst54DirectSchedule.roundCount, PiRLCFirst54Invocations.roundCount,
        NightstreamFPrime.Gadgets.Sampling.First54.candidateCount]
      omega

/-- Each grid selects the same exact form as the direct sampler resolver. -/
theorem grid_logical_form? {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (segment : Fin 8) (offset : Fin (count segment)) :
    (grid program descriptor.lane segment).form? logicalWidth
        (sourceStart + descriptor.source.val * 15504 + descriptor.round.val * 992 +
          descriptor.lane.val * 100 + (position segment offset).val) =
      some (logicalForm geometry descriptor (position segment offset)) := by
  rw [position, show sourceStart + descriptor.source.val * 15504 + descriptor.round.val * 992 +
      descriptor.lane.val * 100 + (start segment + offset.val) =
      sourceStart + descriptor.source.val * 15504 + descriptor.round.val * 992 +
      descriptor.lane.val * 100 + start segment + offset.val by omega, grid_form? geometry descriptor segment offset]
  exact congrArg some (mappedForm_eq_logicalForm geometry descriptor segment offset)

theorem source_of_grid_form?_some {program : Program} {logicalWidth column : Nat}
    {lane : Fin laneCount} {segment : Fin 8} {form : SparseForm logicalWidth}
    (found : (grid program lane segment).form? logicalWidth column = some form) :
    ∃ source : Fin sourceCount, ∃ round : Fin roundCount, ∃ offset : Fin (count segment),
      column = sourceStart + source.val * 15504 + round.val * 992 + lane.val * 100 +
        start segment + offset.val := by
  obtain ⟨source, round, offset, coordinates⟩ := SourceGrid.source_of_form?_some _ found
  refine ⟨source, round, offset, ?_⟩
  simp only [grid, SourceGrid.ofSemantic] at coordinates
  omega

private theorem matching_grid {program : Program} {logicalWidth : Nat}
    (descriptor : Lane) (p : Fin logicalCountPerLane)
    (lane : Fin laneCount) (segment : Fin 8) {form : SparseForm logicalWidth}
    (found : (grid program lane segment).form? logicalWidth
      (sourceStart + descriptor.source.val * 15504 + descriptor.round.val * 992 +
        descriptor.lane.val * 100 + p.val) = some form) :
    lane = descriptor.lane ∧ segment = selected p := by
  obtain ⟨source, round, offset, coordinates⟩ := source_of_grid_form?_some found
  have roundLt : round.val < 8 := round.isLt
  have descriptorRoundLt : descriptor.round.val < 8 := descriptor.round.isLt
  have laneLt : lane.val < 4 := lane.isLt
  have descriptorLaneLt : descriptor.lane.val < 4 := descriptor.lane.isLt
  have positionLt : p.val < 100 := p.isLt
  have endLt : start segment + offset.val < 100 :=
    (position segment offset).isLt
  have sourceEq : source.val = descriptor.source.val := by omega
  have roundEq : round.val = descriptor.round.val := by omega
  have laneEq : lane.val = descriptor.lane.val := by omega
  have positionEq : p.val = start segment + offset.val := by omega
  refine ⟨Fin.ext laneEq, (selected_eq p segment (by omega) ?_).symm⟩
  have offsetLt := offset.isLt
  omega

/-- Only the lane and segment that own the source column can resolve it. -/
theorem lookup_at {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (p : Fin logicalCountPerLane)
    (lane : Fin laneCount) (segment : Fin 8) :
    (grid program lane segment).form? logicalWidth
        (sourceStart + descriptor.source.val * 15504 + descriptor.round.val * 992 +
          descriptor.lane.val * 100 + p.val) =
      if lane = descriptor.lane ∧ segment = selected p then
        some (logicalForm geometry descriptor p) else none := by
  by_cases chosen : lane = descriptor.lane ∧ segment = selected p
  · rw [if_pos chosen]
    rcases chosen with ⟨rfl, rfl⟩
    have bounds := selected_bounds p
    let offset : Fin (count (selected p)) := ⟨p.val - start (selected p), by omega⟩
    have same : position (selected p) offset = p := by
      apply Fin.ext
      simp only [position, offset]
      omega
    simpa only [same] using grid_logical_form? geometry descriptor (selected p) offset
  · rw [if_neg chosen]
    cases found : (grid program lane segment).form? logicalWidth
        (sourceStart + descriptor.source.val * 15504 + descriptor.round.val * 992 +
          descriptor.lane.val * 100 + p.val) with
    | none => rfl
    | some form => exact False.elim (chosen (matching_grid descriptor p lane segment found))

private theorem filterMap_select {α β : Type} [DecidableEq α]
    (items : List α) (target : α) (value : β) (unique : items.Nodup) (member : target ∈ items) :
    items.filterMap (fun item => if item = target then some value else none) = [value] := by
  induction items with
  | nil => simp at member
  | cons head tail ih =>
      have nodup := List.nodup_cons.mp unique
      by_cases same : head = target
      · subst target
        have empty : tail.filterMap (fun item => if item = head then some value else none) = [] := by
          apply List.filterMap_eq_nil_iff.mpr
          intro item itemMem
          have different : item ≠ head := by
            intro same
            exact nodup.1 (same ▸ itemMem)
          simp only [if_neg different]
        simp [empty]
      · have tailMem : target ∈ tail := by
          rcases List.mem_cons.mp member with equal | tailMem
          · exact False.elim (same equal.symm)
          · exact tailMem
        simp only [List.filterMap_cons, if_neg same]
        exact ih nodup.2 tailMem

/-- The complete compact mapping has one result at every digest-lane source. -/
theorem forms_at {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (p : Fin logicalCountPerLane) :
    (grids program).filterMap (fun entry => entry.form? logicalWidth
        (sourceStart + descriptor.source.val * 15504 + descriptor.round.val * 992 +
          descriptor.lane.val * 100 + p.val)) = [logicalForm geometry descriptor p] := by
  let target := Fin.encodeProd (descriptor.lane, selected p)
  have atIndex (index : Fin (laneCount * 8)) :
      (gridAt program index).form? logicalWidth
          (sourceStart + descriptor.source.val * 15504 + descriptor.round.val * 992 +
            descriptor.lane.val * 100 + p.val) =
        if index = target then some (logicalForm geometry descriptor p) else none := by
    rw [gridAt, lookup_at geometry descriptor p]
    have same : ((Fin.decodeProd index).1 = descriptor.lane ∧
        (Fin.decodeProd index).2 = selected p) ↔ index = target := by
      constructor
      · intro coordinates
        have pair : (Fin.decodeProd index : Fin laneCount × Fin 8) =
            (descriptor.lane, selected p) := Prod.ext coordinates.1 coordinates.2
        simpa [target] using congrArg Fin.encodeProd pair
      · intro equality
        simp [equality, target]
    simp only [same]
  simp only [grids, List.filterMap_map, Function.comp_def]
  rw [List.filterMap_congr (fun index _ => atIndex index)]
  apply filterMap_select
  · exact List.nodup_finRange _
  · simp [target]

/-- The unsplit domain is used only to prove that the new grids reject gaps. -/
def envelope (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (logicalBlock program)
    (PiRLCSamplerOrdinaryRetainedGeometry.logicalStart program)
    sourceStart 17 15504 8 992 400 0 3200 400

theorem envelope_form? {program : Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (descriptor : Lane) (p : Fin logicalCountPerLane) :
    (envelope program).form? logicalWidth
        (sourceStart + descriptor.source.val * 15504 + descriptor.round.val * 992 +
          descriptor.lane.val * 100 + p.val) = some (localForm geometry descriptor p) := by
  have sourceLt : descriptor.source.val < 17 := descriptor.source.isLt
  have roundLt : descriptor.round.val < 8 := descriptor.round.isLt
  have laneLt : descriptor.lane.val < 4 := descriptor.lane.isLt
  have positionLt : p.val < 100 := p.isLt
  let offset : Fin 400 := ⟨descriptor.lane.val * 100 + p.val, by omega⟩
  have slotBound : 0 + descriptor.source.val * 3200 + descriptor.round.val * 400 + offset.val <
      (logicalBlock program).slotCount := by
    rw [logicalBlock_slotCount]
    dsimp [offset]
    omega
  have lookup := SourceGrid.form?_ofSemantic (logicalBlock program)
    (PiRLCSamplerOrdinaryRetainedGeometry.logicalStart program)
    sourceStart 17 15504 8 992 400 0 3200 400
    (PiRLCSamplerOrdinaryRetainedGeometry.logicalFits geometry)
    (by decide) (by decide) descriptor.source descriptor.round offset
    (by dsimp [offset]; omega) (by dsimp [offset]; omega) slotBound
  have sameSlot : (⟨0 + descriptor.source.val * 3200 + descriptor.round.val * 400 + offset.val,
      slotBound⟩ : Fin (logicalBlock program).slotCount) = logicalSlot descriptor p := by
    apply Fin.ext
    simp [logicalSlot, laneIndex, Fin.encodeProd, laneCount, logicalCountPerLane,
      roundCount, PiRLCSamplerOrdinaryRows.digestRoundCount, offset]
    omega
  rw [sameSlot] at lookup
  convert lookup using 1
  congr 1
  dsimp [offset]
  omega

/-- Splitting the logical grid preserves rejection outside its old domain. -/
theorem forms_none_of_envelope_none {program : Program} {logicalWidth column : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (missing : (envelope program).form? logicalWidth column = none) :
    (grids program).filterMap (fun entry => entry.form? logicalWidth column) = [] := by
  apply List.filterMap_eq_nil_iff.mpr
  intro entry member
  obtain ⟨index, _indexMem, rfl⟩ := List.mem_map.mp member
  cases found : (gridAt program index).form? logicalWidth column with
  | none => rfl
  | some form =>
      unfold gridAt at found
      obtain ⟨source, round, offset, coordinates⟩ := source_of_grid_form?_some found
      let descriptor : Lane := ⟨source, round, (Fin.decodeProd index).1⟩
      have lookup := envelope_form? geometry descriptor
        (position (Fin.decodeProd index).2 offset)
      have sameColumn : sourceStart + descriptor.source.val * 15504 +
          descriptor.round.val * 992 + descriptor.lane.val * 100 +
          (position (Fin.decodeProd index).2 offset).val = column := by
        dsimp only [descriptor, position]
        omega
      rw [sameColumn, missing] at lookup
      cases lookup

end NightstreamFPrime.Export.Stage1.PiRLCSamplerLogicalGrids
