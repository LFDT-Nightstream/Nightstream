import NightstreamFPrime.Export.Stage1.Wide.PhysicalRelabel
import NightstreamFPrime.Export.Stage1.Wide.MatrixProgram
import NightstreamFPrime.Layout.MatrixProgram.SourceComposition

/-! Rebind ordinary matrix blocks to the wide physical row archive. Embedded
range templates and retained-coordinate maps keep their existing operands. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource

open NightstreamFPrime.Layout NightstreamFPrime.Layout.MatrixProgram

def inverseRanges (extraColumns : Nat) : List SourceProjectionRange :=
  (PhysicalRelabel.columnRanges.take 3).map fun range =>
    {
      packageStart := range.sourceStart
      sourceStart := range.packageStart
      count := range.count + if range.packageStart =
        Layout.Stage1.Spartan.sourceToSpartan Layout.Stage1.PiRLCStarts.commitmentFreshStart
        then extraColumns else 0 }

def inverse (extraColumns : Nat) : SourceProjection := .mapped (inverseRanges extraColumns)

theorem ordinary_ranges_eq : PhysicalRelabel.columnRanges.take 3 =
    [⟨0, 0, 19512839⟩, ⟨19776407, 19568242, 52326⟩,
      ⟨20572364, 19646884, 8212655⟩] := by
  have old := Layout.Stage1.PiRLCStarts.childLogicalStarts_eq
  have current := Layout.Stage1.Wide.PiRLCStarts.childLogicalStarts_eq
  simp only [List.cons.injEq] at old current
  rcases old with ⟨_, oldCommit, _, _, _, oldOutput, _⟩
  rcases current with ⟨_, newCommit, _, _, _, _, _⟩
  rw [PhysicalRelabel.columnRanges, oldCommit, oldOutput, newCommit, Layout.Stage1.PiRLCStarts.phaseLogicalStart_eq,
    Layout.Stage1.PiRLCStarts.commitmentFreshStart_eq,
    Layout.Stage1.Wide.PiRLCStarts.commitmentFreshStart_eq,
    Layout.Stage1.Spartan.spartanColumnCount_eq]
  norm_num [Layout.Stage1.Wide.SourceOrder.column, Layout.Stage1.Wide.SourceOrder.relocate,
    Layout.Stage1.Spartan.sourceToSpartan, Layout.Stage1.Spartan.pilotSourceColumnCount,
    Layout.Stage1.Spartan.proofInputSourceStart, Layout.Stage1.Spartan.piCcsPhaseOffset,
    Layout.Stage1.Spartan.piCcsLocalStart, Layout.Stage1.Spartan.privateColumnCount_eq,
    Layout.Stage1.Wide.SourceOrder.privateColumns_eq]


theorem inverseRanges_eq (extraColumns : Nat) :
    inverseRanges extraColumns =
      [⟨0, 0, 19512839⟩, ⟨19568242, 19776407, 52326⟩,
        ⟨19646884, 20572364, 8212655 + extraColumns⟩] := by
  unfold inverseRanges
  rw [ordinary_ranges_eq, Layout.Stage1.PiRLCStarts.commitmentFreshStart_eq]
  norm_num [Layout.Stage1.Spartan.sourceToSpartan, Layout.Stage1.Spartan.pilotSourceColumnCount,
    Layout.Stage1.Spartan.proofInputSourceStart, Layout.Stage1.Spartan.piCcsPhaseOffset,
    Layout.Stage1.Spartan.piCcsLocalStart]

theorem inverse_unique (extraColumns : Nat) : (inverse extraColumns).Unique := by
  intro source
  simp only [inverseRanges_eq,
    SourceProjectionRange.column?, List.filterMap_cons, List.filterMap_nil]
  split_ifs <;> simp_all <;> omega

theorem inverse_compose_column (extraColumns : Nat) (target : SourceProjection) (source : Nat) :
    ((inverse extraColumns).compose target).column? source =
      ((inverse extraColumns).column? source).bind target.column? :=
  SourceProjection.compose_column _ _ (inverse_unique extraColumns) source

theorem selected_range (ranges : List SourceProjectionRange) (source target : Nat)
    (selected : (SourceProjection.mapped ranges).column? source = some target) :
    ∃ range ∈ ranges, range.column? source = some target := by
  unfold SourceProjection.column? at selected
  cases found : ranges.filterMap (fun range => range.column? source) with
  | nil => simp [found] at selected
  | cons head tail =>
    cases tail with
    | nil =>
      simp only [found, Option.some.injEq] at selected
      subst head
      apply List.mem_filterMap.mp
      rw [found]
      exact List.mem_singleton_self target
    | cons next rest => simp [found] at selected

theorem range_position (range : SourceProjectionRange) (source target : Nat)
    (selected : range.column? source = some target) :
    range.packageStart ≤ source ∧ source - range.packageStart < range.count ∧
      target = range.sourceStart + (source - range.packageStart) := by
  unfold SourceProjectionRange.column? at selected
  split at selected
  · dsimp only at selected
    split at selected
    · exact ⟨by assumption, by assumption, (Option.some.inj selected).symm⟩
    · cases selected
  · cases selected

/-- Ordinary-row emission accepts only columns recovered by the matrix
source projection. This excludes the temporary digit-word bridge. -/
theorem ordinary_column_inverse (extra source target : Nat)
    (emitted : PhysicalRelabel.ordinaryColumn source = .ok target) :
    (inverse extra).column? target = some source := by
  rw [inverse, inverseRanges_eq]
  have selected : PhysicalRelabel.ordinaryProjection.column? source = some target := by
    unfold PhysicalRelabel.ordinaryColumn at emitted
    cases found : PhysicalRelabel.ordinaryProjection.column? source with
    | none => rw [found] at emitted; cases emitted
    | some value =>
      rw [found] at emitted
      exact congrArg some (Except.ok.inj emitted)
  rw [PhysicalRelabel.ordinaryProjection, ordinary_ranges_eq] at selected
  obtain ⟨range, member, selected⟩ := selected_range _ source target selected
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · obtain ⟨_, bounded, value⟩ := range_position _ source target selected
    have bound : source < 19512839 := by simpa using bounded
    have same : target = source := by simpa using value
    subst target
    apply SourceProjection.mapped_three_first_column?
    · simpa using SourceProjectionRange.column?_at ⟨0, 0, 19512839⟩ ⟨source, bound⟩
    · exact SourceProjectionRange.column?_eq_none_of_before _ _ (by dsimp; omega)
    · exact SourceProjectionRange.column?_eq_none_of_before _ _ (by dsimp; omega)
  · obtain ⟨lower, bounded, value⟩ := range_position _ source target selected
    change 19776407 ≤ source at lower
    change source - 19776407 < 52326 at bounded
    change target = 19568242 + (source - 19776407) at value
    subst target
    apply SourceProjection.mapped_three_second_column?
    · exact SourceProjectionRange.column?_eq_none_of_after _ _ (by dsimp; omega)
    · have mappedAt := SourceProjectionRange.column?_at ⟨19568242, 19776407, 52326⟩
        ⟨source - 19776407, bounded⟩
      simpa only [Nat.add_sub_of_le lower] using mappedAt
    · exact SourceProjectionRange.column?_eq_none_of_before _ _ (by dsimp; omega)
  · obtain ⟨lower, bounded, value⟩ := range_position _ source target selected
    change 20572364 ≤ source at lower
    change source - 20572364 < 8212655 at bounded
    change target = 19646884 + (source - 20572364) at value
    subst target
    apply SourceProjection.mapped_three_column?
    · exact SourceProjectionRange.column?_eq_none_of_after _ _ (by dsimp; omega)
    · exact SourceProjectionRange.column?_eq_none_of_after _ _ (by dsimp; omega)
    · have mappedAt := SourceProjectionRange.column?_at ⟨19646884, 20572364, 8212655 + extra⟩
        ⟨source - 20572364, by change source - 20572364 < 8212655 + extra; omega⟩
      simpa only [Nat.add_sub_of_le lower] using mappedAt

private theorem shifted_bound (start index count extra : Nat)
    (lower : start ≤ index) (bounded : index - start < count) :
    index + extra - start < count + extra := by omega

theorem inverse_insert (extra source target : Nat)
    (read : (inverse 0).column? target = some source) :
    (inverse extra).column? (if target < 27859260 then target else target + extra) =
      some (if source < 28784740 then source else source + extra) := by
  rw [inverse, inverseRanges_eq] at read ⊢
  norm_num only [Nat.add_zero] at read
  obtain ⟨range, member, selected⟩ := selected_range _ target source read
  clear read
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · obtain ⟨_, bounded, value⟩ := range_position _ target source selected
    have bound : target < 19512839 := by simpa using bounded
    have same : source = target := by simpa using value
    subst source
    rw [if_pos (by omega), if_pos (by omega)]
    apply SourceProjection.mapped_three_first_column?
    · simpa using SourceProjectionRange.column?_at ⟨0, 0, 19512839⟩ ⟨target, bound⟩
    · exact SourceProjectionRange.column?_eq_none_of_before _ _ (by dsimp; omega)
    · exact SourceProjectionRange.column?_eq_none_of_before _ _ (by dsimp; omega)
  · obtain ⟨lower, bounded, value⟩ := range_position _ target source selected
    change 19568242 ≤ target at lower
    change target - 19568242 < 52326 at bounded
    change source = 19776407 + (target - 19568242) at value
    rw [if_pos (by omega), if_pos (by omega)]
    apply SourceProjection.mapped_three_second_column?
    · exact SourceProjectionRange.column?_eq_none_of_after _ _ (by dsimp; omega)
    · have mappedAt := SourceProjectionRange.column?_at ⟨19568242, 19776407, 52326⟩
        ⟨target - 19568242, bounded⟩
      simpa only [Nat.add_sub_of_le lower, ← value] using mappedAt
    · exact SourceProjectionRange.column?_eq_none_of_before _ _ (by dsimp; omega)
  · obtain ⟨lower, bounded, value⟩ := range_position _ target source selected
    change 19646884 ≤ target at lower
    change target - 19646884 < 8212655 at bounded
    change source = 20572364 + (target - 19646884) at value
    clear selected
    by_cases before : target < 27859260
    · rw [if_pos before, if_pos (by omega)]
      apply SourceProjection.mapped_three_column?
      · exact SourceProjectionRange.column?_eq_none_of_after _ _ (by dsimp; omega)
      · exact SourceProjectionRange.column?_eq_none_of_after _ _ (by dsimp; omega)
      · have mappedAt := SourceProjectionRange.column?_at ⟨19646884, 20572364, 8212655 + extra⟩
          ⟨target - 19646884, by change target - 19646884 < 8212655 + extra; omega⟩
        simpa only [Nat.add_sub_of_le lower, ← value] using mappedAt
    · rw [if_neg before, if_neg (by omega)]
      apply SourceProjection.mapped_three_column?
      · exact SourceProjectionRange.column?_eq_none_of_after _ _ (by dsimp; omega)
      · exact SourceProjectionRange.column?_eq_none_of_after _ _ (by dsimp; omega)
      · have shifted : 19646884 ≤ target + extra := by omega
        have mappedAt := SourceProjectionRange.column?_at ⟨19646884, 20572364, 8212655 + extra⟩
          ⟨target + extra - 19646884, shifted_bound _ _ _ extra lower bounded⟩
        have result : 20572364 + (target + extra - 19646884) = source + extra := by omega
        simpa only [Nat.add_sub_of_le shifted, result] using mappedAt

def moveRange (range : IndexRange) : Except String IndexRange := do
  let first ← PhysicalRelabel.row range.start
  unless 0 < range.count do throw "empty source row range"
  let last ← PhysicalRelabel.row (range.start + range.count - 1)
  unless first + range.count = last + 1 do throw "source row range crosses the removed sampler"
  return { range with start := first }

def schedule : IndexSchedule → Except String IndexSchedule
  | .indexTable indices => return .indexTable (← indices.mapM PhysicalRelabel.row)
  | .rangeList ranges => return .rangeList (← ranges.mapM moveRange)

private theorem row_ok_iff (source target : Nat) :
    PhysicalRelabel.row source = .ok target ↔
      (source < 19385261 ∧ target = source) ∨
      (20394109 ≤ source ∧ target = source - 949909) := by
  simp only [PhysicalRelabel.row, Layout.Stage1.PiRLCStarts.phaseRowStart,
    Layout.Stage1.PiRLCStarts.commitmentRowStart, Layout.Stage1.PiRLCStarts.samplerRowStart,
    Layout.Stage1.Wide.PiRLCStarts.commitmentRowStart, Layout.Stage1.Wide.PiRLCStarts.samplerRowStart,
    Layout.Stage1.Wide.PiRLCStarts.phaseRowStart, Nat.reduceAdd, Nat.reduceSub]
  by_cases before : source < 19385261
  · rw [if_pos before, Except.ok.injEq]
    omega
  · rw [if_neg before]
    by_cases after : 20394109 ≤ source
    · rw [if_pos after, Except.ok.injEq]
      omega
    · rw [if_neg after]
      constructor
      · intro impossible; cases impossible
      · intro impossible; rcases impossible with impossible | impossible <;> omega

private theorem row_interpolate (start count first last index : Nat)
    (beginning : PhysicalRelabel.row start = .ok first)
    (ending : PhysicalRelabel.row (start + count - 1) = .ok last)
    (contiguous : first + count = last + 1) (inside : index < count) :
    PhysicalRelabel.row (start + index) = .ok (first + index) := by
  rw [row_ok_iff] at beginning ending ⊢
  omega

theorem moveRange_correct (before after : IndexRange) (emitted : moveRange before = .ok after) :
    after.count = before.count ∧ ∀ index < before.count,
      PhysicalRelabel.row (before.start + index) = .ok (after.start + index) := by
  cases first : PhysicalRelabel.row before.start with
  | error message => simp [moveRange, first, Bind.bind, Except.bind] at emitted
  | ok start =>
    by_cases positive : 0 < before.count
    · cases last : PhysicalRelabel.row (before.start + before.count - 1) with
      | error message => simp [moveRange, first, positive, last, Bind.bind, Except.bind,
          Pure.pure, Except.pure] at emitted
      | ok stop =>
        by_cases contiguous : start + before.count = stop + 1
        · simp [moveRange, first, positive, last, contiguous, Bind.bind, Except.bind,
            Pure.pure, Except.pure] at emitted
          subst after
          exact ⟨rfl, fun index bound => row_interpolate _ _ _ _ index first last contiguous bound⟩
        · simp [moveRange, first, positive, last, contiguous, Bind.bind, Except.bind,
            Pure.pure, Except.pure] at emitted
    · simp [moveRange, first, positive, Bind.bind, Except.bind] at emitted

private theorem ranges_correct (before after : List IndexRange)
    (pairs : List.Forall₂ (fun a b => moveRange a = .ok b) before after) :
    (after.map IndexRange.count).sum = (before.map IndexRange.count).sum ∧ ∀ index,
      IndexSchedule.index?.select after index =
        (IndexSchedule.index?.select before index).bind (fun row => (PhysicalRelabel.row row).toOption) := by
  induction pairs with
  | nil => exact ⟨rfl, fun _ => rfl⟩
  | @cons a b before after mapped pairs ih =>
    obtain ⟨count, rows⟩ := moveRange_correct a b mapped
    refine ⟨by simp only [List.map_cons, List.sum_cons, count, ih.1], ?_⟩
    intro index
    simp only [IndexSchedule.index?.select, count]
    split
    · rename_i inside
      simp only [Option.bind_some, rows index inside, Except.toOption]
    · exact ih.2 (index - a.count)

private theorem indices_correct (before after : List Nat)
    (pairs : List.Forall₂ (fun a b => PhysicalRelabel.row a = .ok b) before after) :
    ∀ index : Nat, after[index]? = before[index]?.bind (fun row => (PhysicalRelabel.row row).toOption) := by
  induction pairs with
  | nil => intro index; simp
  | @cons a b before after mapped pairs ih =>
    intro index
    cases index with
    | zero => simp [mapped, Except.toOption]
    | succ index => simpa using ih index

theorem schedule_correct (before after : IndexSchedule) (emitted : schedule before = .ok after) :
    after.count = before.count ∧ ∀ index,
      after.index? index = (before.index? index).bind (fun row => (PhysicalRelabel.row row).toOption) := by
  cases before with
  | rangeList ranges =>
    cases mapped : ranges.mapM moveRange with
    | error message => simp [schedule, mapped] at emitted
    | ok moved =>
      simp [schedule, mapped] at emitted
      subst after
      exact ranges_correct ranges moved (PhysicalRelabel.mapM_pairs moveRange ranges moved mapped)
  | indexTable indices =>
    rw [schedule, Array.mapM_eq_mapM_toList] at emitted
    cases mapped : indices.toList.mapM PhysicalRelabel.row with
    | error message => simp [mapped] at emitted
    | ok moved =>
      simp [mapped] at emitted
      subst after
      have pairs := PhysicalRelabel.mapM_pairs PhysicalRelabel.row indices.toList moved mapped
      refine ⟨?_, ?_⟩
      · simpa [IndexSchedule.count] using pairs.length_eq.symm
      · intro index
        simpa [IndexSchedule.index?] using indices_correct indices.toList moved pairs index

def block (extraColumns : Nat) : Block → Except String Block
  | .ordinary value => do
    return .ordinary { value with
      rows := ← schedule value.rows
      projection := (inverse extraColumns).compose value.projection }
  | .mapped width projection inner => return .mapped width projection (← block extraColumns inner)
  | other => .ok other

def relocateProgram (extra : Nat) (before : Layout.MatrixProgram.Program) :
    Except String Layout.MatrixProgram.Program := do
  return ⟨← before.blocks.mapM (block extra)⟩

def program (application : RetainedLayout.Program) (compiled : PiRlcWideSampler.RangePlan.Compiled) :
    Except String Layout.MatrixProgram.Program :=
  relocateProgram (PerApplicationPackage.directAddedPrivateColumnCount application)
    (MatrixProgram.program application compiled)

end NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource
