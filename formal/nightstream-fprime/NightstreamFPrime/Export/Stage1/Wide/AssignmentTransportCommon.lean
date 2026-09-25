import NightstreamFPrime.Export.Stage1.Wide.AssignmentProjection
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalEncodes

/-! The common witness schedule retains the PiCCS hash prefix, the common
state and verifier fields, and the application fields in their original order. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCommon

open NightstreamFPrime.Spec NightstreamFPrime.Layout NightstreamFPrime.Lifecycle
open ProductionRelation PerApplicationCanonicalAssignment
open CanonicalBlockAssignment

variable {program : RetainedLayout.Program}

def hashes (raw : RawValues program) : Schedule :=
  raw.schedule.take 2 ++ [ofBlock (LaterPoseidonRetainedBlocks.piCcsBlock program) raw.retainedSource]

def shared (raw : RawValues program) : Schedule := (raw.schedule.drop 10).take 17

def application (raw : RawValues program) : Schedule := raw.schedule.drop 29

def common (raw : RawValues program) : Schedule := hashes raw ++ shared raw ++ application raw

theorem lookup_before (left right : Schedule) (index : Nat)
    (inside : index < coordinateCount left) :
    coordinateAt (left ++ right) index = coordinateAt left index := by
  induction left generalizing index with
  | nil => simp [coordinateCount] at inside
  | cons entry rest ih =>
    simp only [List.cons_append, coordinateAt]
    split
    · rfl
    · apply ih
      change index < entry.coordinateCount + coordinateCount rest at inside
      omega

private theorem lookup_take (schedule : Schedule) (count index : Nat)
    (inside : index < coordinateCount (schedule.take count)) :
    coordinateAt (schedule.take count) index = coordinateAt schedule index := by
  have same := lookup_before (schedule.take count) (schedule.drop count) index inside
  rw [List.take_append_drop] at same
  exact same.symm

private theorem lookup_drop (schedule : Schedule) (count index : Nat) :
    coordinateAt (schedule.drop count) index =
      coordinateAt schedule (coordinateCount (schedule.take count) + index) := by
  have same := coordinateAt_append_offset (schedule.take count) (schedule.drop count) index
  rw [List.take_append_drop] at same
  exact same.symm

private theorem lookup_slice {sourceWidth : Nat} (block : LowNormBlock.Block sourceWidth)
    (source : Fin sourceWidth → F) (count : Nat) (fits : 0 + count ≤ block.slotCount)
    (index : Nat) (inside : index < (block.slice 0 count fits).coordinateCount) :
    (ofBlock (block.slice 0 count fits) source).coordinateAt index =
      (ofBlock block source).coordinateAt index := by
  have parent : index < block.coordinateCount := by
    have bound := Nat.mul_le_mul_right block.kind.width fits
    simp only [LowNormBlock.Block.slice_coordinateCount, Nat.zero_add] at inside bound
    exact lt_of_lt_of_le inside bound
  unfold BlockValue.coordinateAt
  dsimp only [ofBlock, BlockValue.coordinateCount]
  rw [dif_pos inside, dif_pos parent]
  simp only [ofBlock, LowNormBlock.Block.slice, Nat.zero_add]

private theorem first_count (raw : RawValues program) :
    coordinateCount (raw.schedule.take 2) = 87092200 := by
  simp only [RawValues.schedule, List.take_succ_cons, List.take_zero, coordinateCount,
    Canonical.ofBlock, ofBlock, BlockValue.coordinateCount]
  rw [PiRLCRetainedGeometry.priorPoseidonBlock, PiRLCRetainedGeometry.outputPoseidonBlock,
    LowNormBlock.Block.lift_coordinateCount, LowNormBlock.Block.lift_coordinateCount,
    PoseidonRetainedBlock.priorBlock_coordinateCount, PoseidonRetainedBlock.outputBlock_coordinateCount]

theorem hashes_count (raw : RawValues program) : coordinateCount (hashes raw) = 113903904 := by
  rw [hashes, coordinateCount_append, first_count]
  simp only [coordinateCount, BlockValue.coordinateCount, ofBlock,
    LaterPoseidonRetainedBlocks.piCcsBlock_coordinateCount]

private theorem suffix_count (raw : RawValues program) :
    coordinateCount (application raw) = RetainedLayout.applicationCount program := by
  rfl

private theorem take_ten_count (raw : RawValues program) :
    coordinateCount (raw.schedule.take 10) = 121293090 := by
  have count := PerApplicationCanonicalEncodes.shared_prefix_count raw
  rw [PiRLCRetainedGeometry.prefixLogicalWidth_eq] at count
  change 121293360 = 270 + _ at count
  omega

private theorem take_count (schedule : Schedule) (count : Nat) :
    coordinateCount schedule = coordinateCount (schedule.take count) +
      coordinateCount (schedule.drop count) := by
  rw [← coordinateCount_append, List.take_append_drop]

private theorem take_twentyNine_count (raw : RawValues program) :
    coordinateCount (raw.schedule.take 29) = 149281987 := by
  have all := schedule_width raw
  rw [RetainedLayout.referenceWidth_eq] at all
  have parts := take_count raw.schedule 29
  change coordinateCount raw.schedule = _ + coordinateCount (application raw) at parts
  rw [suffix_count] at parts
  change 270 + coordinateCount raw.schedule = _ at all
  omega

private theorem take_twentySeven_count (raw : RawValues program) :
    coordinateCount (raw.schedule.take 27) = 140293475 := by
  have all := schedule_width raw
  rw [RetainedLayout.referenceWidth_eq] at all
  have parts := take_count raw.schedule 27
  have suffix : coordinateCount (raw.schedule.drop 27) = 8988512 + RetainedLayout.applicationCount program := by
    change (PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock program).coordinateCount +
      ((PiRLCSamplerOrdinaryRetainedBlocks.freshBlock program).coordinateCount +
        coordinateCount (application raw)) = _
    rw [PiRLCSamplerOrdinaryRetainedBlocks.logicalBlock_coordinateCount,
      PiRLCSamplerOrdinaryRetainedBlocks.freshBlock_coordinateCount, suffix_count]
    omega
  rw [suffix] at parts
  change 270 + coordinateCount raw.schedule = _ at all
  omega

theorem shared_count (raw : RawValues program) : coordinateCount (shared raw) = 19000385 := by
  have pieces : raw.schedule.take 27 = raw.schedule.take 10 ++ shared raw := by rfl
  have counts := congrArg coordinateCount pieces
  rw [coordinateCount_append, take_ten_count, take_twentySeven_count] at counts
  omega

theorem common_count (raw : RawValues program) :
    270 + coordinateCount (common raw) = RetainedLayout.commonCount program := by
  rw [common, coordinateCount_append, coordinateCount_append, hashes_count,
    shared_count, suffix_count, RetainedLayout.commonCount_eq]
  omega

private theorem hashes_lookup (raw : RawValues program) (index : Nat)
    (inside : index < 113903904) : coordinateAt (hashes raw) index = coordinateAt raw.schedule index := by
  by_cases first : index < 87092200
  · rw [hashes, lookup_before _ _ index (by rwa [first_count])]
    exact lookup_take raw.schedule 2 index (by rwa [first_count])
  · have firstLe : 87092200 ≤ index := by omega
    have shifted : index = coordinateCount (raw.schedule.take 2) + (index - 87092200) := by
      rw [first_count]
      omega
    rw [hashes, shifted, coordinateAt_append_offset, ← lookup_drop raw.schedule 2]
    change coordinateAt [ofBlock (LaterPoseidonRetainedBlocks.piCcsBlock program) raw.retainedSource]
      (index - 87092200) = coordinateAt (ofBlock (PiRLCRetainedGeometry.laterPoseidonBlock program)
        raw.retainedSource :: _) (index - 87092200)
    have child : index - 87092200 < (LaterPoseidonRetainedBlocks.piCcsBlock program).coordinateCount := by
      rw [LaterPoseidonRetainedBlocks.piCcsBlock_coordinateCount]
      omega
    have parent : index - 87092200 < (PiRLCRetainedGeometry.laterPoseidonBlock program).coordinateCount := by
      rw [PiRLCRetainedGeometry.laterPoseidonBlock_coordinateCount]
      omega
    simp only [coordinateAt, BlockValue.coordinateCount, ofBlock, dif_pos child, dif_pos parent]
    exact lookup_slice _ raw.retainedSource _ _ _ child

theorem common_lookup (raw : RawValues program) (index : Nat) :
    coordinateAt (common raw) index =
      coordinateAt raw.schedule (if index < 113903904 then index
        else if index < 132904289 then 121293090 + (index - 113903904)
        else 149281987 + (index - 132904289)) := by
  by_cases hash : index < 113903904
  · rw [if_pos hash, common, List.append_assoc,
      lookup_before _ _ index (by rwa [hashes_count])]
    exact hashes_lookup raw index hash
  · rw [if_neg hash]
    by_cases middle : index < 132904289
    · rw [if_pos middle]
      have shifted : index = coordinateCount (hashes raw) + (index - 113903904) := by
        rw [hashes_count]; omega
      rw [common, List.append_assoc, shifted, coordinateAt_append_offset,
        lookup_before _ _ _ (by rw [shared_count]; omega)]
      rw [shared, lookup_take _ _ _ (by change _ < coordinateCount (shared raw); rw [shared_count]; omega),
        lookup_drop, take_ten_count]
      rw [hashes_count]
      apply congrArg (coordinateAt raw.schedule)
      omega
    · rw [if_neg middle]
      have shifted : index = coordinateCount (hashes raw ++ shared raw) + (index - 132904289) := by
        rw [coordinateCount_append, hashes_count, shared_count]; omega
      rw [common, shifted, coordinateAt_append_offset, application, lookup_drop, take_twentyNine_count]
      rw [coordinateCount_append, hashes_count, shared_count]
      apply congrArg (coordinateAt raw.schedule)
      omega

private theorem assignment_after {width : Nat} (publicInput : Fin ProductionAssignment.publicWidth → F)
    (schedule : Schedule) (target : Fin width) (afterPublic : 270 ≤ target.val) :
    CanonicalBlockAssignment.assignment publicInput schedule target = coordinateAt schedule (target.val - 270) := by
  unfold CanonicalBlockAssignment.assignment
  exact dif_neg (Nat.not_lt.mpr afterPublic)

private theorem raw_after (raw : RawValues program)
    (target : Fin (PerApplicationFixedPoint.logicalWidth program)) (afterPublic : 270 ≤ target.val) :
    raw.assignment target = coordinateAt raw.schedule (target.val - 270) :=
  assignment_after _ _ _ afterPublic

private theorem project_value (raw : RawValues program)
    (target : Fin (RetainedLayout.logicalWidth program)) (source : Nat)
    (found : CoordinateRecovery.source? program target.val = some source) :
    AssignmentProjection.project program raw.assignment target =
      raw.assignment ⟨source, CoordinateRecovery.source?_lt program target.val source found⟩ := by
  unfold AssignmentProjection.project
  split
  · rename_i value read
    apply congrArg raw.assignment
    apply Fin.ext
    exact Option.some.inj (read.symm.trans found)
  · rename_i read
    rw [found] at read
    cases read

theorem assignment_before (raw : RawValues program)
    (target : Fin (RetainedLayout.logicalWidth program))
    (before : target.val < RetainedLayout.commonCount program) :
    CanonicalBlockAssignment.assignment (encodedHashCells raw.outputDigest) (common raw) target =
      AssignmentProjection.seed program raw.assignment target := by
  rw [AssignmentProjection.seed_before program raw.assignment target before]
  obtain ⟨hashEnd, sharedStart, sharedEnd, applicationStart⟩ := RetainedLayout.boundaries program
  by_cases hash : target.val < 113904174
  · have found : CoordinateRecovery.source? program target.val = some target.val := by
      rw [CoordinateRecovery.source?, hashEnd, if_pos hash]
    rw [project_value raw target target.val found]
    by_cases publicRegion : target.val < 270
    · have bound : target.val < ProductionAssignment.publicWidth := publicRegion
      simp only [RawValues.assignment, Canonical.assignment, CanonicalBlockAssignment.assignment, dif_pos bound]
    · rw [assignment_after _ _ _ (by omega), raw_after raw _ (by exact Nat.not_lt.mp publicRegion),
        common_lookup, if_pos (by omega)]
  · by_cases middle : target.val < 132904559
    · have found : CoordinateRecovery.source? program target.val =
          some (121293360 + (target.val - 113904174)) := by
        rw [CoordinateRecovery.source?, hashEnd, sharedStart, sharedEnd, if_neg hash, if_pos middle]
      rw [project_value raw target _ found]
      rw [assignment_after _ _ _ (by omega),
        raw_after raw _ (by change 270 ≤ 121293360 + (target.val - 113904174); omega),
        common_lookup, if_neg (by omega), if_pos (by omega)]
      apply congrArg (coordinateAt raw.schedule)
      change 121293090 + (target.val - 270 - 113903904) =
        (121293360 + (target.val - 113904174)) - 270
      omega
    · have found : CoordinateRecovery.source? program target.val =
          some (149282257 + (target.val - 132904559)) := by
        rw [CoordinateRecovery.source?, hashEnd, sharedStart, sharedEnd, applicationStart,
          if_neg hash, if_neg middle, if_pos before]
      rw [project_value raw target _ found]
      rw [assignment_after _ _ _ (by omega),
        raw_after raw _ (by change 270 ≤ 149282257 + (target.val - 132904559); omega),
        common_lookup, if_neg (by omega), if_neg (by omega)]
      apply congrArg (coordinateAt raw.schedule)
      change 149281987 + (target.val - 270 - 132904289) =
        (149282257 + (target.val - 132904559)) - 270
      omega

end NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportCommon
