import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectSource

/-!
Owns the compact physical-row schedule for the PiRLC sampler ordinary rows.
The order is source-major, round-major, lane-major, followed by one selector
row for each source. It matches the canonical sampler ordinary source list.

This module selects row indices only. It does not compile matrix forms.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSchedule

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout.Stage1

def laneRange (source round : Nat) (lane : Fin 4) : IndexRange where
  start := PiRLCStarts.digestLaneRowStart source round lane.val
  count := 406

def selectorRange (source : Nat) : IndexRange where
  start := PiRLCStarts.selectorRowStart source + 41023
  count := 1

def windowRanges (source round : Nat) : List IndexRange :=
  [laneRange source round 0, laneRange source round 1,
    laneRange source round 2, laneRange source round 3]

theorem windowRanges_eq_reference (source round : Nat) :
    windowRanges source round =
      (List.finRange 4).map (laneRange source round) := by
  rfl

def sourceRanges (source : Nat) : List IndexRange :=
  (List.range PiRLCSamplerOrdinaryRows.digestRoundCount).flatMap
      (windowRanges source) ++
    [selectorRange source]

/-- The schedule has 544 lane ranges and 17 selector ranges. -/
def ranges : List IndexRange :=
  (List.range PiRLCSamplerOrdinaryRows.sourceCount).flatMap sourceRanges

def rowSchedule : IndexSchedule := .rangeList ranges

/-- Proof-oriented expansion of the compact physical-row schedule. -/
def rowIndexReference : List Nat := ranges.flatMap IndexRange.indices

private theorem sum_map_flatMap {Alpha Beta : Type}
    (items : List Alpha) (children : Alpha → List Beta) (weight : Beta → Nat) :
    ((items.flatMap children).map weight).sum =
      (items.map fun item => ((children item).map weight).sum).sum := by
  induction items with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      simp [inductionHypothesis]

@[simp] theorem laneRange_count (source round : Nat) (lane : Fin 4) :
    (laneRange source round lane).count = 406 := by
  rfl

@[simp] theorem selectorRange_count (source : Nat) :
    (selectorRange source).count = 1 := by
  rfl

@[simp] theorem windowRanges_count (source round : Nat) :
    ((windowRanges source round).map IndexRange.count).sum = 1624 := by
  norm_num [windowRanges, laneRange, Function.comp_def]

@[simp] theorem sourceRanges_count (source : Nat) :
    ((sourceRanges source).map IndexRange.count).sum = 12993 := by
  rw [sourceRanges, List.map_append, List.sum_append,
    sum_map_flatMap]
  simp [PiRLCSamplerOrdinaryRows.digestRoundCount]

@[simp] theorem ranges_count :
    (ranges.map IndexRange.count).sum = 220881 := by
  rw [ranges, sum_map_flatMap]
  simp [PiRLCSamplerOrdinaryRows.sourceCount]

@[simp] theorem rowSchedule_count : rowSchedule.count = 220881 := by
  exact ranges_count

theorem rowSchedule_indices : rowSchedule.indices = rowIndexReference := by
  rfl

theorem rowSchedule_index? (ordinal : Nat) :
    rowSchedule.index? ordinal = rowIndexReference[ordinal]? := by
  rw [IndexSchedule.index?_eq_getElem?, rowSchedule_indices]

private theorem sourceRanges_valid (source minimum limit : Nat)
    (suffix : List IndexRange)
    (minimumLe : minimum ≤ PiRLCStarts.samplerSourceRowStart source + 592)
    (endLe : PiRLCStarts.samplerSourceRowStart source + 59344 ≤ limit)
    (suffixValid : validIndexRanges limit
      (PiRLCStarts.samplerSourceRowStart source + 59344) suffix = true) :
    validIndexRanges limit minimum (sourceRanges source ++ suffix) = true := by
  have rounds : List.range PiRLCSamplerOrdinaryRows.digestRoundCount =
      [0, 1, 2, 3, 4, 5, 6, 7] := by decide
  rw [sourceRanges, rounds]
  simp [windowRanges, laneRange, selectorRange, validIndexRanges,
    IndexRange.endExclusive, PiRLCStarts.digestLaneRowStart,
    PiRLCStarts.windowRowStart, PiRLCStarts.selectorRowStart,
    minimumLe, endLe, suffixValid]
  all_goals omega

theorem rowSchedule_valid : rowSchedule.valid 27584200 = true := by
  change validIndexRanges 27584200 0 ranges = true
  rw [ranges]
  rw [show List.range PiRLCSamplerOrdinaryRows.sourceCount =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] by
    decide]
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
  repeat
    first
    | apply sourceRanges_valid
    | norm_num [PiRLCStarts.samplerSourceRowStart,
        PiRLCStarts.samplerRowStart, PiRLCStarts.phaseRowStart,
        validIndexRanges]

theorem rowSchedule_valid_between :
    validIndexRanges PiDECStarts.phaseRowStart PiRLCStarts.phaseRowStart
      ranges = true := by
  rw [ranges]
  rw [show List.range PiRLCSamplerOrdinaryRows.sourceCount =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] by
    decide]
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
  repeat
    first
    | apply sourceRanges_valid
    | norm_num [PiRLCStarts.samplerSourceRowStart,
        PiRLCStarts.samplerRowStart, PiRLCStarts.phaseRowStart,
        PiDECStarts.phaseRowStart, validIndexRanges]

theorem rowIndexReference_nodup : rowIndexReference.Nodup := by
  rw [← rowSchedule_indices]
  exact IndexSchedule.rangeList_indices_nodup _ _ rowSchedule_valid

theorem rowIndexReference_bounds :
    ∀ index ∈ rowIndexReference,
      PiRLCStarts.phaseRowStart ≤ index ∧ index < PiDECStarts.phaseRowStart := by
  rw [← rowSchedule_indices]
  unfold rowSchedule IndexSchedule.indices
  exact validIndexRanges_indices_bounds _ _ _ rowSchedule_valid_between

theorem laneRows_rowIndices
    {logicalWidth : Nat}
    {publicFits : NightstreamFPrime.Spec.ringDegree *
      NightstreamFPrime.Lifecycle.PaperAlgebra.publicRingColumns ≤
        NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
          logicalWidth}
    (source round : Nat) (lane : Fin 4) :
    (PiRLCSamplerOrdinaryRows.laneRows
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round lane).map Rows.CompiledRow.rowIndex =
      (laneRange source round lane).indices := by
  calc
    _ = List.range' (PiRLCStarts.digestLaneRowStart source round lane.val)
        (PiRLCSamplerOrdinaryRows.laneRows
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          source round lane).length := by
      exact PiCCSArithmetic.compilePacket_rowIndices _ _ _
    _ = List.range' (PiRLCStarts.digestLaneRowStart source round lane.val)
        406 := by
      rw [PiRLCSamplerOrdinaryRows.laneRows_length]
    _ = (laneRange source round lane).indices := by
      rw [IndexRange.indices_eq_range']
      rfl

theorem selectorRows_rowIndices
    (source : Nat) :
    (PiRLCSamplerOrdinaryRows.selectorFinalRows source).map
        Rows.CompiledRow.rowIndex =
      (selectorRange source).indices := by
  calc
    _ = List.range' (PiRLCStarts.selectorRowStart source + 41023)
        (PiRLCSamplerOrdinaryRows.selectorFinalRows source).length := by
      exact PiCCSArithmetic.compilePacket_rowIndices _ _ _
    _ = List.range' (PiRLCStarts.selectorRowStart source + 41023) 1 := by
      rw [PiRLCSamplerOrdinaryRows.selectorFinalRows_length]
    _ = (selectorRange source).indices := by
      rw [IndexRange.indices_eq_range']
      rfl

theorem windowRows_rowIndices
    {logicalWidth : Nat}
    {publicFits : NightstreamFPrime.Spec.ringDegree *
      NightstreamFPrime.Lifecycle.PaperAlgebra.publicRingColumns ≤
        NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
          logicalWidth}
    (source round : Nat) :
    (PiRLCSamplerOrdinaryRows.windowRows
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source round).map Rows.CompiledRow.rowIndex =
      (windowRanges source round).flatMap IndexRange.indices := by
  rw [windowRanges_eq_reference]
  simp [PiRLCSamplerOrdinaryRows.windowRows,
    laneRows_rowIndices, List.map_flatMap, List.flatMap_map,
    Function.comp_def]

theorem sourceRows_rowIndices
    {logicalWidth : Nat}
    {publicFits : NightstreamFPrime.Spec.ringDegree *
      NightstreamFPrime.Lifecycle.PaperAlgebra.publicRingColumns ≤
        NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
          logicalWidth}
    (source : Nat) :
    (PiRLCSamplerOrdinaryRows.sourceRows
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        source).map Rows.CompiledRow.rowIndex =
      (sourceRanges source).flatMap IndexRange.indices := by
  simp [PiRLCSamplerOrdinaryRows.sourceRows, sourceRanges,
    List.map_flatMap, windowRows_rowIndices, selectorRows_rowIndices,
    List.flatMap_assoc, Function.comp_def]

/-- The compact schedule is exactly the physical index stream emitted by the
canonical sampler ordinary builder. -/
theorem arithmeticRows_rowIndices
    {logicalWidth : Nat}
    {publicFits : NightstreamFPrime.Spec.ringDegree *
      NightstreamFPrime.Lifecycle.PaperAlgebra.publicRingColumns ≤
        NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
          logicalWidth} :
    (PiRLCSamplerOrdinaryRows.rows
        (logicalWidth := logicalWidth) (publicFits := publicFits)).map
        Rows.CompiledRow.rowIndex = rowIndexReference := by
  simp [PiRLCSamplerOrdinaryRows.rows, ranges, rowIndexReference,
    List.map_flatMap, sourceRows_rowIndices, List.flatMap_assoc,
    Function.comp_def]

theorem rowSchedule_index?_eq_arithmeticRowIndex?
    {logicalWidth : Nat}
    {publicFits : NightstreamFPrime.Spec.ringDegree *
      NightstreamFPrime.Lifecycle.PaperAlgebra.publicRingColumns ≤
        NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
          logicalWidth}
    (ordinal : Nat) :
    rowSchedule.index? ordinal =
      ((PiRLCSamplerOrdinaryRows.rows
        (logicalWidth := logicalWidth) (publicFits := publicFits))[ordinal]?).map
        Rows.CompiledRow.rowIndex := by
  rw [rowSchedule_index?]
  rw [← arithmeticRows_rowIndices (logicalWidth := logicalWidth)
    (publicFits := publicFits)]
  simp

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSchedule
