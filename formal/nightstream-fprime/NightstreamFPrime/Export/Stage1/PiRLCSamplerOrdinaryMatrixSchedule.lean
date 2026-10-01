import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectSource

/-! The physical row schedule for checked reduction and coefficient words,
in scalar order. Each scalar contributes two intervals around its advance
permutation. The permutation has its own matrix plan. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSchedule

open NightstreamFPrime.Layout.MatrixProgram NightstreamFPrime.Layout.Stage1

def reductionRange (source : Nat) : IndexRange where
  start := PiRLCStarts.rangeRowStart source
  count := 2229

def wordRange (source : Nat) : IndexRange where
  start := PiRLCStarts.challengeWordRowStart source
  count := 54

def sourceRanges (source : Nat) : List IndexRange := [reductionRange source, wordRange source]
def ranges : List IndexRange :=
  (List.range PiRLCSamplerInvocations.sourceCount).flatMap sourceRanges

def rowSchedule : IndexSchedule := .rangeList ranges
def rowIndexReference : List Nat := ranges.flatMap IndexRange.indices

private theorem sum_map_flatMap {Alpha Beta : Type}
    (items : List Alpha) (children : Alpha → List Beta) (weight : Beta → Nat) :
    ((items.flatMap children).map weight).sum =
      (items.map fun item => ((children item).map weight).sum).sum := by
  induction items with
  | nil => rfl
  | cons item rest ih => simp [ih]

@[simp] theorem sourceRanges_count (source : Nat) :
    ((sourceRanges source).map IndexRange.count).sum = 2283 := rfl

@[simp] theorem ranges_count : (ranges.map IndexRange.count).sum = 38811 := by
  rw [ranges, sum_map_flatMap]
  simp [PiRLCSamplerInvocations.sourceCount]

@[simp] theorem rowSchedule_count : rowSchedule.count = 38811 := ranges_count

theorem rowSchedule_indices : rowSchedule.indices = rowIndexReference := rfl

theorem rowSchedule_index? (ordinal : Nat) :
    rowSchedule.index? ordinal = rowIndexReference[ordinal]? := by
  rw [IndexSchedule.index?_eq_getElem?, rowSchedule_indices]

private theorem sourceRanges_valid (source minimum limit : Nat) (suffix : List IndexRange)
    (minimumLe : minimum ≤ PiRLCStarts.samplerSourceRowStart source + 1096)
    (endLe : PiRLCStarts.samplerSourceRowStart source + 4475 ≤ limit)
    (suffixValid : validIndexRanges limit (PiRLCStarts.samplerSourceRowStart source + 4475) suffix = true) :
    validIndexRanges limit minimum (sourceRanges source ++ suffix) = true := by
  simp [sourceRanges, reductionRange, wordRange, validIndexRanges, IndexRange.endExclusive,
    PiRLCStarts.rangeRowStart, PiRLCStarts.challengeWordRowStart, PiRLCStarts.advanceRowStart,
    minimumLe, endLe, suffixValid]
  all_goals omega

private theorem sourceInterval_valid (start count minimum limit : Nat)
    (minimumLe : minimum ≤ PiRLCStarts.samplerSourceRowStart start + 1096)
    (endLe : PiRLCStarts.samplerSourceRowStart (start + count) ≤ limit) :
    validIndexRanges limit minimum ((List.range' start count).flatMap sourceRanges) = true := by
  induction count generalizing start minimum with
  | zero => rfl
  | succ count ih =>
      simp only [List.range'_succ, List.flatMap_cons]
      apply sourceRanges_valid start minimum limit _ minimumLe
      · unfold PiRLCStarts.samplerSourceRowStart at endLe ⊢
        omega
      · apply ih (start := start + 1)
        · unfold PiRLCStarts.samplerSourceRowStart
          omega
        · simpa only [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using endLe

private theorem rowSchedule_valid_of_le (limit : Nat)
    (endLe : PiRLCStarts.samplerSourceRowStart PiRLCSamplerInvocations.sourceCount ≤ limit) :
    rowSchedule.valid limit = true := by
  simp only [rowSchedule, IndexSchedule.valid, ranges, List.range_eq_range']
  apply sourceInterval_valid 0 PiRLCSamplerInvocations.sourceCount
  · omega
  · exact endLe

private theorem samplerEnd_eq :
    PiRLCStarts.samplerSourceRowStart PiRLCSamplerInvocations.sourceCount = PiRLCStarts.commitmentRowStart := by
  unfold PiRLCStarts.samplerSourceRowStart PiRLCStarts.commitmentRowStart
  exact congrArg (PiRLCStarts.samplerRowStart + ·) (by decide :
    PiRLCSamplerInvocations.sourceCount * 4475 = 76075)

theorem rowSchedule_valid : rowSchedule.valid PiRLCStarts.commitmentRowStart = true :=
  @rowSchedule_valid_of_le PiRLCStarts.commitmentRowStart (Nat.le_of_eq samplerEnd_eq)

theorem rowSchedule_valid_between :
    validIndexRanges PiRLCStarts.outputRowStart PiRLCStarts.phaseRowStart ranges = true := by
  simp only [ranges, List.range_eq_range']
  apply sourceInterval_valid 0 PiRLCSamplerInvocations.sourceCount
  all_goals norm_num [PiRLCSamplerInvocations.sourceCount, PiRLCStarts.samplerSourceRowStart,
    PiRLCStarts.samplerRowStart, PiRLCStarts.phaseRowStart, PiRLCStarts.finalBoundaries_eq.1]

theorem rowIndexReference_nodup : rowIndexReference.Nodup := by
  rw [← rowSchedule_indices]
  exact IndexSchedule.rangeList_indices_nodup _ _ rowSchedule_valid

theorem rowIndexReference_bounds : ∀ index ∈ rowIndexReference,
    PiRLCStarts.phaseRowStart ≤ index ∧ index < PiRLCStarts.outputRowStart := by
  rw [← rowSchedule_indices]
  unfold rowSchedule IndexSchedule.indices
  exact validIndexRanges_indices_bounds _ _ _ rowSchedule_valid_between

theorem rangeRows_rowIndices {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth logicalWidth} (source : Nat) :
    (PiRLCSamplerOrdinaryRows.rangeRows (logicalWidth := logicalWidth) (publicFits := publicFits) source).map
      Rows.CompiledRow.rowIndex = (reductionRange source).indices := by
  rw [PiRLCSamplerOrdinaryRows.rangeRows, PiCCSArithmetic.compilePacket_rowIndices]
  rw [← PiRLCSamplerOrdinaryRows.rangeRows, PiRLCSamplerOrdinaryRows.rangeRows_length]
  rw [IndexRange.indices_eq_range']
  rfl

theorem wordRows_rowIndices (source : Nat) :
    (PiRLCSamplerOrdinaryRows.wordRows source).map Rows.CompiledRow.rowIndex = (wordRange source).indices := by
  rw [PiRLCSamplerOrdinaryRows.wordRows, PiCCSArithmetic.compilePacket_rowIndices]
  rw [← PiRLCSamplerOrdinaryRows.wordRows, PiRLCSamplerOrdinaryRows.wordRows_length]
  rw [IndexRange.indices_eq_range']
  rfl

theorem sourceRows_rowIndices {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth logicalWidth} (source : Nat) :
    (PiRLCSamplerOrdinaryRows.sourceRows (logicalWidth := logicalWidth) (publicFits := publicFits) source).map
      Rows.CompiledRow.rowIndex = (sourceRanges source).flatMap IndexRange.indices := by
  simp only [PiRLCSamplerOrdinaryRows.sourceRows, List.map_append, rangeRows_rowIndices,
    wordRows_rowIndices, sourceRanges, List.flatMap_cons, List.flatMap_nil, List.append_nil]

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
