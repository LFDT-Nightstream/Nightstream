import NightstreamFPrime.Export.Stage1.PerApplicationPreservation

/-!
Owns the production construction of compact-row shift compatibility for the
PiRLC First54 and combination invocation families.

The generic row-renaming semantics remain in `PerApplicationPreservation`.
This file proves only the fixed family layout facts selected by the canonical
Lean generators.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationCompactPreservation

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1.PerApplicationPackage
open NightstreamFPrime.Export.Stage1.PerApplicationPreservation
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec

private theorem sum_take_add_getD_le_sum (values : List Nat) (index : Nat) :
    (values.take index).sum + values.getD index 0 ≤ values.sum := by
  induction values generalizing index with
  | nil => simp
  | cons head rest inductionHypothesis =>
      cases index with
      | zero => simp
      | succ previous =>
          simp only [List.take_succ_cons, List.sum_cons, List.getD_cons_succ]
          have tail := inductionHypothesis previous
          omega

private theorem laneFreshPrefix_add_cost_le (lane : Nat) :
    PiRLCCombinationInvocations.laneFreshPrefix lane +
        PiRLCCombinationInvocations.laneFreshCost lane ≤ 8100 := by
  have bound := sum_take_add_getD_le_sum
    PiRLCCombinationInvocations.laneFreshCosts lane
  rw [PiRLCCombinationInvocations.laneFreshCosts_sum] at bound
  exact bound

private theorem laneFreshCost_eq (lane : Fin ringDegree) :
    PiRLCCombinationInvocations.laneFreshCost lane.val =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount lane := by
  unfold PiRLCCombinationInvocations.laneFreshCost
    PiRLCCombinationInvocations.laneFreshCosts
  rw [List.getD_eq_get _ _ ⟨lane.val, by simp⟩]
  simp

private theorem coordinateFreshEnd_le
    {blockCount cellCount : Nat} (block : Fin blockCount)
    (lane : Fin ringDegree) (cell : Fin cellCount) :
    PiRLCCombinationInvocations.coordinateFreshPrefix cellCount block.val
        lane.val cell.val +
      PiRLCCombinationInvocations.laneFreshCost lane.val ≤
        PiRLCCombinationInvocations.sourceFreshCount blockCount cellCount := by
  let cost := PiRLCCombinationInvocations.laneFreshCost lane.val
  let lanePrefix := PiRLCCombinationInvocations.laneFreshPrefix lane.val
  have laneBound : lanePrefix + cost ≤ 8100 :=
    laneFreshPrefix_add_cost_le lane.val
  have cellSucc : cell.val + 1 ≤ cellCount := by omega
  have cellPart : cell.val * cost + cost ≤ cellCount * cost := by
    calc
      cell.val * cost + cost = (cell.val + 1) * cost := by ring
      _ ≤ cellCount * cost := Nat.mul_le_mul_right cost cellSucc
  have lanePart :
      cellCount * lanePrefix + cell.val * cost + cost ≤ cellCount * 8100 := by
    calc
      cellCount * lanePrefix + cell.val * cost + cost =
          cellCount * lanePrefix + (cell.val * cost + cost) := by omega
      _ ≤ cellCount * lanePrefix + cellCount * cost :=
        Nat.add_le_add_left cellPart _
      _ = cellCount * (lanePrefix + cost) := by ring
      _ ≤ cellCount * 8100 := Nat.mul_le_mul_left cellCount laneBound
  have blockSucc : block.val + 1 ≤ blockCount := by omega
  unfold PiRLCCombinationInvocations.coordinateFreshPrefix
    PiRLCCombinationInvocations.sourceFreshCount
  change block.val * cellCount * 8100 + cellCount * lanePrefix +
      cell.val * cost + cost ≤ blockCount * cellCount * 8100
  calc
    block.val * cellCount * 8100 + cellCount * lanePrefix +
        cell.val * cost + cost =
        block.val * cellCount * 8100 +
          (cellCount * lanePrefix + cell.val * cost + cost) := by omega
    _ ≤ block.val * cellCount * 8100 + cellCount * 8100 :=
      Nat.add_le_add_left lanePart _
    _ = (block.val + 1) * (cellCount * 8100) := by ring
    _ ≤ blockCount * (cellCount * 8100) :=
      Nat.mul_le_mul_right (cellCount * 8100) blockSucc
    _ = blockCount * cellCount * 8100 := by ring

private theorem shiftRange_private
    (program : Lifecycle.Stage1.Application.Program)
    (range : CompactInputRange)
    (endBound : ∀ offset, offset < range.inputCount →
      range.columnStart + offset * range.columnStride <
        basePackage.layout.constantColumn) :
    CompactRangeCompatible program range := by
  intro offset offsetBound
  exact shiftColumn_add_of_private program range.columnStart
    (offset * range.columnStride) (endBound offset offsetBound)

private theorem shiftRange_suffix
    (program : Lifecycle.Stage1.Application.Program)
    (range : CompactInputRange)
    (startBound : basePackage.layout.constantColumn ≤ range.columnStart) :
    CompactRangeCompatible program range := by
  intro offset _offsetBound
  exact shiftColumn_add_of_suffix program range.columnStart
    (offset * range.columnStride) startBound

private theorem mappedSourceRange_private
    (program : Lifecycle.Stage1.Application.Program)
    (inputStart inputCount sourceStart stride : Nat)
    (affine : ∀ offset, offset < inputCount →
      Spartan.sourceToSpartan (sourceStart + offset * stride) =
        Spartan.sourceToSpartan sourceStart + offset * stride)
    (mappedBound : ∀ offset, offset < inputCount →
      Spartan.sourceToSpartan (sourceStart + offset * stride) <
        basePackage.layout.constantColumn) :
    CompactRangeCompatible program
      ⟨inputStart, inputCount, Spartan.sourceToSpartan sourceStart, stride⟩ := by
  apply shiftRange_private
  intro offset offsetBound
  rw [← affine offset offsetBound]
  exact mappedBound offset offsetBound

private theorem singletonRange_compatible
    (program : Lifecycle.Stage1.Application.Program)
    (inputStart columnStart stride : Nat) :
    CompactRangeCompatible program ⟨inputStart, 1, columnStart, stride⟩ := by
  intro offset offsetBound
  change offset < 1 at offsetBound
  have offsetZero : offset = 0 := by omega
  subst offset
  simp

private theorem sourceToSpartan_local_lt_constant (source : Nat)
    (sourceLocal : Spartan.piCcsPhaseOffset ≤ source)
    (upper : Spartan.piCcsLocalStart + (source - Spartan.piCcsPhaseOffset) <
      Spartan.constantColumn) :
    Spartan.sourceToSpartan source < Spartan.constantColumn := by
  unfold Spartan.sourceToSpartan
  rw [if_neg (by
    norm_num [Spartan.pilotSourceColumnCount, Spartan.piCcsPhaseOffset]
      at sourceLocal ⊢
    omega), if_neg (by
    norm_num [Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset]
      at sourceLocal ⊢
    omega), if_neg (by omega)]
  exact upper

private theorem pilotPriorPrivateColumn_private (column : Nat)
    (upper : column < PilotProduction.priorPublicInputStart) :
    Spartan.sourceToSpartan column < basePackage.layout.constantColumn := by
  have affine := Spartan.sourceToSpartan_add_of_pilotPriorPrivate 0 column
    (by simpa using upper)
  have mappedZero : Spartan.sourceToSpartan 0 = 0 := rfl
  rw [Nat.zero_add, mappedZero, Nat.zero_add] at affine
  rw [affine]
  norm_num [basePackage, Data.circuitPackage_layout, Data.physicalLayout,
    Spartan.constantColumn, PilotProduction.priorPublicInputStart,
    PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq]
    at upper ⊢
  omega

private theorem proofInputColumn_private (column : Nat)
    (lower : Spartan.proofInputSourceStart ≤ column)
    (upper : column < Spartan.piCcsPhaseOffset) :
    Spartan.sourceToSpartan column < basePackage.layout.constantColumn := by
  unfold Spartan.sourceToSpartan
  rw [if_neg (by
    norm_num [Spartan.pilotSourceColumnCount, Spartan.proofInputSourceStart]
      at lower ⊢
    omega), if_neg (by omega), if_pos upper]
  norm_num [basePackage, Data.circuitPackage_layout, Data.physicalLayout,
    Spartan.pilotInputPrivateColumnCount, Spartan.proofInputSourceStart,
    Spartan.piCcsPhaseOffset, Spartan.constantColumn] at lower upper ⊢
  omega

private theorem pilotPriorPublicColumn_suffix (column : Nat)
    (lower : PilotProduction.priorPublicInputStart ≤ column)
    (upper : column < PilotProduction.outputPreimageStart) :
    basePackage.layout.constantColumn ≤ Spartan.sourceToSpartan column := by
  unfold Spartan.sourceToSpartan
  rw [if_pos (by
    norm_num [Spartan.pilotSourceColumnCount,
      PilotProduction.outputPreimageStart, PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
      NightstreamFPrime.Lifecycle.PriorStateHash.publicWidth_eq]
      at upper ⊢
    omega)]
  unfold PilotSpartan.sourceToSpartan
  rw [if_neg (by
    simpa [PilotSpartan.priorPublicStart,
      PilotProduction.priorPublicInputStart] using Nat.not_lt.mpr lower),
    if_pos (by
      simpa [PilotSpartan.outputPreimageStart,
        PilotProduction.outputPreimageStart] using upper)]
  unfold Spartan.liftPilotColumn
  rw [if_neg (by
    norm_num [PilotSpartan.firstPublicStart, PilotSpartan.privateColumnCount_value,
      Spartan.pilotInputPrivateColumnCount] at lower ⊢
    omega), if_neg (by
    norm_num [PilotSpartan.firstPublicStart, PilotSpartan.privateColumnCount_value,
      Spartan.pilotPrivateColumnCount] at lower ⊢
    omega)]
  norm_num [basePackage, Data.circuitPackage_layout, Data.physicalLayout,
    Spartan.privateColumnCount, Spartan.constantColumn]

private theorem samplerColumn_private (column : Nat)
    (lower : PiRLCStarts.phaseLogicalStart ≤ column)
    (upper : column < PiRLCStarts.commitmentLogicalStart) :
    Spartan.sourceToSpartan column < basePackage.layout.constantColumn := by
  have sourceLocal : Spartan.piCcsPhaseOffset ≤ column := by
    have lowerValue : 20064823 ≤ column := by
      simpa [PiRLCStarts.phaseLogicalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset] using lower
    norm_num [Spartan.piCcsPhaseOffset] at lowerValue ⊢
    omega
  apply sourceToSpartan_local_lt_constant column
  · exact sourceLocal
  · norm_num [basePackage, Data.circuitPackage_layout, Data.physicalLayout,
      Spartan.piCcsLocalStart, Spartan.piCcsPhaseOffset,
      Spartan.constantColumn] at sourceLocal ⊢
    have upperValue : column < 20328391 := by
      simpa [PiRLCStarts.commitmentLogicalStart,
        PiRLCStarts.phaseLogicalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset,
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.commitmentOffset,
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.logicalPrivateCount,
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.sourceCount,
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.Sampler.logicalPrivateCount]
        using upper
    omega

private theorem samplerRange_compatible
    (program : Lifecycle.Stage1.Application.Program)
    (inputStart inputCount sourceStart stride : Nat)
    (sourceLower : PiRLCStarts.phaseLogicalStart ≤ sourceStart)
    (sourceUpper : ∀ offset, offset < inputCount →
      sourceStart + offset * stride < PiRLCStarts.commitmentLogicalStart) :
    CompactRangeCompatible program
      ⟨inputStart, inputCount, Spartan.sourceToSpartan sourceStart, stride⟩ := by
  have sourceLocal : Spartan.piCcsPhaseOffset ≤ sourceStart := by
    have lowerValue : 20064823 ≤ sourceStart := by
      simpa [PiRLCStarts.phaseLogicalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset] using sourceLower
    norm_num [Spartan.piCcsPhaseOffset] at lowerValue ⊢
    omega
  apply mappedSourceRange_private
  · intro offset _offsetBound
    apply Spartan.sourceToSpartan_add_of_piCcsLocal
    exact sourceLocal
  · intro offset offsetBound
    apply samplerColumn_private
    · exact Nat.le_trans sourceLower (Nat.le_add_right sourceStart _)
    · exact sourceUpper offset offsetBound

private theorem samplerFreshInterval_private (sourceStart count : Nat)
    (sourceLower : PiRLCStarts.phaseLogicalStart ≤ sourceStart)
    (sourceUpper : sourceStart + count ≤ PiRLCStarts.commitmentFreshStart) :
    Spartan.sourceToSpartan sourceStart + count ≤
      basePackage.layout.constantColumn := by
  have sourceLocal : Spartan.piCcsPhaseOffset ≤ sourceStart := by
    have lowerValue : 20064823 ≤ sourceStart := by
      simpa [PiRLCStarts.phaseLogicalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset] using sourceLower
    norm_num [Spartan.piCcsPhaseOffset] at lowerValue ⊢
    omega
  have affine := Spartan.sourceToSpartan_add_of_piCcsLocal sourceStart count
    sourceLocal
  rw [← affine]
  have mappedUpper : Spartan.sourceToSpartan (sourceStart + count) <
      basePackage.layout.constantColumn := by
    apply sourceToSpartan_local_lt_constant
    · exact Nat.le_trans sourceLocal (Nat.le_add_right sourceStart count)
    · norm_num [basePackage, Data.circuitPackage_layout, Data.physicalLayout,
        Spartan.piCcsLocalStart, Spartan.piCcsPhaseOffset,
        Spartan.constantColumn, PiRLCStarts.phaseLogicalStart,
        PiRLCStarts.commitmentFreshStart_eq] at sourceUpper ⊢
      omega
  exact mappedUpper.le

private theorem piRlcFreshInterval_private (sourceStart count : Nat)
    (sourceLocal : Spartan.piCcsPhaseOffset ≤ sourceStart)
    (sourceUpper : sourceStart + count ≤ PiRLCStarts.outputFreshStart) :
    Spartan.sourceToSpartan sourceStart + count ≤
      basePackage.layout.constantColumn := by
  have outputValue : PiRLCStarts.outputFreshStart = 28973248 := by rfl
  rw [outputValue] at sourceUpper
  have affine := Spartan.sourceToSpartan_add_of_piCcsLocal sourceStart count
    sourceLocal
  rw [← affine]
  have mappedUpper : Spartan.sourceToSpartan (sourceStart + count) <
      basePackage.layout.constantColumn := by
    apply sourceToSpartan_local_lt_constant
    · exact Nat.le_trans sourceLocal (Nat.le_add_right sourceStart count)
    · norm_num [basePackage, Data.circuitPackage_layout, Data.physicalLayout,
        Spartan.piCcsLocalStart, Spartan.piCcsPhaseOffset,
        Spartan.constantColumn]
        at sourceUpper ⊢
      omega
  exact mappedUpper.le

set_option maxRecDepth 100000 in -- fixed-size: 55 First54 position templates
private theorem firstPosition_layout
    (program : Lifecycle.Stage1.Application.Program) (source : Nat)
    (slot : Fin First54Step.slotCount)
    (sourceLt : source < PiRLCFirst54Invocations.sourceCount) :
    CompactInvocationPrivate program
      (PiRLCFirst54Invocations.positionInvocation source 0 slot.val) 0 := by
  constructor
  · simp only [Nat.add_zero]
    rw [PiRLCFirst54Invocations.positionInvocation_localStart]
    change Spartan.sourceToSpartan (PiRLCStarts.selectorFreshStart source) ≤
      basePackage.layout.constantColumn
    apply samplerFreshInterval_private _ 0
    · unfold PiRLCStarts.selectorFreshStart
        PiRLCStarts.samplerSourceFreshStart
      norm_num [PiRLCStarts.samplerFreshStart,
        PiRLCStarts.phaseFreshStart_eq, PiRLCStarts.phaseLogicalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
      omega
    · unfold PiRLCStarts.selectorFreshStart
        PiRLCStarts.samplerSourceFreshStart
      norm_num [PiRLCFirst54Invocations.sourceCount,
        PiRLCStarts.samplerFreshStart, PiRLCStarts.phaseFreshStart_eq,
        PiRLCStarts.commitmentFreshStart_eq] at sourceLt ⊢
      omega
  · intro range member
    rw [PiRLCFirst54Invocations.positionInvocation_zero_inputRanges] at member
    have choices : range =
        { inputStart := 0
          inputCount := 1
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.rejectSourceColumn source 0)
          columnStride := 1 } ∨
      range =
        { inputStart := 1
          inputCount := 1
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.positionSourceStart source 0 + slot.val)
          columnStride := 1 } := by
      simpa only [PiRLCFirst54Invocations.firstPositionInputRanges,
        List.mem_cons, List.not_mem_nil, or_false] using member
    rcases choices with rfl | rfl
    · exact singletonRange_compatible program 0
        (PiRLCFirst54Invocations.finalColumn
          (PiRLCFirst54Invocations.rejectSourceColumn source 0)) 1
    · exact singletonRange_compatible program 1
        (PiRLCFirst54Invocations.finalColumn
          (PiRLCFirst54Invocations.positionSourceStart source 0 + slot.val)) 1

set_option maxRecDepth 100000 in -- fixed-size: 54 First54 value templates
private theorem firstValue_layout
    (program : Lifecycle.Stage1.Application.Program) (source : Nat)
    (slot : Fin First54ValueStep.outputCount)
    (sourceLt : source < PiRLCFirst54Invocations.sourceCount) :
    CompactInvocationPrivate program
      (PiRLCFirst54Invocations.valueInvocation source 0 slot.val) 4 := by
  constructor
  ·
    rw [PiRLCFirst54Invocations.valueInvocation_localStart]
    change Spartan.sourceToSpartan
        (PiRLCStarts.selectorFreshStart source +
          PiRLCFirst54Invocations.valueFreshPrefix 0 slot.val) + 4 ≤
      basePackage.layout.constantColumn
    apply samplerFreshInterval_private _ 4
    · unfold PiRLCStarts.selectorFreshStart
        PiRLCStarts.samplerSourceFreshStart
        PiRLCFirst54Invocations.valueFreshPrefix
        PiRLCFirst54Invocations.positionFreshCount
      norm_num [PiRLCStarts.samplerFreshStart,
        PiRLCStarts.phaseFreshStart_eq, PiRLCStarts.phaseLogicalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
      omega
    · unfold PiRLCStarts.selectorFreshStart
        PiRLCStarts.samplerSourceFreshStart
        PiRLCFirst54Invocations.valueFreshPrefix
        PiRLCFirst54Invocations.positionFreshCount
      have slotLt := slot.isLt
      norm_num [PiRLCFirst54Invocations.sourceCount,
        First54ValueStep.outputCount, PiRLCStarts.samplerFreshStart,
        PiRLCStarts.phaseFreshStart_eq, PiRLCStarts.commitmentFreshStart_eq]
        at sourceLt slotLt ⊢
      omega
  · intro range member
    rw [PiRLCFirst54Invocations.valueInvocation_zero_inputRanges] at member
    have choices : range =
        { inputStart := 0
          inputCount := 1
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.rejectSourceColumn source 0)
          columnStride := 1 } ∨
      range =
        { inputStart := 1
          inputCount := 1
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.remainderSourceColumn source 0)
          columnStride := 1 } ∨
      range =
        { inputStart := 2
          inputCount := 1
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.valueSourceStart source 0 + slot.val)
          columnStride := 1 } := by
      simpa only [PiRLCFirst54Invocations.firstValueInputRanges,
        List.mem_cons, List.not_mem_nil, or_false] using member
    rcases choices with rfl | rfl | rfl
    · exact singletonRange_compatible program 0
        (PiRLCFirst54Invocations.finalColumn
          (PiRLCFirst54Invocations.rejectSourceColumn source 0)) 1
    · exact singletonRange_compatible program 1
        (PiRLCFirst54Invocations.finalColumn
          (PiRLCFirst54Invocations.remainderSourceColumn source 0)) 1
    · exact singletonRange_compatible program 2
        (PiRLCFirst54Invocations.finalColumn
          (PiRLCFirst54Invocations.valueSourceStart source 0 + slot.val)) 1

private def laterPositionLocalCount (slot : Nat) : Nat :=
  if slot = 0 then 0 else if slot = 54 then 3 else 6

private theorem laterPositionLocalCount_le (slot : Nat) :
    laterPositionLocalCount slot ≤ 6 := by
  unfold laterPositionLocalCount
  by_cases zero : slot = 0
  · simp [zero]
  · by_cases last : slot = 54
    · simp [zero, last]
    · simp [zero, last]

set_option maxRecDepth 100000 in -- fixed-size: 55 First54 position templates
private theorem laterPosition_layout
    (program : Lifecycle.Stage1.Application.Program) (source round : Nat)
    (slot : Fin First54Step.slotCount)
    (sourceLt : source < PiRLCFirst54Invocations.sourceCount)
    (roundLt : round + 1 < PiRLCFirst54Invocations.roundCount) :
    CompactInvocationPrivate program
      (PiRLCFirst54Invocations.positionInvocation source (round + 1) slot.val)
      (laterPositionLocalCount slot.val) := by
  constructor
  · rw [PiRLCFirst54Invocations.positionInvocation_localStart]
    change Spartan.sourceToSpartan
        (PiRLCStarts.selectorFreshStart source +
          PiRLCFirst54Invocations.roundFreshPrefix (round + 1) +
          PiRLCFirst54Invocations.positionFreshPrefix (round + 1) slot.val) +
          laterPositionLocalCount slot.val ≤
      basePackage.layout.constantColumn
    apply samplerFreshInterval_private _ (laterPositionLocalCount slot.val)
    · unfold PiRLCStarts.selectorFreshStart
        PiRLCStarts.samplerSourceFreshStart
        PiRLCFirst54Invocations.roundFreshPrefix
        PiRLCFirst54Invocations.positionFreshPrefix
      norm_num [PiRLCStarts.samplerFreshStart,
        PiRLCStarts.phaseFreshStart_eq, PiRLCStarts.phaseLogicalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
      omega
    · have slotLt := slot.isLt
      have countLe := laterPositionLocalCount_le slot.val
      unfold PiRLCStarts.selectorFreshStart
        PiRLCStarts.samplerSourceFreshStart
      by_cases slotZero : slot.val = 0
      · simp [PiRLCFirst54Invocations.roundFreshPrefix,
          PiRLCFirst54Invocations.positionFreshPrefix, slotZero]
        rw [show PiRLCStarts.samplerFreshStart = 20380717 by rfl,
          PiRLCStarts.commitmentFreshStart_eq]
        norm_num [PiRLCFirst54Invocations.sourceCount,
          PiRLCFirst54Invocations.roundCount, First54.candidateCount,
          First54Step.slotCount, laterPositionLocalCount]
          at sourceLt roundLt slotLt countLe ⊢
        omega
      · simp [PiRLCFirst54Invocations.roundFreshPrefix,
          PiRLCFirst54Invocations.positionFreshPrefix, slotZero]
        rw [show PiRLCStarts.samplerFreshStart = 20380717 by rfl,
          PiRLCStarts.commitmentFreshStart_eq]
        norm_num [PiRLCFirst54Invocations.sourceCount,
          PiRLCFirst54Invocations.roundCount, First54.candidateCount,
          First54Step.slotCount] at sourceLt roundLt slotLt countLe ⊢
        omega
  · intro range member
    rw [PiRLCFirst54Invocations.positionInvocation_succ_inputRanges] at member
    have choices : range =
        { inputStart := 0
          inputCount := 1
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.rejectSourceColumn source (round + 1))
          columnStride := 1 } ∨
      range =
        { inputStart := 1
          inputCount := First54Step.slotCount
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.previousPositionSourceStart source
              (round + 1))
          columnStride := 1 } ∨
      range =
        { inputStart := 56
          inputCount := 1
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.positionSourceStart source (round + 1) +
              slot.val)
          columnStride := 1 } := by
      simpa only [PiRLCFirst54Invocations.laterPositionInputRanges,
        List.mem_cons, List.not_mem_nil, or_false] using member
    rcases choices with rfl | rfl | rfl
    · exact singletonRange_compatible program 0
        (PiRLCFirst54Invocations.finalColumn
          (PiRLCFirst54Invocations.rejectSourceColumn source (round + 1))) 1
    · apply samplerRange_compatible
      · unfold PiRLCFirst54Invocations.previousPositionSourceStart
          PiRLCFirst54Invocations.positionSourceStart
          First54.positionOffset PiRLCStarts.selectorLogicalStart
          PiRLCStarts.samplerSourceLogicalStart
          First54.roundPrivateCount
        change 20064823 ≤
          20064823 + source * 15504 + 8528 + round * 109
        omega
      · intro offset offsetLt
        unfold PiRLCFirst54Invocations.previousPositionSourceStart
          PiRLCFirst54Invocations.positionSourceStart
          First54.positionOffset PiRLCStarts.selectorLogicalStart
          PiRLCStarts.samplerSourceLogicalStart
          First54.roundPrivateCount
        rw [show PiRLCStarts.samplerLogicalStart = 20064823 by rfl,
          show PiRLCStarts.commitmentLogicalStart = 20328391 by rfl]
        norm_num [PiRLCFirst54Invocations.sourceCount,
          PiRLCFirst54Invocations.roundCount, First54.candidateCount,
          First54Step.slotCount, First54ValueStep.outputCount]
          at sourceLt roundLt offsetLt ⊢
        omega
    · exact singletonRange_compatible program 56
        (PiRLCFirst54Invocations.finalColumn
          (PiRLCFirst54Invocations.positionSourceStart source (round + 1) +
            slot.val)) 1

set_option maxRecDepth 100000 in -- fixed-size: 54 First54 value templates
private theorem laterValue_layout
    (program : Lifecycle.Stage1.Application.Program) (source round : Nat)
    (slot : Fin First54ValueStep.outputCount)
    (sourceLt : source < PiRLCFirst54Invocations.sourceCount)
    (roundLt : round + 1 < PiRLCFirst54Invocations.roundCount) :
    CompactInvocationPrivate program
      (PiRLCFirst54Invocations.valueInvocation source (round + 1) slot.val) 4 := by
  constructor
  · rw [PiRLCFirst54Invocations.valueInvocation_localStart]
    change Spartan.sourceToSpartan
        (PiRLCStarts.selectorFreshStart source +
          PiRLCFirst54Invocations.roundFreshPrefix (round + 1) +
          PiRLCFirst54Invocations.valueFreshPrefix (round + 1) slot.val) + 4 ≤
      basePackage.layout.constantColumn
    apply samplerFreshInterval_private _ 4
    · unfold PiRLCStarts.selectorFreshStart
        PiRLCStarts.samplerSourceFreshStart
        PiRLCFirst54Invocations.roundFreshPrefix
        PiRLCFirst54Invocations.valueFreshPrefix
        PiRLCFirst54Invocations.positionFreshCount
      norm_num [PiRLCStarts.samplerFreshStart,
        PiRLCStarts.phaseFreshStart_eq, PiRLCStarts.phaseLogicalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset]
      omega
    · have slotLt := slot.isLt
      unfold PiRLCStarts.selectorFreshStart
        PiRLCStarts.samplerSourceFreshStart
        PiRLCFirst54Invocations.roundFreshPrefix
        PiRLCFirst54Invocations.valueFreshPrefix
        PiRLCFirst54Invocations.positionFreshCount
      rw [show PiRLCStarts.samplerFreshStart = 20380717 by rfl,
        PiRLCStarts.commitmentFreshStart_eq]
      norm_num [PiRLCFirst54Invocations.sourceCount,
        PiRLCFirst54Invocations.roundCount, First54.candidateCount,
        First54ValueStep.outputCount] at sourceLt roundLt slotLt ⊢
      omega
  · intro range member
    rw [PiRLCFirst54Invocations.valueInvocation_succ_inputRanges] at member
    have choices : range =
        { inputStart := 0
          inputCount := 1
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.rejectSourceColumn source (round + 1))
          columnStride := 1 } ∨
      range =
        { inputStart := 1
          inputCount := 1
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.remainderSourceColumn source (round + 1))
          columnStride := 1 } ∨
      range =
        { inputStart := 2
          inputCount := First54Step.slotCount
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.previousPositionSourceStart source
              (round + 1))
          columnStride := 1 } ∨
      range =
        { inputStart := 57
          inputCount := First54ValueStep.outputCount
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.previousValueSourceStart source
              (round + 1))
          columnStride := 1 } ∨
      range =
        { inputStart := 111
          inputCount := 1
          columnStart := PiRLCFirst54Invocations.finalColumn
            (PiRLCFirst54Invocations.valueSourceStart source (round + 1) +
              slot.val)
          columnStride := 1 } := by
      simpa only [PiRLCFirst54Invocations.laterValueInputRanges,
        List.mem_cons, List.not_mem_nil, or_false] using member
    rcases choices with rfl | rfl | rfl | rfl | rfl
    · exact singletonRange_compatible program 0
        (PiRLCFirst54Invocations.finalColumn
          (PiRLCFirst54Invocations.rejectSourceColumn source (round + 1))) 1
    · exact singletonRange_compatible program 1
        (PiRLCFirst54Invocations.finalColumn
          (PiRLCFirst54Invocations.remainderSourceColumn source (round + 1))) 1
    · apply samplerRange_compatible
      · unfold PiRLCFirst54Invocations.previousPositionSourceStart
          PiRLCFirst54Invocations.positionSourceStart
          First54.positionOffset PiRLCStarts.selectorLogicalStart
          PiRLCStarts.samplerSourceLogicalStart First54.roundPrivateCount
        change 20064823 ≤
          20064823 + source * 15504 + 8528 + round * 109
        omega
      · intro offset offsetLt
        unfold PiRLCFirst54Invocations.previousPositionSourceStart
          PiRLCFirst54Invocations.positionSourceStart
          First54.positionOffset PiRLCStarts.selectorLogicalStart
          PiRLCStarts.samplerSourceLogicalStart First54.roundPrivateCount
        rw [show PiRLCStarts.samplerLogicalStart = 20064823 by rfl,
          show PiRLCStarts.commitmentLogicalStart = 20328391 by rfl]
        norm_num [PiRLCFirst54Invocations.sourceCount,
          PiRLCFirst54Invocations.roundCount, First54.candidateCount,
          First54Step.slotCount, First54ValueStep.outputCount]
          at sourceLt roundLt offsetLt ⊢
        omega
    · apply samplerRange_compatible
      · unfold PiRLCFirst54Invocations.previousValueSourceStart
          PiRLCFirst54Invocations.valueSourceStart First54.valueOffset
          First54.positionOffset PiRLCStarts.selectorLogicalStart
          PiRLCStarts.samplerSourceLogicalStart First54.roundPrivateCount
        change 20064823 ≤
          20064823 + source * 15504 + 8528 + round * 109 + 55
        omega
      · intro offset offsetLt
        unfold PiRLCFirst54Invocations.previousValueSourceStart
          PiRLCFirst54Invocations.valueSourceStart First54.valueOffset
          First54.positionOffset PiRLCStarts.selectorLogicalStart
          PiRLCStarts.samplerSourceLogicalStart First54.roundPrivateCount
        rw [show PiRLCStarts.samplerLogicalStart = 20064823 by rfl,
          show PiRLCStarts.commitmentLogicalStart = 20328391 by rfl]
        norm_num [PiRLCFirst54Invocations.sourceCount,
          PiRLCFirst54Invocations.roundCount, First54.candidateCount,
          First54Step.slotCount, First54ValueStep.outputCount]
          at sourceLt roundLt offsetLt ⊢
        omega
    · exact singletonRange_compatible program 111
        (PiRLCFirst54Invocations.finalColumn
          (PiRLCFirst54Invocations.valueSourceStart source (round + 1) +
            slot.val)) 1

set_option maxRecDepth 100000 in -- fixed-size: 55 First54 position templates
private theorem firstPosition_row
    (program : Lifecycle.Stage1.Application.Program) (source : Nat)
    (slot : Fin First54Step.slotCount)
    (sourceLt : source < PiRLCFirst54Invocations.sourceCount)
    (row : CompactTemplateRow)
    (rowMember : row ∈
      (PiRLCFirst54Templates.firstPositionTemplate slot).rows) :
    instantiateCompactRow
        (shiftCompactRowInvocation program
          (PiRLCFirst54Invocations.positionInvocation source 0 slot.val)) row =
      CompactRows.renameRow (shiftColumn program)
        (instantiateCompactRow
          (PiRLCFirst54Invocations.positionInvocation source 0 slot.val) row) := by
  have sourceMember := rowMember
  change row ∈ (CompactRows.compactConstraintTemplate
    PiRLCFirst54Templates.firstPositionInputCount
    PiRLCFirst54Templates.firstPositionOutputInput
    (PiRLCFirst54Templates.firstPositionRecipe slot)).rows at sourceMember
  have within := compactConstraintTemplate_rowWithin
    PiRLCFirst54Templates.firstPositionInputCount
    PiRLCFirst54Templates.firstPositionOutputInput
    (PiRLCFirst54Templates.firstPositionRecipe slot) row
    (PiRLCFirst54Templates.firstPosition_constraint_varsBelow slot)
    sourceMember
  rw [PiRLCFirst54Templates.firstPosition_constraintFreshCount] at within
  exact instantiateCompactRow_mapColumns_of_within
    (shiftCompactRowInvocation program
      (PiRLCFirst54Invocations.positionInvocation source 0 slot.val))
    (PiRLCFirst54Invocations.positionInvocation source 0 slot.val)
    (shiftColumn program) PiRLCFirst54Templates.firstPositionInputCount 0 row
    within
    (shiftedCompactColumn program
      (PiRLCFirst54Invocations.positionInvocation source 0 slot.val)
      PiRLCFirst54Templates.firstPositionInputCount 0
      (firstPosition_layout program source slot sourceLt))

set_option maxRecDepth 100000 in -- fixed-size: 55 First54 position templates
private theorem laterPosition_row
    (program : Lifecycle.Stage1.Application.Program) (source round : Nat)
    (slot : Fin First54Step.slotCount)
    (sourceLt : source < PiRLCFirst54Invocations.sourceCount)
    (roundLt : round + 1 < PiRLCFirst54Invocations.roundCount)
    (row : CompactTemplateRow)
    (rowMember : row ∈
      (PiRLCFirst54Templates.laterPositionTemplate slot).rows) :
    instantiateCompactRow
        (shiftCompactRowInvocation program
          (PiRLCFirst54Invocations.positionInvocation source (round + 1)
            slot.val)) row =
      CompactRows.renameRow (shiftColumn program)
        (instantiateCompactRow
          (PiRLCFirst54Invocations.positionInvocation source (round + 1)
            slot.val) row) := by
  have sourceMember := rowMember
  change row ∈ (CompactRows.compactConstraintTemplate
    PiRLCFirst54Templates.laterPositionInputCount
    PiRLCFirst54Templates.laterPositionOutputInput
    (PiRLCFirst54Templates.laterPositionRecipe slot)).rows at sourceMember
  have within := compactConstraintTemplate_rowWithin
    PiRLCFirst54Templates.laterPositionInputCount
    PiRLCFirst54Templates.laterPositionOutputInput
    (PiRLCFirst54Templates.laterPositionRecipe slot) row
    (PiRLCFirst54Templates.laterPosition_constraint_varsBelow slot)
    sourceMember
  rw [PiRLCFirst54Templates.laterPosition_constraintFreshCount] at within
  change CompactTemplateRowWithin
    PiRLCFirst54Templates.laterPositionInputCount
    (laterPositionLocalCount slot.val) row at within
  exact instantiateCompactRow_mapColumns_of_within
    (shiftCompactRowInvocation program
      (PiRLCFirst54Invocations.positionInvocation source (round + 1) slot.val))
    (PiRLCFirst54Invocations.positionInvocation source (round + 1) slot.val)
    (shiftColumn program) PiRLCFirst54Templates.laterPositionInputCount
    (laterPositionLocalCount slot.val) row within
    (shiftedCompactColumn program
      (PiRLCFirst54Invocations.positionInvocation source (round + 1) slot.val)
      PiRLCFirst54Templates.laterPositionInputCount
      (laterPositionLocalCount slot.val)
      (laterPosition_layout program source round slot sourceLt roundLt))

set_option maxRecDepth 100000 in -- fixed-size: 54 First54 value templates
private theorem firstValue_row
    (program : Lifecycle.Stage1.Application.Program) (source : Nat)
    (slot : Fin First54ValueStep.outputCount)
    (sourceLt : source < PiRLCFirst54Invocations.sourceCount)
    (row : CompactTemplateRow)
    (rowMember : row ∈
      (PiRLCFirst54Templates.firstValueTemplate slot).rows) :
    instantiateCompactRow
        (shiftCompactRowInvocation program
          (PiRLCFirst54Invocations.valueInvocation source 0 slot.val)) row =
      CompactRows.renameRow (shiftColumn program)
        (instantiateCompactRow
          (PiRLCFirst54Invocations.valueInvocation source 0 slot.val) row) := by
  have sourceMember := rowMember
  change row ∈ (CompactRows.compactConstraintTemplate
    PiRLCFirst54Templates.firstValueInputCount
    PiRLCFirst54Templates.firstValueOutputInput
    (PiRLCFirst54Templates.firstValueRecipe slot)).rows at sourceMember
  have within := compactConstraintTemplate_rowWithin
    PiRLCFirst54Templates.firstValueInputCount
    PiRLCFirst54Templates.firstValueOutputInput
    (PiRLCFirst54Templates.firstValueRecipe slot) row
    (PiRLCFirst54Templates.firstValue_constraint_varsBelow slot) sourceMember
  rw [PiRLCFirst54Templates.firstValue_constraintFreshCount] at within
  exact instantiateCompactRow_mapColumns_of_within
    (shiftCompactRowInvocation program
      (PiRLCFirst54Invocations.valueInvocation source 0 slot.val))
    (PiRLCFirst54Invocations.valueInvocation source 0 slot.val)
    (shiftColumn program) PiRLCFirst54Templates.firstValueInputCount 4 row
    within
    (shiftedCompactColumn program
      (PiRLCFirst54Invocations.valueInvocation source 0 slot.val)
      PiRLCFirst54Templates.firstValueInputCount 4
      (firstValue_layout program source slot sourceLt))

set_option maxRecDepth 100000 in -- fixed-size: 54 First54 value templates
private theorem laterValue_row
    (program : Lifecycle.Stage1.Application.Program) (source round : Nat)
    (slot : Fin First54ValueStep.outputCount)
    (sourceLt : source < PiRLCFirst54Invocations.sourceCount)
    (roundLt : round + 1 < PiRLCFirst54Invocations.roundCount)
    (row : CompactTemplateRow)
    (rowMember : row ∈
      (PiRLCFirst54Templates.laterValueTemplate slot).rows) :
    instantiateCompactRow
        (shiftCompactRowInvocation program
          (PiRLCFirst54Invocations.valueInvocation source (round + 1)
            slot.val)) row =
      CompactRows.renameRow (shiftColumn program)
        (instantiateCompactRow
          (PiRLCFirst54Invocations.valueInvocation source (round + 1)
            slot.val) row) := by
  have sourceMember := rowMember
  change row ∈ (CompactRows.compactConstraintTemplate
    PiRLCFirst54Templates.laterValueInputCount
    PiRLCFirst54Templates.laterValueOutputInput
    (PiRLCFirst54Templates.laterValueRecipe slot)).rows at sourceMember
  have within := compactConstraintTemplate_rowWithin
    PiRLCFirst54Templates.laterValueInputCount
    PiRLCFirst54Templates.laterValueOutputInput
    (PiRLCFirst54Templates.laterValueRecipe slot) row
    (PiRLCFirst54Templates.laterValue_constraint_varsBelow slot) sourceMember
  rw [PiRLCFirst54Templates.laterValue_constraintFreshCount] at within
  exact instantiateCompactRow_mapColumns_of_within
    (shiftCompactRowInvocation program
      (PiRLCFirst54Invocations.valueInvocation source (round + 1) slot.val))
    (PiRLCFirst54Invocations.valueInvocation source (round + 1) slot.val)
    (shiftColumn program) PiRLCFirst54Templates.laterValueInputCount 4 row
    within
    (shiftedCompactColumn program
      (PiRLCFirst54Invocations.valueInvocation source (round + 1) slot.val)
      PiRLCFirst54Templates.laterValueInputCount 4
      (laterValue_layout program source round slot sourceLt roundLt))

private theorem baseTemplates_eq_first54PackageTemplates :
    basePackage.compactRowTemplates =
      PiRLCFirst54Invocations.packageTemplates := by
  change Data.compactRowTemplates () = _
  rw [Data.compactRowTemplates_eq]
  rfl

theorem first54Rows
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : CompactRowInvocation)
    (invocationMember : invocation ∈ PiRLCFirst54Invocations.invocations)
    (template : CompactRowTemplate)
    (templateEquation : basePackage.compactRowTemplates[
      invocation.templateIndex]? = some template)
    (row : CompactTemplateRow) (rowMember : row ∈ template.rows) :
    instantiateCompactRow
        (shiftCompactRowInvocation program invocation) row =
      CompactRows.renameRow (shiftColumn program)
        (instantiateCompactRow invocation row) := by
  unfold PiRLCFirst54Invocations.invocations at invocationMember
  rcases List.mem_flatMap.mp invocationMember with
    ⟨source, sourceMember, sourceInvocationMember⟩
  have sourceLt := List.mem_range.mp sourceMember
  unfold PiRLCFirst54Invocations.sourceInvocations at sourceInvocationMember
  rcases List.mem_flatMap.mp sourceInvocationMember with
    ⟨round, roundMember, roundInvocationMember⟩
  have roundLt := List.mem_range.mp roundMember
  unfold PiRLCFirst54Invocations.roundInvocations at roundInvocationMember
  rcases List.mem_append.mp roundInvocationMember with
      positionMember | valueMember
  · unfold PiRLCFirst54Invocations.positionInvocations at positionMember
    rcases List.mem_map.mp positionMember with ⟨slot, _slotMember, rfl⟩
    rw [baseTemplates_eq_first54PackageTemplates] at templateEquation
    cases round with
    | zero =>
        rw [PiRLCFirst54Invocations.positionInvocation_zero_template]
          at templateEquation
        have equals := Option.some.inj templateEquation
        subst template
        exact firstPosition_row program source slot sourceLt row rowMember
    | succ previous =>
        rw [PiRLCFirst54Invocations.positionInvocation_succ_template]
          at templateEquation
        have equals := Option.some.inj templateEquation
        subst template
        exact laterPosition_row program source previous slot sourceLt roundLt
          row rowMember
  · unfold PiRLCFirst54Invocations.valueInvocations at valueMember
    rcases List.mem_map.mp valueMember with ⟨slot, _slotMember, rfl⟩
    rw [baseTemplates_eq_first54PackageTemplates] at templateEquation
    cases round with
    | zero =>
        rw [PiRLCFirst54Invocations.valueInvocation_zero_template]
          at templateEquation
        have equals := Option.some.inj templateEquation
        subst template
        exact firstValue_row program source slot sourceLt row rowMember
    | succ previous =>
        rw [PiRLCFirst54Invocations.valueInvocation_succ_template]
          at templateEquation
        have equals := Option.some.inj templateEquation
        subst template
        exact laterValue_row program source previous slot sourceLt roundLt
          row rowMember

private theorem combination_layout
    (program : Lifecycle.Stage1.Application.Program)
    (logicalStart rowStart freshStart blockCount cellCount valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount)
    (freshStartLocal : Spartan.piCcsPhaseOffset ≤ freshStart)
    (familyEnd : freshStart +
      PiRLCCombinationInvocations.sourceCount *
        PiRLCCombinationInvocations.sourceFreshCount blockCount cellCount ≤
      PiRLCStarts.outputFreshStart)
    (valueCompatible : CompactRangeCompatible program
      { inputStart := PiRLCCombinationTemplates.valueInputStart
        inputCount := ringDegree
        columnStart := Spartan.sourceToSpartan
          (valueSourceStart source.val block.val cell.val)
        columnStride := valueStride }) :
    CompactInvocationPrivate program
      (PiRLCCombinationInvocations.invocation logicalStart rowStart freshStart
        blockCount cellCount valueStride source.val block.val lane.val cell.val
        valueSourceStart)
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount
        lane) := by
  constructor
  · rw [PiRLCCombinationInvocations.invocation_localStart]
    apply piRlcFreshInterval_private
    · exact PiRLCCombinationInvocations.invocationFreshSource_local
        freshStart blockCount cellCount source.val block.val lane.val cell.val
        freshStartLocal
    · have coordinate := coordinateFreshEnd_le block lane cell
      rw [laneFreshCost_eq] at coordinate
      have sourceSucc : source.val + 1 ≤
          PiRLCCombinationInvocations.sourceCount := by omega
      have sourceStep := Nat.mul_le_mul_right
        (PiRLCCombinationInvocations.sourceFreshCount blockCount cellCount)
        sourceSucc
      unfold PiRLCCombinationInvocations.invocationFreshSource
      calc
        freshStart + source.val *
              PiRLCCombinationInvocations.sourceFreshCount blockCount cellCount +
            PiRLCCombinationInvocations.coordinateFreshPrefix cellCount
              block.val lane.val cell.val +
            NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount
              lane ≤
            freshStart + source.val *
              PiRLCCombinationInvocations.sourceFreshCount blockCount cellCount +
              PiRLCCombinationInvocations.sourceFreshCount blockCount
                cellCount := by omega
        _ = freshStart + (source.val + 1) *
              PiRLCCombinationInvocations.sourceFreshCount blockCount
                cellCount := by ring
        _ ≤ freshStart + PiRLCCombinationInvocations.sourceCount *
              PiRLCCombinationInvocations.sourceFreshCount blockCount
                cellCount := Nat.add_le_add_left sourceStep freshStart
        _ ≤ PiRLCStarts.outputFreshStart := familyEnd
  · intro range member
    rw [PiRLCCombinationInvocations.invocation_inputRanges] at member
    let index := PiRLCCombinationInvocations.logicalIndex cellCount block.val
      lane.val cell.val
    let priorSource := if source.val = 0 then 0 else
      logicalStart + (source.val - 1) *
        PiRLCCombinationInvocations.stepSize blockCount cellCount + index
    let outputSource := logicalStart + source.val *
      PiRLCCombinationInvocations.stepSize blockCount cellCount + index
    have choices : range =
        { inputStart := PiRLCCombinationTemplates.challengeInputStart
          inputCount := ringDegree
          columnStart := Spartan.sourceToSpartan
            (PiRLCCombinationInvocations.challengeSourceStart source.val)
          columnStride := 1 } ∨
      range =
        { inputStart := PiRLCCombinationTemplates.valueInputStart
          inputCount := ringDegree
          columnStart := Spartan.sourceToSpartan
            (valueSourceStart source.val block.val cell.val)
          columnStride := valueStride } ∨
      range =
        { inputStart := PiRLCCombinationTemplates.priorInput
          inputCount := 1
          columnStart := Spartan.sourceToSpartan priorSource
          columnStride := 1 } ∨
      range =
        { inputStart := PiRLCCombinationTemplates.outputInput
          inputCount := 1
          columnStart := Spartan.sourceToSpartan outputSource
          columnStride := 1 } := by
      simpa only [PiRLCCombinationInvocations.inputRanges, index, priorSource,
        outputSource, List.mem_cons, List.not_mem_nil, or_false] using member
    rcases choices with rfl | rfl | rfl | rfl
    · apply samplerRange_compatible
      · unfold PiRLCCombinationInvocations.challengeSourceStart
        rw [PiRLCStarts.challengeWordStart_eq]
        omega
      · intro offset offsetLt
        unfold PiRLCCombinationInvocations.challengeSourceStart
        rw [PiRLCStarts.challengeWordStart_eq]
        have sourceLt := source.isLt
        rw [show PiRLCStarts.phaseLogicalStart = 20064823 by rfl,
          show PiRLCStarts.commitmentLogicalStart = 20328391 by rfl]
        norm_num [PiRLCCombinationInvocations.sourceCount, ringDegree]
          at sourceLt offsetLt ⊢
        omega
    · exact valueCompatible
    · exact singletonRange_compatible program PiRLCCombinationTemplates.priorInput
        (Spartan.sourceToSpartan priorSource) 1
    · exact singletonRange_compatible program PiRLCCombinationTemplates.outputInput
        (Spartan.sourceToSpartan outputSource) 1

private theorem commitmentValueRange_compatible
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin 22) (cell : Fin 1) :
    CompactRangeCompatible program
      { inputStart := PiRLCCombinationTemplates.valueInputStart
        inputCount := ringDegree
        columnStart := Spartan.sourceToSpartan
          (PiRLCCombinationInvocations.commitmentValueSourceStart source.val
            block.val cell.val)
        columnStride := 1 } := by
  have sourceLt := source.isLt
  have blockLt := block.isLt
  apply mappedSourceRange_private
  · intro offset offsetLt
    simpa using PiRLCCombinationInvocations.commitmentValueSource_affine
      source.val block.val cell.val offset sourceLt blockLt offsetLt
  · intro offset offsetLt
    by_cases first : source.val = 0
    · unfold PiRLCCombinationInvocations.commitmentValueSourceStart
      rw [if_pos first]
      apply proofInputColumn_private
      · norm_num [PiRLCCombinationInvocations.sourceCount,
          PiRLCCombinationTemplates.valueInputStart,
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.freshCommitmentStart,
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.proofInputStart,
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart,
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextWords,
          Spartan.proofInputSourceStart, ringDegree] at blockLt offsetLt ⊢
        omega
      · norm_num [NightstreamFPrime.Layout.Stage1.PiCCSInputs.freshCommitmentStart,
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.proofInputStart,
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart,
          NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextWords,
          Spartan.piCcsPhaseOffset, ringDegree] at blockLt offsetLt ⊢
        omega
    · unfold PiRLCCombinationInvocations.commitmentValueSourceStart
      rw [if_neg first]
      apply pilotPriorPrivateColumn_private
      norm_num [NightstreamFPrime.Layout.Stage1.PiCCSInputs.runningCommitmentStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.runningGroupStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.runningGroupsStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.priorRunningStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.runningGroupWords,
        PilotProduction.priorPublicInputStart,
        PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
        PiRLCCombinationInvocations.sourceCount, ringDegree]
        at sourceLt blockLt offsetLt ⊢
      omega

private theorem publicInputValueRange_compatible
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin 5) (cell : Fin 1) :
    CompactRangeCompatible program
      { inputStart := PiRLCCombinationTemplates.valueInputStart
        inputCount := ringDegree
        columnStart := Spartan.sourceToSpartan
          (PiRLCCombinationInvocations.publicInputValueSourceStart source.val
            block.val cell.val)
        columnStride := 1 } := by
  have sourceLt := source.isLt
  have blockLt := block.isLt
  by_cases first : source.val = 0
  · apply shiftRange_suffix
    unfold PiRLCCombinationInvocations.publicInputValueSourceStart
    rw [if_pos first]
    apply pilotPriorPublicColumn_suffix
    · omega
    · norm_num [PilotProduction.outputPreimageStart,
        PilotProduction.priorPublicInputStart, PilotProduction.priorPreimageStart,
        PilotProduction.stateHashWords_eq,
        NightstreamFPrime.Lifecycle.PriorStateHash.publicWidth_eq, ringDegree]
        at blockLt ⊢
      omega
  · apply mappedSourceRange_private
    · intro offset offsetLt
      simpa using PiRLCCombinationInvocations.publicInputValueSource_affine
        source.val block.val cell.val offset sourceLt blockLt offsetLt
    · intro offset offsetLt
      unfold PiRLCCombinationInvocations.publicInputValueSourceStart
      rw [if_neg first]
      apply pilotPriorPrivateColumn_private
      norm_num [NightstreamFPrime.Layout.Stage1.PiCCSInputs.runningPublicStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.runningGroupStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.runningGroupsStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.priorRunningStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.runningGroupWords,
        PilotProduction.priorPublicInputStart,
        PilotProduction.priorPreimageStart, PilotProduction.stateHashWords_eq,
        PiRLCCombinationInvocations.sourceCount, ringDegree]
        at sourceLt blockLt offsetLt ⊢
      omega

private theorem evalKValueRange_compatible
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin 1) (cell : Fin 2) :
    CompactRangeCompatible program
      { inputStart := PiRLCCombinationTemplates.valueInputStart
        inputCount := ringDegree
        columnStart := Spartan.sourceToSpartan
          (PiRLCCombinationInvocations.evalKValueSourceStart source.val
            block.val cell.val)
        columnStride := 2 } := by
  have sourceLt := source.isLt
  have cellLt := cell.isLt
  apply mappedSourceRange_private
  · intro offset offsetLt
    exact PiRLCCombinationInvocations.evalKValueSource_affine source.val
      block.val cell.val offset sourceLt cellLt offsetLt
  · intro offset offsetLt
    apply proofInputColumn_private
    · norm_num [PiRLCCombinationInvocations.evalKValueSourceStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.outputEvaluationStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.roundMessageStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.freshCommitmentStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.proofInputStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextWords,
        Spartan.proofInputSourceStart] at sourceLt cellLt offsetLt ⊢
      omega
    · norm_num [PiRLCCombinationInvocations.evalKValueSourceStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.outputEvaluationStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.roundMessageStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.freshCommitmentStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.proofInputStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextWords,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.freshCommitmentWords,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.roundMessageWords,
        Spartan.piCcsPhaseOffset, PiRLCCombinationInvocations.sourceCount,
        ringDegree] at sourceLt cellLt offsetLt ⊢
      omega

private theorem evalAValueRange_compatible
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin 14) (cell : Fin 2) :
    CompactRangeCompatible program
      { inputStart := PiRLCCombinationTemplates.valueInputStart
        inputCount := ringDegree
        columnStart := Spartan.sourceToSpartan
          (PiRLCCombinationInvocations.evalAValueSourceStart source.val
            block.val cell.val)
        columnStride := 2 } := by
  have sourceLt := source.isLt
  have blockLt := block.isLt
  have cellLt := cell.isLt
  apply mappedSourceRange_private
  · intro offset offsetLt
    exact PiRLCCombinationInvocations.evalAValueSource_affine source.val
      block.val cell.val offset sourceLt blockLt cellLt offsetLt
  · intro offset offsetLt
    apply proofInputColumn_private
    · norm_num [PiRLCCombinationInvocations.evalAValueSourceStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.outputEvaluationStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.roundMessageStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.freshCommitmentStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.proofInputStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextWords,
        Spartan.proofInputSourceStart] at sourceLt blockLt cellLt offsetLt ⊢
      omega
    · norm_num [PiRLCCombinationInvocations.evalAValueSourceStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.outputEvaluationStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.roundMessageStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.freshCommitmentStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.proofInputStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextStart,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.expectedContextWords,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.freshCommitmentWords,
        NightstreamFPrime.Layout.Stage1.PiCCSInputs.roundMessageWords,
        Spartan.piCcsPhaseOffset, PiRLCCombinationInvocations.sourceCount,
        ringDegree] at sourceLt blockLt cellLt offsetLt ⊢
      omega

private theorem commitment_layout
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin 22) (lane : Fin ringDegree) (cell : Fin 1) :
    CompactInvocationPrivate program
      (PiRLCCombinationInvocations.invocation
        PiRLCStarts.commitmentLogicalStart PiRLCStarts.commitmentRowStart
        PiRLCStarts.commitmentFreshStart 22 1 1 source.val block.val lane.val
        cell.val PiRLCCombinationInvocations.commitmentValueSourceStart)
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount
        lane) := by
  apply combination_layout
  · exact PiRLCCombinationInvocations.commitmentFreshStart_local
  · change 21124348 + 17 * (22 * 1 * 8100) ≤ 28973248
    norm_num
  · exact commitmentValueRange_compatible program source block cell

private theorem publicInput_layout
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin 5) (lane : Fin ringDegree) (cell : Fin 1) :
    CompactInvocationPrivate program
      (PiRLCCombinationInvocations.invocation
        PiRLCStarts.publicInputLogicalStart PiRLCStarts.publicInputRowStart
        PiRLCStarts.publicInputFreshStart 5 1 1 source.val block.val lane.val
        cell.val PiRLCCombinationInvocations.publicInputValueSourceStart)
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount
        lane) := by
  apply combination_layout
  · exact PiRLCCombinationInvocations.publicInputFreshStart_local
  · change 24153748 + 17 * (5 * 1 * 8100) ≤ 28973248
    norm_num
  · exact publicInputValueRange_compatible program source block cell

private theorem evalK_layout
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin 1) (lane : Fin ringDegree) (cell : Fin 2) :
    CompactInvocationPrivate program
      (PiRLCCombinationInvocations.invocation
        PiRLCStarts.evalKLogicalStart PiRLCStarts.evalKRowStart
        PiRLCStarts.evalKFreshStart 1 2 2 source.val block.val lane.val
        cell.val PiRLCCombinationInvocations.evalKValueSourceStart)
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount
        lane) := by
  apply combination_layout
  · exact PiRLCCombinationInvocations.evalKFreshStart_local
  · change 24842248 + 17 * (1 * 2 * 8100) ≤ 28973248
    norm_num
  · exact evalKValueRange_compatible program source block cell

private theorem evalA_layout
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin 14) (lane : Fin ringDegree) (cell : Fin 2) :
    CompactInvocationPrivate program
      (PiRLCCombinationInvocations.invocation
        PiRLCStarts.evalALogicalStart PiRLCStarts.evalARowStart
        PiRLCStarts.evalAFreshStart 14 2 2 source.val block.val lane.val
        cell.val PiRLCCombinationInvocations.evalAValueSourceStart)
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount
        lane) := by
  apply combination_layout
  · exact PiRLCCombinationInvocations.evalAFreshStart_local
  · change 25117648 + 17 * (14 * 2 * 8100) ≤ 28973248
    norm_num
  · exact evalAValueRange_compatible program source block cell

set_option maxRecDepth 100000 in -- fixed-size: 54 normalized lane templates
private theorem combination_row
    (program : Lifecycle.Stage1.Application.Program)
    (logicalStart rowStart freshStart blockCount cellCount valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount)
    (layout : CompactInvocationPrivate program
      (PiRLCCombinationInvocations.invocation logicalStart rowStart freshStart
        blockCount cellCount valueStride source.val block.val lane.val cell.val
        valueSourceStart)
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount lane))
    (row : CompactTemplateRow)
    (rowMember : row ∈
      (PiRLCCombinationTemplates.template
        (PiRLCCombinationInvocations.firstSource source.val) lane).rows) :
    instantiateCompactRow
        (shiftCompactRowInvocation program
          (PiRLCCombinationInvocations.invocation logicalStart rowStart
            freshStart blockCount cellCount valueStride source.val block.val
            lane.val cell.val valueSourceStart)) row =
      CompactRows.renameRow (shiftColumn program)
        (instantiateCompactRow
          (PiRLCCombinationInvocations.invocation logicalStart rowStart
            freshStart blockCount cellCount valueStride source.val block.val
            lane.val cell.val valueSourceStart) row) := by
  have sourceMember := rowMember
  change row ∈ (CompactRows.compactTemplate
    PiRLCCombinationTemplates.inputCount PiRLCCombinationTemplates.outputInput
    (PiRLCCombinationTemplates.outputRecipe
      (PiRLCCombinationInvocations.firstSource source.val) lane)).rows
    at sourceMember
  have within := compactTemplate_rowWithin PiRLCCombinationTemplates.inputCount
    PiRLCCombinationTemplates.outputInput
    (PiRLCCombinationTemplates.outputRecipe
      (PiRLCCombinationInvocations.firstSource source.val) lane) row
    (PiRLCCombinationTemplates.constraint_varsBelow
      (PiRLCCombinationInvocations.firstSource source.val) lane) sourceMember
  have count : Layout.R1CS.mulCount
      (Expr.var PiRLCCombinationTemplates.outputInput -
        PiRLCCombinationTemplates.outputRecipe
          (PiRLCCombinationInvocations.firstSource source.val) lane) =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount lane := by
    simpa [PiRLCCombinationTemplates.template,
      CompactRows.compactTemplate] using
      (PiRLCCombinationTemplates.template_localColumnCount
        (PiRLCCombinationInvocations.firstSource source.val) lane)
  rw [count] at within
  exact instantiateCompactRow_mapColumns_of_within
    (shiftCompactRowInvocation program
      (PiRLCCombinationInvocations.invocation logicalStart rowStart freshStart
        blockCount cellCount valueStride source.val block.val lane.val cell.val
        valueSourceStart))
    (PiRLCCombinationInvocations.invocation logicalStart rowStart freshStart
      blockCount cellCount valueStride source.val block.val lane.val cell.val
      valueSourceStart)
    (shiftColumn program) PiRLCCombinationTemplates.inputCount
    (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount lane)
    row within
    (shiftedCompactColumn program
      (PiRLCCombinationInvocations.invocation logicalStart rowStart freshStart
        blockCount cellCount valueStride source.val block.val lane.val cell.val
        valueSourceStart) PiRLCCombinationTemplates.inputCount
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount lane)
      layout)

private theorem combinationTemplateSelection
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (lane : Fin ringDegree) :
    basePackage.compactRowTemplates[
        PiRLCCombinationTemplates.templateIndex source.val lane.val]? =
      some (PiRLCCombinationTemplates.template
        (PiRLCCombinationInvocations.firstSource source.val) lane) := by
  rw [baseTemplates_eq_first54PackageTemplates]
  unfold PiRLCFirst54Invocations.packageTemplates
  rw [List.getElem?_append_left
    (PiRLCCombinationTemplates.templateIndex_lt source.val lane)]
  exact PiRLCCombinationTemplates.template_getElem? source.val lane

private theorem combinationFamilyRows
    (program : Lifecycle.Stage1.Application.Program)
    (logicalStart rowStart freshStart blockCount cellCount valueStride : Nat)
    (valueSourceStart : Nat → Nat → Nat → Nat)
    (familyLayout : ∀ source : Fin PiRLCCombinationInvocations.sourceCount,
      ∀ block : Fin blockCount, ∀ lane : Fin ringDegree,
        ∀ cell : Fin cellCount,
          CompactInvocationPrivate program
            (PiRLCCombinationInvocations.invocation logicalStart rowStart
              freshStart blockCount cellCount valueStride source.val block.val
              lane.val cell.val valueSourceStart)
            (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount
              lane))
    (invocation : CompactRowInvocation)
    (invocationMember : invocation ∈
      PiRLCCombinationInvocations.familyInvocations logicalStart rowStart
        freshStart blockCount cellCount valueStride valueSourceStart)
    (template : CompactRowTemplate)
    (templateEquation : basePackage.compactRowTemplates[
      invocation.templateIndex]? = some template)
    (row : CompactTemplateRow) (rowMember : row ∈ template.rows) :
    instantiateCompactRow
        (shiftCompactRowInvocation program invocation) row =
      CompactRows.renameRow (shiftColumn program)
        (instantiateCompactRow invocation row) := by
  unfold PiRLCCombinationInvocations.familyInvocations at invocationMember
  rcases List.mem_flatMap.mp invocationMember with
    ⟨source, sourceMember, indexedMember⟩
  let sourceFin : Fin PiRLCCombinationInvocations.sourceCount :=
    ⟨source, List.mem_range.mp sourceMember⟩
  rcases List.mem_ofFn.mp indexedMember with ⟨index, rfl⟩
  let coordinates :=
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationStep.coordinates index
  have selected := combinationTemplateSelection sourceFin coordinates.2.1
  change basePackage.compactRowTemplates[
      PiRLCCombinationTemplates.templateIndex source coordinates.2.1.val]? =
    some template at templateEquation
  have selectedSource : basePackage.compactRowTemplates[
      PiRLCCombinationTemplates.templateIndex source coordinates.2.1.val]? =
    some (PiRLCCombinationTemplates.template
      (PiRLCCombinationInvocations.firstSource source) coordinates.2.1) := by
    simpa [sourceFin] using selected
  rw [selectedSource] at templateEquation
  have equals := Option.some.inj templateEquation
  subst template
  simpa [sourceFin, coordinates] using
    (combination_row program logicalStart rowStart freshStart blockCount
    cellCount valueStride valueSourceStart sourceFin coordinates.1
    coordinates.2.1 coordinates.2.2
    (familyLayout sourceFin coordinates.1 coordinates.2.1 coordinates.2.2)
    row rowMember)

theorem combinationRows
    (program : Lifecycle.Stage1.Application.Program)
    (invocation : CompactRowInvocation)
    (invocationMember : invocation ∈ PiRLCCombinationInvocations.invocations)
    (template : CompactRowTemplate)
    (templateEquation : basePackage.compactRowTemplates[
      invocation.templateIndex]? = some template)
    (row : CompactTemplateRow) (rowMember : row ∈ template.rows) :
    instantiateCompactRow
        (shiftCompactRowInvocation program invocation) row =
      CompactRows.renameRow (shiftColumn program)
        (instantiateCompactRow invocation row) := by
  unfold PiRLCCombinationInvocations.invocations at invocationMember
  simp only [List.mem_append] at invocationMember
  rcases invocationMember with
      ((commitmentMember | publicInputMember) | evalKMember) | evalAMember
  · exact combinationFamilyRows program PiRLCStarts.commitmentLogicalStart
      PiRLCStarts.commitmentRowStart PiRLCStarts.commitmentFreshStart 22 1 1
      PiRLCCombinationInvocations.commitmentValueSourceStart
      (fun source block lane cell =>
        commitment_layout program source block lane cell)
      invocation commitmentMember template templateEquation row rowMember
  · exact combinationFamilyRows program PiRLCStarts.publicInputLogicalStart
      PiRLCStarts.publicInputRowStart PiRLCStarts.publicInputFreshStart 5 1 1
      PiRLCCombinationInvocations.publicInputValueSourceStart
      (fun source block lane cell =>
        publicInput_layout program source block lane cell)
      invocation publicInputMember template templateEquation row rowMember
  · exact combinationFamilyRows program PiRLCStarts.evalKLogicalStart
      PiRLCStarts.evalKRowStart PiRLCStarts.evalKFreshStart 1 2 2
      PiRLCCombinationInvocations.evalKValueSourceStart
      (fun source block lane cell => evalK_layout program source block lane cell)
      invocation evalKMember template templateEquation row rowMember
  · exact combinationFamilyRows program PiRLCStarts.evalALogicalStart
      PiRLCStarts.evalARowStart PiRLCStarts.evalAFreshStart 14 2 2
      PiRLCCombinationInvocations.evalAValueSourceStart
      (fun source block lane cell => evalA_layout program source block lane cell)
      invocation evalAMember template templateEquation row rowMember

end NightstreamFPrime.Export.Stage1.PerApplicationCompactPreservation
