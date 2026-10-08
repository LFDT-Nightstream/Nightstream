import NightstreamFPrime.Export.Stage1.PerApplicationPreservation

/-!
Owns the production construction of compact-row shift compatibility for the
PiRLC combination invocation families.

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
      PilotProduction.priorPublicInputStart] using! Nat.not_lt.mpr lower),
    if_pos (by
      simpa [PilotSpartan.outputPreimageStart,
        PilotProduction.outputPreimageStart] using! upper)]
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
    have lowerValue : 7207123 ≤ column := by
      simpa [PiRLCStarts.phaseLogicalStart,
        NightstreamFPrime.Layout.Stage1.PiRLCInputs.phaseOffset] using lower
    norm_num [Spartan.piCcsPhaseOffset] at lowerValue ⊢
    omega
  apply sourceToSpartan_local_lt_constant column
  · exact sourceLocal
  · norm_num [basePackage, Data.circuitPackage_layout, Data.physicalLayout,
      Spartan.piCcsLocalStart, Spartan.piCcsPhaseOffset,
      Spartan.constantColumn] at sourceLocal ⊢
    have upperValue : column < 7279662 := by
      change column < 7279662 at upper
      exact upper
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
    have lowerValue : 7207123 ≤ sourceStart := by
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

private theorem piRlcFreshInterval_private (sourceStart count : Nat)
    (sourceLocal : Spartan.piCcsPhaseOffset ≤ sourceStart)
    (sourceUpper : sourceStart + count ≤ PiRLCStarts.outputFreshStart) :
    Spartan.sourceToSpartan sourceStart + count ≤
      basePackage.layout.constantColumn := by
  have outputValue : PiRLCStarts.outputFreshStart = 12410976 := by rfl
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
        rw [show PiRLCStarts.phaseLogicalStart = 7207123 by rfl,
          show PiRLCStarts.commitmentLogicalStart = 7279662 by rfl]
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
    (block : Fin 4) (cell : Fin 2) :
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
  · change 7316076 + 17 * (22 * 1 * 8100) ≤ 12410976
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
  · change 10345476 + 17 * (5 * 1 * 8100) ≤ 12410976
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
  · change 11033976 + 17 * (1 * 2 * 8100) ≤ 12410976
    norm_num
  · exact evalKValueRange_compatible program source block cell

private theorem evalA_layout
    (program : Lifecycle.Stage1.Application.Program)
    (source : Fin PiRLCCombinationInvocations.sourceCount)
    (block : Fin 4) (lane : Fin ringDegree) (cell : Fin 2) :
    CompactInvocationPrivate program
      (PiRLCCombinationInvocations.invocation
        PiRLCStarts.evalALogicalStart PiRLCStarts.evalARowStart
        PiRLCStarts.evalAFreshStart 4 2 2 source.val block.val lane.val
        cell.val PiRLCCombinationInvocations.evalAValueSourceStart)
      (NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount
        lane) := by
  apply combination_layout
  · exact PiRLCCombinationInvocations.evalAFreshStart_local
  · change 11309376 + 17 * (4 * 2 * 8100) ≤ 12410976
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
  change PiRLCCombinationTemplates.templates[
    PiRLCCombinationTemplates.templateIndex source.val lane.val]? = _
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
      PiRLCStarts.evalARowStart PiRLCStarts.evalAFreshStart 4 2 2
      PiRLCCombinationInvocations.evalAValueSourceStart
      (fun source block lane cell => evalA_layout program source block lane cell)
      invocation evalAMember template templateEquation row rowMember

end NightstreamFPrime.Export.Stage1.PerApplicationCompactPreservation
