import NightstreamFPrime.Export.Stage1.PiRLCProductSchedule

/-!
Owns the fixed column separation of the selected PiRLC combination recipes.
Each invocation writes scratch only inside the combination fresh interval;
its normalized inputs and required output remain outside that interval.
The bounds are structural in source/block/lane/cell coordinates.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchGeometry

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Export.Package
open PiRLCProductSchedule
open PiRLCCombinationInvocations

abbrev Descriptor := PiRLCProductSchedule.Descriptor

def scratchStart : Nat := Spartan.sourceToSpartan PiRLCStarts.commitmentFreshStart

def scratchEnd : Nat := Spartan.sourceToSpartan PiRLCStarts.outputFreshStart

@[simp] theorem scratchStart_eq : scratchStart = 20572364 := by rfl
@[simp] theorem scratchEnd_eq : scratchEnd = 28421264 := by rfl

def inputColumn (descriptor : Descriptor) : Nat → Nat :=
  CompactRows.inputColumnOfRanges descriptor.compactInvocation.inputRanges

def scratchCount (descriptor : Descriptor) : Nat :=
  R1CS.mulCount (Expr.var PiRLCCombinationTemplates.outputInput -
    PiRLCCombinationTemplates.outputRecipe (firstSource descriptor.source.val)
      descriptor.lane)

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

theorem laneFreshPrefix_add_cost_le (lane : Nat) :
    laneFreshPrefix lane + laneFreshCost lane ≤ 8100 := by
  have bound := sum_take_add_getD_le_sum laneFreshCosts lane
  rw [laneFreshCosts_sum] at bound
  exact bound

theorem laneFreshCost_eq (lane : Fin ringDegree) :
    laneFreshCost lane.val =
      NightstreamFPrime.Layout.PiRLC.v1_1.CombinationStep.laneFreshCount lane := by
  unfold laneFreshCost laneFreshCosts
  rw [List.getD_eq_get _ _ ⟨lane.val, by simp⟩]
  simp

theorem coordinateFreshEnd_le
    {blockCount cellCount : Nat} (block : Fin blockCount)
    (lane : Fin ringDegree) (cell : Fin cellCount) :
    coordinateFreshPrefix cellCount block.val lane.val cell.val +
      laneFreshCost lane.val ≤ sourceFreshCount blockCount cellCount := by
  let cost := laneFreshCost lane.val
  let lanePrefix := laneFreshPrefix lane.val
  have laneBound : lanePrefix + cost ≤ 8100 := laneFreshPrefix_add_cost_le lane.val
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
      _ ≤ cellCount * lanePrefix + cellCount * cost := Nat.add_le_add_left cellPart _
      _ = cellCount * (lanePrefix + cost) := by ring
      _ ≤ cellCount * 8100 := Nat.mul_le_mul_left cellCount laneBound
  have blockSucc : block.val + 1 ≤ blockCount := by omega
  unfold coordinateFreshPrefix sourceFreshCount
  change block.val * cellCount * 8100 + cellCount * lanePrefix +
      cell.val * cost + cost ≤ blockCount * cellCount * 8100
  calc
    block.val * cellCount * 8100 + cellCount * lanePrefix +
        cell.val * cost + cost =
        block.val * cellCount * 8100 +
          (cellCount * lanePrefix + cell.val * cost + cost) := by omega
    _ ≤ block.val * cellCount * 8100 + cellCount * 8100 := Nat.add_le_add_left lanePart _
    _ = (block.val + 1) * (cellCount * 8100) := by ring
    _ ≤ blockCount * (cellCount * 8100) :=
      Nat.mul_le_mul_right (cellCount * 8100) blockSucc
    _ = blockCount * cellCount * 8100 := by ring

private def familyFreshStart : Family → Nat
  | .commitment => PiRLCStarts.commitmentFreshStart
  | .publicInput => PiRLCStarts.publicInputFreshStart
  | .evalK => PiRLCStarts.evalKFreshStart
  | .evalA => PiRLCStarts.evalAFreshStart

private theorem familyFreshStart_mapped (family : Family) :
    Spartan.sourceToSpartan (familyFreshStart family) =
      match family with
      | .commitment => 20572364
      | .publicInput => 23601764
      | .evalK => 24290264
      | .evalA => 24565664 := by
  cases family <;> rfl

private theorem localStart_eq (descriptor : Descriptor) :
    descriptor.compactInvocation.localStart =
      Spartan.sourceToSpartan (familyFreshStart descriptor.family) +
        descriptor.source.val * sourceFreshCount descriptor.family.blockCount
          descriptor.family.cellCount +
        coordinateFreshPrefix descriptor.family.cellCount descriptor.block.val
          descriptor.lane.val descriptor.cell.val := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp only [PiRLCProductSchedule.Descriptor.compactInvocation, invocation_localStart,
      invocationFreshSource, familyFreshStart, Family.blockCount, Family.cellCount]
  all_goals
    rw [Nat.add_assoc, Spartan.sourceToSpartan_add_of_piCcsLocal]
    · omega
    · first
      | exact commitmentFreshStart_local
      | exact publicInputFreshStart_local
      | exact evalKFreshStart_local
      | exact evalAFreshStart_local

private theorem scratchCount_eq (descriptor : Descriptor) :
    scratchCount descriptor = laneFreshCost descriptor.lane.val := by
  change (PiRLCCombinationTemplates.template (firstSource descriptor.source.val)
    descriptor.lane).localColumnCount = _
  rw [PiRLCCombinationTemplates.template_localColumnCount, laneFreshCost_eq]

/-- Every local allocation is contained in the one discarded scratch interval. -/
theorem scratch_contained (descriptor : Descriptor) :
    scratchStart ≤ descriptor.compactInvocation.localStart ∧
      descriptor.compactInvocation.localStart + scratchCount descriptor ≤ scratchEnd := by
  have coordinateBound := coordinateFreshEnd_le descriptor.block descriptor.lane descriptor.cell
  have sourceSucc : descriptor.source.val + 1 ≤ sourceCount := by
    have := descriptor.source.isLt
    omega
  have sourceBound := Nat.mul_le_mul_right
    (sourceFreshCount descriptor.family.blockCount descriptor.family.cellCount) sourceSucc
  rw [scratchCount_eq, localStart_eq, scratchStart_eq, scratchEnd_eq,
    familyFreshStart_mapped]
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    norm_num [Family.blockCount, Family.cellCount,
      sourceFreshCount, sourceCount, Nat.add_mul] at coordinateBound sourceBound ⊢ <;>
    omega

theorem localStart_ge (descriptor : Descriptor) :
    110 ≤ descriptor.compactInvocation.localStart := by
  have lower := (scratch_contained descriptor).1
  rw [scratchStart_eq] at lower
  omega

private def valueStart (descriptor : Descriptor) : Nat :=
  match descriptor.family with
  | .commitment => commitmentValueSourceStart descriptor.source.val descriptor.block.val descriptor.cell.val
  | .publicInput => publicInputValueSourceStart descriptor.source.val descriptor.block.val descriptor.cell.val
  | .evalK => evalKValueSourceStart descriptor.source.val descriptor.block.val descriptor.cell.val
  | .evalA => evalAValueSourceStart descriptor.source.val descriptor.block.val descriptor.cell.val

def sourceColumn (descriptor : Descriptor) (input : Nat) : Nat :=
  if input < 54 then challengeSourceStart descriptor.source.val + input
  else if input < 108 then valueStart descriptor +
    (input - 54) * descriptor.family.valueStride
  else if input = 108 then
    if descriptor.source.val = 0 then 0 else descriptor.priorColumn
  else if input = 109 then descriptor.outputColumn
  else 0

theorem inputColumn_eq (descriptor : Descriptor) (input : Nat)
    (bounded : input < 110) :
    inputColumn descriptor input = Spartan.sourceToSpartan (sourceColumn descriptor input) := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family
  · exact inputColumnOfRanges_eq _ _ _ _ _ _ _ _ input
      commitmentValueSourceStart
      (by
        intro offset bound
        simpa only [Nat.mul_one] using
          (commitmentValueSource_affine source.val block.val cell.val
            offset source.isLt block.isLt bound)) bounded
  · exact inputColumnOfRanges_eq _ _ _ _ _ _ _ _ input
      publicInputValueSourceStart
      (by
        intro offset bound
        simpa only [Nat.mul_one] using
          (publicInputValueSource_affine source.val block.val cell.val
            offset source.isLt block.isLt bound)) bounded
  · exact inputColumnOfRanges_eq _ _ _ _ _ _ _ _ input
      evalKValueSourceStart
      (fun offset bound => evalKValueSource_affine source.val block.val cell.val
        offset source.isLt cell.isLt bound) bounded
  · exact inputColumnOfRanges_eq _ _ _ _ _ _ _ _ input
      evalAValueSourceStart
      (fun offset bound => evalAValueSource_affine source.val block.val cell.val
        offset source.isLt block.isLt cell.isLt bound) bounded

private theorem source_input_before_output (descriptor : Descriptor) (input : Nat)
    (bounded : input < 109) :
    sourceColumn descriptor input < descriptor.outputColumn := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    have sourceBound := source.isLt
  all_goals
    have blockBound := block.isLt
    have laneBound := lane.isLt
    have cellBound := cell.isLt
    norm_num [sourceColumn, valueStart,
      PiRLCProductSchedule.Descriptor.outputColumn, PiRLCProductSchedule.Descriptor.priorColumn,
      PiRLCProductSchedule.Descriptor.logicalIndex, Family.logicalStart,
      Family.blockCount, Family.cellCount, Family.valueStride, logicalIndex, stepSize,
      sourceCount, ringDegree, challengeSourceStart, PiRLCStarts.challengeWordStart_eq,
      PiRLCStarts.phaseLogicalStart_eq,
      commitmentValueSourceStart, publicInputValueSourceStart, evalKValueSourceStart, evalAValueSourceStart,
      PiCCSInputs.freshCommitmentStart, PiCCSInputs.runningCommitmentStart,
      PiCCSInputs.runningPublicStart, PiCCSInputs.runningGroupStart,
      PiCCSInputs.runningGroupsStart, PiCCSInputs.priorRunningStart,
      PiCCSInputs.runningGroupWords, PiCCSInputs.outputEvaluationStart,
      PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentWords,
      PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
      PiCCSInputs.expectedContextWords, PiCCSInputs.roundMessageWords,
      PilotProduction.priorPublicInputStart, PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq, PiRLCStarts.commitmentLogicalStart,
      PiRLCStarts.publicInputLogicalStart, PiRLCStarts.evalKLogicalStart,
      PiRLCStarts.evalALogicalStart, PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.commitmentOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.publicInputOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.evalKOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.evalAOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.logicalPrivateCount_eq]
      at sourceBound blockBound laneBound cellBound ⊢
  all_goals split_ifs <;> omega

/-- Required source outputs precede the multiplication scratch allocation. -/
theorem outputSource_before (descriptor : Descriptor) :
    descriptor.outputColumn < PiRLCStarts.commitmentFreshStart := by
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    have sourceBound := source.isLt
  all_goals
    have blockBound := block.isLt
    have laneBound := lane.isLt
    have cellBound := cell.isLt
    norm_num [PiRLCProductSchedule.Descriptor.outputColumn,
      PiRLCProductSchedule.Descriptor.logicalIndex, Family.logicalStart,
      Family.blockCount, Family.cellCount, stepSize, logicalIndex, sourceCount,
      ringDegree, PiRLCStarts.commitmentFreshStart_eq] at sourceBound blockBound laneBound cellBound ⊢
    norm_num [PiRLCStarts.commitmentLogicalStart, PiRLCStarts.publicInputLogicalStart,
      PiRLCStarts.evalKLogicalStart, PiRLCStarts.evalALogicalStart,
      PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.commitmentOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.publicInputOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.evalKOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.evalAOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.SamplerChain.logicalPrivateCount_eq]
    omega

/-- The required output is a private column before the discarded interval. -/
theorem output_before_scratch (descriptor : Descriptor) :
    inputColumn descriptor 109 < scratchStart := by
  have startLocal : Spartan.piCcsPhaseOffset ≤ descriptor.family.logicalStart := by
    cases descriptor.family <;> decide
  have outputLocal : Spartan.piCcsPhaseOffset ≤ descriptor.outputColumn := by
    unfold PiRLCProductSchedule.Descriptor.outputColumn
    omega
  have outputEq : inputColumn descriptor 109 =
      Spartan.sourceToSpartan descriptor.outputColumn := by
    simpa [sourceColumn] using inputColumn_eq descriptor 109 (by decide)
  rw [outputEq]
  exact Spartan.sourceToSpartan_lt_of_piCcsLocal _ _ outputLocal
    (outputSource_before descriptor)

private theorem scratchEnd_le_private : scratchEnd ≤ Spartan.privateColumnCount := by
  have bound := Spartan.sourceColumnCount_ge_piDecPhaseOffset
  change 28470790 ≤ Spartan.SourceColumnCount at bound
  rw [scratchEnd_eq]
  norm_num [Spartan.privateColumnCount, Spartan.appendedPrivateColumnCount,
    Spartan.pilotPrivateColumnCount, Spartan.pilotSourceColumnCount,
    Spartan.expectedContextColumnCount]
  omega

/-- All normalized inputs, including the output slot, avoid global scratch. -/
theorem inputs_outside_scratch (descriptor : Descriptor) (input : Nat)
    (bounded : input < 110) :
    inputColumn descriptor input < scratchStart ∨ scratchEnd ≤ inputColumn descriptor input := by
  have before : sourceColumn descriptor input < PiRLCStarts.commitmentFreshStart := by
    by_cases recipe : input < 109
    · exact lt_trans (source_input_before_output descriptor input recipe) (outputSource_before descriptor)
    · have output : input = 109 := by omega
      subst input
      simpa [sourceColumn] using outputSource_before descriptor
  rw [inputColumn_eq descriptor input bounded]
  rcases Spartan.sourceToSpartan_before_piCcsLocal _ _ commitmentFreshStart_local before with
    earlier | publicColumn
  · exact Or.inl earlier
  · exact Or.inr (by have := scratchEnd_le_private; omega)

theorem inputs_outside_local (descriptor : Descriptor) (input : Nat)
    (bounded : input < 110) :
    inputColumn descriptor input < descriptor.compactInvocation.localStart ∨
      descriptor.compactInvocation.localStart + scratchCount descriptor ≤ inputColumn descriptor input := by
  have contained := scratch_contained descriptor
  rcases inputs_outside_scratch descriptor input bounded with earlier | later
  · exact Or.inl (by omega)
  · exact Or.inr (by omega)

theorem output_distinct (descriptor : Descriptor) (input : Nat)
    (bounded : input < 109) :
    inputColumn descriptor input ≠ inputColumn descriptor 109 := by
  have before := source_input_before_output descriptor input bounded
  have outputBound : descriptor.outputColumn < Spartan.SourceColumnCount := by
    have earlier := outputSource_before descriptor
    have endpoint := Spartan.sourceColumnCount_ge_piDecPhaseOffset
    change 28470790 ≤ Spartan.SourceColumnCount at endpoint
    rw [PiRLCStarts.commitmentFreshStart_eq] at earlier
    omega
  rw [inputColumn_eq descriptor input (by omega), inputColumn_eq descriptor 109 (by decide)]
  have outputEq : sourceColumn descriptor 109 = descriptor.outputColumn := by
    simp [sourceColumn]
  rw [outputEq]
  intro equal
  have same := Spartan.sourceToSpartan_injective (lt_trans before outputBound) outputBound equal
  omega

end NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchGeometry
