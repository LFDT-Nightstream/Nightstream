import NightstreamFPrime.Export.Stage1.PiRLCCombinationOrdinaryReadSupport
import NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchCustody
import NightstreamFPrime.Layout.Stage1.RunningTransitionValues

/-!
Structural read exclusion for running-transition scratch. The source owners,
not an emitted artifact or a caller support premise, establish the selected
ordinary-row and logical-constraint support. This module does not establish
complete stored-event execution or select a changed package.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.RunningTransitionScratchCustody

open NightstreamFPrime.Circuit NightstreamFPrime.Spec
open NightstreamFPrime.Layout NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Export.Stage1 NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def sourceScratchStart : Nat := RunningTransitionLayout.logicalColumnCount
def sourceScratchEnd : Nat := 28411244
def scratchStart : Nat := 28114829
def scratchEnd : Nat := 28410966

def SourceOutside (column : Nat) : Prop :=
  column < sourceScratchStart ∨ sourceScratchEnd ≤ column

def Outside (column : Nat) : Prop := column < scratchStart ∨ scratchEnd ≤ column

theorem endpoints :
    sourceScratchStart = 28115107 ∧ sourceScratchEnd = 28411244 ∧
      scratchStart = 28114829 ∧ scratchEnd = 28410966 := by
  exact ⟨rfl, rfl, rfl, rfl⟩

/-- Numeric endpoints are bound to the existing layout by its named equalities. -/
theorem endpoint_geometry :
    sourceScratchEnd = RunningTransitionLayout.physicalEnd ∧
      scratchStart = Spartan.sourceToSpartan sourceScratchStart ∧
      scratchEnd = Spartan.sourceToSpartan sourceScratchEnd := by
  refine ⟨?_, ?_, ?_⟩
  · exact Spartan.sourceColumnCount_eq.symm.trans Spartan.sourceColumnCount_eq_physicalEnd
  · rw [sourceScratchStart, RunningTransitionLayout.logicalColumnCount_eq]
    rfl
  · rw [show sourceScratchEnd = Spartan.SourceColumnCount from Spartan.sourceColumnCount_eq.symm,
      Spartan.sourceToSpartan_sourceColumnCount, Spartan.privateColumnCount_eq]
    rfl

/-- Transport uses the proved inverse of the existing source permutation. -/
theorem mapped_outside (column : Nat) (bounded : column < Spartan.SourceColumnCount)
    (outside : SourceOutside column) : Outside (Spartan.sourceToSpartan column) := by
  by_contra failure
  have lower : 28114829 ≤ Spartan.sourceToSpartan column := by
    change ¬ (_ < 28114829 ∨ 28410966 ≤ _) at failure
    omega
  have upper : Spartan.sourceToSpartan column < 28410966 := by
    change ¬ (_ < 28114829 ∨ 28410966 ≤ _) at failure
    omega
  have inverse := Spartan.spartanToSource_sourceToSpartan column bounded
  unfold Spartan.spartanToSource at inverse
  rw [if_neg (by change ¬ _ < 98786; omega),
    if_neg (by change ¬ _ < 128074; omega),
    if_neg (by change ¬ _ < 14751526; omega),
    if_pos (by rw [Spartan.privateColumnCount_eq]; omega)] at inverse
  have coordinate := Option.some.inj inverse
  change 14751804 + (Spartan.sourceToSpartan column - 14751526) = column at coordinate
  change column < 28115107 ∨ 28411244 ≤ column at outside
  omega

theorem shifted_outside (application : Stage1.Application.Program) (column : Nat)
    (outside : Outside column) :
    Outside (PerApplicationPackage.shiftColumn application column) := by
  unfold PerApplicationPackage.shiftColumn
  split_ifs with before
  · exact outside
  · right
    have constant := Package.circuitPackage_layout_values.2.2.1
    change PerApplicationPackage.basePackage.layout.constantColumn = 28410966 at constant
    rw [constant] at before
    change 28410966 ≤ _
    omega

private theorem mapped_before (column : Nat) (before : column < sourceScratchStart) :
    Outside (Spartan.sourceToSpartan column) := by
  apply mapped_outside column _ (Or.inl before)
  exact Nat.lt_of_lt_of_le before (by
    rw [Spartan.sourceColumnCount_eq]
    change 28115107 ≤ 28411244
    decide)

theorem piCcsSource_before (column : Nat)
    (source : PiCCSOrdinarySourceSupport.Source column) : column < sourceScratchStart := by
  have bound := PiRLCCombinationPiCCSReadSupport.source_before column source
  change column < 20198868 at bound
  change column < 28115107
  omega

theorem piDecSource_before (column : Nat)
    (source : PiDECSourceSupport.Source column) : column < sourceScratchStart := by
  change column < 28115107
  rcases source with ((parent | proof) | logical) | fresh
  · rcases parent with commitment | publicInput | evalK | evalA
    · have upper := commitment.2
      change column < 20139234 + 1188 at upper
      omega
    · have upper := publicInput.2
      change column < 20144742 + 270 at upper
      omega
    · have upper := evalK.2
      change column < 20146740 + 108 at upper
      omega
    · have upper := evalA.2
      change column < 20171040 + 1512 at upper
      omega
  · have upper := proof.2
    change column < 28047768 + 49248 at upper
    omega
  · have upper := logical.2
    change column < 28097016 + 270 at upper
    omega
  · have upper := fresh.2
    change column < 28097286 + 17820 at upper
    omega

theorem samplerSource_before (column : Nat)
    (source : PiRLCSamplerOrdinaryDirectSource.Source column) :
    column < sourceScratchStart :=
  lt_of_lt_of_le source.beforeCombination (by decide)

theorem runningLogical_before (column : Nat)
    (source : RunningTransitionSourceSupport.Logical column) : column < sourceScratchStart := by
  change column < 28115107
  rcases source with (state | output | point | piDec) | inverse
  · have upper := state.2
    change column < 28 + 11 at upper
    omega
  · have upper := output.2
    change column < 49663 + 49393 at upper
    omega
  · rcases point with ⟨coordinate, c0 | c1⟩
    all_goals subst column
    all_goals
      have bound := coordinate.isLt
      change coordinate.val < 28 at bound
      rw [PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [RunningTransitionInputs.roundStride, RunningTransitionInputs.roundSampleC0Offset,
        RunningTransitionInputs.roundSampleC1Offset]
      omega
  · have upper := (RunningTransitionSourceSupport.piDecField_inRange piDec).2
    change column < 28047768 + 49248 at upper
    omega
  · subst column
    decide

/-- The actual logical transition reads no cells from its lowering scratch. -/
theorem runningLogicalConstraints_supported (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :
    ∀ expression ∈ RunningTransitionLayout.logicalConstraints logicalWidth publicFits,
      expression.VarsSatisfy SourceOutside := by
  intro expression member
  exact (RunningTransitionSourceSupport.logicalConstraints_varsSatisfy
    logicalWidth publicFits expression member).mono expression
      (fun column supported => Or.inl (runningLogical_before column supported))

/-- Exactly the PiCCS, sampler and PiDEC ordinary prefix before the transition. -/
def nonTransitionRows : List R1CS.Row :=
  ((PiCCSArithmetic.arithmeticRows Data.logicalWidth Data.publicFits ++
    PiRLCSamplerOrdinaryRows.rows (logicalWidth := Data.logicalWidth)
      (publicFits := Data.publicFits)) ++
    (PiDECArithmetic.canonicalPlan Data.logicalWidth Data.publicFits).rows).map
      Rows.CompiledRow.toR1CS

/-- The proved prefix is the actual selected arithmetic row prefix. -/
theorem arithmeticRows_split :
    (Data.arithmeticRows ()).map Rows.CompiledRow.toR1CS = nonTransitionRows ++
      (RunningTransitionArithmetic.canonicalPlan Data.logicalWidth Data.publicFits).rows.map
        Rows.CompiledRow.toR1CS := by
  simp only [Data.arithmeticRows, nonTransitionRows, List.map_append]

theorem nonTransitionRows_supported
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits) :
    ∀ row ∈ nonTransitionRows, row.VarsSatisfy Outside := by
  intro row member
  simp only [nonTransitionRows, List.map_append, List.mem_append] at member
  rcases member with (piCcs | sampler) | piDec
  · have support := PiCCSOrdinaryDirectSupport.sourceRows_varsSatisfy relation row piCcs
    apply support.mono row
    intro column source
    rcases source with ⟨original, support, rfl⟩
    exact mapped_before original (piCcsSource_before original support)
  · have support := PiRLCSamplerOrdinaryDirectSource.sourceRows_varsSatisfy row sampler
    apply support.mono row
    intro column source
    rcases source with ⟨original, support, rfl⟩
    exact mapped_before original (samplerSource_before original support)
  · rw [← PiDECOrdinaryDirectSource.sourceRows_eq_canonical] at piDec
    simp only [PiDECOrdinaryDirectSource.sourceRows, List.mem_append] at piDec
    have support : row.VarsSatisfy PiDECSourceSupport.Target := by
      rcases piDec with ((publicInput | commitment) | evalK) | evalA
      · exact PiDECOrdinaryDirectSource.publicRows_varsSatisfy relation row publicInput
      · exact PiDECOrdinaryDirectSource.commitmentRows_varsSatisfy relation row commitment
      · exact PiDECOrdinaryDirectSource.evalKRows_varsSatisfy relation row evalK
      · exact PiDECOrdinaryDirectSource.evalARows_varsSatisfy relation row evalA
    apply support.mono row
    intro column source
    rcases source with ⟨original, support, rfl⟩
    exact mapped_before original (piDecSource_before original support)

private theorem applicationSource_outside (application : Stage1.Application.Program)
    (column : Nat) (source : ApplicationDirectSource.SourceAllowed application column) :
    Outside column := by
  rcases source with input | witness | output | localRange
  · rcases input with ⟨index, rfl⟩
    rw [ApplicationInputs.inputColumn_value]
    have bound := index.isLt
    change index.val < 4 at bound
    left
    change 35 + index.val < 28114829
    omega
  · rcases witness with ⟨index, rfl⟩
    right
    unfold ApplicationInputs.witnessColumn ApplicationInputs.witnessStart
    rw [Spartan.privateColumnCount_eq]
    change 28410966 ≤ 28410966 + index.val
    omega
  · rcases output with ⟨index, rfl⟩
    rw [ApplicationInputs.outputColumn_value]
    have bound := index.isLt
    change index.val < 4 at bound
    left
    change 49428 + index.val < 28114829
    omega
  · right
    have lower := localRange.1
    unfold ApplicationInputs.localStart ApplicationInputs.witnessStart at lower
    rw [Spartan.privateColumnCount_eq] at lower
    change 28410966 ≤ column
    omega

theorem applicationRows_supported (application : Stage1.Application.Program) :
    ∀ row ∈ ApplicationDirectSource.sourceRows application, row.VarsSatisfy Outside := by
  intro row member
  exact (ApplicationDirectSource.sourceRows_varsSatisfy application row member).mono row
    (applicationSource_outside application)

/-- Changing only transition scratch preserves the selected ordinary prefix. -/
theorem nonTransitionRows_hold_iff
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (left right : Env) (agree : ∀ column, Outside column → left column = right column) :
    R1CS.RowsHold left nonTransitionRows ↔ R1CS.RowsHold right nonTransitionRows := by
  constructor
  · exact R1CS.rowsHold_of_agree nonTransitionRows Outside left right
      (nonTransitionRows_supported relation) (fun column allowed => (agree column allowed).symm)
  · exact R1CS.rowsHold_of_agree nonTransitionRows Outside right left
      (nonTransitionRows_supported relation) agree


open NightstreamFPrime.Layout.ProductionRelation
open PerApplicationCanonicalAssignment PerApplicationAssignmentPlan

private theorem shiftedSource_before (application : Stage1.Application.Program) (column : Nat)
    (before : column < PiRLCStarts.commitmentFreshStart) :
    Outside (PerApplicationPackage.shiftColumn application (Spartan.sourceToSpartan column)) := by
  apply shifted_outside
  apply mapped_before
  change column < 20198868 at before
  change column < 28115107
  omega

private theorem shiftedSource_beforeTransition (application : Stage1.Application.Program)
    (column : Nat) (before : column < sourceScratchStart) :
    Outside (PerApplicationPackage.shiftColumn application (Spartan.sourceToSpartan column)) :=
  shifted_outside application _ (mapped_before column before)

private theorem baseWidth_after (application : Stage1.Application.Program) :
    scratchEnd ≤ PiRLCProductPlan.baseSourceWidth application := by
  have bound := PiRLCProductPlan.basePackage_fits application
  have constant : PiRLCProductPlan.basePackage.layout.constantColumn = 28410966 :=
    Package.circuitPackage_layout_values.2.2.1
  rw [constant] at bound
  exact bound

private theorem samplerLogical_before
    (source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount)
    (position : Fin PiRLCSamplerOrdinaryRetainedBlocks.logicalCountPerSource) :
    PiRLCSamplerOrdinaryRetainedBlocks.logicalSource source position <
      PiRLCStarts.commitmentFreshStart :=
  (PiRLCSamplerOrdinaryDirectSource.Source.logical source.val position.val
    source.isLt position.isLt).beforeCombination

private theorem samplerFresh_before
    (source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount)
    (position : Fin PiRLCSamplerOrdinaryRetainedBlocks.freshCountPerSource) :
    PiRLCSamplerOrdinaryRetainedBlocks.freshSource source position <
      PiRLCStarts.commitmentFreshStart :=
  (PiRLCSamplerOrdinaryDirectSource.Source.fresh source.val position.val
    source.isLt position.isLt).beforeCombination

private theorem proofLogical_before
    (slot : Fin PiCCSOrdinaryRetainedBlocks.proofLogicalCount) :
    PiCCSOrdinaryRetainedBlocks.proofLogicalSource slot <
      PiRLCStarts.commitmentFreshStart := by
  have bound := slot.isLt
  change slot.val < 114878 at bound
  unfold PiCCSOrdinaryRetainedBlocks.proofLogicalSource
  split
  · change 14722516 + slot.val < 20198868
    omega
  · split
    · rename_i selected
      let index : Fin PiCCSOrdinaryRetainedBlocks.transcriptOutputCount :=
        ⟨slot.val - PiCCSOrdinaryRetainedBlocks.proofInputCount, by omega⟩
      let decoded : Fin PiCCSOrdinaryRetainedBlocks.transcriptInvocationCount ×
          Fin Poseidon2.width := Fin.decodeProd index
      have invocationBound := decoded.1.isLt
      have laneBound := decoded.2.isLt
      change decoded.1.val < 718 at invocationBound
      change decoded.2.val < 8 at laneBound
      change 14751804 + decoded.1.val * 592 + 584 + decoded.2.val < 20198868
      omega
    · change PiCCSStarts.initialClaimLogicalStart +
          (slot.val - (29288 + 5744)) < 20198868
      have start : PiCCSStarts.initialClaimLogicalStart + 114878 ≤ 20198868 := by decide
      omega

private theorem poseidonBlock_outside (sourceWidth invocationCount : Nat)
    (witnessStart : Fin invocationCount → Nat)
    (witnessBound : ∀ invocation,
      witnessStart invocation + PoseidonScheduleTrace.localColumnCount ≤ sourceWidth)
    (before : ∀ invocation,
      witnessStart invocation + PoseidonScheduleTrace.localColumnCount ≤ 20198590)
    (slot : Fin (invocationCount * PoseidonRetainedSlots.rows.length)) :
    Outside ((Layout.ProductionRelation.PoseidonRetainedBlock.block sourceWidth
      invocationCount witnessStart witnessBound).source slot).val := by
  let indices : Fin invocationCount × Fin PoseidonRetainedSlots.rows.length :=
    Fin.decodeProd slot
  have startBound := before indices.1
  have localBound := (PoseidonRetainedSlots.localOutput indices.2).isLt
  left
  change witnessStart indices.1 +
    (PoseidonRetainedSlots.localOutput indices.2).val < 28114829
  omega

/-- All existing retained blocks except the one being replaced avoid the scratch.
The conclusion follows from their actual source functions for every slot. -/
theorem retainedNonTransitionSource_outside (application : Stage1.Application.Program)
    (kind : BlockKind) (different : kind ≠ .runningFresh)
    (slot : Fin (kind.template application).block.slotCount) :
    Outside ((kind.template application).block.source slot).val := by
  cases kind with
  | priorPoseidon =>
      exact poseidonBlock_outside _ _ _ PoseidonRetainedBlock.priorWitnessStart_bound
        PiRLCCombinationScratchPoseidon.priorWitnessEnd_le slot
  | outputPoseidon =>
      exact poseidonBlock_outside _ _ _ PoseidonRetainedBlock.outputWitnessStart_bound
        PiRLCCombinationScratchPoseidon.outputWitnessEnd_le slot
  | laterPoseidon =>
      exact poseidonBlock_outside _ _ _ PoseidonRetainedBlock.laterWitnessStart_bound
        PiRLCCombinationScratchPoseidon.laterWitnessEnd_le slot
  | productGroup =>
      right
      have baseBound := baseWidth_after application
      change scratchEnd ≤ PiRLCProductPlan.baseSourceWidth application + _
      omega
  | challengeWords =>
      let position : PiRLCProductSourceBlocks.Challenge := Fin.decodeProd slot
      apply shiftedSource_before
      exact (PiRLCSamplerOrdinaryDirectSource.Source.word position.1.val position.2.val
        position.1.isLt position.2.isLt).beforeCombination
  | productOutput =>
      exact shiftedSource_before application _
        (PiRLCCombinationScratchGeometry.outputSource_before
          (PiRLCProductSchedule.descriptor slot))
  | priorPoseidonInput =>
      left
      have bound := slot.isLt
      change slot.val < 49393 at bound
      change 0 + slot.val < 28114829
      omega
  | outputPoseidonInput =>
      left
      have bound := slot.isLt
      change slot.val < 49393 at bound
      change 49393 + slot.val < 28114829
      omega
  | runningPiDec =>
      apply shiftedSource_beforeTransition
      have bound := slot.isLt
      change slot.val < 49248 at bound
      change 28047768 + slot.val < 28115107
      omega
  | runningFresh => exact False.elim (different rfl)
  | piCcsFreshPublicInput =>
      apply shiftedSource_before
      have bound := slot.isLt
      change slot.val < 270 at bound
      change 49393 + slot.val < 20198868
      omega
  | piCcsPriorLast =>
      apply shifted_outside
      left
      have bound := slot.isLt
      change slot.val < 592 at bound
      change PoseidonRetainedBlock.priorWitnessStart
        PiCCSOrdinaryRetainedBlocks.priorLastInvocation + slot.val < 28114829
      have endBound := PiRLCCombinationScratchPoseidon.priorWitnessEnd_le
        PiCCSOrdinaryRetainedBlocks.priorLastInvocation
      change _ + 592 ≤ 20198590 at endBound
      omega
  | piCcsOutputLast =>
      apply shifted_outside
      left
      have bound := slot.isLt
      change slot.val < 592 at bound
      change PoseidonRetainedBlock.outputWitnessStart
        PiCCSOrdinaryRetainedBlocks.outputLastInvocation + slot.val < 28114829
      have endBound := PiRLCCombinationScratchPoseidon.outputWitnessEnd_le
        PiCCSOrdinaryRetainedBlocks.outputLastInvocation
      change _ + 592 ≤ 20198590 at endBound
      omega
  | piCcsExpectedContext =>
      apply shiftedSource_before
      have bound := slot.isLt
      change slot.val < 4 at bound
      change 14722512 + slot.val < 20198868
      omega
  | piCcsProofLogical =>
      exact shiftedSource_before application _ (proofLogical_before slot)
  | piCcsOutputEndpoint =>
      apply shiftedSource_before
      have bound := slot.isLt
      change slot.val < 8 at bound
      change 19333210 + slot.val < 20198868
      omega
  | piCcsFresh =>
      apply shiftedSource_before
      have bound := slot.isLt
      change slot.val < 731605 at bound
      change 19333218 + slot.val < 20198868
      omega
  | pilotCanonicalLocal =>
      dsimp only [BlockKind.template, PilotOrdinaryRetainedBlocks.canonicalLocalBlock,
        PiCCSOrdinaryRetainedBlocks.sourceFieldBlock,
        RunningTransitionRetainedBlocks.packageSourceColumn,
        PiRLCRetainedPreservation.baseSourceColumn,
         ProductRetainedBlock.baseColumn,
        PiRLCProductPlan.shiftedPackageColumn, FieldSuffixBlock.baseColumn]
      change Outside (PerApplicationPackage.shiftColumn application
        (Spartan.sourceToSpartan
          (PriorStateHash.hashEnd PilotProduction.priorInterface
            PilotProduction.witnessOffset + slot.val)))
      apply shiftedSource_before
      have bound := slot.isLt
      change slot.val < 264 at bound
      unfold PriorStateHash.hashEnd
      rw [PilotProduction.priorHashLogicalLength_eq, PilotProduction.witnessOffset_eq]
      change _ < 20198868
      omega
  | pilotCanonicalFresh =>
      apply shiftedSource_before
      have bound := slot.isLt
      change slot.val < 788 at bound
      have start : PilotValues.logicalColumnCount + 788 ≤ PiRLCStarts.commitmentFreshStart := by decide
      omega
  | pilotOutputDigest =>
      apply shiftedSource_before
      have bound := slot.isLt
      change slot.val < 4 at bound
      change 99056 + slot.val < 20198868
      omega
  | piDecLogical =>
      apply shiftedSource_beforeTransition
      have bound := slot.isLt
      change slot.val < 270 at bound
      change 28097016 + slot.val < 28115107
      omega
  | piDecFresh =>
      apply shiftedSource_beforeTransition
      have bound := slot.isLt
      change slot.val < 17820 at bound
      change 28097286 + slot.val < 28115107
      omega
  | samplerLogical =>
      exact shiftedSource_before application _
        (samplerLogical_before
          (PiRLCSamplerOrdinaryRetainedBlocks.logicalDescriptor slot).1
          (PiRLCSamplerOrdinaryRetainedBlocks.logicalDescriptor slot).2)
  | samplerFresh =>
      exact shiftedSource_before application _
        (samplerFresh_before
          (PiRLCSamplerOrdinaryRetainedBlocks.freshDescriptor slot).1
          (PiRLCSamplerOrdinaryRetainedBlocks.freshDescriptor slot).2)
  | applicationWitness =>
      right
      change 28410966 ≤ 28410966 + slot.val
      omega
  | applicationLocal =>
      right
      change 28410966 ≤ 28410966 + application.witnessWordCount + slot.val
      omega

/-- Unchanged retained values need no assumed support for their source functions. -/
theorem retainedNonTransitionValues_eq (application : Stage1.Application.Program)
    (kind : BlockKind) (different : kind ≠ .runningFresh)
    (left right : Env) (agree : ∀ column, Outside column → left column = right column)
    (slot : Fin (kind.template application).block.slotCount) :
    left ((kind.template application).block.source slot).val =
      right ((kind.template application).block.source slot).val :=
  agree _ (retainedNonTransitionSource_outside application kind different slot)

end NightstreamFPrime.Export.Stage1.RunningTransitionScratchCustody
