import NightstreamFPrime.Export.Stage1.PiRLCCombinationDirectWitness
import NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchPoseidon
import NightstreamFPrime.Export.Stage1.PerApplicationAssignmentTransportExecution
import NightstreamFPrime.Export.Stage1.PerApplicationSourceAssignment
import NightstreamFPrime.Layout.Stage1.SpartanValues

/-!
Owns the read boundary of the discarded PiRLC multiplication scratch cells.
The retained CCS assignment and its public digest must be unchanged when only
this physical interval changes. No equality is claimed for the old scratch.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchCustody

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open PerApplicationCanonicalAssignment
open PerApplicationAssignmentPlan

def scratchStart : Nat := Spartan.sourceToSpartan PiRLCStarts.commitmentFreshStart

def scratchEnd : Nat := Spartan.sourceToSpartan PiRLCStarts.outputFreshStart

@[simp] theorem scratchStart_eq : scratchStart = 21124070 := by rfl

@[simp] theorem scratchEnd_eq : scratchEnd = 28972970 := by rfl

@[simp] theorem scratchCount_eq : scratchEnd - scratchStart = 7848900 := by rfl

def Outside (column : Nat) : Prop := column < scratchStart ∨ scratchEnd ≤ column

private theorem baseConstant_eq :
    PerApplicationPackage.basePackage.layout.constantColumn = 29336446 :=
  Package.circuitPackage_layout_values.2.2.1

private theorem shifted_outside (application : Program) (column : Nat)
    (outside : Outside column) :
    Outside (PerApplicationPackage.shiftColumn application column) := by
  unfold PerApplicationPackage.shiftColumn
  split_ifs with before
  · exact outside
  · right
    rw [baseConstant_eq] at before
    rw [scratchEnd_eq]
    omega

/-- The source permutation cannot move a cell outside the combination scratch
interval into that interval. Public columns move to the final public suffix. -/
private theorem mapped_outside (column : Nat)
    (bounded : column < Spartan.SourceColumnCount)
    (outside : column < PiRLCStarts.commitmentFreshStart ∨
      PiRLCStarts.outputFreshStart ≤ column) :
    Outside (Spartan.sourceToSpartan column) := by
  by_contra failure
  have lower : 21124070 ≤ Spartan.sourceToSpartan column := by
    simp only [Outside, scratchStart_eq, scratchEnd_eq] at failure
    omega
  have upper : Spartan.sourceToSpartan column < 28972970 := by
    simp only [Outside, scratchStart_eq, scratchEnd_eq] at failure
    omega
  have inverse := Spartan.spartanToSource_sourceToSpartan column bounded
  unfold Spartan.spartanToSource at inverse
  rw [if_neg (by change ¬ _ < 98786; omega),
    if_neg (by change ¬ _ < 128074; omega),
    if_neg (by change ¬ _ < 14751526; omega),
    if_pos (by rw [Spartan.privateColumnCount_eq]; omega)] at inverse
  have coordinate := Option.some.inj inverse
  change 14751804 + (Spartan.sourceToSpartan column - 14751526) = column at coordinate
  change column < 21124348 ∨ 28973248 ≤ column at outside
  omega

private theorem baseWidth_after (application : Program) :
    scratchEnd ≤ PiRLCProductPlan.baseSourceWidth application := by
  have bound := PiRLCProductPlan.basePackage_fits application
  have constant : PiRLCProductPlan.basePackage.layout.constantColumn = 29336446 :=
    Package.circuitPackage_layout_values.2.2.1
  rw [constant] at bound
  rw [scratchEnd_eq]
  omega

private theorem samplerLogical_before
    (descriptor : PiRLCSamplerOrdinaryRetainedBlocks.Lane)
    (position : Fin PiRLCSamplerOrdinaryRetainedBlocks.logicalCountPerLane) :
    PiRLCSamplerOrdinaryRetainedBlocks.logicalSource descriptor position <
      PiRLCStarts.commitmentFreshStart := by
  rcases descriptor with ⟨source, round, lane⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have laneBound := lane.isLt
  have positionBound := position.isLt
  change source.val < 17 at sourceBound
  change round.val < 8 at roundBound
  change lane.val < 4 at laneBound
  change position.val < 100 at positionBound
  change _ < 21124348
  norm_num [PiRLCSamplerOrdinaryRetainedBlocks.logicalSource,
    PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
    PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
    Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  omega

private theorem samplerFresh_before
    (descriptor : PiRLCSamplerOrdinaryRetainedBlocks.Lane)
    (position : Fin PiRLCSamplerOrdinaryRetainedBlocks.freshCountPerLane) :
    PiRLCSamplerOrdinaryRetainedBlocks.freshSource descriptor position <
      PiRLCStarts.commitmentFreshStart := by
  rcases descriptor with ⟨source, round, lane⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have laneBound := lane.isLt
  have positionBound := position.isLt
  change source.val < 17 at sourceBound
  change round.val < 8 at roundBound
  change lane.val < 4 at laneBound
  change position.val < 303 at positionBound
  change _ < 21124348
  norm_num [PiRLCSamplerOrdinaryRetainedBlocks.freshSource,
    PiRLCStarts.digestLaneFreshStart, PiRLCStarts.windowFreshStart,
    PiRLCStarts.samplerSourceFreshStart, PiRLCStarts.samplerFreshStart,
    PiRLCStarts.phaseFreshStart_eq]
  omega

private theorem first54Reject_before
    (descriptor : PiRLCFirst54DirectSchedule.Candidate) :
    descriptor.rejectColumn < PiRLCStarts.commitmentFreshStart := by
  rcases descriptor with ⟨source, round⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  change source.val < 17 at sourceBound
  change round.val < 64 at roundBound
  change _ < 21124348
  norm_num [PiRLCFirst54DirectSchedule.Candidate.rejectColumn,
    PiRLCFirst54Invocations.rejectSourceColumn,
    PiRLCFirst54Invocations.decoderLogicalStart,
    PiRLCFirst54Invocations.candidateDigestRound,
    PiRLCFirst54Invocations.candidateLane,
    PiRLCFirst54Invocations.candidatePart,
    Gadgets.Range.CanonicalU64.auxiliaryCount,
    Gadgets.Sampling.Candidate16Five.auxiliaryCount,
    PiRLCStarts.digestLaneLogicalStart,
    PiRLCStarts.windowLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset, Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  omega

private theorem first54Symbol_before
    (descriptor : PiRLCFirst54DirectSchedule.Candidate) :
    descriptor.symbolColumn < PiRLCStarts.commitmentFreshStart := by
  rcases descriptor with ⟨source, round⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  change source.val < 17 at sourceBound
  change round.val < 64 at roundBound
  change _ < 21124348
  norm_num [PiRLCFirst54DirectSchedule.Candidate.symbolColumn,
    PiRLCFirst54Invocations.remainderSourceColumn,
    PiRLCFirst54Invocations.decoderLogicalStart,
    PiRLCFirst54Invocations.candidateDigestRound,
    PiRLCFirst54Invocations.candidateLane,
    PiRLCFirst54Invocations.candidatePart,
    Gadgets.Range.CanonicalU64.auxiliaryCount,
    Gadgets.Sampling.Candidate16Five.auxiliaryCount,
    PiRLCStarts.digestLaneLogicalStart,
    PiRLCStarts.windowLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset, Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  omega

private theorem first54Position_before
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    descriptor.positionColumn < PiRLCStarts.commitmentFreshStart := by
  rcases descriptor with ⟨⟨source, round⟩, slot⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have slotBound := slot.isLt
  change source.val < 17 at sourceBound
  change round.val < 64 at roundBound
  change slot.val < 55 at slotBound
  change _ < 21124348
  norm_num [PiRLCFirst54DirectSchedule.Position.positionColumn,
    PiRLCFirst54Invocations.positionSourceStart,
    Gadgets.Sampling.First54.positionOffset,
    Gadgets.Sampling.First54.roundPrivateCount,
    Gadgets.Sampling.First54Step.slotCount,
    Gadgets.Sampling.First54ValueStep.outputCount,
    PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset, Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  omega

private theorem first54Value_before
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    descriptor.valueColumn < PiRLCStarts.commitmentFreshStart := by
  rcases descriptor with ⟨⟨source, round⟩, slot⟩
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have slotBound := slot.isLt
  change source.val < 17 at sourceBound
  change round.val < 64 at roundBound
  change slot.val < 54 at slotBound
  change _ < 21124348
  norm_num [PiRLCFirst54DirectSchedule.Value.valueColumn,
    PiRLCFirst54Invocations.valueSourceStart,
    Gadgets.Sampling.First54.valueOffset,
    Gadgets.Sampling.First54.positionOffset,
    Gadgets.Sampling.First54.roundPrivateCount,
    Gadgets.Sampling.First54Step.slotCount,
    Gadgets.Sampling.First54ValueStep.outputCount,
    PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset, Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  omega

private theorem shiftedSource_before (application : Program) (column : Nat)
    (before : column < PiRLCStarts.commitmentFreshStart) :
    Outside (PerApplicationPackage.shiftColumn application
      (Spartan.sourceToSpartan column)) := by
  apply shifted_outside
  apply mapped_outside column
  · exact lt_of_lt_of_le before (by
      rw [PiRLCStarts.commitmentFreshStart_eq, Spartan.sourceColumnCount_eq]
      decide)
  · exact Or.inl before

private theorem shiftedSource_after (application : Program) (column : Nat)
    (after : PiRLCStarts.outputFreshStart ≤ column) :
    Outside (PerApplicationPackage.shiftColumn application
      (Spartan.sourceToSpartan column)) := by
  apply shifted_outside
  right
  change 28973248 ≤ column at after
  rw [Spartan.sourceToSpartan, if_neg (by change ¬ column < 14722512; omega),
    if_neg (by change ¬ column < 14722516; omega),
    if_neg (by change ¬ column < 14751804; omega)]
  change 28972970 ≤ 14751526 + (column - 14751804)
  omega

private theorem proofLogical_before
    (slot : Fin PiCCSOrdinaryRetainedBlocks.proofLogicalCount) :
    PiCCSOrdinaryRetainedBlocks.proofLogicalSource slot <
      PiRLCStarts.commitmentFreshStart := by
  have bound := slot.isLt
  change slot.val < 114878 at bound
  unfold PiCCSOrdinaryRetainedBlocks.proofLogicalSource
  split
  · change 14722516 + slot.val < 21124348
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
      change 14751804 + decoded.1.val * 592 + 584 + decoded.2.val < 21124348
      omega
    · change PiCCSStarts.initialClaimLogicalStart +
          (slot.val - (29288 + 5744)) < 21124348
      have start : PiCCSStarts.initialClaimLogicalStart + 114878 ≤ 21124348 := by decide
      omega

private theorem poseidonBlock_outside (sourceWidth invocationCount : Nat)
    (witnessStart : Fin invocationCount → Nat)
    (witnessBound : ∀ invocation,
      witnessStart invocation + PoseidonScheduleTrace.localColumnCount ≤ sourceWidth)
    (before : ∀ invocation,
      witnessStart invocation + PoseidonScheduleTrace.localColumnCount ≤ 21124070)
    (slot : Fin (invocationCount * PoseidonRetainedSlots.rows.length)) :
    Outside ((Layout.ProductionRelation.PoseidonRetainedBlock.block sourceWidth
      invocationCount witnessStart witnessBound).source slot).val := by
  let indices : Fin invocationCount × Fin PoseidonRetainedSlots.rows.length :=
    Fin.decodeProd slot
  have startBound := before indices.1
  have localBound := (PoseidonRetainedSlots.localOutput indices.2).isLt
  left
  change witnessStart indices.1 +
    (PoseidonRetainedSlots.localOutput indices.2).val < 21124070
  omega

/-- Every retained block source avoids the discarded physical scratch. The
two derived suffixes are beyond the complete physical base. -/
theorem retainedSource_outside (application : Program) (kind : BlockKind)
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
  | first54Reject =>
      exact shiftedSource_before application _
        (first54Reject_before (PiRLCFirst54DirectSchedule.candidate slot))
  | first54Symbol =>
      exact shiftedSource_before application _
        (first54Symbol_before (PiRLCFirst54DirectSchedule.candidate slot))
  | first54Position =>
      exact shiftedSource_before application _
        (first54Position_before (PiRLCFirst54DirectSchedule.position slot))
  | first54Value =>
      exact shiftedSource_before application _
        (first54Value_before (PiRLCFirst54DirectSchedule.value slot))
  | first54Product =>
      right
      have baseBound := baseWidth_after application
      change scratchEnd ≤ PiRLCProductPlan.baseSourceWidth application + _ + _
      omega
  | productOutput =>
      exact shiftedSource_before application _
        (PiRLCCombinationScratchGeometry.outputSource_before
          (PiRLCProductSchedule.descriptor slot))
  | priorPoseidonInput =>
      left
      have bound := slot.isLt
      change slot.val < 49393 at bound
      change 0 + slot.val < 21124070
      omega
  | outputPoseidonInput =>
      left
      have bound := slot.isLt
      change slot.val < 49393 at bound
      change 49393 + slot.val < 21124070
      omega
  | runningPiDec =>
      apply shiftedSource_after
      change 28973248 ≤ 28973248 + slot.val
      omega
  | runningInverse =>
      simp only [BlockKind.template, RunningTransitionReducedRetainedBlocks.inverseBlock]
      change Outside (RunningTransitionReducedRetainedBlocks.inverseSource application).val
      rw [(RunningTransitionReducedRetainedBlocks.source_addresses application).1]
      exact Or.inr (by rw [scratchEnd_eq]; decide)
  | runningFlag =>
      simp only [BlockKind.template, RunningTransitionReducedRetainedBlocks.flagBlock]
      change Outside (RunningTransitionReducedRetainedBlocks.flagSource application).val
      rw [(RunningTransitionReducedRetainedBlocks.source_addresses application).2]
      exact Or.inr (by rw [scratchEnd_eq]; decide)
  | piCcsFreshPublicInput =>
      apply shiftedSource_before
      have bound := slot.isLt
      change slot.val < 270 at bound
      change 49393 + slot.val < 21124348
      omega
  | piCcsPriorLast =>
      apply shifted_outside
      left
      have bound := slot.isLt
      change slot.val < 592 at bound
      change PoseidonRetainedBlock.priorWitnessStart
        PiCCSOrdinaryRetainedBlocks.priorLastInvocation + slot.val < 21124070
      have endBound := PiRLCCombinationScratchPoseidon.priorWitnessEnd_le
        PiCCSOrdinaryRetainedBlocks.priorLastInvocation
      change _ + 592 ≤ 21124070 at endBound
      omega
  | piCcsOutputLast =>
      apply shifted_outside
      left
      have bound := slot.isLt
      change slot.val < 592 at bound
      change PoseidonRetainedBlock.outputWitnessStart
        PiCCSOrdinaryRetainedBlocks.outputLastInvocation + slot.val < 21124070
      have endBound := PiRLCCombinationScratchPoseidon.outputWitnessEnd_le
        PiCCSOrdinaryRetainedBlocks.outputLastInvocation
      change _ + 592 ≤ 21124070 at endBound
      omega
  | piCcsExpectedContext =>
      apply shiftedSource_before
      have bound := slot.isLt
      change slot.val < 4 at bound
      change 14722512 + slot.val < 21124348
      omega
  | piCcsProofLogical =>
      exact shiftedSource_before application _ (proofLogical_before slot)
  | piCcsOutputEndpoint =>
      apply shiftedSource_before
      have bound := slot.isLt
      change slot.val < 8 at bound
      change 19333210 + slot.val < 21124348
      omega
  | piCcsFresh =>
      apply shiftedSource_before
      have bound := slot.isLt
      change slot.val < 731605 at bound
      change 19333218 + slot.val < 21124348
      omega
  | pilotCanonicalLocal =>
      dsimp only [BlockKind.template, PilotOrdinaryRetainedBlocks.canonicalLocalBlock,
        PiCCSOrdinaryRetainedBlocks.sourceFieldBlock,
        RunningTransitionRetainedBlocks.packageSourceColumn,
        PiRLCRetainedPreservation.baseSourceColumn,
        PiRLCFirst54DirectPlan.prefixColumn, ProductRetainedBlock.baseColumn,
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
      change _ < 21124348
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
      change 99056 + slot.val < 21124348
      omega
  | piDecLogical =>
      apply shiftedSource_after
      have start : PiRLCStarts.outputFreshStart ≤ PiDECStarts.phaseLogicalStart := by decide
      change PiRLCStarts.outputFreshStart ≤ PiDECStarts.phaseLogicalStart + slot.val
      omega
  | piDecFresh =>
      apply shiftedSource_after
      have start : PiRLCStarts.outputFreshStart ≤ PiDECStarts.phaseFreshStart := by decide
      change PiRLCStarts.outputFreshStart ≤ PiDECStarts.phaseFreshStart + slot.val
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
      change 28972970 ≤ 29336446 + slot.val
      omega
  | applicationLocal =>
      right
      change 28972970 ≤ 29336446 + application.witnessWordCount + slot.val
      omega

private theorem blockCoordinate_congr {sourceWidth : Nat}
    (block : LowNormBlock.Block sourceWidth) (left right : Fin sourceWidth → F)
    (agree : ∀ slot, left (block.source slot) = right (block.source slot))
    (coordinate : Nat) :
    (Canonical.ofBlock block left).coordinateAt coordinate =
      (Canonical.ofBlock block right).coordinateAt coordinate := by
  simp only [Canonical.ofBlock, CanonicalBlockAssignment.ofBlock,
    CanonicalBlockAssignment.BlockValue.coordinateAt,
    CanonicalBlockAssignment.BlockValue.coordinateCount]
  split
  · rw [agree]
  · rfl

/-- Only selected block reads are required for assignment equality. Source
functions may differ at every source column which no retained slot reads. -/
private theorem coordinateAt_congr {application : Program}
    (left right : RawValues application)
    (agree : ∀ (kind : BlockKind) (slot : Fin (kind.template application).block.slotCount),
      (kind.template application).source left
          ((kind.template application).block.source slot) =
        (kind.template application).source right
          ((kind.template application).block.source slot))
    (kinds : List BlockKind) (coordinate : Nat) :
    CanonicalBlockAssignment.coordinateAt (kinds.map (BlockKind.expand left)) coordinate =
      CanonicalBlockAssignment.coordinateAt (kinds.map (BlockKind.expand right)) coordinate := by
  induction kinds generalizing coordinate with
  | nil => rfl
  | cons kind rest inductionHypothesis =>
    simp only [List.map_cons, CanonicalBlockAssignment.coordinateAt]
    have count : (kind.expand left).coordinateCount =
        (kind.expand right).coordinateCount := rfl
    rw [count]
    split
    · exact blockCoordinate_congr (kind.template application).block
        ((kind.template application).source left)
        ((kind.template application).source right) (agree kind) coordinate
    · exact inductionHypothesis _

private theorem retainedRead_congr {application : Program}
    (left right : RawValues application)
    (base : ∀ column, Outside column.val → left.base column = right.base column)
    (groups : left.groupValue = right.groupValue)
    (products : left.products = right.products)
    (column : Fin (PiRLCRetainedGeometry.sourceWidth application))
    (outside : Outside column.val) :
    left.retainedSource column = right.retainedSource column := by
  simp only [RawValues.retainedSource, PiRLCRetainedPreservation.sourceAssignment,
    PiRLCFirst54DirectPlan.sourceAssignment, FieldSuffixBlock.sourceAssignment,
    PiRLCProductPlan.sourceAssignment, ProductRetainedBlock.sourceAssignment]
  split
  · split
    · exact base _ outside
    · rw [groups]
  · rw [products]

private theorem sourceEnv_congr {application : Program}
    (left right : PerApplicationAssignmentTransportExecution.BaseValues application)
    (agree : ∀ column, Outside column.val → left column = right column)
    (column : Nat) (outside : Outside column) :
    SourceCompiler.sourceEnv left column = SourceCompiler.sourceEnv right column := by
  unfold SourceCompiler.sourceEnv
  split
  · exact agree _ outside
  · rfl

private theorem baseBlockValue_congr {application : Program}
    (left right : PerApplicationAssignmentTransportExecution.BaseValues application)
    (agree : ∀ column, Outside column.val → left column = right column)
    (kind : BlockKind) (slot : Nat) :
    PerApplicationAssignmentTransportProducts.baseBlockValue application left kind slot =
      PerApplicationAssignmentTransportProducts.baseBlockValue application right kind slot := by
  unfold PerApplicationAssignmentTransportProducts.baseBlockValue
  split
  · dsimp only
    split
    · apply agree
      exact retainedSource_outside application kind _
    · rfl
  · rfl

private theorem productValue_before (descriptor : PiRLCProductSchedule.Descriptor)
    (lane : Fin ringDegree) :
    descriptor.valueColumn lane < PiRLCStarts.commitmentFreshStart := by
  rcases descriptor with ⟨family, source, block, productLane, cell⟩
  cases family <;>
    simp only [PiRLCProductSchedule.Descriptor.valueColumn,
      PiRLCProductSchedule.Family.blockCount,
      PiRLCProductSchedule.Family.cellCount] at *
  all_goals
    have sourceBound := source.isLt
    have blockBound := block.isLt
    have cellBound := cell.isLt
    have laneBound := lane.isLt
    norm_num [PiRLCCombinationInvocations.sourceCount, ringDegree,
      PiRLCCombinationInvocations.commitmentValueSourceStart,
      PiRLCCombinationInvocations.publicInputValueSourceStart,
      PiRLCCombinationInvocations.evalKValueSourceStart,
      PiRLCCombinationInvocations.evalAValueSourceStart,
      PiCCSInputs.freshCommitmentStart, PiCCSInputs.runningCommitmentStart,
      PiCCSInputs.runningPublicStart, PiCCSInputs.runningGroupStart,
      PiCCSInputs.runningGroupsStart, PiCCSInputs.priorRunningStart,
      PiCCSInputs.runningGroupWords, PiCCSInputs.outputEvaluationStart,
      PiCCSInputs.roundMessageStart, PiCCSInputs.freshCommitmentWords,
      PiCCSInputs.proofInputStart, PiCCSInputs.expectedContextStart,
      PiCCSInputs.expectedContextWords, PiCCSInputs.roundMessageWords,
      PilotProduction.priorPublicInputStart, PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq,
      PiRLCStarts.commitmentFreshStart_eq] at sourceBound blockBound cellBound laneBound ⊢
  all_goals (try split) <;> omega

private theorem recipeIndex_eq (application : Program)
    (descriptor : PiRLCProductSchedule.Descriptor) :
    PerApplicationAssignmentTransportProducts.invocationIndex
      (PerApplicationAssignmentTransport.phi81GroupRecipe application) descriptor =
        descriptor.invocation.val := by
  rw [PiRLCProductSchedule.Descriptor.invocation_val]
  rcases descriptor with ⟨family, source, block, lane, cell⟩
  cases family <;>
    simp [PerApplicationAssignmentTransportProducts.invocationIndex,
      PerApplicationAssignmentTransportProducts.familyOffset,
      PerApplicationAssignmentTransportProducts.familyShape,
      PerApplicationAssignmentTransportProducts.familyOrdinal,
      PerApplicationAssignmentTransportProducts.shapeInvocationCount,
      PerApplicationAssignmentTransport.phi81GroupRecipe,
      PerApplicationAssignmentTransport.phi81FamilyShapes,
      PiRLCProductSchedule.Family.blockCount,
      PiRLCProductSchedule.Family.cellCount, PiRLCCombinationInvocations.sourceCount,
      ringDegree, List.getD]
  all_goals omega

private theorem derivedGroups_congr {application : Program}
    (left right : PerApplicationAssignmentTransportExecution.BaseValues application)
    (agree : ∀ column, Outside column.val → left column = right column) :
    (PerApplicationAssignmentTransportExecution.canonicalRawValues application left).groupValue =
      (PerApplicationAssignmentTransportExecution.canonicalRawValues application right).groupValue := by
  have challengeEq (descriptor : PiRLCProductSchedule.Descriptor) :
      PerApplicationAssignmentTransportProducts.challengeRing
        (PerApplicationAssignmentTransport.phi81GroupRecipe application) application left descriptor =
      PerApplicationAssignmentTransportProducts.challengeRing
        (PerApplicationAssignmentTransport.phi81GroupRecipe application) application right descriptor := by
    funext lane
    unfold PerApplicationAssignmentTransportProducts.challengeRing
    rw [baseBlockValue_congr left right agree]
  have valueEq (descriptor : PiRLCProductSchedule.Descriptor) :
      PerApplicationAssignmentTransportProducts.valueRing
        (PerApplicationAssignmentTransport.phi81GroupRecipe application) application left descriptor =
      PerApplicationAssignmentTransportProducts.valueRing
        (PerApplicationAssignmentTransport.phi81GroupRecipe application) application right descriptor := by
    funext lane
    unfold PerApplicationAssignmentTransportProducts.valueRing
    rw [recipeIndex_eq]
    dsimp only [PerApplicationAssignmentTransport.phi81GroupRecipe]
    rw [PerApplicationAssignmentTransport.phi81ValueSources_at,
      PiRLCProductSchedule.descriptor_invocation]
    apply sourceEnv_congr left right agree
    exact shiftedSource_before application _ (productValue_before _ _)
  funext invocation group
  change PerApplicationAssignmentTransportProducts.phi81GroupValue
      (PerApplicationAssignmentTransport.phi81GroupRecipe application) application left
      invocation group.val =
    PerApplicationAssignmentTransportProducts.phi81GroupValue
      (PerApplicationAssignmentTransport.phi81GroupRecipe application) application right
      invocation group.val
  simp only [PerApplicationAssignmentTransportProducts.phi81GroupValue,
    challengeEq, valueEq]

private theorem derivedProducts_congr {application : Program}
    (left right : PerApplicationAssignmentTransportExecution.BaseValues application)
    (agree : ∀ column, Outside column.val → left column = right column) :
    (PerApplicationAssignmentTransportExecution.canonicalRawValues application left).products =
      (PerApplicationAssignmentTransportExecution.canonicalRawValues application right).products := by
  funext candidate
  change PerApplicationAssignmentTransportProducts.first54ProductValue
      PerApplicationAssignmentTransport.first54ProductRecipe application left candidate.val =
    PerApplicationAssignmentTransportProducts.first54ProductValue
      PerApplicationAssignmentTransport.first54ProductRecipe application right candidate.val
  unfold PerApplicationAssignmentTransportProducts.first54ProductValue
  rw [baseBlockValue_congr left right agree, baseBlockValue_congr left right agree]

private theorem outputDigest_congr {application : Program}
    (left right : RawValues application)
    (agree : ∀ column, Outside column.val → left.base column = right.base column) :
    left.outputDigest = right.outputDigest := by
  unfold RawValues.outputDigest
  apply congrArg List.ofFn
  funext lane
  change RunningTransitionDirectPlan.transitionEnv application left.base
      (Spartan.liftPilotColumn
        (PilotSpartan.sourceToSpartan (PilotProduction.outputDigestStart + lane.val))) =
    RunningTransitionDirectPlan.transitionEnv application right.base
      (Spartan.liftPilotColumn
        (PilotSpartan.sourceToSpartan (PilotProduction.outputDigestStart + lane.val)))
  have pilotBound : PilotProduction.outputDigestStart + lane.val <
      Spartan.pilotSourceColumnCount := by
    change 99056 + lane.val < 14722512
    have bound := lane.isLt
    change lane.val < 4 at bound
    omega
  have mapped : Spartan.sourceToSpartan (PilotProduction.outputDigestStart + lane.val) =
      Spartan.liftPilotColumn
        (PilotSpartan.sourceToSpartan (PilotProduction.outputDigestStart + lane.val)) := by
    rw [Spartan.sourceToSpartan, if_pos pilotBound]
  rw [← mapped]
  have sourceBound : PilotProduction.outputDigestStart + lane.val < Spartan.SourceColumnCount :=
    lt_of_lt_of_le pilotBound (by rw [Spartan.sourceColumnCount_eq]; decide)
  have before : PilotProduction.outputDigestStart + lane.val < PiCCSInputs.phaseOffset :=
    lt_of_lt_of_le pilotBound (by decide)
  rw [RunningTransitionDirectPlan.transitionEnv_of_outside application left.base _
      sourceBound (Or.inl before),
    RunningTransitionDirectPlan.transitionEnv_of_outside application right.base _
      sourceBound (Or.inl before)]
  apply sourceEnv_congr left.base right.base agree
  exact retainedSource_outside application .pilotOutputDigest lane

/-- Changing only discarded combination scratch leaves every coordinate of
the complete CCS carrier unchanged, including the encoded public digest. -/
theorem completeAssignment_congr (application : Program)
    (left right : PerApplicationAssignmentTransportExecution.BaseValues application)
    (agree : ∀ column, Outside column.val → left column = right column) :
    (PerApplicationAssignmentTransportExecution.canonicalRawValues application left).completeAssignment =
      (PerApplicationAssignmentTransportExecution.canonicalRawValues application right).completeAssignment := by
  let first := PerApplicationAssignmentTransportExecution.canonicalRawValues application left
  let second := PerApplicationAssignmentTransportExecution.canonicalRawValues application right
  have digest : first.outputDigest = second.outputDigest :=
    outputDigest_congr first second agree
  have groups : first.groupValue = second.groupValue := derivedGroups_congr left right agree
  have products : first.products = second.products := derivedProducts_congr left right agree
  have reads : ∀ (kind : BlockKind) (slot : Fin (kind.template application).block.slotCount),
      (kind.template application).source first
          ((kind.template application).block.source slot) =
        (kind.template application).source second
          ((kind.template application).block.source slot) := by
    intro kind slot
    have outside := retainedSource_outside application kind slot
    cases kind <;> dsimp only [BlockKind.template] at outside ⊢
    all_goals first
    | exact retainedRead_congr first second agree groups products _ outside
    | exact agree _ outside
  have coordinates := coordinateAt_congr first second reads canonicalKinds
  have assignments : first.assignment = second.assignment := by
    funext column
    unfold RawValues.assignment Canonical.assignment CanonicalBlockAssignment.assignment
    rw [digest]
    split
    · rfl
    · rw [← expand_eq_schedule first, ← expand_eq_schedule second]
      exact coordinates _
  funext column
  unfold RawValues.completeAssignment
  rw [assignments]

/-- The public digest itself, before its fixed low-norm public encoding, is
also independent of all discarded multiplication scratch. -/
theorem canonicalOutputDigest_congr (application : Program)
    (left right : PerApplicationAssignmentTransportExecution.BaseValues application)
    (agree : ∀ column, Outside column.val → left column = right column) :
    (PerApplicationAssignmentTransportExecution.canonicalRawValues application left).outputDigest =
      (PerApplicationAssignmentTransportExecution.canonicalRawValues application right).outputDigest :=
  outputDigest_congr _ _ agree

/-- Every declared private segment except the generated main-witness segment
is disjoint from combination scratch. This includes every caller-owned base
input segment and, more strongly, both later generated-witness segments. -/
theorem baseInputSegment_outside (segment : Export.Package.Segment)
    (member : segment ∈ Data.privateSegments)
    (inputRole : segment.role ≠ Data.Role.witness)
    (index : Fin segment.length) : Outside (segment.start + index.val) := by
  unfold Data.privateSegments at member
  rcases List.mem_append.mp member with before | after
  · rcases List.mem_append.mp before with initial | outputs
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at initial
      rcases initial with rfl | rfl | rfl | rfl
      all_goals
        left
        have bound := index.isLt
        norm_num [scratchStart_eq, Data.proofInputStart,
          Spartan.pilotInputPrivateColumnCount, PilotValues.stateHashWords,
          PilotValues.stateHashBaseWords, PiCCSInputs.freshCommitmentWords,
          PiCCSInputs.roundMessageWords] at bound ⊢
        omega
    · unfold Data.piCcsOutputSegments at outputs
      rcases List.mem_flatMap.mp outputs with ⟨source, _, selected⟩
      simp only [List.mem_cons, List.not_mem_nil, or_false] at selected
      have sourceBound := source.isLt
      change source.val < 17 at sourceBound
      rcases selected with rfl | rfl
      · left
        have bound := index.isLt
        change index.val < 108 at bound
        change 100534 + source.val * 1620 + index.val < 21124070
        omega
      · left
        have bound := index.isLt
        change index.val < 1512 at bound
        change 100534 + source.val * 1620 + 108 + index.val < 21124070
        omega
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at after
    rcases after with rfl | rfl | rfl | rfl | rfl | rfl | rfl
    · exact False.elim (inputRole rfl)
    all_goals
      right
      exact Nat.le_trans (by decide) (Nat.le_add_right _ index.val)

private theorem column_le_shifted (application : Program) (column : Nat) :
    column ≤ PerApplicationPackage.shiftColumn application column := by
  unfold PerApplicationPackage.shiftColumn
  split_ifs <;> omega

/-- The declared public segments are outside scratch before and after the
application insertion. Thus the verifier context and public inputs cannot be
changed by omission of multiplication scratch. -/
theorem publicSegment_outside (application : Program)
    (segment : Export.Package.Segment) (member : segment ∈ Data.publicSegments)
    (index : Fin segment.length) :
    Outside ((PerApplicationPackage.shiftSegment application segment).start + index.val) := by
  unfold Data.publicSegments at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  all_goals
    right
    dsimp only [PerApplicationPackage.shiftSegment]
    apply Nat.le_trans _ (Nat.le_add_right _ index.val)
    apply Nat.le_trans _ (column_le_shifted application _)
    norm_num [scratchEnd_eq, Spartan.expectedContextPublicStart,
      Spartan.privateColumnCount_eq, Spartan.pilotPublicColumnCount,
      PilotValues.priorPublicInputWords]

/-- Both application-private segments follow the old physical base. The
statement includes arbitrary application witness and local lengths. -/
theorem applicationSegment_outside (application : Program)
    (segment : Export.Package.Segment)
    (member : segment ∈ [PerApplicationPackage.applicationWitnessSegment application,
      PerApplicationPackage.applicationLocalSegment application])
    (index : Fin segment.length) : Outside (segment.start + index.val) := by
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · right
    change scratchEnd ≤ PerApplicationPackage.basePackage.layout.constantColumn + index.val
    rw [scratchEnd_eq, baseConstant_eq]
    omega
  · right
    change 28972970 ≤ 29336446 + application.witnessWordCount + index.val
    omega

private theorem copiedBase_congr (application : Program) (left right : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F)
    (agree : ∀ column, Outside column → left column = right column)
    (column : Fin (PiRLCProductPlan.baseSourceWidth application))
    (outside : Outside column.val) :
    PerApplicationSourceAssignment.ofCompleted application left applicationPrivate column =
      PerApplicationSourceAssignment.ofCompleted application right applicationPrivate column := by
  unfold PerApplicationSourceAssignment.ofCompleted
  split
  · exact agree _ outside
  · split
    · rfl
    · rename_i after
      apply agree
      right
      rw [baseConstant_eq] at after
      rw [scratchEnd_eq]
      omega

/-- At the end of the combination phase, direct output writes give
the same complete CCS carrier and public digest as successful full execution.
The full execution exists for every input environment; no scratch values,
source agreement, or row-validity premise is supplied by the caller.
Preservation through the later producer schedule is a separate obligation. -/
theorem directPhase_preserves_completeAssignment (application : Program) (env : Env)
    (applicationPrivate : Fin (PerApplicationPackage.addedPrivateColumnCount application) → F) :
    ∃ completed, PiRLCCombinationDirectWitness.fullPhase env = some completed ∧
      (PerApplicationAssignmentTransportExecution.canonicalRawValues application
        (PerApplicationSourceAssignment.ofCompleted application completed applicationPrivate)).completeAssignment =
      (PerApplicationAssignmentTransportExecution.canonicalRawValues application
        (PerApplicationSourceAssignment.ofCompleted application
          (PiRLCCombinationDirectWitness.directPhase env) applicationPrivate)).completeAssignment ∧
      (PerApplicationAssignmentTransportExecution.canonicalRawValues application
        (PerApplicationSourceAssignment.ofCompleted application completed applicationPrivate)).outputDigest =
      (PerApplicationAssignmentTransportExecution.canonicalRawValues application
        (PerApplicationSourceAssignment.ofCompleted application
          (PiRLCCombinationDirectWitness.directPhase env) applicationPrivate)).outputDigest := by
  rcases PiRLCCombinationDirectWitness.fullPhase_agrees_directPhase env with
    ⟨completed, success, agrees⟩
  have copy := copiedBase_congr application completed
    (PiRLCCombinationDirectWitness.directPhase env) applicationPrivate agrees
  exact ⟨completed, success, completeAssignment_congr application _ _ copy,
    canonicalOutputDigest_congr application _ _ copy⟩

end NightstreamFPrime.Export.Stage1.PiRLCCombinationScratchCustody
