import NightstreamFPrime.Export.Stage1.PiRLCSamplerRetainedCustody

/-!
Owns exact source-column custody for the First54 part of each retained PiRLC
sampler. The proof uses source-relative offsets and does not inspect row data.

This module does not prove First54 recurrence semantics or close PiRLC status.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerSelectorCustody

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def selectorColumn
    (source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount)
    (inner : Nat) : Nat :=
  PiRLCStarts.samplerSourceLogicalStart source.val + inner

theorem poseidonSource_eq
    (source round : Nat) (lane : Fin 4) :
    PiRLCSamplerOrdinaryDirectSource.poseidonSource source round lane =
      PiRLCStarts.samplerLogicalStart + source * 15504 + 584 +
        round * 992 + lane.val := by
  cases round with
  | zero =>
      simp [PiRLCSamplerOrdinaryDirectSource.poseidonSource,
        PiRLCStarts.samplerSourceLogicalStart]
  | succ previous =>
      simp [PiRLCSamplerOrdinaryDirectSource.poseidonSource,
        SamplerChain.sourceOffset,
        Sampler.windowOffset, Sampler.windowBase,
        DigestWindow.permutationOffset, Sampler.logicalPrivateCount,
        Sampler.entryPrivateCount, DigestWindow.logicalPrivateCount,
        DigestLane.logicalPrivateCount]
      omega

theorem logicalSource_eq
    (descriptor : PiRLCSamplerOrdinaryRetainedBlocks.Lane)
    (position : Fin
      PiRLCSamplerOrdinaryRetainedBlocks.logicalCountPerLane) :
    PiRLCSamplerOrdinaryRetainedBlocks.logicalSource descriptor position =
      PiRLCStarts.samplerLogicalStart + descriptor.source.val * 15504 + 592 +
        descriptor.round.val * 992 + descriptor.lane.val * 100 +
          position.val := by
  simp [PiRLCSamplerOrdinaryRetainedBlocks.logicalSource,
    PiRLCStarts.digestLaneLogicalStart, PiRLCStarts.windowLogicalStart,
    PiRLCStarts.samplerSourceLogicalStart]

theorem selectorSource_eq (source : Nat) :
    PiRLCSamplerOrdinaryDirectSource.selectorSource source =
      PiRLCStarts.samplerLogicalStart + source * 15504 + 15449 := by
  norm_num [PiRLCSamplerOrdinaryDirectSource.selectorSource,
    PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    NightstreamFPrime.Gadgets.Sampling.First54.positionOffset,
    NightstreamFPrime.Gadgets.Sampling.First54.candidateCount,
    NightstreamFPrime.Gadgets.Sampling.First54.roundPrivateCount,
    NightstreamFPrime.Gadgets.Sampling.First54.fullSlot,
    NightstreamFPrime.Gadgets.Sampling.First54Step.fullSlot,
    NightstreamFPrime.Gadgets.Sampling.First54Step.slotCount,
    NightstreamFPrime.Gadgets.Sampling.First54ValueStep.outputCount]

theorem stateSource_eq
    (location : PiRLCSamplerRetainedCustody.StateLocation) :
    location.sourceColumn =
      PiRLCStarts.samplerLogicalStart + location.source.val * 15504 + 584 +
        location.step.val * 992 + location.lane.val := by
  simp [PiRLCSamplerRetainedCustody.StateLocation.sourceColumn,
    Sampler.logicalPrivateCount,
    DigestWindow.logicalPrivateCount,
    PiRLCSamplerRetainedCustody.stateOutputOffset]

private theorem selectorColumn_lt_sourceColumnCount
    (source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount)
    (inner : Nat) (innerLt : inner < 15504) :
    selectorColumn source inner < Spartan.SourceColumnCount := by
  have sourceLt := source.isLt
  change source.val < 17 at sourceLt
  rw [Spartan.sourceColumnCount_eq]
  norm_num [selectorColumn, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  omega

private theorem selectorColumn_lt_samplerFreshStart
    (source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount)
    (inner : Nat) (innerLt : inner < 15504) :
    selectorColumn source inner < PiRLCStarts.samplerFreshStart := by
  have sourceLt := source.isLt
  change source.val < 17 at sourceLt
  unfold PiRLCStarts.samplerFreshStart
  rw [PiRLCStarts.phaseFreshStart_eq]
  norm_num [selectorColumn, PiRLCStarts.samplerSourceLogicalStart,
    PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
    PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
  omega

private theorem ordinaryTarget_none
    (source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount)
    (inner : Nat) (selectorLower : 8528 ≤ inner)
    (innerLt : inner < 15504) (notFinal : inner ≠ 15449) :
    PiRLCSamplerOrdinaryDirectPlan.classifyTarget
        (Spartan.sourceToSpartan (selectorColumn source inner)) = none := by
  have columnLt := selectorColumn_lt_sourceColumnCount source inner innerLt
  unfold PiRLCSamplerOrdinaryDirectPlan.classifyTarget
  rw [Spartan.spartanToSource_sourceToSpartan _ columnLt]
  cases found : PiRLCSamplerOrdinaryDirectPlan.classifySource
      (selectorColumn source inner) with
  | none => exact found
  | some location =>
      have owns := PiRLCSamplerOrdinaryDirectPlan.classifySource_sound found
      exfalso
      cases location with
      | poseidon descriptor sourceLane =>
          rcases descriptor with ⟨other, round, digestLane⟩
          simp only [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn] at owns
          rw [poseidonSource_eq] at owns
          have sourceLt := source.isLt
          have otherLt := other.isLt
          have roundLt := round.isLt
          have sourceLaneLt := sourceLane.isLt
          change source.val < 17 at sourceLt
          change other.val < 17 at otherLt
          change round.val < 8 at roundLt
          change sourceLane.val < 4 at sourceLaneLt
          change PiRLCStarts.samplerLogicalStart + other.val * 15504 + 584 +
              round.val * 992 + sourceLane.val =
            PiRLCStarts.samplerLogicalStart + source.val * 15504 + inner at owns
          by_cases same : other.val = source.val
          · rw [same] at owns
            omega
          · rcases Nat.lt_or_gt_of_ne same with previous | next
            · omega
            · omega
      | logical descriptor position =>
          simp only [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn] at owns
          rw [logicalSource_eq] at owns
          have sourceLt := source.isLt
          have otherLt := descriptor.source.isLt
          have roundLt := descriptor.round.isLt
          have laneLt := descriptor.lane.isLt
          have positionLt := position.isLt
          change source.val < 17 at sourceLt
          change descriptor.source.val < 17 at otherLt
          change descriptor.round.val < 8 at roundLt
          change descriptor.lane.val < 4 at laneLt
          change position.val < 100 at positionLt
          change PiRLCStarts.samplerLogicalStart +
              descriptor.source.val * 15504 + 592 +
                descriptor.round.val * 992 + descriptor.lane.val * 100 +
                  position.val =
            PiRLCStarts.samplerLogicalStart + source.val * 15504 + inner at owns
          by_cases same : descriptor.source.val = source.val
          · rw [same] at owns
            omega
          · rcases Nat.lt_or_gt_of_ne same with previous | next
            · omega
            · omega
      | fresh descriptor position =>
          simp only [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn] at owns
          have freshLower : PiRLCStarts.samplerFreshStart ≤
              PiRLCSamplerOrdinaryRetainedBlocks.freshSource descriptor
                position := by
            simp [PiRLCSamplerOrdinaryRetainedBlocks.freshSource,
              PiRLCStarts.digestLaneFreshStart,
              PiRLCStarts.windowFreshStart,
              PiRLCStarts.samplerSourceFreshStart]
            omega
          have beforeFresh := selectorColumn_lt_samplerFreshStart source inner
            innerLt
          rw [owns] at freshLower
          omega
      | selector other =>
          simp only [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn] at owns
          rw [selectorSource_eq] at owns
          have sourceLt := source.isLt
          have otherLt := other.isLt
          change source.val < 17 at sourceLt
          change other.val < 17 at otherLt
          change PiRLCStarts.samplerLogicalStart + other.val * 15504 + 15449 =
            PiRLCStarts.samplerLogicalStart + source.val * 15504 + inner at owns
          have sameSource : other.val = source.val := by omega
          have sameInner : inner = 15449 := by omega
          exact notFinal sameInner

private theorem stateTarget_none
    (source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount)
    (inner : Nat) (selectorLower : 8528 ≤ inner)
    (innerLt : inner < 15504) :
    PiRLCSamplerRetainedCustody.classifyStateTarget
        (Spartan.sourceToSpartan (selectorColumn source inner)) = none := by
  have columnLt := selectorColumn_lt_sourceColumnCount source inner innerLt
  unfold PiRLCSamplerRetainedCustody.classifyStateTarget
  rw [Spartan.spartanToSource_sourceToSpartan _ columnLt]
  cases found : PiRLCSamplerRetainedCustody.classifyStateSource
      (selectorColumn source inner) with
  | none => exact found
  | some location =>
      have owns := PiRLCSamplerRetainedCustody.classifyStateSource_sound found
      exfalso
      rw [stateSource_eq] at owns
      have sourceLt := source.isLt
      have otherLt := location.source.isLt
      have stepLt := location.step.isLt
      have laneLt := location.lane.isLt
      change source.val < 17 at sourceLt
      change location.source.val < 17 at otherLt
      change location.step.val < 9 at stepLt
      change location.lane.val < 8 at laneLt
      change PiRLCStarts.samplerLogicalStart + location.source.val * 15504 +
          584 + location.step.val * 992 + location.lane.val =
        PiRLCStarts.samplerLogicalStart + source.val * 15504 + inner at owns
      by_cases same : location.source.val = source.val
      · rw [same] at owns
        omega
      · rcases Nat.lt_or_gt_of_ne same with previous | next
        · omega
        · omega

/-- Every non-final First54 source-relative column keeps the canonical
direct-plan base value in the complete retained sampler environment. -/
theorem semanticEnv_eq_baseEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (source : Fin PiRLCSamplerOrdinaryRetainedBlocks.sourceCount)
    (inner : Nat) (selectorLower : 8528 ≤ inner)
    (innerLt : inner < 15504) (notFinal : inner ≠ 15449) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        (selectorColumn source inner) =
      PiRLCFirst54DirectPlan.baseEnv program base
        (selectorColumn source inner) := by
  have ordinaryNone := ordinaryTarget_none source inner selectorLower innerLt
    notFinal
  have stateNone := stateTarget_none source inner selectorLower innerLt
  have privateBound : selectorColumn source inner <
      PiRLCProductPlan.basePackage.layout.constantColumn := by
    have constant : PiRLCProductPlan.basePackage.layout.constantColumn =
        29336446 :=
      NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.1
    rw [constant]
    have sourceLt := source.isLt
    change source.val < 17 at sourceLt
    norm_num [selectorColumn, PiRLCStarts.samplerSourceLogicalStart,
      PiRLCStarts.samplerLogicalStart, PiRLCStarts.phaseLogicalStart,
      PiRLCInputs.phaseOffset,
      NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]
    omega
  unfold Spartan.pullback PiRLCSamplerRetainedCustody.semanticEnv
  rw [ordinaryNone, stateNone]
  exact (PiRLCSamplerRetainedCustody.baseEnv_eq_transitionEnv program base
    (selectorColumn source inner) privateBound (by
      right
      norm_num [selectorColumn, PiCCSInputs.phaseOffset_eq,
        PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq,
        PiRLCStarts.samplerSourceLogicalStart, PiRLCStarts.samplerLogicalStart,
        PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
        NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset] <;> omega)).symm

def positionInner (descriptor : PiRLCFirst54DirectSchedule.Position) : Nat :=
  8528 + descriptor.candidate.round.val * 109 + descriptor.slot.val

def valueInner (descriptor : PiRLCFirst54DirectSchedule.Value) : Nat :=
  8528 + descriptor.candidate.round.val * 109 + 55 + descriptor.slot.val

theorem positionColumn_eq_selectorColumn
    (descriptor : PiRLCFirst54DirectSchedule.Position) :
    descriptor.positionColumn =
      selectorColumn descriptor.candidate.source (positionInner descriptor) := by
  simp [PiRLCFirst54DirectSchedule.Position.positionColumn,
    PiRLCFirst54Invocations.positionSourceStart, First54.positionOffset,
    PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    selectorColumn, positionInner, First54.roundPrivateCount,
    First54Step.slotCount, First54ValueStep.outputCount]
  omega

theorem valueColumn_eq_selectorColumn
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    descriptor.valueColumn =
      selectorColumn descriptor.candidate.source (valueInner descriptor) := by
  simp [PiRLCFirst54DirectSchedule.Value.valueColumn,
    PiRLCFirst54Invocations.valueSourceStart, First54.valueOffset,
    First54.positionOffset, PiRLCStarts.selectorLogicalStart,
    PiRLCStarts.samplerSourceLogicalStart, selectorColumn, valueInner,
    First54.roundPrivateCount, First54Step.slotCount,
    First54ValueStep.outputCount]
  omega

/-- Every First54 position output except the unique final-full slot keeps the
exact direct-plan base value. -/
theorem semanticEnv_position_eq_baseEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCFirst54DirectSchedule.Position)
    (notFinal : descriptor.positionColumn ≠
      PiRLCSamplerOrdinaryDirectSource.selectorSource
        descriptor.candidate.source.val) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        descriptor.positionColumn =
      PiRLCFirst54DirectPlan.baseEnv program base
        descriptor.positionColumn := by
  have roundLt := descriptor.candidate.round.isLt
  have slotLt := descriptor.slot.isLt
  change descriptor.candidate.round.val < 64 at roundLt
  change descriptor.slot.val < 55 at slotLt
  have lower : 8528 ≤ positionInner descriptor := by
    simp [positionInner]
    omega
  have upper : positionInner descriptor < 15504 := by
    simp [positionInner]
    omega
  have innerNot : positionInner descriptor ≠ 15449 := by
    intro same
    apply notFinal
    rw [positionColumn_eq_selectorColumn, selectorSource_eq]
    simp [selectorColumn, PiRLCStarts.samplerSourceLogicalStart, same]
  rw [positionColumn_eq_selectorColumn]
  exact semanticEnv_eq_baseEnv geometry assignment base
    descriptor.candidate.source (positionInner descriptor) lower upper innerNot

/-- Every First54 value output keeps the exact direct-plan base value. -/
theorem semanticEnv_value_eq_baseEnv
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (descriptor : PiRLCFirst54DirectSchedule.Value) :
    Spartan.pullback
        (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)
        descriptor.valueColumn =
      PiRLCFirst54DirectPlan.baseEnv program base descriptor.valueColumn := by
  have roundLt := descriptor.candidate.round.isLt
  have slotLt := descriptor.slot.isLt
  change descriptor.candidate.round.val < 64 at roundLt
  change descriptor.slot.val < 54 at slotLt
  have lower : 8528 ≤ valueInner descriptor := by
    simp [valueInner]
    omega
  have upper : valueInner descriptor < 15504 := by
    simp [valueInner]
    omega
  have innerNot : valueInner descriptor ≠ 15449 := by
    simp [valueInner]
    omega
  rw [valueColumn_eq_selectorColumn]
  exact semanticEnv_eq_baseEnv geometry assignment base
    descriptor.candidate.source (valueInner descriptor) lower upper innerNot

theorem priorPosition_eval_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (candidate : PiRLCFirst54DirectSchedule.Candidate)
    (slot : Fin First54Step.slotCount) :
    (First54.priorPosition
        (PiRLCStarts.selectorLogicalStart candidate.source.val)
        candidate.round.val slot).eval
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) =
      (First54.priorPosition
        (PiRLCStarts.selectorLogicalStart candidate.source.val)
        candidate.round.val slot).eval
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  rcases candidate with ⟨source, ⟨round, roundLt⟩⟩
  cases round with
  | zero =>
      by_cases first : slot.val = 0 <;>
        simp [First54.priorPosition, First54.initialPosition, first]
  | succ previous =>
      let previousRound : Fin PiRLCFirst54DirectSchedule.roundCount :=
        ⟨previous, by
          change previous + 1 < 64 at roundLt
          change previous < 64
          omega⟩
      let descriptor : PiRLCFirst54DirectSchedule.Position :=
        ⟨⟨source, previousRound⟩, slot⟩
      have notFinal : descriptor.positionColumn ≠
          PiRLCSamplerOrdinaryDirectSource.selectorSource source.val := by
        intro same
        rw [positionColumn_eq_selectorColumn, selectorSource_eq] at same
        change PiRLCStarts.samplerLogicalStart + source.val * 15504 +
            (8528 + previous * 109 + slot.val) =
          PiRLCStarts.samplerLogicalStart + source.val * 15504 + 15449 at same
        have slotLt := slot.isLt
        change slot.val < 55 at slotLt
        change previous + 1 < 64 at roundLt
        omega
      have custody := semanticEnv_position_eq_baseEnv geometry assignment base
        descriptor notFinal
      simpa [First54.priorPosition, descriptor, previousRound,
        PiRLCFirst54DirectSchedule.Position.positionColumn,
        PiRLCFirst54Invocations.positionSourceStart] using custody

theorem priorOutput_eval_eq
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PiRLCSamplerOrdinaryRetainedGeometry.Geometry program
      logicalWidth) (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (candidate : PiRLCFirst54DirectSchedule.Candidate)
    (slot : Fin First54ValueStep.outputCount) :
    (First54.priorOutput
        (PiRLCStarts.selectorLogicalStart candidate.source.val)
        candidate.round.val slot).eval
        (Spartan.pullback
          (PiRLCSamplerRetainedCustody.semanticEnv geometry assignment base)) =
      (First54.priorOutput
        (PiRLCStarts.selectorLogicalStart candidate.source.val)
        candidate.round.val slot).eval
        (PiRLCFirst54DirectPlan.baseEnv program base) := by
  rcases candidate with ⟨source, ⟨round, roundLt⟩⟩
  cases round with
  | zero => simp [First54.priorOutput]
  | succ previous =>
      let previousRound : Fin PiRLCFirst54DirectSchedule.roundCount :=
        ⟨previous, by
          change previous + 1 < 64 at roundLt
          change previous < 64
          omega⟩
      let descriptor : PiRLCFirst54DirectSchedule.Value :=
        ⟨⟨source, previousRound⟩, slot⟩
      have custody := semanticEnv_value_eq_baseEnv geometry assignment base
        descriptor
      simpa [First54.priorOutput, descriptor, previousRound,
        PiRLCFirst54DirectSchedule.Value.valueColumn,
        PiRLCFirst54Invocations.valueSourceStart] using custody

end NightstreamFPrime.Export.Stage1.PiRLCSamplerSelectorCustody
