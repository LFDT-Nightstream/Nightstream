import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSchedule
import NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryDirectPlan
import NightstreamFPrime.Export.MatrixProgram.Ordinary

/-!
Owns the four compact source grids for the PiRLC sampler ordinary matrix
block. The grids cover Poseidon2 outputs, digest-lane logical values,
digest-lane fresh values, and final selector outputs.

Every grid preserves the exact direct source resolver. Gap rejection and the
complete substitution theorem are proved in this module.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSubstitution

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open PiRLCSamplerOrdinaryRetainedBlocks
open PiRLCSamplerOrdinaryRetainedGeometry

abbrev Program := Lifecycle.Stage1.Application.Program

def poseidonSourceStart : Nat :=
  Spartan.sourceToSpartan (PiRLCStarts.samplerLogicalStart + 584)

def logicalSourceStart : Nat :=
  Spartan.sourceToSpartan (PiRLCStarts.samplerLogicalStart + 592)

def freshSourceStart : Nat :=
  Spartan.sourceToSpartan PiRLCStarts.samplerFreshStart

def selectorSourceStart : Nat :=
  Spartan.sourceToSpartan (PiRLCStarts.samplerLogicalStart + 15449)

def poseidonGrid (program : Program) : SourceGrid :=
  SourceGrid.externalOfSemantic
    (PiRLCSamplerPoseidonPlan.retainedBlock program)
    (PiRLCSamplerPoseidonPlan.retainedStart program)
    poseidonSourceStart 17 15504 8 992 4 78 774 86

def logicalGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (logicalBlock program) (logicalStart program)
    logicalSourceStart 17 15504 8 992 400 0 3200 400

def freshGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (freshBlock program) (freshStart program)
    freshSourceStart 17 43743 1 9696 9696 0 9696 0

def selectorGrid (program : Program) : SourceGrid :=
  SourceGrid.ofSemantic (PiRLCFirst54RetainedBlocks.positionBlock program)
    (PiRLCRetainedGeometry.positionStart program)
    selectorSourceStart 17 15504 1 1 1 3519 3520 0

def substitution (program : Program) : SourceSubstitution where
  ranges := []
  grids := [poseidonGrid program, logicalGrid program, freshGrid program,
    selectorGrid program]

def poseidonDescriptor (source : Fin sourceCount) (round : Fin roundCount) :
    Lane :=
  { source
    round
    lane := ⟨0, by decide⟩ }

def logicalOffset (lane : Fin laneCount)
    (position : Fin logicalCountPerLane) : Fin 400 :=
  ⟨lane.val * 100 + position.val, by
    have laneBound := lane.isLt
    have positionBound := position.isLt
    change lane.val < 4 at laneBound
    change position.val < 100 at positionBound
    omega⟩

def freshOffset (round : Fin roundCount) (lane : Fin laneCount)
    (position : Fin freshCountPerLane) : Fin 9696 :=
  ⟨round.val * 1212 + lane.val * 303 + position.val, by
    have roundBound := round.isLt
    have laneBound := lane.isLt
    have positionBound := position.isLt
    change round.val < 8 at roundBound
    change lane.val < 4 at laneBound
    change position.val < 303 at positionBound
    omega⟩

private theorem samplerLogical_after_piCcs :
    Spartan.piCcsPhaseOffset ≤ PiRLCStarts.samplerLogicalStart := by
  norm_num [Spartan.piCcsPhaseOffset, PiRLCStarts.samplerLogicalStart,
    PiRLCStarts.phaseLogicalStart, PiRLCInputs.phaseOffset,
    NightstreamFPrime.Lifecycle.PiRLC.v1_1.Formal.samplerOffset]

private theorem samplerFresh_after_piCcs :
    Spartan.piCcsPhaseOffset ≤ PiRLCStarts.samplerFreshStart := by
  change Spartan.piCcsPhaseOffset ≤ PiRLCStarts.phaseFreshStart
  rw [PiRLCStarts.phaseFreshStart_eq]
  norm_num [Spartan.piCcsPhaseOffset]

theorem logicalSourceStart_eq :
    logicalSourceStart = poseidonSourceStart + 8 := by
  have affine := Spartan.sourceToSpartan_add_of_piCcsLocal
    (PiRLCStarts.samplerLogicalStart + 584) 8 (by
      exact Nat.le_trans samplerLogical_after_piCcs (by omega))
  simpa [logicalSourceStart, poseidonSourceStart] using affine

theorem selectorSourceStart_eq :
    selectorSourceStart = logicalSourceStart + 14857 := by
  have affine := Spartan.sourceToSpartan_add_of_piCcsLocal
    (PiRLCStarts.samplerLogicalStart + 592) 14857 (by
      exact Nat.le_trans samplerLogical_after_piCcs (by omega))
  simpa [selectorSourceStart, logicalSourceStart] using affine

theorem freshSourceStart_eq :
    freshSourceStart = logicalSourceStart + 311630 := by
  have startEq : PiRLCStarts.samplerFreshStart =
      PiRLCStarts.samplerLogicalStart + 592 + 311630 := by
    unfold PiRLCStarts.samplerFreshStart
    rw [PiRLCStarts.phaseFreshStart_eq]
    rfl
  have affine := Spartan.sourceToSpartan_add_of_piCcsLocal
    (PiRLCStarts.samplerLogicalStart + 592) 311630 (by
      exact Nat.le_trans samplerLogical_after_piCcs (by omega))
  unfold freshSourceStart logicalSourceStart
  rw [startEq]
  exact affine

theorem poseidonSource_formula (source round : Nat) (lane : Fin 4) :
    PiRLCSamplerOrdinaryDirectSource.poseidonSource source round lane =
      PiRLCStarts.samplerLogicalStart + source * 15504 + 584 +
        round * 992 + lane.val := by
  cases round with
  | zero =>
      simp [PiRLCSamplerOrdinaryDirectSource.poseidonSource,
        PiRLCStarts.samplerSourceLogicalStart]
  | succ previous =>
      simp [PiRLCSamplerOrdinaryDirectSource.poseidonSource,
        DigestWindow.permutationOffset, Sampler.windowOffset,
        Sampler.windowBase, SamplerChain.sourceOffset,
        Sampler.entryPrivateCount, Sampler.logicalPrivateCount,
        DigestWindow.logicalPrivateCount, DigestLane.logicalPrivateCount,
        PiRLCStarts.samplerSourceLogicalStart,
        PiRLCStarts.samplerLogicalStart]
      omega

theorem logicalSource_formula (descriptor : Lane)
    (position : Fin logicalCountPerLane) :
    logicalSource descriptor position =
      PiRLCStarts.samplerLogicalStart + descriptor.source.val * 15504 + 592 +
        descriptor.round.val * 992 + descriptor.lane.val * 100 +
          position.val := by
  rfl

theorem freshSource_formula (descriptor : Lane)
    (position : Fin freshCountPerLane) :
    freshSource descriptor position =
      PiRLCStarts.samplerFreshStart + descriptor.source.val * 43743 +
        descriptor.round.val * 1212 + descriptor.lane.val * 303 +
          position.val := by
  rfl

theorem selectorSource_formula (source : Nat) :
    PiRLCSamplerOrdinaryDirectSource.selectorSource source =
      PiRLCStarts.samplerLogicalStart + source * 15504 + 15449 := by
  simp [PiRLCSamplerOrdinaryDirectSource.selectorSource,
    PiRLCStarts.selectorLogicalStart, PiRLCStarts.samplerSourceLogicalStart,
    First54.positionOffset, First54.candidateCount,
    First54.roundPrivateCount, First54.fullSlot, First54Step.fullSlot,
    First54Step.slotCount, First54ValueStep.outputCount]

theorem poseidonTarget (source : Fin sourceCount) (round : Fin roundCount)
    (lane : Fin 4) :
    Spartan.sourceToSpartan
        ((PiRLCSamplerOrdinaryDirectPlan.Location.poseidon
          (poseidonDescriptor source round) lane).sourceColumn) =
      poseidonSourceStart + source.val * 15504 + round.val * 992 + lane.val := by
  rw [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
    poseidonDescriptor]
  rw [poseidonSource_formula]
  let delta := source.val * 15504 + round.val * 992 + lane.val
  have affine := Spartan.sourceToSpartan_add_of_piCcsLocal
    (PiRLCStarts.samplerLogicalStart + 584) delta (by
      exact Nat.le_trans samplerLogical_after_piCcs (by omega))
  calc
    Spartan.sourceToSpartan
        (PiRLCStarts.samplerLogicalStart + source.val * 15504 + 584 +
          round.val * 992 + lane.val) =
        Spartan.sourceToSpartan
          ((PiRLCStarts.samplerLogicalStart + 584) + delta) := by
      apply congrArg Spartan.sourceToSpartan
      dsimp [delta]
      omega
    _ = Spartan.sourceToSpartan
        (PiRLCStarts.samplerLogicalStart + 584) + delta := affine
    _ = poseidonSourceStart + source.val * 15504 +
        round.val * 992 + lane.val := by
      unfold poseidonSourceStart
      dsimp [delta]
      omega

theorem logicalTarget (descriptor : Lane)
    (position : Fin logicalCountPerLane) :
    Spartan.sourceToSpartan
        ((PiRLCSamplerOrdinaryDirectPlan.Location.logical descriptor
          position).sourceColumn) =
      logicalSourceStart + descriptor.source.val * 15504 +
        descriptor.round.val * 992 + (logicalOffset descriptor.lane position).val := by
  rw [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
    logicalSource_formula]
  let delta := descriptor.source.val * 15504 +
    descriptor.round.val * 992 + descriptor.lane.val * 100 + position.val
  have affine := Spartan.sourceToSpartan_add_of_piCcsLocal
    (PiRLCStarts.samplerLogicalStart + 592) delta (by
      exact Nat.le_trans samplerLogical_after_piCcs (by omega))
  calc
    Spartan.sourceToSpartan
        (PiRLCStarts.samplerLogicalStart + descriptor.source.val * 15504 +
          592 + descriptor.round.val * 992 + descriptor.lane.val * 100 +
          position.val) =
        Spartan.sourceToSpartan
          ((PiRLCStarts.samplerLogicalStart + 592) + delta) := by
      apply congrArg Spartan.sourceToSpartan
      dsimp [delta]
      omega
    _ = Spartan.sourceToSpartan
        (PiRLCStarts.samplerLogicalStart + 592) + delta := affine
    _ = logicalSourceStart + descriptor.source.val * 15504 +
        descriptor.round.val * 992 +
          (logicalOffset descriptor.lane position).val := by
      unfold logicalSourceStart logicalOffset
      dsimp [delta]
      omega

theorem freshTarget (descriptor : Lane)
    (position : Fin freshCountPerLane) :
    Spartan.sourceToSpartan
        ((PiRLCSamplerOrdinaryDirectPlan.Location.fresh descriptor
          position).sourceColumn) =
      freshSourceStart + descriptor.source.val * 43743 +
        (freshOffset descriptor.round descriptor.lane position).val := by
  rw [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
    freshSource_formula]
  let delta := descriptor.source.val * 43743 +
    descriptor.round.val * 1212 + descriptor.lane.val * 303 + position.val
  have affine := Spartan.sourceToSpartan_add_of_piCcsLocal
    PiRLCStarts.samplerFreshStart delta samplerFresh_after_piCcs
  calc
    Spartan.sourceToSpartan
        (PiRLCStarts.samplerFreshStart + descriptor.source.val * 43743 +
          descriptor.round.val * 1212 + descriptor.lane.val * 303 +
          position.val) =
        Spartan.sourceToSpartan (PiRLCStarts.samplerFreshStart + delta) := by
      apply congrArg Spartan.sourceToSpartan
      dsimp [delta]
      omega
    _ = Spartan.sourceToSpartan PiRLCStarts.samplerFreshStart + delta := affine
    _ = freshSourceStart + descriptor.source.val * 43743 +
        (freshOffset descriptor.round descriptor.lane position).val := by
      unfold freshSourceStart freshOffset
      dsimp [delta]
      omega

theorem selectorTarget (source : Fin sourceCount) :
    Spartan.sourceToSpartan
        ((PiRLCSamplerOrdinaryDirectPlan.Location.selector source).sourceColumn) =
      selectorSourceStart + source.val * 15504 := by
  rw [PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
    selectorSource_formula]
  have affine := Spartan.sourceToSpartan_add_of_piCcsLocal
    (PiRLCStarts.samplerLogicalStart + 15449) (source.val * 15504) (by
      exact Nat.le_trans samplerLogical_after_piCcs (by omega))
  calc
    Spartan.sourceToSpartan
        (PiRLCStarts.samplerLogicalStart + source.val * 15504 + 15449) =
        Spartan.sourceToSpartan
          ((PiRLCStarts.samplerLogicalStart + 15449) +
            source.val * 15504) := by
      apply congrArg Spartan.sourceToSpartan
      omega
    _ = Spartan.sourceToSpartan
        (PiRLCStarts.samplerLogicalStart + 15449) + source.val * 15504 := affine
    _ = selectorSourceStart + source.val * 15504 := by
      rfl

theorem logicalGrid_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (descriptor : Lane) (position : Fin logicalCountPerLane) :
    (logicalGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.logical descriptor
            position).sourceColumn)) =
      some ((PiRLCSamplerOrdinaryDirectPlan.Location.logical descriptor
        position).form geometry) := by
  rw [logicalTarget]
  have sourceBound := descriptor.source.isLt
  have roundBound := descriptor.round.isLt
  have laneBound := descriptor.lane.isLt
  have positionBound := position.isLt
  change descriptor.source.val < 17 at sourceBound
  change descriptor.round.val < 8 at roundBound
  change descriptor.lane.val < 4 at laneBound
  change position.val < 100 at positionBound
  have direct := SourceGrid.form?_ofSemantic
    (logicalBlock program) (logicalStart program)
    logicalSourceStart 17 15504 8 992 400 0 3200 400
    (logicalFits geometry) (by decide) (by decide)
    descriptor.source descriptor.round
    (logicalOffset descriptor.lane position)
    (by unfold logicalOffset; omega)
    (by unfold logicalOffset; omega)
    (by
      rw [logicalBlock_slotCount]
      unfold logicalOffset
      omega)
  have slotEq :
      (⟨descriptor.source.val * 3200 + descriptor.round.val * 400 +
          (logicalOffset descriptor.lane position).val,
        by
          rw [logicalBlock_slotCount]
          unfold logicalOffset
          omega⟩ : Fin (logicalBlock program).slotCount) =
        logicalSlot descriptor position := by
    apply Fin.ext
    simp [logicalSlot, laneIndex, Fin.encodeProd, logicalOffset,
      sourceCount, roundCount, laneCount, logicalCountPerLane,
      PiRLCSamplerOrdinaryRows.digestRoundCount]
    omega
  simp only [Nat.zero_add, Nat.zero_mul, Nat.mul_zero, Nat.add_zero] at direct
  rw [slotEq] at direct
  simpa [logicalGrid, PiRLCSamplerOrdinaryDirectPlan.Location.form] using direct

theorem freshGrid_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (descriptor : Lane) (position : Fin freshCountPerLane) :
    (freshGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.fresh descriptor
            position).sourceColumn)) =
      some ((PiRLCSamplerOrdinaryDirectPlan.Location.fresh descriptor
        position).form geometry) := by
  rw [freshTarget]
  have sourceBound := descriptor.source.isLt
  have roundBound := descriptor.round.isLt
  have laneBound := descriptor.lane.isLt
  have positionBound := position.isLt
  change descriptor.source.val < 17 at sourceBound
  change descriptor.round.val < 8 at roundBound
  change descriptor.lane.val < 4 at laneBound
  change position.val < 303 at positionBound
  have direct := SourceGrid.form?_ofSemantic
    (freshBlock program) (freshStart program)
    freshSourceStart 17 43743 1 9696 9696 0 9696 0
    (freshFits geometry) (by decide) (by decide)
    descriptor.source ⟨0, by decide⟩
    (freshOffset descriptor.round descriptor.lane position)
    (by unfold freshOffset; omega)
    (by unfold freshOffset; omega)
    (by
      rw [freshBlock_slotCount]
      unfold freshOffset
      omega)
  have slotEq :
      (⟨descriptor.source.val * 9696 +
          (freshOffset descriptor.round descriptor.lane position).val,
        by
          rw [freshBlock_slotCount]
          unfold freshOffset
          omega⟩ : Fin (freshBlock program).slotCount) =
        freshSlot descriptor position := by
    apply Fin.ext
    simp [freshSlot, laneIndex, Fin.encodeProd, freshOffset,
      sourceCount, roundCount, laneCount, freshCountPerLane,
      PiRLCSamplerOrdinaryRows.digestRoundCount]
    omega
  simp only [Nat.zero_add, Nat.zero_mul, Nat.mul_zero, Nat.add_zero] at direct
  rw [slotEq] at direct
  simpa [freshGrid, PiRLCSamplerOrdinaryDirectPlan.Location.form] using direct

theorem selectorGrid_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (source : Fin sourceCount) :
    (selectorGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.selector source).sourceColumn)) =
      some ((PiRLCSamplerOrdinaryDirectPlan.Location.selector source).form
        geometry) := by
  rw [selectorTarget]
  have sourceBound := source.isLt
  change source.val < 17 at sourceBound
  have direct := SourceGrid.form?_ofSemantic
    (PiRLCFirst54RetainedBlocks.positionBlock program)
    (PiRLCRetainedGeometry.positionStart program)
    selectorSourceStart 17 15504 1 1 1 3519 3520 0
    (PiRLCRetainedGeometry.positionFits
      (PiRLCSamplerOrdinaryDirectPlan.piRlcGeometry geometry))
    (by decide) (by decide) source ⟨0, by decide⟩ ⟨0, by decide⟩
    (by omega) (by omega)
    (by
      rw [PiRLCFirst54RetainedBlocks.positionBlock_slotCount]
      omega)
  have slotEq :
      (⟨3519 + source.val * 3520,
        by
          rw [PiRLCFirst54RetainedBlocks.positionBlock_slotCount]
          omega⟩ :
        Fin (PiRLCFirst54RetainedBlocks.positionBlock program).slotCount) =
      PiRLCFirst54DirectSchedule.positionIndex
        (PiRLCFirst54DirectPlan.finalPositionDescriptor source) := by
    apply Fin.ext
    simp [PiRLCFirst54DirectSchedule.positionIndex,
      PiRLCFirst54DirectSchedule.candidateIndex,
      PiRLCFirst54DirectPlan.finalPositionDescriptor, Fin.encodeProd,
      PiRLCFirst54DirectSchedule.sourceCount,
      PiRLCFirst54DirectSchedule.roundCount,
      PiRLCFirst54Invocations.roundCount,
      First54.candidateCount, First54Step.slotCount, First54Step.fullSlot]
    omega
  simp only [Nat.zero_add, Nat.zero_mul, Nat.mul_zero, Nat.add_zero] at direct
  rw [slotEq] at direct
  simpa [selectorGrid, PiRLCSamplerOrdinaryDirectPlan.Location.form] using direct

theorem poseidonGrid_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount) (lane : Fin 4) :
    (poseidonGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.poseidon
            (poseidonDescriptor source round) lane).sourceColumn)) =
      some ((PiRLCSamplerOrdinaryDirectPlan.Location.poseidon
        (poseidonDescriptor source round) lane).form geometry) := by
  rw [poseidonTarget]
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have laneBound := lane.isLt
  change source.val < 17 at sourceBound
  change round.val < 8 at roundBound
  change lane.val < 4 at laneBound
  have direct := SourceGrid.form?_externalOfSemantic
    (PiRLCSamplerPoseidonPlan.retainedBlock program)
    (PiRLCSamplerPoseidonPlan.retainedStart program)
    poseidonSourceStart 17 15504 8 992 4 78 774 86
    (PiRLCSamplerPoseidonPlan.retainedFits
      (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry))
    (by decide) (by decide) source round lane
    (by omega) (by omega) (by omega)
    (by
      intro selected
      have selectedBound := selected.isLt
      rw [PiRLCSamplerPoseidonPlan.retainedBlock_slotCount]
      omega)
  have outputEq :
      SparseLayer.external (fun selected : Fin 8 =>
        (PiRLCSamplerPoseidonPlan.retainedBlock program).form
          (PiRLCSamplerPoseidonPlan.retainedStart program)
          (PiRLCSamplerPoseidonPlan.retainedFits
            (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry))
          ⟨78 + source.val * 774 + round.val * 86 + selected.val,
            by
              have selectedBound := selected.isLt
              rw [PiRLCSamplerPoseidonPlan.retainedBlock_slotCount]
              omega⟩) ⟨lane.val, by omega⟩ =
        (PiRLCSamplerOrdinaryDirectPlan.Location.poseidon
          (poseidonDescriptor source round) lane).form geometry := by
    unfold PiRLCSamplerOrdinaryDirectPlan.Location.form
      PiRLCSamplerPoseidonPlan.interface
      PoseidonRetainedFamily.familyInterface
      PoseidonRetainedFamily.outputState
    apply congrArg (fun state => SparseLayer.external state
      (DigestWindow.rateLane lane))
    funext selected
    apply congrArg ((PiRLCSamplerPoseidonPlan.retainedBlock program).form
      (PiRLCSamplerPoseidonPlan.retainedStart program)
      (PiRLCSamplerPoseidonPlan.retainedFits
        (PiRLCSamplerOrdinaryDirectPlan.poseidonGeometry geometry)))
    apply Fin.ext
    simp [poseidonDescriptor,
      PiRLCSamplerOrdinaryDirectPlan.Location.poseidonInvocation,
      PiRLCSamplerPoseidonPlan.invocation,
      PoseidonRetainedFamily.slot, Fin.encodeProd,
      PoseidonRetainedSlots.finalRow_val, DigestWindow.rateLane,
      PiRLCSamplerPoseidonPlan.invocationsPerSource]
    omega
  rw [outputEq] at direct
  simpa [poseidonGrid] using direct

theorem freshGrid_none_poseidon
    {program : Program} {logicalWidth : Nat}
    (source : Fin sourceCount) (round : Fin roundCount) (lane : Fin 4) :
    (freshGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.poseidon
            (poseidonDescriptor source round) lane).sourceColumn)) = none := by
  rw [poseidonTarget]
  apply SourceGrid.form?_eq_none_of_before
  simp only [freshGrid, SourceGrid.ofSemantic]
  rw [freshSourceStart_eq, logicalSourceStart_eq]
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have laneBound := lane.isLt
  change source.val < 17 at sourceBound
  change round.val < 8 at roundBound
  change lane.val < 4 at laneBound
  omega

theorem freshGrid_none_logical
    {program : Program} {logicalWidth : Nat}
    (descriptor : Lane) (position : Fin logicalCountPerLane) :
    (freshGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.logical descriptor
            position).sourceColumn)) = none := by
  rw [logicalTarget]
  apply SourceGrid.form?_eq_none_of_before
  simp only [freshGrid, SourceGrid.ofSemantic]
  rw [freshSourceStart_eq]
  have sourceBound := descriptor.source.isLt
  have roundBound := descriptor.round.isLt
  have offsetBound := (logicalOffset descriptor.lane position).isLt
  change descriptor.source.val < 17 at sourceBound
  change descriptor.round.val < 8 at roundBound
  change (logicalOffset descriptor.lane position).val < 400 at offsetBound
  omega

theorem freshGrid_none_selector
    {program : Program} {logicalWidth : Nat}
    (source : Fin sourceCount) :
    (freshGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.selector source).sourceColumn)) =
      none := by
  rw [selectorTarget]
  apply SourceGrid.form?_eq_none_of_before
  simp only [freshGrid, SourceGrid.ofSemantic]
  rw [freshSourceStart_eq, selectorSourceStart_eq]
  have sourceBound := source.isLt
  change source.val < 17 at sourceBound
  omega

theorem poseidonGrid_none_fresh
    {program : Program} {logicalWidth : Nat}
    (descriptor : Lane) (position : Fin freshCountPerLane) :
    (poseidonGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.fresh descriptor
            position).sourceColumn)) = none := by
  rw [freshTarget]
  apply SourceGrid.form?_eq_none_of_after
  · norm_num [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic]
  · simp only [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic]
    rw [freshSourceStart_eq, logicalSourceStart_eq]
    omega

theorem logicalGrid_none_fresh
    {program : Program} {logicalWidth : Nat}
    (descriptor : Lane) (position : Fin freshCountPerLane) :
    (logicalGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.fresh descriptor
            position).sourceColumn)) = none := by
  rw [freshTarget]
  apply SourceGrid.form?_eq_none_of_after
  · norm_num [logicalGrid, SourceGrid.ofSemantic]
  · simp only [logicalGrid, SourceGrid.ofSemantic]
    rw [freshSourceStart_eq]
    omega

theorem selectorGrid_none_fresh
    {program : Program} {logicalWidth : Nat}
    (descriptor : Lane) (position : Fin freshCountPerLane) :
    (selectorGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.fresh descriptor
            position).sourceColumn)) = none := by
  rw [freshTarget]
  apply SourceGrid.form?_eq_none_of_after
  · norm_num [selectorGrid, SourceGrid.ofSemantic]
  · simp only [selectorGrid, SourceGrid.ofSemantic]
    rw [freshSourceStart_eq, selectorSourceStart_eq]
    omega

theorem poseidonGrid_none_logical
    {program : Program} {logicalWidth : Nat}
    (descriptor : Lane) (position : Fin logicalCountPerLane) :
    (poseidonGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.logical descriptor
            position).sourceColumn)) = none := by
  rw [logicalTarget]
  have sourceBound := descriptor.source.isLt
  have roundBound := descriptor.round.isLt
  have offsetBound := (logicalOffset descriptor.lane position).isLt
  change descriptor.source.val < 17 at sourceBound
  change descriptor.round.val < 8 at roundBound
  change (logicalOffset descriptor.lane position).val < 400 at offsetBound
  have rejected := SourceGrid.form?_eq_none_at_gap
    (poseidonGrid program) logicalWidth descriptor.source descriptor.round
    (8 + (logicalOffset descriptor.lane position).val)
    (by norm_num [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic])
    (by norm_num [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic])
    (by simp [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic]; omega)
    (by simp [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic]; omega)
    (by simp [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic]; omega)
  have coordinateEq :
      logicalSourceStart + descriptor.source.val * 15504 +
          descriptor.round.val * 992 +
            (logicalOffset descriptor.lane position).val =
        poseidonSourceStart + descriptor.source.val * 15504 +
          descriptor.round.val * 992 +
            (8 + (logicalOffset descriptor.lane position).val) := by
    rw [logicalSourceStart_eq]
    omega
  rw [coordinateEq]
  exact rejected

theorem poseidonGrid_none_selector
    {program : Program} {logicalWidth : Nat}
    (source : Fin sourceCount) :
    (poseidonGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.selector source).sourceColumn)) =
      none := by
  rw [selectorTarget]
  have rejected := SourceGrid.form?_eq_none_at_minorAfter
    (poseidonGrid program) logicalWidth source 14 977
    (by norm_num [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic])
    (by norm_num [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic])
    (by norm_num [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic])
    (by norm_num [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic])
    (by norm_num [poseidonGrid, SourceGrid.externalOfSemantic,
      SourceGrid.ofSemantic])
  have coordinateEq :
      selectorSourceStart + source.val * 15504 =
        poseidonSourceStart + source.val * 15504 + 14 * 992 + 977 := by
    rw [selectorSourceStart_eq, logicalSourceStart_eq]
    omega
  rw [coordinateEq]
  exact rejected

theorem logicalGrid_none_selector
    {program : Program} {logicalWidth : Nat}
    (source : Fin sourceCount) :
    (logicalGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.selector source).sourceColumn)) =
      none := by
  rw [selectorTarget]
  have rejected := SourceGrid.form?_eq_none_at_minorAfter
    (logicalGrid program) logicalWidth source 14 969
    (by norm_num [logicalGrid, SourceGrid.ofSemantic])
    (by norm_num [logicalGrid, SourceGrid.ofSemantic])
    (by norm_num [logicalGrid, SourceGrid.ofSemantic])
    (by norm_num [logicalGrid, SourceGrid.ofSemantic])
    (by norm_num [logicalGrid, SourceGrid.ofSemantic])
  have coordinateEq :
      selectorSourceStart + source.val * 15504 =
        logicalSourceStart + source.val * 15504 + 14 * 992 + 969 := by
    rw [selectorSourceStart_eq]
    omega
  rw [coordinateEq]
  exact rejected

theorem selectorGrid_none_logical
    {program : Program} {logicalWidth : Nat}
    (descriptor : Lane) (position : Fin logicalCountPerLane) :
    (selectorGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.logical descriptor
            position).sourceColumn)) = none := by
  rw [logicalTarget]
  have sourceBound := descriptor.source.isLt
  have roundBound := descriptor.round.isLt
  have offsetBound := (logicalOffset descriptor.lane position).isLt
  change descriptor.source.val < 17 at sourceBound
  change descriptor.round.val < 8 at roundBound
  change (logicalOffset descriptor.lane position).val < 400 at offsetBound
  cases sourceEq : descriptor.source.val with
  | zero =>
      apply SourceGrid.form?_eq_none_of_before
      simp only [selectorGrid, SourceGrid.ofSemantic]
      rw [selectorSourceStart_eq]
      omega
  | succ previous =>
      let major : Fin 17 := ⟨previous, by omega⟩
      let minor := 647 + descriptor.round.val * 992 +
        (logicalOffset descriptor.lane position).val
      have rejected := SourceGrid.form?_eq_none_at_minorAfter
        (selectorGrid program) logicalWidth major minor 0
        (by norm_num [selectorGrid, SourceGrid.ofSemantic])
        (by norm_num [selectorGrid, SourceGrid.ofSemantic])
        (by
          simp [selectorGrid, SourceGrid.ofSemantic, minor]
          omega)
        (by norm_num [selectorGrid, SourceGrid.ofSemantic])
        (by
          simp [selectorGrid, SourceGrid.ofSemantic, minor]
          omega)
      have coordinateEq :
          logicalSourceStart + descriptor.source.val * 15504 +
              descriptor.round.val * 992 +
                (logicalOffset descriptor.lane position).val =
            selectorSourceStart + major.val * 15504 + minor * 1 + 0 := by
        rw [selectorSourceStart_eq, sourceEq]
        simp only [major, minor]
        omega
      rw [← sourceEq, coordinateEq]
      exact rejected

theorem selectorGrid_none_poseidon
    {program : Program} {logicalWidth : Nat}
    (source : Fin sourceCount) (round : Fin roundCount) (lane : Fin 4) :
    (selectorGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.poseidon
            (poseidonDescriptor source round) lane).sourceColumn)) = none := by
  rw [poseidonTarget]
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have laneBound := lane.isLt
  change source.val < 17 at sourceBound
  change round.val < 8 at roundBound
  change lane.val < 4 at laneBound
  cases sourceEq : source.val with
  | zero =>
      apply SourceGrid.form?_eq_none_of_before
      simp only [selectorGrid, SourceGrid.ofSemantic]
      rw [selectorSourceStart_eq, logicalSourceStart_eq]
      omega
  | succ previous =>
      let major : Fin 17 := ⟨previous, by omega⟩
      let minor := 639 + round.val * 992 + lane.val
      have rejected := SourceGrid.form?_eq_none_at_minorAfter
        (selectorGrid program) logicalWidth major minor 0
        (by norm_num [selectorGrid, SourceGrid.ofSemantic])
        (by norm_num [selectorGrid, SourceGrid.ofSemantic])
        (by
          simp [selectorGrid, SourceGrid.ofSemantic, minor]
          omega)
        (by norm_num [selectorGrid, SourceGrid.ofSemantic])
        (by
          simp [selectorGrid, SourceGrid.ofSemantic, minor]
          omega)
      have coordinateEq :
          poseidonSourceStart + source.val * 15504 + round.val * 992 +
              lane.val =
            selectorSourceStart + major.val * 15504 + minor * 1 + 0 := by
        rw [selectorSourceStart_eq, logicalSourceStart_eq, sourceEq]
        simp only [major, minor]
        omega
      rw [← sourceEq, coordinateEq]
      exact rejected

theorem logicalGrid_none_poseidon
    {program : Program} {logicalWidth : Nat}
    (source : Fin sourceCount) (round : Fin roundCount) (lane : Fin 4) :
    (logicalGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.poseidon
            (poseidonDescriptor source round) lane).sourceColumn)) = none := by
  rw [poseidonTarget]
  have sourceBound := source.isLt
  have roundBound := round.isLt
  have laneBound := lane.isLt
  change source.val < 17 at sourceBound
  change round.val < 8 at roundBound
  change lane.val < 4 at laneBound
  cases roundEq : round.val with
  | succ previousRound =>
      let minor : Fin 8 := ⟨previousRound, by omega⟩
      have rejected := SourceGrid.form?_eq_none_at_gap
        (logicalGrid program) logicalWidth source minor (984 + lane.val)
        (by norm_num [logicalGrid, SourceGrid.ofSemantic])
        (by norm_num [logicalGrid, SourceGrid.ofSemantic])
        (by simp [logicalGrid, SourceGrid.ofSemantic, minor]; omega)
        (by simp [logicalGrid, SourceGrid.ofSemantic]; omega)
        (by simp [logicalGrid, SourceGrid.ofSemantic]; omega)
      have coordinateEq :
          poseidonSourceStart + source.val * 15504 + round.val * 992 +
              lane.val =
            logicalSourceStart + source.val * 15504 + minor.val * 992 +
              (984 + lane.val) := by
        rw [logicalSourceStart_eq, roundEq]
        simp only [minor]
        omega
      rw [← roundEq, coordinateEq]
      exact rejected
  | zero =>
      cases sourceEq : source.val with
      | zero =>
          apply SourceGrid.form?_eq_none_of_before
          simp only [logicalGrid, SourceGrid.ofSemantic]
          rw [logicalSourceStart_eq]
          omega
      | succ previousSource =>
          let major : Fin 17 := ⟨previousSource, by omega⟩
          have rejected := SourceGrid.form?_eq_none_at_minorAfter
            (logicalGrid program) logicalWidth major 15 (616 + lane.val)
            (by norm_num [logicalGrid, SourceGrid.ofSemantic])
            (by norm_num [logicalGrid, SourceGrid.ofSemantic])
            (by simp [logicalGrid, SourceGrid.ofSemantic]; omega)
            (by simp [logicalGrid, SourceGrid.ofSemantic]; omega)
            (by norm_num [logicalGrid, SourceGrid.ofSemantic])
          have coordinateEq :
              poseidonSourceStart + source.val * 15504 + round.val * 992 +
                  lane.val =
                logicalSourceStart + major.val * 15504 + 15 * 992 +
                  (616 + lane.val) := by
            rw [logicalSourceStart_eq, sourceEq, roundEq]
            simp only [major]
            omega
          rw [← sourceEq, ← roundEq, coordinateEq]
          exact rejected

theorem substitution_poseidon_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (source : Fin sourceCount) (round : Fin roundCount) (lane : Fin 4) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.poseidon
            (poseidonDescriptor source round) lane).sourceColumn)) =
      some ((PiRLCSamplerOrdinaryDirectPlan.Location.poseidon
        (poseidonDescriptor source round) lane).form geometry) := by
  simp [substitution, SourceSubstitution.form?,
    poseidonGrid_form? geometry source round lane,
    logicalGrid_none_poseidon source round lane,
    freshGrid_none_poseidon source round lane,
    selectorGrid_none_poseidon source round lane]

theorem substitution_logical_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (descriptor : Lane) (position : Fin logicalCountPerLane) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.logical descriptor
            position).sourceColumn)) =
      some ((PiRLCSamplerOrdinaryDirectPlan.Location.logical descriptor
        position).form geometry) := by
  simp [substitution, SourceSubstitution.form?,
    poseidonGrid_none_logical descriptor position,
    logicalGrid_form? geometry descriptor position,
    freshGrid_none_logical descriptor position,
    selectorGrid_none_logical descriptor position]

theorem substitution_fresh_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (descriptor : Lane) (position : Fin freshCountPerLane) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.fresh descriptor
            position).sourceColumn)) =
      some ((PiRLCSamplerOrdinaryDirectPlan.Location.fresh descriptor
        position).form geometry) := by
  simp [substitution, SourceSubstitution.form?,
    poseidonGrid_none_fresh descriptor position,
    logicalGrid_none_fresh descriptor position,
    freshGrid_form? geometry descriptor position,
    selectorGrid_none_fresh descriptor position]

theorem substitution_selector_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (source : Fin sourceCount) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          ((PiRLCSamplerOrdinaryDirectPlan.Location.selector source).sourceColumn)) =
      some ((PiRLCSamplerOrdinaryDirectPlan.Location.selector source).form
        geometry) := by
  simp [substitution, SourceSubstitution.form?,
    poseidonGrid_none_selector source,
    logicalGrid_none_selector source,
    freshGrid_none_selector source,
    selectorGrid_form? geometry source]

theorem substitution_location_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (location : PiRLCSamplerOrdinaryDirectPlan.Location) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan location.sourceColumn) =
      some (location.form geometry) := by
  cases location with
  | poseidon descriptor sourceLane =>
      rcases descriptor with ⟨source, round, unusedLane⟩
      simpa [poseidonDescriptor,
        PiRLCSamplerOrdinaryDirectPlan.Location.sourceColumn,
        PiRLCSamplerOrdinaryDirectPlan.Location.form,
        PiRLCSamplerOrdinaryDirectPlan.Location.poseidonInvocation] using
        (substitution_poseidon_form? geometry source round sourceLane)
  | logical descriptor position =>
      exact substitution_logical_form? geometry descriptor position
  | fresh descriptor position =>
      exact substitution_fresh_form? geometry descriptor position
  | selector source =>
      exact substitution_selector_form? geometry source

theorem substitution_agrees_on_target
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (column : Fin Spartan.spartanColumnCount)
    (support : PiRLCSamplerOrdinaryDirectSource.Target column.val) :
    (substitution program).form? logicalWidth column.val =
      some ((PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry).form column) := by
  rcases support with ⟨source, sourceSupport, mapped⟩
  have sourceBound : source < Spartan.SourceColumnCount := by
    have complete :=
      PiRLCSamplerOrdinaryDirectPlan.classifySource_complete sourceSupport
    cases found : PiRLCSamplerOrdinaryDirectPlan.classifySource source with
    | none =>
        simp [found] at complete
    | some location =>
        have owns :=
          PiRLCSamplerOrdinaryDirectPlan.classifySource_sound found
        have bounded := location.sourceColumn_lt
        rw [owns] at bounded
        exact bounded
  have inverse := Spartan.spartanToSource_sourceToSpartan source sourceBound
  rw [mapped] at inverse
  rcases PiRLCSamplerOrdinaryDirectPlan.classifyTarget_complete
      ⟨source, sourceSupport, mapped⟩ with ⟨decoded, found⟩
  have decodedFound :
      PiRLCSamplerOrdinaryDirectPlan.classifySource source = some decoded := by
    unfold PiRLCSamplerOrdinaryDirectPlan.classifyTarget at found
    rw [inverse] at found
    exact found
  have owns :=
    PiRLCSamplerOrdinaryDirectPlan.classifySource_sound decodedFound
  have target :
      Spartan.sourceToSpartan decoded.sourceColumn = column.val := by
    rw [owns, mapped]
  change (substitution program).form? logicalWidth column.val =
    some (match PiRLCSamplerOrdinaryDirectPlan.classifyTarget column.val with
      | none => .empty
      | some location => location.form geometry)
  rw [found]
  simpa only [target] using substitution_location_form? geometry decoded

variable {relationLogicalWidth : Nat}
  {relationPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth relationLogicalWidth}

private theorem programRow_support (index : Fin 220881) :
    (PiRLCSamplerOrdinaryDirectSource.programRow
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index).VarsSatisfy
        PiRLCSamplerOrdinaryDirectSource.Target := by
  exact PiRLCSamplerOrdinaryDirectSource.sourceRows_varsSatisfy _
    (List.get_mem _
      (PiRLCSamplerOrdinaryDirectSource.sourceListIndex
        (logicalWidth := relationLogicalWidth)
        (publicFits := relationPublicFits) index))

theorem substitution_agrees_on_programRow
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 220881) :
    let row := PiRLCSamplerOrdinaryDirectSource.programRow
      (logicalWidth := relationLogicalWidth)
      (publicFits := relationPublicFits) index
    Ordinary.AgreesOnTerms (substitution program)
        (PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry) row.a.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry) row.b.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (PiRLCSamplerOrdinaryDirectPlan.sourceMap geometry) row.c.terms := by
  dsimp only
  have scope := programRow_support
    (relationLogicalWidth := relationLogicalWidth)
    (relationPublicFits := relationPublicFits) index
  refine ⟨?_, ?_, ?_⟩
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.1 term member)
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.2.1 term member)
  · intro term member bounded
    exact substitution_agrees_on_target geometry ⟨term.1, bounded⟩
      (scope.2.2 term member)

end NightstreamFPrime.Export.Stage1.PiRLCSamplerOrdinaryMatrixSubstitution
