import NightstreamFPrime.Export.MatrixProgram.Ordinary
import NightstreamFPrime.Export.MatrixProgram.Program
import NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection
import NightstreamFPrime.Export.Stage1.PiCCSOrdinaryDirectPlan

/-!
Owns the package-carried sparse substitution for canonical PiCCS ordinary
rows. This module is built range by range and proves every wire range equal to
the existing direct source resolver before the range enters a package.

This module currently owns the verifier-context range. It does not yet claim
the complete PiCCS substitution table or package integration.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSOrdinaryMatrixProgram

open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open PiCCSOrdinaryRetainedBlocks
open PiCCSOrdinaryRetainedGeometry

abbrev Program := Lifecycle.Stage1.Application.Program

def priorInputRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (PiRLCPoseidonGeometry.priorInputBlock program)
    (PiRLCPoseidonGeometry.priorInputStart program)
    (Spartan.sourceToSpartan PilotProduction.priorPreimageStart)
    PilotProduction.stateHashWords 0

theorem priorInputRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PilotProduction.stateHashWords) :
    (priorInputRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PilotProduction.priorPreimageStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.priorInput index).form
        geometry) := by
  have upper : PilotProduction.priorPreimageStart + index.val <
      PilotProduction.priorPublicInputStart := by
    have bound : index.val < PilotProduction.stateHashWords := index.isLt
    norm_num [PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq] at bound ⊢
    exact bound
  rw [Spartan.sourceToSpartan_add_of_pilotPriorPrivate
    PilotProduction.priorPreimageStart index.val upper]
  rw [PiCCSOrdinaryDirectPlan.Location.priorInput_form_eq_pilot]
  simpa only [priorInputRange, Nat.zero_add] using!
    (SourceRange.form?_ofSemantic (PiRLCPoseidonGeometry.priorInputBlock program)
      (PiRLCPoseidonGeometry.priorInputStart program)
      (Spartan.sourceToSpartan PilotProduction.priorPreimageStart)
      PilotProduction.stateHashWords 0
      (PiRLCPoseidonGeometry.priorInputFits (pilotGeometry geometry))
      (by rfl) index)

def freshPublicInputRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (freshPublicInputBlock program)
    (freshPublicInputStart program)
    (Spartan.sourceToSpartan PilotProduction.priorPublicInputStart) 270 0

theorem freshPublicInputRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 270) :
    (freshPublicInputRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PilotProduction.priorPublicInputStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.freshPublicInput index).form
        geometry) := by
  have upper : PilotProduction.priorPublicInputStart + index.val <
      PilotProduction.outputPreimageStart := by
    have bound : index.val < 270 := index.isLt
    norm_num [PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      Lifecycle.PriorStateHash.publicWidth,
      Lifecycle.PaperAlgebra.publicRingColumns, Spec.ringDegree] at bound ⊢
  rw [Spartan.sourceToSpartan_add_of_pilotPriorPublic
    PilotProduction.priorPublicInputStart index.val (Nat.le_refl _) upper]
  simpa [freshPublicInputRange,
    PiCCSOrdinaryDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (freshPublicInputBlock program)
      (freshPublicInputStart program)
      (Spartan.sourceToSpartan PilotProduction.priorPublicInputStart)
      270 0 (freshPublicInputFits geometry) (by rfl) index)

private theorem sourceToSpartan_outputInput_add
    (offset : Fin PilotProduction.stateHashWords) :
    Spartan.sourceToSpartan
        (PilotProduction.outputPreimageStart + offset.val) =
      Spartan.sourceToSpartan PilotProduction.outputPreimageStart +
        offset.val := by
  rw [← PilotSpartan.sourceBoundaries_eq.2.1]
  have bound : offset.val < PilotProduction.stateHashWords := offset.isLt
  have sumPilot : PilotSpartan.outputPreimageStart + offset.val <
      Spartan.pilotSourceColumnCount := by
    rw [PilotSpartan.outputPreimageStart_value]
    norm_num [PilotProduction.stateHashWords_eq,
      Spartan.pilotSourceColumnCount] at bound ⊢
    omega
  have startPilot : PilotSpartan.outputPreimageStart <
      Spartan.pilotSourceColumnCount := by omega
  have sumNotPrior : ¬ PilotSpartan.outputPreimageStart + offset.val <
      PilotSpartan.priorPublicStart := by
    rw [PilotSpartan.outputPreimageStart_value,
      PilotSpartan.priorPublicStart_value]
    omega
  have startNotPrior : ¬ PilotSpartan.outputPreimageStart <
      PilotSpartan.priorPublicStart := by
    rw [PilotSpartan.outputPreimageStart_value,
      PilotSpartan.priorPublicStart_value]
    omega
  have sumNotOutput : ¬ PilotSpartan.outputPreimageStart + offset.val <
      PilotSpartan.outputPreimageStart := by
    omega
  have startNotOutput : ¬ PilotSpartan.outputPreimageStart <
      PilotSpartan.outputPreimageStart := by
    omega
  have sumBeforeDigest : PilotSpartan.outputPreimageStart + offset.val <
      PilotSpartan.outputDigestStart := by
    rw [PilotSpartan.outputPreimageStart_value,
      PilotSpartan.outputDigestStart_value]
    norm_num [PilotProduction.stateHashWords_eq] at bound ⊢
    omega
  have startBeforeDigest : PilotSpartan.outputPreimageStart <
      PilotSpartan.outputDigestStart := by
    rw [PilotSpartan.outputPreimageStart_value,
      PilotSpartan.outputDigestStart_value]
    omega
  have pilotAffine :
      PilotSpartan.sourceToSpartan
          (PilotSpartan.outputPreimageStart + offset.val) =
        PilotSpartan.sourceToSpartan PilotSpartan.outputPreimageStart +
          offset.val := by
    unfold PilotSpartan.sourceToSpartan
    rw [if_neg sumNotPrior, if_neg sumNotOutput, if_pos sumBeforeDigest,
      if_neg startNotPrior, if_neg startNotOutput, if_pos startBeforeDigest]
    omega
  unfold Spartan.sourceToSpartan
  rw [if_pos sumPilot, if_pos startPilot, pilotAffine]
  exact Spartan.liftPilotColumn_add_of_input
    (PilotSpartan.sourceToSpartan PilotSpartan.outputPreimageStart)
    offset.val (by
      have mappedStart :
          PilotSpartan.sourceToSpartan PilotSpartan.outputPreimageStart =
            PilotSpartan.secondPrivateStart := by
        unfold PilotSpartan.sourceToSpartan
        rw [if_neg startNotPrior, if_neg startNotOutput,
          if_pos startBeforeDigest]
        omega
      rw [mappedStart, PilotSpartan.secondPrivateStart_value]
      norm_num [Spartan.pilotInputPrivateColumnCount,
        PilotProduction.stateHashWords_eq] at bound ⊢
      omega)

def outputInputRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (PiRLCPoseidonGeometry.outputInputBlock program)
    (PiRLCPoseidonGeometry.outputInputStart program)
    (Spartan.sourceToSpartan PilotProduction.outputPreimageStart)
    PilotProduction.stateHashWords 0

theorem outputInputRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PilotProduction.stateHashWords) :
    (outputInputRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PilotProduction.outputPreimageStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.outputInput index).form
        geometry) := by
  rw [sourceToSpartan_outputInput_add]
  rw [PiCCSOrdinaryDirectPlan.Location.outputInput_form_eq_pilot]
  simpa only [outputInputRange, Nat.zero_add] using!
    (SourceRange.form?_ofSemantic (PiRLCPoseidonGeometry.outputInputBlock program)
      (PiRLCPoseidonGeometry.outputInputStart program)
      (Spartan.sourceToSpartan PilotProduction.outputPreimageStart)
      PilotProduction.stateHashWords 0
      (PiRLCPoseidonGeometry.outputInputFits (pilotGeometry geometry))
      (by rfl) index)

def freshRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (freshBlock program) (freshStart program)
    (Spartan.sourceToSpartan PiCCSArithmetic.initialClaimFreshStart)
    freshCount 0

theorem freshRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin freshCount) :
    (freshRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSArithmetic.initialClaimFreshStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.fresh index).form geometry) := by
  have phaseBound : Spartan.piCcsPhaseOffset ≤
      PiCCSArithmetic.initialClaimFreshStart := by
    unfold PiCCSArithmetic.initialClaimFreshStart
      PiCCSStarts.initialClaimFreshStart PiCCSStarts.roundTranscriptFreshStart
      PiCCSStarts.challengeFreshStart PiCCSStarts.statementAbsorptionFreshStart
      PiCCSStarts.statementBindingFreshStart PiCCSStarts.logicalFreshBase
    rw [PiCCSInputs.phaseOffset_eq]
    norm_num [Spartan.piCcsPhaseOffset]
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiCCSArithmetic.initialClaimFreshStart index.val phaseBound]
  simpa [freshRange, PiCCSOrdinaryDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (freshBlock program) (freshStart program)
      (Spartan.sourceToSpartan PiCCSArithmetic.initialClaimFreshStart)
      freshCount 0 (freshFits geometry) (by rfl) index)

def proofInputRangeCount : Nat := proofInputCount

def proofInputIndex (index : Fin proofInputRangeCount) :
    Fin proofLogicalCount :=
  proofInputSlot index

def proofInputRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (proofLogicalBlock program) (proofLogicalStart program)
    (Spartan.sourceToSpartan PiCCSInputs.priorChildrenStart)
    proofInputRangeCount 0

theorem proofInputRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin proofInputRangeCount) :
    (proofInputRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSInputs.priorChildrenStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (proofInputIndex index)).form geometry) := by
  have lower : Spartan.proofInputSourceStart ≤ PiCCSInputs.priorChildrenStart := by
    norm_num [Spartan.proofInputSourceStart, PiCCSInputs.priorChildrenStart_eq]
  have upper : PiCCSInputs.priorChildrenStart + index.val <
      Spartan.piCcsPhaseOffset := by
    have bound := index.isLt
    norm_num [PiCCSInputs.priorChildrenStart_eq,
      proofInputRangeCount, Spartan.proofInputColumnCount,
      Spartan.piCcsPhaseOffset] at bound ⊢
    omega
  rw [Spartan.sourceToSpartan_add_of_proofInput
    PiCCSInputs.priorChildrenStart index.val lower upper]
  rw [proofInputIndex, PiCCSOrdinaryDirectPlan.Location.form_proofInput]
  simpa only [proofInputRange, proofInputSlot, Nat.zero_add] using!
    (SourceRange.form?_ofSemantic (proofLogicalBlock program)
      (proofLogicalStart program)
      (Spartan.sourceToSpartan PiCCSInputs.priorChildrenStart)
      proofInputRangeCount 0 (proofLogicalFits geometry)
      (by
        change proofInputRangeCount ≤ proofLogicalCount
        norm_num [proofInputRangeCount, proofInputCount_eq,
          proofLogicalCount_eq]) index)

def transcriptOutputSourceStart : Nat := PiCCSInputs.phaseOffset + 1080

def transcriptOutputGrid (program : Program) : SourceGrid :=
  PiCCSTranscriptOutputForms.transcriptGrid program

theorem transcriptOutputGrid_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin transcriptOutputCount) :
    (transcriptOutputGrid program).form? logicalWidth
        (Spartan.sourceToSpartan (transcriptOutputSource index)) =
      some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (transcriptOutputSlot index)).form geometry) := by
  rw [PiCCSOrdinaryDirectPlan.Location.form_transcriptOutput]
  let decoded : Fin transcriptInvocationCount × Fin Spec.Poseidon2.width :=
    Fin.decodeProd index
  have sourceEq : transcriptOutputSource index =
      PiCCSTranscriptOutputForms.transcriptSource decoded.1 decoded.2 := by
    unfold transcriptOutputSource PiCCSTranscriptOutputForms.transcriptSource
      PiCCSTranscriptOutputForms.transcriptSourceStart
    change PiCCSInputs.phaseOffset + decoded.1.val * 1096 + 1080 + decoded.2.val = _
    omega
  rw [sourceEq]
  exact PiCCSTranscriptOutputForms.transcriptGrid_form? (poseidonGeometry geometry)
    decoded.1 decoded.2

def ordinaryLogicalRangeCount : Nat := ordinaryLogicalCount

def ordinaryLogicalIndex (index : Fin ordinaryLogicalRangeCount) :
    Fin proofLogicalCount :=
  ordinaryLogicalSlot index

def ordinaryLogicalRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (proofLogicalBlock program) (proofLogicalStart program)
    (Spartan.sourceToSpartan PiCCSStarts.initialClaimLogicalStart)
    ordinaryLogicalRangeCount (proofInputCount + transcriptOutputCount)

theorem ordinaryLogicalRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin ordinaryLogicalRangeCount) :
    (ordinaryLogicalRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSStarts.initialClaimLogicalStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (ordinaryLogicalIndex index)).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiCCSStarts.initialClaimLogicalStart index.val (by
      unfold PiCCSStarts.initialClaimLogicalStart
      rw [PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [Spartan.piCcsPhaseOffset])]
  rw [ordinaryLogicalIndex, PiCCSOrdinaryDirectPlan.Location.form_ordinaryLogical]
  simpa only [ordinaryLogicalRange, ordinaryLogicalSlot] using!
    (SourceRange.form?_ofSemantic (proofLogicalBlock program)
      (proofLogicalStart program)
      (Spartan.sourceToSpartan PiCCSStarts.initialClaimLogicalStart)
      ordinaryLogicalRangeCount (proofInputCount + transcriptOutputCount)
      (proofLogicalFits geometry) (by
        change proofInputCount + transcriptOutputCount +
          ordinaryLogicalRangeCount ≤ proofLogicalCount
        norm_num [ordinaryLogicalRangeCount, proofLogicalCount,
          ordinaryLogicalCount]) index)

/-- The four verifier-context source columns remain one contiguous public
range after the Spartan permutation. -/
def expectedContextRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (expectedContextBlock program)
    (expectedContextStart program) Spartan.expectedContextPublicStart
    PiCCSInputs.expectedContextWords 0

/-- The encoded verifier-context range reconstructs the exact direct
PiCCS retained form. -/
theorem expectedContextRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (lane : Fin 4) :
    (expectedContextRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSInputs.expectedContextStart + lane.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.expectedContext lane).form
        geometry) := by
  rw [Spartan.sourceToSpartan_expectedContext]
  simpa [expectedContextRange, PiCCSOrdinaryDirectPlan.Location.form,
    PiCCSInputs.expectedContextWords] using
    (SourceRange.form?_ofSemantic (expectedContextBlock program)
      (expectedContextStart program) Spartan.expectedContextPublicStart
      PiCCSInputs.expectedContextWords 0 (expectedContextFits geometry)
      (by simp [expectedContextBlock, sourceFieldBlock,
        PiCCSInputs.expectedContextWords]) lane)

private theorem rangeValues (program : Program) :
    (priorInputRange program).sourceStart = 0 ∧
    (priorInputRange program).sourceCount = 27819 ∧
    (outputInputRange program).sourceStart = 27819 ∧
    (outputInputRange program).sourceCount = 27819 ∧
    (proofInputRange program).sourceStart = 55638 ∧
    (proofInputRange program).sourceCount = 15462 ∧
    (ordinaryLogicalRange program).sourceStart = 5357516 ∧
    (ordinaryLogicalRange program).sourceCount = 29591 ∧
    (freshRange program).sourceStart = 6225547 ∧
    (freshRange program).sourceCount = 2956 ∧
    (freshPublicInputRange program).sourceStart = 11464597 ∧
    (freshPublicInputRange program).sourceCount = 270 ∧
    (expectedContextRange program).sourceStart = 11464871 ∧
    (expectedContextRange program).sourceCount = 4 := by
  unfold priorInputRange outputInputRange proofInputRange ordinaryLogicalRange
    freshRange freshPublicInputRange expectedContextRange SourceRange.ofSemantic
    PiCCSArithmetic.initialClaimFreshStart PiCCSStarts.initialClaimFreshStart
    PiCCSStarts.roundTranscriptFreshStart PiCCSStarts.challengeFreshStart
    PiCCSStarts.statementAbsorptionFreshStart PiCCSStarts.statementBindingFreshStart
    PiCCSStarts.logicalFreshBase PiCCSStarts.initialClaimLogicalStart
  rw [PiCCSInputs.phaseOffset_eq, PiCCSStarts.roundTranscriptWitnessStart_eq]
  norm_num [proofInputRangeCount,
    proofInputCount_eq,
    ordinaryLogicalRangeCount,
    ordinaryLogicalCount_eq,
    PilotProduction.stateHashWords_eq,
    PilotProduction.priorPreimageStart,
    PilotProduction.priorPublicInputStart,
    PilotProduction.outputPreimageStart,
    Lifecycle.PriorStateHash.publicWidth,
    Lifecycle.PaperAlgebra.publicRingColumns,
    Spec.ringDegree,
    freshCount,
    PiCCSInputs.expectedContextWords,
    Spartan.sourceToSpartan,
    Spartan.liftPilotColumn,
    PilotSpartan.sourceToSpartan,
    Spartan.pilotSourceColumnCount,
    Spartan.proofInputSourceStart,
    Spartan.piCcsPhaseOffset,
    Spartan.piCcsLocalStart,
    Spartan.expectedContextPublicStart,
    Spartan.pilotInputPrivateColumnCount,
    Spartan.pilotPrivateColumnCount,
    Spartan.proofInputColumnCount,
    Spartan.privateColumnCount_eq,
    PilotSpartan.priorPublicStart_value,
    PilotSpartan.outputPreimageStart_value,
    PilotSpartan.outputDigestStart_value,
    PilotSpartan.witnessStart_value,
    PilotSpartan.secondPrivateStart_value,
    PilotSpartan.witnessPrivateStart_value,
    PilotSpartan.firstPublicStart_value,
    PilotSpartan.secondPublicStart_value,
    PiCCSInputs.priorChildrenStart_eq]

private theorem transcriptGridValues (program : Program) :
    (transcriptOutputGrid program).sourceStart = 5158028 ∧
      (transcriptOutputGrid program).majorCount = 183 ∧
      (transcriptOutputGrid program).majorSourceStride = 1096 ∧
      (transcriptOutputGrid program).minorCount = 1 ∧
      (transcriptOutputGrid program).minorSourceStride = 16 := by
  unfold transcriptOutputGrid PiCCSTranscriptOutputForms.transcriptGrid
    SourceGrid.externalOfSemantic SourceGrid.ofSemantic
    PiCCSTranscriptOutputForms.transcriptSourceStart
  rw [PiCCSInputs.phaseOffset_eq]
  norm_num [PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq,
    Spartan.sourceToSpartan, Spartan.pilotSourceColumnCount,
    Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset,
    Spartan.piCcsLocalStart]

private theorem priorTarget_eq (program : Program)
    (index : Fin PilotProduction.stateHashWords) :
    Spartan.sourceToSpartan
        (PilotProduction.priorPreimageStart + index.val) = index.val := by
  have upper : PilotProduction.priorPreimageStart + index.val <
      PilotProduction.priorPublicInputStart := by
    have bound : index.val < PilotProduction.stateHashWords := index.isLt
    norm_num [PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq] at bound ⊢
    exact bound
  rw [Spartan.sourceToSpartan_add_of_pilotPriorPrivate
    PilotProduction.priorPreimageStart index.val upper]
  change (priorInputRange program).sourceStart + index.val = index.val
  have values := rangeValues program
  omega

private theorem outputTarget_eq (program : Program)
    (index : Fin PilotProduction.stateHashWords) :
    Spartan.sourceToSpartan
        (PilotProduction.outputPreimageStart + index.val) =
      27819 + index.val := by
  rw [sourceToSpartan_outputInput_add]
  change (outputInputRange program).sourceStart + index.val =
    27819 + index.val
  have values := rangeValues program
  omega

private theorem proofInputTarget_eq (program : Program)
    (index : Fin proofInputRangeCount) :
    Spartan.sourceToSpartan (PiCCSInputs.priorChildrenStart + index.val) =
      55638 + index.val := by
  have lower : Spartan.proofInputSourceStart ≤ PiCCSInputs.priorChildrenStart := by
    norm_num [Spartan.proofInputSourceStart, PiCCSInputs.priorChildrenStart_eq]
  have upper : PiCCSInputs.priorChildrenStart + index.val <
      Spartan.piCcsPhaseOffset := by
    have bound := index.isLt
    norm_num [PiCCSInputs.priorChildrenStart_eq,
      proofInputRangeCount, Spartan.proofInputColumnCount,
      Spartan.piCcsPhaseOffset] at bound ⊢
    omega
  rw [Spartan.sourceToSpartan_add_of_proofInput
    PiCCSInputs.priorChildrenStart index.val lower upper]
  change (proofInputRange program).sourceStart + index.val =
    55638 + index.val
  have values := rangeValues program
  omega

private theorem ordinaryLogicalTarget_eq (program : Program)
    (index : Fin ordinaryLogicalRangeCount) :
    Spartan.sourceToSpartan
        (PiCCSStarts.initialClaimLogicalStart + index.val) =
      5357516 + index.val := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiCCSStarts.initialClaimLogicalStart index.val (by
      unfold PiCCSStarts.initialClaimLogicalStart
      rw [PiCCSStarts.roundTranscriptWitnessStart_eq]
      norm_num [Spartan.piCcsPhaseOffset])]
  change (ordinaryLogicalRange program).sourceStart + index.val =
    5357516 + index.val
  have values := rangeValues program
  omega

private theorem transcriptOutputTarget_eq (index : Fin transcriptOutputCount) :
    let decoded : Fin transcriptInvocationCount × Fin Spec.Poseidon2.width :=
      Fin.decodeProd index
    Spartan.sourceToSpartan (transcriptOutputSource index) =
      5158028 + decoded.1.val * 1096 + decoded.2.val := by
  let decoded : Fin transcriptInvocationCount × Fin Spec.Poseidon2.width :=
    Fin.decodeProd index
  dsimp only
  unfold transcriptOutputSource
  change Spartan.sourceToSpartan
      (PiCCSInputs.phaseOffset + decoded.1.val * 1096 + 1080 + decoded.2.val) = _
  calc
    _ = Spartan.sourceToSpartan
        (transcriptOutputSourceStart +
          (decoded.1.val * 1096 + decoded.2.val)) := by
        apply congrArg Spartan.sourceToSpartan
        unfold transcriptOutputSourceStart
        omega
    _ = Spartan.sourceToSpartan transcriptOutputSourceStart +
        decoded.1.val * 1096 + decoded.2.val := by
      have mapped := Spartan.sourceToSpartan_add_of_piCcsLocal
        transcriptOutputSourceStart
        (decoded.1.val * 1096 + decoded.2.val) (by
          unfold transcriptOutputSourceStart
          rw [PiCCSInputs.phaseOffset_eq]
          norm_num [Spartan.piCcsPhaseOffset])
      simpa only [Nat.add_assoc] using mapped
    _ = _ := by
      have startEq : Spartan.sourceToSpartan transcriptOutputSourceStart =
          5158028 := by
        unfold transcriptOutputSourceStart
        rw [PiCCSInputs.phaseOffset_eq]
        norm_num [Spartan.sourceToSpartan,
          Spartan.pilotSourceColumnCount, Spartan.proofInputSourceStart,
          Spartan.piCcsPhaseOffset, Spartan.piCcsLocalStart,
          PiCCSInputs.phaseOffset_eq]
      rw [startEq]

private theorem freshTarget_eq (program : Program) (index : Fin freshCount) :
    Spartan.sourceToSpartan
        (PiCCSArithmetic.initialClaimFreshStart + index.val) =
      6225547 + index.val := by
  have phaseBound : Spartan.piCcsPhaseOffset ≤
      PiCCSArithmetic.initialClaimFreshStart := by
    unfold PiCCSArithmetic.initialClaimFreshStart
      PiCCSStarts.initialClaimFreshStart PiCCSStarts.roundTranscriptFreshStart
      PiCCSStarts.challengeFreshStart PiCCSStarts.statementAbsorptionFreshStart
      PiCCSStarts.statementBindingFreshStart PiCCSStarts.logicalFreshBase
    rw [PiCCSInputs.phaseOffset_eq]
    norm_num [Spartan.piCcsPhaseOffset]
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiCCSArithmetic.initialClaimFreshStart index.val phaseBound]
  change (freshRange program).sourceStart + index.val =
    6225547 + index.val
  have values := rangeValues program
  omega

private theorem freshPublicTarget_eq (program : Program) (index : Fin 270) :
    Spartan.sourceToSpartan
        (PilotProduction.priorPublicInputStart + index.val) =
      11464597 + index.val := by
  have upper : PilotProduction.priorPublicInputStart + index.val <
      PilotProduction.outputPreimageStart := by
    have bound : index.val < 270 := index.isLt
    norm_num [PilotProduction.outputPreimageStart,
      PilotProduction.priorPublicInputStart,
      Lifecycle.PriorStateHash.publicWidth,
      Lifecycle.PaperAlgebra.publicRingColumns, Spec.ringDegree] at bound ⊢
  rw [Spartan.sourceToSpartan_add_of_pilotPriorPublic
    PilotProduction.priorPublicInputStart index.val (Nat.le_refl _) upper]
  change (freshPublicInputRange program).sourceStart + index.val =
    11464597 + index.val
  have values := rangeValues program
  omega

private theorem expectedContextTarget_eq (lane : Fin 4) :
    Spartan.sourceToSpartan (PiCCSInputs.expectedContextStart + lane.val) =
      11464871 + lane.val := by
  rw [Spartan.sourceToSpartan_expectedContext]
  rfl

/-- Complete PiCCS ordinary source substitution in increasing post-Spartan
column order. -/
def substitution (program : Program) : SourceSubstitution where
  ranges := [priorInputRange program, outputInputRange program,
    proofInputRange program, ordinaryLogicalRange program, freshRange program,
    freshPublicInputRange program, expectedContextRange program]
  grids := [transcriptOutputGrid program]

/-- Exact physical packet intervals whose expanded R1CS rows form the
PiCCS ordinary source program. -/
def rowSchedule : IndexSchedule := .rangeList [
    ⟨PiCCSArithmetic.statementBindingRowStart, 4712⟩,
    ⟨PiCCSArithmetic.initialClaimRowStart, 12957⟩,
    ⟨PiCCSArithmetic.sumcheckRowStart, 728⟩,
    ⟨PiCCSArithmetic.evalKRowStart, 3364⟩,
    ⟨PiCCSArithmetic.evalARowStart, 11140⟩,
    ⟨PiCCSArithmetic.ccsRowStart, 23⟩,
    ⟨PiCCSArithmetic.normRowStart, 800⟩,
    ⟨PiCCSArithmetic.finalIdentityRowStart, 3593⟩]

/-- Proof-oriented reference expansion of the eight row intervals. The wire
interpreter uses `IndexSchedule.index?` and does not build this list. -/
def rowIndexReference : List Nat :=
  List.range' PiCCSArithmetic.statementBindingRowStart 4712 ++
    List.range' PiCCSArithmetic.initialClaimRowStart 12957 ++
    List.range' PiCCSArithmetic.sumcheckRowStart 728 ++
    List.range' PiCCSArithmetic.evalKRowStart 3364 ++
    List.range' PiCCSArithmetic.evalARowStart 11140 ++
    List.range' PiCCSArithmetic.ccsRowStart 23 ++
    List.range' PiCCSArithmetic.normRowStart 800 ++
    List.range' PiCCSArithmetic.finalIdentityRowStart 3593

theorem rowSchedule_indices : rowSchedule.indices = rowIndexReference := by
  simp [rowSchedule, IndexSchedule.indices, IndexRange.indices_eq_range',
    rowIndexReference]

theorem rowSchedule_index? (ordinal : Nat) :
    rowSchedule.index? ordinal = rowIndexReference[ordinal]? := by
  rw [IndexSchedule.index?_eq_getElem?, rowSchedule_indices]

@[simp] theorem rowSchedule_count : rowSchedule.count = 37317 := by
  rfl

theorem rowSchedule_valid :
    rowSchedule.valid PerApplicationPackage.basePackage.layout.rowCount =
      true := by
  rw [show PerApplicationPackage.basePackage.layout.rowCount = 11377912 by
    exact Package.circuitPackage_layout_values.1]
  decide

theorem rowIndexReference_nodup : rowIndexReference.Nodup := by
  rw [← rowSchedule_indices]
  exact IndexSchedule.rangeList_indices_nodup _ _ rowSchedule_valid

theorem rowIndexReference_bounds :
    ∀ index ∈ rowIndexReference,
      PiCCSArithmetic.statementBindingRowStart ≤ index ∧
        index < PiRLCStarts.phaseRowStart := by
  rw [← rowSchedule_indices]
  unfold rowSchedule IndexSchedule.indices
  exact validIndexRanges_indices_bounds _ _ _ (by decide)

private theorem packetRowIndices (rowStart freshStart count : Nat)
    (constraints : List Circuit.Expr)
    (lengthEq : (PiCCSArithmetic.compilePacket rowStart freshStart
      constraints).length = count) :
    (PiCCSArithmetic.compilePacket rowStart freshStart constraints).map
        Rows.CompiledRow.rowIndex =
      List.range' rowStart count := by
  rw [PiCCSArithmetic.compilePacket_rowIndices, lengthEq]

/-- The wire row schedule is exactly the physical row-index stream emitted by
the canonical PiCCS arithmetic builder. -/
theorem arithmeticRows_rowIndices
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits) :
    (PiCCSArithmetic.arithmeticRows logicalWidth publicFits).map
        Rows.CompiledRow.rowIndex = rowIndexReference := by
  have statement := packetRowIndices PiCCSArithmetic.statementBindingRowStart
    PiCCSArithmetic.statementBindingFreshStart 4712
    (PiCCSArithmetic.statementBindingConstraints logicalWidth publicFits)
    (PiCCSArithmetic.statementBindingRows_length logicalWidth publicFits relation)
  have initial := packetRowIndices PiCCSArithmetic.initialClaimRowStart
    PiCCSArithmetic.initialClaimFreshStart 12957
    (PiCCSArithmetic.initialClaimConstraints logicalWidth publicFits)
    (PiCCSArithmetic.initialClaimRows_length logicalWidth publicFits relation)
  have sumcheck := packetRowIndices PiCCSArithmetic.sumcheckRowStart
    PiCCSArithmetic.sumcheckFreshStart 728
    (PiCCSArithmetic.sumcheckConstraints logicalWidth publicFits)
    (PiCCSArithmetic.sumcheckRows_length logicalWidth publicFits relation)
  have evalK := packetRowIndices PiCCSArithmetic.evalKRowStart
    PiCCSArithmetic.evalKFreshStart 3364
    (PiCCSArithmetic.evalKConstraints logicalWidth publicFits)
    (PiCCSArithmetic.evalKRows_length logicalWidth publicFits relation)
  have evalA := packetRowIndices PiCCSArithmetic.evalARowStart
    PiCCSArithmetic.evalAFreshStart 11140
    (PiCCSArithmetic.evalAConstraints logicalWidth publicFits)
    (PiCCSArithmetic.evalARows_length logicalWidth publicFits relation)
  have ccs := packetRowIndices PiCCSArithmetic.ccsRowStart
    PiCCSArithmetic.ccsFreshStart 23
    (PiCCSArithmetic.ccsConstraints logicalWidth publicFits)
    (PiCCSArithmetic.ccsRows_length logicalWidth publicFits relation)
  have norm := packetRowIndices PiCCSArithmetic.normRowStart
    PiCCSArithmetic.normFreshStart 800
    (PiCCSArithmetic.normConstraints logicalWidth publicFits)
    (PiCCSArithmetic.normRows_length logicalWidth publicFits relation)
  have finalIdentity := packetRowIndices
    PiCCSArithmetic.finalIdentityRowStart PiCCSArithmetic.finalIdentityFreshStart
    3593 (PiCCSArithmetic.finalIdentityConstraints logicalWidth publicFits)
    (PiCCSArithmetic.finalIdentityRows_length logicalWidth publicFits relation)
  unfold PiCCSArithmetic.arithmeticRows rowIndexReference
  simp only [List.map_append]
  unfold PiCCSArithmetic.statementBindingRows
    PiCCSArithmetic.initialClaimRows PiCCSArithmetic.sumcheckRows
    PiCCSArithmetic.evalKRows PiCCSArithmetic.evalARows
    PiCCSArithmetic.ccsRows PiCCSArithmetic.normRows
    PiCCSArithmetic.finalIdentityRows
  rw [statement, initial, sumcheck, evalK, evalA, ccs, norm, finalIdentity]

/-- Streaming schedule selection returns exactly the physical index carried
by the canonical compiled row at the same dense ordinal. -/
theorem rowSchedule_index?_eq_arithmeticRowIndex?
    {logicalWidth : Nat}
    {publicFits : Spec.ringDegree * Lifecycle.PaperAlgebra.publicRingColumns ≤
      Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation logicalWidth
      publicFits)
    (ordinal : Nat) :
    rowSchedule.index? ordinal =
      ((PiCCSArithmetic.arithmeticRows logicalWidth publicFits)[ordinal]?).map
        Rows.CompiledRow.rowIndex := by
  rw [rowSchedule_index?, ← arithmeticRows_rowIndices relation]
  simp

/-- Complete package operands for the PiCCS ordinary matrix block. -/
def block {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : Ordinary.Block where
  rows := rowSchedule
  oneColumn := (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val
  substitution := substitution program
  projection := PerApplicationSourceProjection.base program

@[simp] theorem block_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (block geometry).rowCount = 37317 := by
  rfl

def matrixProgram {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program where
  blocks := [.ordinary (block geometry)]

@[simp] theorem matrixProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 37317 := by
  rfl

theorem substitution_priorInput_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PilotProduction.stateHashWords) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PilotProduction.priorPreimageStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.priorInput index).form
        geometry) := by
  have target := priorTarget_eq program index
  have selected := priorInputRange_form? geometry index
  rw [target] at selected
  have values := rangeValues program
  have indexBound : index.val < 27819 := by
    simpa only [PilotProduction.stateHashWords_eq] using index.isLt
  have outputNone := SourceRange.form?_eq_none_of_before
    (outputInputRange program) logicalWidth index.val (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_before
    (proofInputRange program) logicalWidth index.val (by omega)
  have ordinaryNone := SourceRange.form?_eq_none_of_before
    (ordinaryLogicalRange program) logicalWidth index.val (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (freshRange program) logicalWidth index.val (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (freshPublicInputRange program) logicalWidth index.val (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth index.val (by omega)
  have gridNone := SourceGrid.form?_eq_none_of_before
    (transcriptOutputGrid program) logicalWidth index.val (by
      rw [(transcriptGridValues program).1]
      omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, selected, outputNone,
    proofInputNone, ordinaryNone, freshNone, publicNone, contextNone, gridNone]

theorem substitution_outputInput_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin PilotProduction.stateHashWords) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PilotProduction.outputPreimageStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.outputInput index).form
        geometry) := by
  have target := outputTarget_eq program index
  have selected := outputInputRange_form? geometry index
  rw [target] at selected
  have values := rangeValues program
  have indexBound : index.val < 27819 := by
    simpa only [PilotProduction.stateHashWords_eq] using index.isLt
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorInputRange program) logicalWidth (27819 + index.val) (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_before
    (proofInputRange program) logicalWidth (27819 + index.val) (by omega)
  have ordinaryNone := SourceRange.form?_eq_none_of_before
    (ordinaryLogicalRange program) logicalWidth (27819 + index.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (freshRange program) logicalWidth (27819 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (freshPublicInputRange program) logicalWidth (27819 + index.val) (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth (27819 + index.val) (by omega)
  have gridNone := SourceGrid.form?_eq_none_of_before
    (transcriptOutputGrid program) logicalWidth (27819 + index.val) (by
      rw [(transcriptGridValues program).1]
      omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, selected,
    proofInputNone, ordinaryNone, freshNone, publicNone, contextNone, gridNone]

theorem substitution_proofInput_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin proofInputRangeCount) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSInputs.priorChildrenStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (proofInputIndex index)).form geometry) := by
  have target := proofInputTarget_eq program index
  have selected := proofInputRange_form? geometry index
  rw [target] at selected
  have values := rangeValues program
  have indexBound : index.val < 15462 := by
    simpa only [proofInputRangeCount, proofInputCount_eq]
      using index.isLt
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorInputRange program) logicalWidth (55638 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputInputRange program) logicalWidth (55638 + index.val) (by omega)
  have ordinaryNone := SourceRange.form?_eq_none_of_before
    (ordinaryLogicalRange program) logicalWidth (55638 + index.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (freshRange program) logicalWidth (55638 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (freshPublicInputRange program) logicalWidth (55638 + index.val) (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth (55638 + index.val) (by omega)
  have gridNone := SourceGrid.form?_eq_none_of_before
    (transcriptOutputGrid program) logicalWidth (55638 + index.val) (by
      rw [(transcriptGridValues program).1]
      omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, outputNone,
    selected, ordinaryNone, freshNone, publicNone, contextNone, gridNone]

theorem substitution_transcriptOutput_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin transcriptOutputCount) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan (transcriptOutputSource index)) =
      some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (transcriptOutputSlot index)).form geometry) := by
  let decoded : Fin transcriptInvocationCount × Fin Spec.Poseidon2.width :=
    Fin.decodeProd index
  have target : Spartan.sourceToSpartan (transcriptOutputSource index) =
      5158028 + decoded.1.val * 1096 + decoded.2.val := by
    simpa only [decoded] using transcriptOutputTarget_eq index
  have selected := transcriptOutputGrid_form? geometry index
  rw [target] at selected
  have values := rangeValues program
  have invocationBound := decoded.1.isLt
  change decoded.1.val < 183 at invocationBound
  have laneBound := decoded.2.isLt
  change decoded.2.val < 16 at laneBound
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorInputRange program) logicalWidth
      (5158028 + decoded.1.val * 1096 + decoded.2.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputInputRange program) logicalWidth
      (5158028 + decoded.1.val * 1096 + decoded.2.val) (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_after
    (proofInputRange program) logicalWidth
      (5158028 + decoded.1.val * 1096 + decoded.2.val) (by omega)
  have ordinaryNone := SourceRange.form?_eq_none_of_before
    (ordinaryLogicalRange program) logicalWidth
      (5158028 + decoded.1.val * 1096 + decoded.2.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (freshRange program) logicalWidth
      (5158028 + decoded.1.val * 1096 + decoded.2.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (freshPublicInputRange program) logicalWidth
      (5158028 + decoded.1.val * 1096 + decoded.2.val) (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth
      (5158028 + decoded.1.val * 1096 + decoded.2.val) (by omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, outputNone,
    proofInputNone, ordinaryNone, freshNone, publicNone, contextNone, selected]

private theorem transcriptOutputGrid_form?_none_at_ordinary
    {program : Program} {logicalWidth : Nat}
    (index : Fin ordinaryLogicalRangeCount) :
    (transcriptOutputGrid program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSStarts.initialClaimLogicalStart + index.val)) = none := by
  rw [ordinaryLogicalTarget_eq program index]
  change (transcriptOutputGrid program).form? logicalWidth
    (5357516 + index.val) = none
  rcases transcriptGridValues program with
    ⟨gridStart, gridCount, gridStride, minorCount, minorStride⟩
  have indexBound := index.isLt
  change index.val < 29591 at indexBound
  by_cases after : 1080 ≤ index.val
  · apply SourceGrid.form?_eq_none_of_after
      (transcriptOutputGrid program) logicalWidth (5357516 + index.val)
    · rw [gridStride]
      omega
    · rw [gridStart, gridCount, gridStride]
      omega
  · have small : index.val < 1080 := Nat.lt_of_not_ge after
    have majorBound : 182 < (transcriptOutputGrid program).majorCount := by
      rw [gridCount]
      omega
    let major : Fin (transcriptOutputGrid program).majorCount :=
      ⟨182, majorBound⟩
    have modBound : index.val % 16 < 16 := Nat.mod_lt _ (by omega)
    have divmod := Nat.mod_add_div index.val 16
    have rejected := SourceGrid.form?_eq_none_at_minorAfter
      (transcriptOutputGrid program) logicalWidth major
      (1 + index.val / 16) (index.val % 16)
      (by rw [gridStride]; omega)
      (by rw [minorStride]; omega)
      (by rw [gridStride, minorStride]; omega)
      (by rw [minorStride]; exact modBound)
      (by rw [minorCount]; omega)
    have sourceEq :
        (transcriptOutputGrid program).sourceStart +
            major.val * (transcriptOutputGrid program).majorSourceStride +
            (1 + index.val / 16) *
              (transcriptOutputGrid program).minorSourceStride +
            index.val % 16 =
          5357516 + index.val := by
      rw [gridStart, gridStride, minorStride]
      change 5158028 + 182 * 1096 +
          (1 + index.val / 16) * 16 + index.val % 16 =
        5357516 + index.val
      omega
    rw [← sourceEq]
    exact rejected

theorem substitution_ordinaryLogical_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin ordinaryLogicalRangeCount) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSStarts.initialClaimLogicalStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (ordinaryLogicalIndex index)).form geometry) := by
  have target := ordinaryLogicalTarget_eq program index
  have selected := ordinaryLogicalRange_form? geometry index
  rw [target] at selected
  have values := rangeValues program
  have indexBound : index.val < 29591 := by
    simpa only [ordinaryLogicalRangeCount, ordinaryLogicalCount_eq]
      using index.isLt
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorInputRange program) logicalWidth (5357516 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputInputRange program) logicalWidth (5357516 + index.val) (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_after
    (proofInputRange program) logicalWidth (5357516 + index.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (freshRange program) logicalWidth (5357516 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (freshPublicInputRange program) logicalWidth (5357516 + index.val) (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth (5357516 + index.val) (by omega)
  have gridNone := transcriptOutputGrid_form?_none_at_ordinary
    (program := program) (logicalWidth := logicalWidth) index
  rw [target] at gridNone
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, outputNone,
    proofInputNone, selected, freshNone, publicNone, contextNone, gridNone]

theorem substitution_fresh_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin freshCount) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSArithmetic.initialClaimFreshStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.fresh index).form geometry) := by
  have target := freshTarget_eq program index
  have selected := freshRange_form? geometry index
  rw [target] at selected
  have values := rangeValues program
  have indexBound : index.val < 2956 := by
    simpa only [freshCount] using index.isLt
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorInputRange program) logicalWidth (6225547 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputInputRange program) logicalWidth (6225547 + index.val) (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_after
    (proofInputRange program) logicalWidth (6225547 + index.val) (by omega)
  have ordinaryNone := SourceRange.form?_eq_none_of_after
    (ordinaryLogicalRange program) logicalWidth (6225547 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (freshPublicInputRange program) logicalWidth (6225547 + index.val) (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth (6225547 + index.val) (by omega)
  have gridNone := SourceGrid.form?_eq_none_of_after
    (transcriptOutputGrid program) logicalWidth (6225547 + index.val)
    (by rw [(transcriptGridValues program).2.2.1]; omega)
    (by
      rcases transcriptGridValues program with
        ⟨gridStart, gridCount, gridStride, _, _⟩
      rw [gridStart, gridCount, gridStride]
      omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, outputNone,
    proofInputNone, ordinaryNone, selected, publicNone, contextNone, gridNone]

theorem substitution_freshPublicInput_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (index : Fin 270) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PilotProduction.priorPublicInputStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.freshPublicInput index).form
        geometry) := by
  have target := freshPublicTarget_eq program index
  have selected := freshPublicInputRange_form? geometry index
  rw [target] at selected
  have values := rangeValues program
  have indexBound : index.val < 270 := index.isLt
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorInputRange program) logicalWidth (11464597 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputInputRange program) logicalWidth (11464597 + index.val) (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_after
    (proofInputRange program) logicalWidth (11464597 + index.val) (by omega)
  have ordinaryNone := SourceRange.form?_eq_none_of_after
    (ordinaryLogicalRange program) logicalWidth (11464597 + index.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_after
    (freshRange program) logicalWidth (11464597 + index.val) (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth (11464597 + index.val) (by omega)
  have gridNone := SourceGrid.form?_eq_none_of_after
    (transcriptOutputGrid program) logicalWidth (11464597 + index.val)
    (by rw [(transcriptGridValues program).2.2.1]; omega)
    (by
      rcases transcriptGridValues program with
        ⟨gridStart, gridCount, gridStride, _, _⟩
      rw [gridStart, gridCount, gridStride]
      omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, outputNone,
    proofInputNone, ordinaryNone, freshNone, selected, contextNone, gridNone]

theorem substitution_expectedContext_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) (lane : Fin 4) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSInputs.expectedContextStart + lane.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.expectedContext lane).form
        geometry) := by
  have target := expectedContextTarget_eq lane
  have selected := expectedContextRange_form? geometry lane
  rw [target] at selected
  have values := rangeValues program
  have laneBound : lane.val < 4 := lane.isLt
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorInputRange program) logicalWidth (11464871 + lane.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputInputRange program) logicalWidth (11464871 + lane.val) (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_after
    (proofInputRange program) logicalWidth (11464871 + lane.val) (by omega)
  have ordinaryNone := SourceRange.form?_eq_none_of_after
    (ordinaryLogicalRange program) logicalWidth (11464871 + lane.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_after
    (freshRange program) logicalWidth (11464871 + lane.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_after
    (freshPublicInputRange program) logicalWidth (11464871 + lane.val) (by omega)
  have gridNone := SourceGrid.form?_eq_none_of_after
    (transcriptOutputGrid program) logicalWidth (11464871 + lane.val)
    (by rw [(transcriptGridValues program).2.2.1]; omega)
    (by
      rcases transcriptGridValues program with
        ⟨gridStart, gridCount, gridStride, _, _⟩
      rw [gridStart, gridCount, gridStride]
      omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, outputNone,
    proofInputNone, ordinaryNone, freshNone, publicNone, selected, gridNone]

/-- The complete sparse substitution reconstructs every semantic PiCCS
location at its exact post-Spartan source column. -/
theorem substitution_location_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (location : PiCCSOrdinaryDirectPlan.Location) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan location.sourceColumn) =
      some (location.form geometry) := by
  cases location with
  | priorInput index =>
      exact substitution_priorInput_form? geometry index
  | freshPublicInput index =>
      exact substitution_freshPublicInput_form? geometry index
  | outputInput index =>
      exact substitution_outputInput_form? geometry index
  | expectedContext index =>
      exact substitution_expectedContext_form? geometry index
  | fresh index =>
      exact substitution_fresh_form? geometry index
  | proofLogical index =>
      by_cases before : index.val < proofInputCount
      · let inputIndex : Fin proofInputRangeCount := ⟨index.val, before⟩
        have indexEq : proofInputIndex inputIndex = index := by
          apply Fin.ext
          rfl
        have sourceEq : proofLogicalSource index =
            PiCCSInputs.priorChildrenStart + inputIndex.val := by
          rw [← indexEq]
          change proofLogicalSource (proofInputSlot inputIndex) = _
          rw [proofLogicalSource_proofInput]
        simpa [PiCCSOrdinaryDirectPlan.Location.sourceColumn, sourceEq,
          indexEq] using
          (substitution_proofInput_form? geometry inputIndex)
      · have proofLower : proofInputCount ≤ index.val :=
          Nat.le_of_not_gt before
        by_cases inTranscript :
            index.val < proofInputCount + transcriptOutputCount
        · have transcriptBound : index.val - proofInputCount <
              transcriptOutputCount := by omega
          let transcriptIndex : Fin transcriptOutputCount :=
            ⟨index.val - proofInputCount, transcriptBound⟩
          have indexEq : transcriptOutputSlot transcriptIndex = index := by
            apply Fin.ext
            change proofInputCount + (index.val - proofInputCount) = index.val
            omega
          have sourceEq : proofLogicalSource index =
              transcriptOutputSource transcriptIndex := by
            rw [← indexEq, proofLogicalSource_transcriptOutput]
          simpa [PiCCSOrdinaryDirectPlan.Location.sourceColumn, sourceEq,
            indexEq] using
            (substitution_transcriptOutput_form? geometry transcriptIndex)
        · have ordinaryLower :
              proofInputCount + transcriptOutputCount ≤ index.val :=
            Nat.le_of_not_gt inTranscript
          have ordinaryBound :
              index.val - (proofInputCount + transcriptOutputCount) <
                ordinaryLogicalRangeCount := by
            have upper := index.isLt
            norm_num [proofLogicalCount_eq, proofInputCount_eq,
              transcriptOutputCount_eq, ordinaryLogicalRangeCount,
              ordinaryLogicalCount_eq] at upper ⊢
            omega
          let ordinaryIndex : Fin ordinaryLogicalRangeCount :=
            ⟨index.val - (proofInputCount + transcriptOutputCount),
              ordinaryBound⟩
          have indexEq : ordinaryLogicalIndex ordinaryIndex = index := by
            apply Fin.ext
            change proofInputCount + transcriptOutputCount +
                (index.val - (proofInputCount + transcriptOutputCount)) =
              index.val
            omega
          have sourceEq : proofLogicalSource index =
              PiCCSStarts.initialClaimLogicalStart + ordinaryIndex.val := by
            rw [← indexEq]
            exact proofLogicalSource_ordinaryLogical ordinaryIndex
          simpa [PiCCSOrdinaryDirectPlan.Location.sourceColumn, sourceEq,
            indexEq] using
            (substitution_ordinaryLogical_form? geometry ordinaryIndex)

/-- On every source column used by a canonical PiCCS row, the package
substitution is exactly the proof-oriented direct source map. -/
theorem substitution_agrees_on_target
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (column : Fin Spartan.spartanColumnCount)
    (support : PiCCSOrdinarySourceSupport.Target column.val) :
    (substitution program).form? logicalWidth column.val =
      some ((PiCCSOrdinaryDirectPlan.sourceMap geometry).form column) := by
  rcases PiCCSOrdinaryDirectPlan.classifyTarget_complete support with
    ⟨decoded, found, mapped⟩
  change (substitution program).form? logicalWidth column.val =
    some (match PiCCSOrdinaryDirectPlan.classifyTarget column.val with
      | none => PiCCSOrdinaryDirectPlan.endpointForm geometry column
      | some value => value.location.form geometry)
  rw [found]
  have target :
      Spartan.sourceToSpartan decoded.location.sourceColumn = column.val := by
    rw [decoded.owns, mapped]
  simpa only [target] using
    (substitution_location_form? geometry decoded.location)

private theorem programRow_support
    {relationLogicalWidth : Nat}
    {relationPublicFits : Spec.ringDegree *
      Lifecycle.PaperAlgebra.publicRingColumns ≤
        Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
          relationLogicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (index : Fin 37317) :
    (PiCCSOrdinaryDirectSource.programRow relation index).VarsSatisfy
      PiCCSOrdinarySourceSupport.Target := by
  exact PiCCSOrdinaryDirectSupport.sourceRows_varsSatisfy relation _
    (List.get_mem _
      (PiCCSOrdinaryDirectSource.sourceListIndex relation index))

/-- The package substitution agrees with the direct compiler on every term
of every canonical indexed PiCCS ordinary row. -/
theorem substitution_agrees_on_programRow
    {program : Program} {logicalWidth relationLogicalWidth : Nat}
    {relationPublicFits : Spec.ringDegree *
      Lifecycle.PaperAlgebra.publicRingColumns ≤
        Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
          relationLogicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (index : Fin 37317) :
    let row := PiCCSOrdinaryDirectSource.programRow relation index
    Ordinary.AgreesOnTerms (substitution program)
        (PiCCSOrdinaryDirectPlan.sourceMap geometry) row.a.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (PiCCSOrdinaryDirectPlan.sourceMap geometry) row.b.terms ∧
      Ordinary.AgreesOnTerms (substitution program)
        (PiCCSOrdinaryDirectPlan.sourceMap geometry) row.c.terms := by
  dsimp only
  have scope := programRow_support relation index
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

/-- Compiling one indexed canonical PiCCS row through the package operands
returns exactly the proof-oriented direct 4-matrix row. -/
theorem compile_programRow?
    {program : Program} {logicalWidth relationLogicalWidth : Nat}
    {relationPublicFits : Spec.ringDegree *
      Lifecycle.PaperAlgebra.publicRingColumns ≤
        Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
          relationLogicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (index : Fin 37317)
    (bounded : Layout.ProductionRelation.SourceCompiler.RowBounded
      Spartan.spartanColumnCount
      (PiCCSOrdinaryDirectSource.programRow relation index)) :
    Ordinary.compileRow? (substitution program) logicalWidth
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry).val
        (PiCCSOrdinaryDirectSource.programRow relation index) =
      some (Layout.ProductionRelation.SourceCompiler.compileRow
        (PiCCSOrdinaryDirectPlan.sourceMap geometry)
        (PiCCSOrdinaryRetainedGeometry.oneColumn geometry)
        (PiCCSOrdinaryDirectSource.programRow relation index)
        bounded) := by
  rcases substitution_agrees_on_programRow relation geometry index with
    ⟨agreesA, agreesB, agreesC⟩
  exact Ordinary.compileRow?_eq_compileRow (substitution program)
    (PiCCSOrdinaryDirectPlan.sourceMap geometry)
    (PiCCSOrdinaryRetainedGeometry.oneColumn geometry)
    (PiCCSOrdinaryDirectSource.programRow relation index)
    bounded agreesA agreesB agreesC

end NightstreamFPrime.Export.Stage1.PiCCSOrdinaryMatrixProgram
