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

open NightstreamFPrime.Export.MatrixProgram
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open PiCCSOrdinaryRetainedBlocks
open PiCCSOrdinaryRetainedGeometry

abbrev Program := Lifecycle.Stage1.Application.Program

def priorInputRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (priorInputBlock program) (priorInputStart program)
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
  simpa [priorInputRange, PiCCSOrdinaryDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (priorInputBlock program)
      (priorInputStart program)
      (Spartan.sourceToSpartan PilotProduction.priorPreimageStart)
      PilotProduction.stateHashWords 0 (priorInputFits geometry)
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
  SourceRange.ofSemantic (outputInputBlock program) (outputInputStart program)
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
  simpa [outputInputRange, PiCCSOrdinaryDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (outputInputBlock program)
      (outputInputStart program)
      (Spartan.sourceToSpartan PilotProduction.outputPreimageStart)
      PilotProduction.stateHashWords 0 (outputInputFits geometry)
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
    norm_num [Spartan.piCcsPhaseOffset,
      PiCCSArithmetic.initialClaimFreshStart,
      PiCCSStarts.initialClaimFreshStart,
      PiCCSStarts.roundTranscriptFreshStart,
      PiCCSStarts.challengeFreshStart,
      PiCCSStarts.statementAbsorptionFreshStart,
      PiCCSStarts.statementBindingFreshStart,
      PiCCSStarts.logicalFreshBase, PiCCSInputs.phaseOffset_eq]
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiCCSArithmetic.initialClaimFreshStart index.val phaseBound]
  simpa [freshRange, PiCCSOrdinaryDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (freshBlock program) (freshStart program)
      (Spartan.sourceToSpartan PiCCSArithmetic.initialClaimFreshStart)
      freshCount 0 (freshFits geometry) (by rfl) index)

def proofInputRangeCount : Nat := Spartan.proofInputColumnCount

def proofLocalRangeCount : Nat := proofLogicalCount - proofInputRangeCount

def proofInputIndex (index : Fin proofInputRangeCount) :
    Fin proofLogicalCount :=
  ⟨index.val, by
    have bound := index.isLt
    norm_num [proofInputRangeCount, Spartan.proofInputColumnCount,
      proofLogicalCount_eq] at bound ⊢
    omega⟩

def proofLocalIndex (index : Fin proofLocalRangeCount) :
    Fin proofLogicalCount :=
  ⟨proofInputRangeCount + index.val, by
    have bound := index.isLt
    norm_num [proofLocalRangeCount, proofInputRangeCount,
      Spartan.proofInputColumnCount, proofLogicalCount_eq] at bound ⊢
    omega⟩

def proofInputRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (proofLogicalBlock program) (proofLogicalStart program)
    (Spartan.sourceToSpartan PiCCSInputs.proofInputStart)
    proofInputRangeCount 0

theorem proofInputRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin proofInputRangeCount) :
    (proofInputRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSInputs.proofInputStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (proofInputIndex index)).form geometry) := by
  have lower : Spartan.proofInputSourceStart ≤ PiCCSInputs.proofInputStart := by
    norm_num [Spartan.proofInputSourceStart, PiCCSInputs.proofInputStart_eq]
  have upper : PiCCSInputs.proofInputStart + index.val <
      Spartan.piCcsPhaseOffset := by
    have bound := index.isLt
    norm_num [PiCCSInputs.proofInputStart_eq,
      proofInputRangeCount, Spartan.proofInputColumnCount,
      Spartan.piCcsPhaseOffset] at bound ⊢
    omega
  rw [Spartan.sourceToSpartan_add_of_proofInput
    PiCCSInputs.proofInputStart index.val lower upper]
  simpa [proofInputRange, proofInputIndex,
    PiCCSOrdinaryDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (proofLogicalBlock program)
      (proofLogicalStart program)
      (Spartan.sourceToSpartan PiCCSInputs.proofInputStart)
      proofInputRangeCount 0 (proofLogicalFits geometry)
      (by
        change proofInputRangeCount ≤ proofLogicalCount
        norm_num [proofInputRangeCount, Spartan.proofInputColumnCount,
          proofLogicalCount_eq]) index)

def proofLocalRange (program : Program) : SourceRange :=
  SourceRange.ofSemantic (proofLogicalBlock program) (proofLogicalStart program)
    (Spartan.sourceToSpartan Spartan.piCcsPhaseOffset)
    proofLocalRangeCount proofInputRangeCount

theorem proofLocalRange_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin proofLocalRangeCount) :
    (proofLocalRange program).form? logicalWidth
        (Spartan.sourceToSpartan
          (Spartan.piCcsPhaseOffset + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (proofLocalIndex index)).form geometry) := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    Spartan.piCcsPhaseOffset index.val (Nat.le_refl _)]
  simpa [proofLocalRange, proofLocalIndex,
    PiCCSOrdinaryDirectPlan.Location.form] using
    (SourceRange.form?_ofSemantic (proofLogicalBlock program)
      (proofLogicalStart program)
      (Spartan.sourceToSpartan Spartan.piCcsPhaseOffset)
      proofLocalRangeCount proofInputRangeCount (proofLogicalFits geometry)
      (by
        change proofInputRangeCount + proofLocalRangeCount ≤ proofLogicalCount
        norm_num [proofLocalRangeCount, proofInputRangeCount,
          Spartan.proofInputColumnCount, proofLogicalCount_eq]) index)

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
    (priorInputRange program).sourceCount = 45937 ∧
    (outputInputRange program).sourceStart = 45937 ∧
    (outputInputRange program).sourceCount = 45937 ∧
    (proofInputRange program).sourceStart = 91874 ∧
    (proofInputRange program).sourceCount = 29072 ∧
    (proofLocalRange program).sourceStart = 13721422 ∧
    (proofLocalRange program).sourceCount = 472934 ∧
    (freshRange program).sourceStart = 18270868 ∧
    (freshRange program).sourceCount = 731605 ∧
    (freshPublicInputRange program).sourceStart = 27695711 ∧
    (freshPublicInputRange program).sourceCount = 270 ∧
    (expectedContextRange program).sourceStart = 27695985 ∧
    (expectedContextRange program).sourceCount = 4 := by
  norm_num [priorInputRange, outputInputRange, proofInputRange,
    proofLocalRange, freshRange, freshPublicInputRange,
    expectedContextRange, SourceRange.ofSemantic, proofInputRangeCount,
    proofLocalRangeCount, proofLogicalCount_eq,
    PilotProduction.stateHashWords_eq,
    PilotProduction.priorPreimageStart,
    PilotProduction.priorPublicInputStart,
    PilotProduction.outputPreimageStart,
    Lifecycle.PriorStateHash.publicWidth,
    Lifecycle.PaperAlgebra.publicRingColumns, Spec.ringDegree,
    freshCount, PiCCSInputs.expectedContextWords,
    Spartan.sourceToSpartan, Spartan.liftPilotColumn,
    PilotSpartan.sourceToSpartan, Spartan.pilotSourceColumnCount,
    Spartan.proofInputSourceStart, Spartan.piCcsPhaseOffset,
    Spartan.piCcsLocalStart, Spartan.expectedContextPublicStart,
    Spartan.pilotInputPrivateColumnCount, Spartan.pilotPrivateColumnCount,
    Spartan.proofInputColumnCount, Spartan.privateColumnCount,
    PilotSpartan.priorPublicStart_value,
    PilotSpartan.outputPreimageStart_value,
    PilotSpartan.outputDigestStart_value,
    PilotSpartan.witnessStart_value,
    PilotSpartan.secondPrivateStart_value,
    PilotSpartan.witnessPrivateStart_value,
    PilotSpartan.firstPublicStart_value,
    PilotSpartan.secondPublicStart_value,
    PiCCSInputs.proofInputStart_eq,
    PiCCSArithmetic.initialClaimFreshStart,
    PiCCSStarts.initialClaimFreshStart,
    PiCCSStarts.roundTranscriptFreshStart,
    PiCCSStarts.challengeFreshStart,
    PiCCSStarts.statementAbsorptionFreshStart,
    PiCCSStarts.statementBindingFreshStart,
    PiCCSStarts.logicalFreshBase, PiCCSInputs.phaseOffset_eq]

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
      45937 + index.val := by
  rw [sourceToSpartan_outputInput_add]
  change (outputInputRange program).sourceStart + index.val =
    45937 + index.val
  have values := rangeValues program
  omega

private theorem proofInputTarget_eq (program : Program)
    (index : Fin proofInputRangeCount) :
    Spartan.sourceToSpartan (PiCCSInputs.proofInputStart + index.val) =
      91874 + index.val := by
  have lower : Spartan.proofInputSourceStart ≤ PiCCSInputs.proofInputStart := by
    norm_num [Spartan.proofInputSourceStart, PiCCSInputs.proofInputStart_eq]
  have upper : PiCCSInputs.proofInputStart + index.val <
      Spartan.piCcsPhaseOffset := by
    have bound := index.isLt
    norm_num [PiCCSInputs.proofInputStart_eq,
      proofInputRangeCount, Spartan.proofInputColumnCount,
      Spartan.piCcsPhaseOffset] at bound ⊢
    omega
  rw [Spartan.sourceToSpartan_add_of_proofInput
    PiCCSInputs.proofInputStart index.val lower upper]
  change (proofInputRange program).sourceStart + index.val =
    91874 + index.val
  have values := rangeValues program
  omega

private theorem proofLocalTarget_eq (program : Program)
    (index : Fin proofLocalRangeCount) :
    Spartan.sourceToSpartan (Spartan.piCcsPhaseOffset + index.val) =
      13721422 + index.val := by
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    Spartan.piCcsPhaseOffset index.val (Nat.le_refl _)]
  change (proofLocalRange program).sourceStart + index.val =
    13721422 + index.val
  have values := rangeValues program
  omega

private theorem freshTarget_eq (program : Program) (index : Fin freshCount) :
    Spartan.sourceToSpartan
        (PiCCSArithmetic.initialClaimFreshStart + index.val) =
      18270868 + index.val := by
  have phaseBound : Spartan.piCcsPhaseOffset ≤
      PiCCSArithmetic.initialClaimFreshStart := by
    norm_num [Spartan.piCcsPhaseOffset,
      PiCCSArithmetic.initialClaimFreshStart,
      PiCCSStarts.initialClaimFreshStart,
      PiCCSStarts.roundTranscriptFreshStart,
      PiCCSStarts.challengeFreshStart,
      PiCCSStarts.statementAbsorptionFreshStart,
      PiCCSStarts.statementBindingFreshStart,
      PiCCSStarts.logicalFreshBase, PiCCSInputs.phaseOffset_eq]
  rw [Spartan.sourceToSpartan_add_of_piCcsLocal
    PiCCSArithmetic.initialClaimFreshStart index.val phaseBound]
  change (freshRange program).sourceStart + index.val =
    18270868 + index.val
  have values := rangeValues program
  omega

private theorem freshPublicTarget_eq (program : Program) (index : Fin 270) :
    Spartan.sourceToSpartan
        (PilotProduction.priorPublicInputStart + index.val) =
      27695711 + index.val := by
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
    27695711 + index.val
  have values := rangeValues program
  omega

private theorem expectedContextTarget_eq (lane : Fin 4) :
    Spartan.sourceToSpartan (PiCCSInputs.expectedContextStart + lane.val) =
      27695985 + lane.val := by
  rw [Spartan.sourceToSpartan_expectedContext]
  rfl

/-- Complete PiCCS ordinary source substitution in increasing post-Spartan
column order. -/
def substitution (program : Program) : SourceSubstitution where
  ranges := [priorInputRange program, outputInputRange program,
    proofInputRange program, proofLocalRange program, freshRange program,
    freshPublicInputRange program, expectedContextRange program]

/-- Exact physical packet intervals whose expanded R1CS rows form the
PiCCS ordinary source program. -/
def rowSchedule : IndexSchedule := .rangeList [
    ⟨PiCCSArithmetic.statementBindingRowStart, 160⟩,
    ⟨PiCCSArithmetic.initialClaimRowStart, 116631⟩,
    ⟨PiCCSArithmetic.sumcheckRowStart, 424657⟩,
    ⟨PiCCSArithmetic.evalKRowStart, 8542⟩,
    ⟨PiCCSArithmetic.evalARowStart, 109630⟩,
    ⟨PiCCSArithmetic.ccsRowStart, 20794⟩,
    ⟨PiCCSArithmetic.normRowStart, 752⟩,
    ⟨PiCCSArithmetic.finalIdentityRowStart, 130503⟩]

/-- Proof-oriented reference expansion of the eight row intervals. The wire
interpreter uses `IndexSchedule.index?` and does not build this list. -/
def rowIndexReference : List Nat :=
  List.range' PiCCSArithmetic.statementBindingRowStart 160 ++
    List.range' PiCCSArithmetic.initialClaimRowStart 116631 ++
    List.range' PiCCSArithmetic.sumcheckRowStart 424657 ++
    List.range' PiCCSArithmetic.evalKRowStart 8542 ++
    List.range' PiCCSArithmetic.evalARowStart 109630 ++
    List.range' PiCCSArithmetic.ccsRowStart 20794 ++
    List.range' PiCCSArithmetic.normRowStart 752 ++
    List.range' PiCCSArithmetic.finalIdentityRowStart 130503

theorem rowSchedule_indices : rowSchedule.indices = rowIndexReference := by
  simp [rowSchedule, IndexSchedule.indices, IndexRange.indices_eq_range',
    rowIndexReference]

theorem rowSchedule_index? (ordinal : Nat) :
    rowSchedule.index? ordinal = rowIndexReference[ordinal]? := by
  rw [IndexSchedule.index?_eq_getElem?, rowSchedule_indices]

@[simp] theorem rowSchedule_count : rowSchedule.count = 811669 := by
  rfl

theorem rowSchedule_valid :
    rowSchedule.valid PerApplicationPackage.basePackage.layout.rowCount =
      true := by
  rw [show PerApplicationPackage.basePackage.layout.rowCount = 27584200 by
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
    PiCCSArithmetic.statementBindingFreshStart 160
    (PiCCSArithmetic.statementBindingConstraints logicalWidth publicFits)
    (PiCCSArithmetic.statementBindingRows_length logicalWidth publicFits relation)
  have initial := packetRowIndices PiCCSArithmetic.initialClaimRowStart
    PiCCSArithmetic.initialClaimFreshStart 116631
    (PiCCSArithmetic.initialClaimConstraints logicalWidth publicFits)
    (PiCCSArithmetic.initialClaimRows_length logicalWidth publicFits relation)
  have sumcheck := packetRowIndices PiCCSArithmetic.sumcheckRowStart
    PiCCSArithmetic.sumcheckFreshStart 424657
    (PiCCSArithmetic.sumcheckConstraints logicalWidth publicFits)
    (PiCCSArithmetic.sumcheckRows_length logicalWidth publicFits relation)
  have evalK := packetRowIndices PiCCSArithmetic.evalKRowStart
    PiCCSArithmetic.evalKFreshStart 8542
    (PiCCSArithmetic.evalKConstraints logicalWidth publicFits)
    (PiCCSArithmetic.evalKRows_length logicalWidth publicFits relation)
  have evalA := packetRowIndices PiCCSArithmetic.evalARowStart
    PiCCSArithmetic.evalAFreshStart 109630
    (PiCCSArithmetic.evalAConstraints logicalWidth publicFits)
    (PiCCSArithmetic.evalARows_length logicalWidth publicFits relation)
  have ccs := packetRowIndices PiCCSArithmetic.ccsRowStart
    PiCCSArithmetic.ccsFreshStart 20794
    (PiCCSArithmetic.ccsConstraints logicalWidth publicFits)
    (PiCCSArithmetic.ccsRows_length logicalWidth publicFits relation)
  have norm := packetRowIndices PiCCSArithmetic.normRowStart
    PiCCSArithmetic.normFreshStart 752
    (PiCCSArithmetic.normConstraints logicalWidth publicFits)
    (PiCCSArithmetic.normRows_length logicalWidth publicFits relation)
  have finalIdentity := packetRowIndices
    PiCCSArithmetic.finalIdentityRowStart PiCCSArithmetic.finalIdentityFreshStart
    130503 (PiCCSArithmetic.finalIdentityConstraints logicalWidth publicFits)
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
    (block geometry).rowCount = 811669 := by
  rfl

def matrixProgram {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) : MatrixProgram.Program where
  blocks := [.ordinary (block geometry)]

@[simp] theorem matrixProgram_rowCount
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth) :
    (matrixProgram geometry).rowCount = 811669 := by
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
  have indexBound : index.val < 45937 := by
    simpa only [PilotProduction.stateHashWords_eq] using index.isLt
  have outputNone := SourceRange.form?_eq_none_of_before
    (outputInputRange program) logicalWidth index.val (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_before
    (proofInputRange program) logicalWidth index.val (by omega)
  have proofLocalNone := SourceRange.form?_eq_none_of_before
    (proofLocalRange program) logicalWidth index.val (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (freshRange program) logicalWidth index.val (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (freshPublicInputRange program) logicalWidth index.val (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth index.val (by omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, selected, outputNone,
    proofInputNone, proofLocalNone, freshNone, publicNone, contextNone]

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
  have indexBound : index.val < 45937 := by
    simpa only [PilotProduction.stateHashWords_eq] using index.isLt
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorInputRange program) logicalWidth (45937 + index.val) (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_before
    (proofInputRange program) logicalWidth (45937 + index.val) (by omega)
  have proofLocalNone := SourceRange.form?_eq_none_of_before
    (proofLocalRange program) logicalWidth (45937 + index.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (freshRange program) logicalWidth (45937 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (freshPublicInputRange program) logicalWidth (45937 + index.val) (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth (45937 + index.val) (by omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, selected,
    proofInputNone, proofLocalNone, freshNone, publicNone, contextNone]

theorem substitution_proofInput_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin proofInputRangeCount) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          (PiCCSInputs.proofInputStart + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (proofInputIndex index)).form geometry) := by
  have target := proofInputTarget_eq program index
  have selected := proofInputRange_form? geometry index
  rw [target] at selected
  have values := rangeValues program
  have indexBound : index.val < 29072 := by
    simpa only [proofInputRangeCount, Spartan.proofInputColumnCount]
      using index.isLt
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorInputRange program) logicalWidth (91874 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputInputRange program) logicalWidth (91874 + index.val) (by omega)
  have proofLocalNone := SourceRange.form?_eq_none_of_before
    (proofLocalRange program) logicalWidth (91874 + index.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (freshRange program) logicalWidth (91874 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (freshPublicInputRange program) logicalWidth (91874 + index.val) (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth (91874 + index.val) (by omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, outputNone,
    selected, proofLocalNone, freshNone, publicNone, contextNone]

theorem substitution_proofLocal_form?
    {program : Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (index : Fin proofLocalRangeCount) :
    (substitution program).form? logicalWidth
        (Spartan.sourceToSpartan
          (Spartan.piCcsPhaseOffset + index.val)) =
      some ((PiCCSOrdinaryDirectPlan.Location.proofLogical
        (proofLocalIndex index)).form geometry) := by
  have target := proofLocalTarget_eq program index
  have selected := proofLocalRange_form? geometry index
  rw [target] at selected
  have values := rangeValues program
  have indexBound : index.val < 472934 := by
    simpa only [proofLocalRangeCount, proofInputRangeCount,
      Spartan.proofInputColumnCount, proofLogicalCount_eq] using index.isLt
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorInputRange program) logicalWidth (13721422 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputInputRange program) logicalWidth (13721422 + index.val) (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_after
    (proofInputRange program) logicalWidth (13721422 + index.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_before
    (freshRange program) logicalWidth (13721422 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (freshPublicInputRange program) logicalWidth (13721422 + index.val) (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth (13721422 + index.val) (by omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, outputNone,
    proofInputNone, selected, freshNone, publicNone, contextNone]

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
  have indexBound : index.val < 731605 := by
    simpa only [freshCount] using index.isLt
  have priorNone := SourceRange.form?_eq_none_of_after
    (priorInputRange program) logicalWidth (18270868 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputInputRange program) logicalWidth (18270868 + index.val) (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_after
    (proofInputRange program) logicalWidth (18270868 + index.val) (by omega)
  have proofLocalNone := SourceRange.form?_eq_none_of_after
    (proofLocalRange program) logicalWidth (18270868 + index.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_before
    (freshPublicInputRange program) logicalWidth (18270868 + index.val) (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth (18270868 + index.val) (by omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, outputNone,
    proofInputNone, proofLocalNone, selected, publicNone, contextNone]

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
    (priorInputRange program) logicalWidth (27695711 + index.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputInputRange program) logicalWidth (27695711 + index.val) (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_after
    (proofInputRange program) logicalWidth (27695711 + index.val) (by omega)
  have proofLocalNone := SourceRange.form?_eq_none_of_after
    (proofLocalRange program) logicalWidth (27695711 + index.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_after
    (freshRange program) logicalWidth (27695711 + index.val) (by omega)
  have contextNone := SourceRange.form?_eq_none_of_before
    (expectedContextRange program) logicalWidth (27695711 + index.val) (by omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, outputNone,
    proofInputNone, proofLocalNone, freshNone, selected, contextNone]

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
    (priorInputRange program) logicalWidth (27695985 + lane.val) (by omega)
  have outputNone := SourceRange.form?_eq_none_of_after
    (outputInputRange program) logicalWidth (27695985 + lane.val) (by omega)
  have proofInputNone := SourceRange.form?_eq_none_of_after
    (proofInputRange program) logicalWidth (27695985 + lane.val) (by omega)
  have proofLocalNone := SourceRange.form?_eq_none_of_after
    (proofLocalRange program) logicalWidth (27695985 + lane.val) (by omega)
  have freshNone := SourceRange.form?_eq_none_of_after
    (freshRange program) logicalWidth (27695985 + lane.val) (by omega)
  have publicNone := SourceRange.form?_eq_none_of_after
    (freshPublicInputRange program) logicalWidth (27695985 + lane.val) (by omega)
  rw [target]
  simp [substitution, SourceSubstitution.form?, priorNone, outputNone,
    proofInputNone, proofLocalNone, freshNone, publicNone, selected]

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
      by_cases before : index.val < proofInputRangeCount
      · let inputIndex : Fin proofInputRangeCount := ⟨index.val, before⟩
        have indexEq : proofInputIndex inputIndex = index := by
          apply Fin.ext
          rfl
        simpa [PiCCSOrdinaryDirectPlan.Location.sourceColumn, indexEq] using
          (substitution_proofInput_form? geometry inputIndex)
      · have lower : proofInputRangeCount ≤ index.val :=
          Nat.le_of_not_gt before
        have localBound : index.val - proofInputRangeCount <
            proofLocalRangeCount := by
          have upper := index.isLt
          norm_num [proofInputRangeCount, proofLocalRangeCount,
            Spartan.proofInputColumnCount, proofLogicalCount_eq] at lower upper ⊢
          omega
        let localIndex : Fin proofLocalRangeCount :=
          ⟨index.val - proofInputRangeCount, localBound⟩
        have indexEq : proofLocalIndex localIndex = index := by
          apply Fin.ext
          change proofInputRangeCount +
              (index.val - proofInputRangeCount) = index.val
          omega
        have sourceEq : Spartan.piCcsPhaseOffset + localIndex.val =
            PiCCSInputs.proofInputStart + index.val := by
          norm_num [localIndex, proofInputRangeCount,
            Spartan.proofInputColumnCount, Spartan.piCcsPhaseOffset,
            PiCCSInputs.proofInputStart_eq] at lower ⊢
          omega
        simpa [PiCCSOrdinaryDirectPlan.Location.sourceColumn, sourceEq,
          indexEq] using (substitution_proofLocal_form? geometry localIndex)

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
      | none => .empty
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
    (index : Fin 811669) :
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
    (index : Fin 811669) :
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
returns exactly the proof-oriented direct 14-matrix row. -/
theorem compile_programRow?
    {program : Program} {logicalWidth relationLogicalWidth : Nat}
    {relationPublicFits : Spec.ringDegree *
      Lifecycle.PaperAlgebra.publicRingColumns ≤
        Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.carrierWidth
          relationLogicalWidth}
    (relation : Lifecycle.ProductionKey.LogicalRelation relationLogicalWidth
      relationPublicFits)
    (geometry : Geometry program logicalWidth)
    (index : Fin 811669)
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
