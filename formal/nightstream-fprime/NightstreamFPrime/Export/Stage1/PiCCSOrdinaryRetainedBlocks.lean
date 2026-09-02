import NightstreamFPrime.Export.Stage1.PiRLCRetainedPreservation
import NightstreamFPrime.Export.Stage1.PoseidonInputRetainedBlock
import NightstreamFPrime.Export.Stage1.PoseidonRetainedBlock
import NightstreamFPrime.Export.Stage1.RunningTransitionRetainedBlocks
import NightstreamFPrime.Layout.Stage1.PiCCSOrdinarySourceSupportData

/-!
Owns a conservative retained-field candidate for the PiCCS ordinary rows.

These six blocks cover the two authoritative state preimages, the last local
permutation interval of both pilot hashes, the proof/PiCCS logical interval,
and the exact R1CS-fresh interval. This module proves only ownership and
footprint. A separate source-support theorem must prove complete row coverage
before these blocks can enter the final direct plan.
-/

namespace NightstreamFPrime.Export.Stage1.PiCCSOrdinaryRetainedBlocks

open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.sourceWidth program

private theorem baseConstant_le_total :
    PerApplicationPackage.basePackage.layout.constantColumn ≤
      PerApplicationPackage.basePackage.layout.totalColumnCount := by
  change (Data.circuitPackage ()).layout.constantColumn ≤
    (Data.circuitPackage ()).layout.totalColumnCount
  have values :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values
  omega

private theorem finalColumn_lt_basePackage (column : Nat)
    (bound : column < PerApplicationPackage.basePackage.layout.totalColumnCount) :
    column < PiRLCProductPlan.basePackage.layout.totalColumnCount := by
  exact bound

/-- Embed one final package column into the nested direct-plan source domain. -/
def packageSourceColumn (program : Lifecycle.Stage1.Application.Program)
    (column : Nat)
    (bound : column < PerApplicationPackage.basePackage.layout.totalColumnCount) :
    Fin (sourceWidth program) :=
  PiRLCRetainedPreservation.baseSourceColumn program <|
    PiRLCProductPlan.shiftedPackageColumn program column
      (finalColumn_lt_basePackage column bound)

def packageFieldBlock (program : Lifecycle.Stage1.Application.Program)
    (count start : Nat)
    (bounded : start + count ≤
      PerApplicationPackage.basePackage.layout.totalColumnCount) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := count
  source := fun index => packageSourceColumn program (start + index.val) (by
    have indexBound := index.isLt
    omega)

def sourceFieldBlock (program : Lifecycle.Stage1.Application.Program)
    (count start : Nat) (bounded : start + count ≤ Spartan.SourceColumnCount) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := count
  source := fun index =>
    RunningTransitionRetainedBlocks.packageSourceColumn program
      (start + index.val) (by
        have indexBound := index.isLt
        omega)

def priorInputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program PilotProduction.stateHashWords
    PilotProduction.priorPreimageStart (by
      rw [Spartan.sourceColumnCount_eq]
      norm_num [PilotProduction.priorPreimageStart,
        PilotProduction.stateHashWords_eq])

def outputInputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program PilotProduction.stateHashWords
    PilotProduction.outputPreimageStart (by
      rw [Spartan.sourceColumnCount_eq]
      norm_num [PilotProduction.outputPreimageStart,
        PilotProduction.priorPublicInputStart,
        PilotProduction.priorPreimageStart,
        PilotProduction.stateHashWords_eq,
        Lifecycle.PriorStateHash.publicWidth,
        Lifecycle.PaperAlgebra.publicRingColumns, Spec.ringDegree])

def freshPublicInputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program 270 PilotProduction.priorPublicInputStart (by
    rw [Spartan.sourceColumnCount_eq]
    norm_num [PilotProduction.priorPublicInputStart,
      PilotProduction.priorPreimageStart,
      PilotProduction.stateHashWords_eq])

def priorLastInvocation : Fin PoseidonRetainedBlock.priorInvocationCount :=
  ⟨12349, by rw [PoseidonRetainedBlock.priorInvocationCount_eq]; omega⟩

def outputLastInvocation : Fin PoseidonRetainedBlock.outputInvocationCount :=
  ⟨12349, by rw [PoseidonRetainedBlock.outputInvocationCount_eq]; omega⟩

def priorLastBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  packageFieldBlock program 592
    (PoseidonRetainedBlock.priorWitnessStart priorLastInvocation) (by
      exact Nat.le_trans
        (PoseidonRetainedBlock.priorWitnessStart_bound priorLastInvocation)
        baseConstant_le_total)

def outputLastBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  packageFieldBlock program 592
    (PoseidonRetainedBlock.outputWitnessStart outputLastInvocation) (by
      exact Nat.le_trans
        (PoseidonRetainedBlock.outputWitnessStart_bound outputLastInvocation)
        baseConstant_le_total)

def expectedContextBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program PiCCSInputs.expectedContextWords
    PiCCSInputs.expectedContextStart (by
      rw [PiCCSInputs.expectedContextStart_eq, Spartan.sourceColumnCount_eq]
      norm_num [PiCCSInputs.expectedContextWords])

def proofInputCount : Nat :=
  PiCCSOrdinarySourceSupport.proofInputCount

def transcriptInvocationCount : Nat :=
  PiCCSOrdinarySourceSupport.transcriptInvocationCount

def transcriptOutputCount : Nat :=
  PiCCSOrdinarySourceSupport.transcriptOutputCount

def ordinaryLogicalCount : Nat :=
  PiCCSOrdinarySourceSupport.ordinaryLogicalCount

/-- Exact compact slot count: proof inputs, transcript output lanes, then the
non-transcript PiCCS logical suffix. -/
def proofLogicalCount : Nat :=
  proofInputCount + transcriptOutputCount + ordinaryLogicalCount

@[simp] theorem proofInputCount_eq : proofInputCount = 29288 := by
  exact PiCCSOrdinarySourceSupport.proofInputCount_eq

@[simp] theorem transcriptInvocationCount_eq :
    transcriptInvocationCount = 718 := by
  exact PiCCSOrdinarySourceSupport.transcriptInvocationCount_eq

@[simp] theorem transcriptOutputCount_eq : transcriptOutputCount = 5744 := by
  exact PiCCSOrdinarySourceSupport.transcriptOutputCount_eq

@[simp] theorem ordinaryLogicalCount_eq : ordinaryLogicalCount = 79846 := by
  exact PiCCSOrdinarySourceSupport.ordinaryLogicalCount_eq

@[simp] theorem proofLogicalCount_eq : proofLogicalCount = 114878 := by
  norm_num [proofLogicalCount, proofInputCount_eq, transcriptOutputCount_eq,
    ordinaryLogicalCount_eq]

def transcriptOutputSource (index : Fin transcriptOutputCount) : Nat :=
  let decoded : Fin transcriptInvocationCount × Fin Spec.Poseidon2.width :=
    Fin.decodeProd index
  PiCCSInputs.phaseOffset + decoded.1.val * 592 + 584 + decoded.2.val

@[simp] theorem transcriptOutputSource_encodeProd
    (invocation : Fin transcriptInvocationCount)
    (lane : Fin Spec.Poseidon2.width) :
    transcriptOutputSource (Fin.encodeProd (invocation, lane)) =
      PiCCSInputs.phaseOffset + invocation.val * 592 + 584 + lane.val := by
  let decoded : Fin transcriptInvocationCount × Fin Spec.Poseidon2.width :=
    Fin.decodeProd (Fin.encodeProd (invocation, lane))
  change PiCCSInputs.phaseOffset + decoded.1.val * 592 + 584 + decoded.2.val = _
  have decodedEq : decoded = (invocation, lane) := by
    exact Fin.decodeProd_encodeProd (invocation, lane)
  rw [decodedEq]

theorem transcriptOutputSource_support (index : Fin transcriptOutputCount) :
    PiCCSOrdinarySourceSupport.Source (transcriptOutputSource index) := by
  let decoded : Fin transcriptInvocationCount × Fin Spec.Poseidon2.width :=
    Fin.decodeProd index
  apply PiCCSOrdinarySourceSupport.transcript_output_source
  exact ⟨decoded.1, decoded.2, rfl⟩

theorem transcriptOutputSource_lt (index : Fin transcriptOutputCount) :
    transcriptOutputSource index < Spartan.SourceColumnCount :=
  PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount
    (transcriptOutputSource_support index)

def proofLogicalSource (index : Fin proofLogicalCount) : Nat :=
  if proof : index.val < proofInputCount then
    PiCCSInputs.proofInputStart + index.val
  else if transcript : index.val < proofInputCount + transcriptOutputCount then
    transcriptOutputSource
      ⟨index.val - proofInputCount, by omega⟩
  else
    PiCCSStarts.initialClaimLogicalStart +
      (index.val - (proofInputCount + transcriptOutputCount))

theorem proofLogicalSource_support (index : Fin proofLogicalCount) :
    PiCCSOrdinarySourceSupport.Source (proofLogicalSource index) := by
  unfold proofLogicalSource
  split
  · rename_i proof
    apply PiCCSOrdinarySourceSupport.external_source
    apply PiCCSOrdinarySourceSupport.external_proof
    unfold PiCCSOrdinarySourceSupport.InRange
    have proofBound : index.val < 29288 := by
      simpa only [proofInputCount_eq] using proof
    rw [PiCCSInputs.proofInputStart_eq, PiCCSInputs.phaseOffset_eq]
    constructor <;> omega
  · split
    · exact transcriptOutputSource_support _
    · rename_i notProof notTranscript
      apply PiCCSOrdinarySourceSupport.ordinary_logical_source
      unfold PiCCSOrdinarySourceSupport.OrdinaryLogical
        PiCCSOrdinarySourceSupport.InRange
      have indexBound : index.val < 114878 := by
        simpa only [proofLogicalCount_eq] using index.isLt
      rw [PiCCSOrdinarySourceSupport.ordinaryLogicalCount_eq]
      have notTranscriptNumeric : ¬index.val < 35032 := by
        simpa only [proofInputCount_eq, transcriptOutputCount_eq] using
          notTranscript
      simp only [proofInputCount_eq, transcriptOutputCount_eq]
      constructor <;> omega

theorem proofLogicalSource_lt (index : Fin proofLogicalCount) :
    proofLogicalSource index < Spartan.SourceColumnCount :=
  PiCCSOrdinarySourceSupport.source_lt_sourceColumnCount
    (proofLogicalSource_support index)

def proofInputSlot (index : Fin proofInputCount) : Fin proofLogicalCount :=
  ⟨index.val, by
    have bound : index.val < 29288 := by
      simpa only [proofInputCount_eq] using index.isLt
    rw [proofLogicalCount_eq]
    omega⟩

def transcriptOutputSlot (index : Fin transcriptOutputCount) :
    Fin proofLogicalCount :=
  ⟨proofInputCount + index.val, by
    have bound : index.val < 5744 := by
      simpa only [transcriptOutputCount_eq] using index.isLt
    rw [proofLogicalCount_eq, proofInputCount_eq]
    omega⟩

def ordinaryLogicalSlot (index : Fin ordinaryLogicalCount) :
    Fin proofLogicalCount :=
  ⟨proofInputCount + transcriptOutputCount + index.val, by
    have bound : index.val < 79846 := by
      simpa only [ordinaryLogicalCount_eq] using index.isLt
    rw [proofLogicalCount_eq, proofInputCount_eq, transcriptOutputCount_eq]
    omega⟩

@[simp] theorem proofLogicalSource_proofInput
    (index : Fin proofInputCount) :
    proofLogicalSource (proofInputSlot index) =
      PiCCSInputs.proofInputStart + index.val := by
  unfold proofLogicalSource proofInputSlot
  rw [dif_pos index.isLt]

@[simp] theorem proofLogicalSource_transcriptOutput
    (index : Fin transcriptOutputCount) :
    proofLogicalSource (transcriptOutputSlot index) =
      transcriptOutputSource index := by
  have indexBound : index.val < transcriptOutputCount := index.isLt
  have notProof : ¬proofInputCount + index.val < proofInputCount := by omega
  have transcript : proofInputCount + index.val <
      proofInputCount + transcriptOutputCount := by omega
  unfold proofLogicalSource transcriptOutputSlot
  rw [dif_neg notProof, dif_pos transcript]
  apply congrArg transcriptOutputSource
  apply Fin.ext
  simp

@[simp] theorem proofLogicalSource_ordinaryLogical
    (index : Fin ordinaryLogicalCount) :
    proofLogicalSource (ordinaryLogicalSlot index) =
      PiCCSStarts.initialClaimLogicalStart + index.val := by
  have notProof : ¬proofInputCount + transcriptOutputCount + index.val <
      proofInputCount := by omega
  have notTranscript : ¬proofInputCount + transcriptOutputCount + index.val <
      proofInputCount + transcriptOutputCount := by omega
  unfold proofLogicalSource ordinaryLogicalSlot
  rw [dif_neg notProof, dif_neg notTranscript]
  simp

def proofLogicalBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := proofLogicalCount
  source := fun index =>
    RunningTransitionRetainedBlocks.packageSourceColumn program
      (proofLogicalSource index) (proofLogicalSource_lt index)

/-- The final eight output-binding variables are the declared post-PiCCS
state. They are outside `proofLogicalBlock` and require one exact endpoint
block for the direct transcript pins. -/
def outputEndpointStart : Nat := PiCCSStarts.logicalFreshBase - 8

def outputEndpointBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program 8 outputEndpointStart (by
    rw [outputEndpointStart, PiCCSStarts.logicalFreshBase,
      PiCCSInputs.phaseOffset_eq, Spartan.sourceColumnCount_eq]
    norm_num)

def freshCount : Nat := 731605

def freshBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program freshCount PiCCSArithmetic.initialClaimFreshStart (by
    unfold PiCCSArithmetic.initialClaimFreshStart
      PiCCSStarts.initialClaimFreshStart PiCCSStarts.roundTranscriptFreshStart
      PiCCSStarts.challengeFreshStart PiCCSStarts.statementAbsorptionFreshStart
      PiCCSStarts.statementBindingFreshStart PiCCSStarts.logicalFreshBase
    rw [PiCCSInputs.phaseOffset_eq, Spartan.sourceColumnCount_eq]
    norm_num [freshCount])

@[simp] theorem retainedSlotCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    (priorInputBlock program).slotCount +
      (outputInputBlock program).slotCount +
      (freshPublicInputBlock program).slotCount +
      (priorLastBlock program).slotCount +
      (outputLastBlock program).slotCount +
      (expectedContextBlock program).slotCount +
      (proofLogicalBlock program).slotCount +
      (outputEndpointBlock program).slotCount +
      (freshBlock program).slotCount = 946735 := by
  norm_num [priorInputBlock, outputInputBlock, freshPublicInputBlock,
    priorLastBlock, outputLastBlock, expectedContextBlock, proofLogicalBlock,
    outputEndpointBlock, freshBlock, packageFieldBlock, sourceFieldBlock,
    proofLogicalCount_eq, freshCount, Data.priorChain, Data.outputChain,
    Data.liftPilotChain, PilotData.priorChain, PilotData.outputChain,
    PilotValues.stateHashWords, PilotValues.stateHashBaseWords,
    PilotProduction.stateHashWords_eq,
    PiCCSInputs.expectedContextWords]

def retainedCoordinateCount (program : Lifecycle.Stage1.Application.Program) :
    Nat :=
  (priorInputBlock program).coordinateCount +
    (outputInputBlock program).coordinateCount +
    (freshPublicInputBlock program).coordinateCount +
    (priorLastBlock program).coordinateCount +
    (outputLastBlock program).coordinateCount +
    (expectedContextBlock program).coordinateCount +
    (proofLogicalBlock program).coordinateCount +
    (outputEndpointBlock program).coordinateCount +
    (freshBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 38816135 := by
  simp only [retainedCoordinateCount, LowNormBlock.Block.coordinateCount,
    priorInputBlock, outputInputBlock, freshPublicInputBlock, priorLastBlock,
    outputLastBlock, expectedContextBlock, proofLogicalBlock, freshBlock,
    outputEndpointBlock, packageFieldBlock, sourceFieldBlock]
  rw [proofLogicalCount_eq]
  norm_num [freshCount, LowNormSlot.Kind.width, BalancedTernary.width,
    Data.priorChain, Data.outputChain, Data.liftPilotChain,
    PilotData.priorChain, PilotData.outputChain, PilotValues.stateHashWords,
    PilotValues.stateHashBaseWords, PilotProduction.stateHashWords_eq,
    PiCCSInputs.expectedContextWords]

end NightstreamFPrime.Export.Stage1.PiCCSOrdinaryRetainedBlocks
