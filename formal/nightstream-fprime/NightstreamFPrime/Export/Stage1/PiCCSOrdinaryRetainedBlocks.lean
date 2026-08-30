import NightstreamFPrime.Export.Stage1.PiRLCRetainedPreservation
import NightstreamFPrime.Export.Stage1.PoseidonInputRetainedBlock
import NightstreamFPrime.Export.Stage1.PoseidonRetainedBlock
import NightstreamFPrime.Export.Stage1.RunningTransitionRetainedBlocks

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
  ⟨11485, by rw [PoseidonRetainedBlock.priorInvocationCount_eq]; omega⟩

def outputLastInvocation : Fin PoseidonRetainedBlock.outputInvocationCount :=
  ⟨11485, by rw [PoseidonRetainedBlock.outputInvocationCount_eq]; omega⟩

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

def proofLogicalCount : Nat :=
  PiCCSStarts.outputBindingWitnessStart - PiCCSInputs.proofInputStart

def proofLogicalBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  sourceFieldBlock program proofLogicalCount PiCCSInputs.proofInputStart (by
    rw [proofLogicalCount, PiCCSStarts.outputBindingWitnessStart_eq,
      PiCCSInputs.proofInputStart_eq,
      Spartan.sourceColumnCount_eq]
    norm_num)

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

@[simp] theorem proofLogicalCount_eq : proofLogicalCount = 502006 := by
  rw [proofLogicalCount, PiCCSStarts.outputBindingWitnessStart_eq,
    PiCCSInputs.proofInputStart_eq]

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
      (freshBlock program).slotCount = 1326951 := by
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
    retainedCoordinateCount program = 54404991 := by
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
