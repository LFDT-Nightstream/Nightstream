import NightstreamFPrime.Export.Stage1.PiRLCRetainedPreservation
import NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupport

/-!
Owns the compact retained field blocks needed by the direct running-transition
rows. Every source first follows the established Spartan permutation, then the
per-application package shift, and finally the existing nested PiRLC source
embedding.

This module does not compile rows or claim that a final assignment encodes the
blocks.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionRetainedBlocks

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle

def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.sourceWidth program

private theorem target_lt_basePackage (source : Nat)
    (bound : source < Spartan.SourceColumnCount) :
    Spartan.sourceToSpartan source <
      PiRLCProductPlan.basePackage.layout.totalColumnCount := by
  have mapped := Spartan.sourceToSpartan_lt source bound
  have total : PiRLCProductPlan.basePackage.layout.totalColumnCount =
      29336725 :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.2.2
  rw [total]
  simpa [Spartan.spartanColumnCount] using mapped

/-- Canonical nested source column for one pre-Spartan transition source. -/
def packageSourceColumn (program : Lifecycle.Stage1.Application.Program)
    (source : Nat) (bound : source < Spartan.SourceColumnCount) :
    Fin (sourceWidth program) :=
  PiRLCRetainedPreservation.baseSourceColumn program <|
    PiRLCProductPlan.shiftedPackageColumn program
      (Spartan.sourceToSpartan source) (target_lt_basePackage source bound)

/-- Generic field block over one proved finite transition-source family. -/
def fieldBlock (program : Lifecycle.Stage1.Application.Program)
    (count : Nat) (source : Fin count → Nat)
    (bounded : ∀ index, source index < Spartan.SourceColumnCount) :
    LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := count
  source := fun index => packageSourceColumn program (source index) (bounded index)

def stateBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  fieldBlock program RunningTransitionSourceSupport.stateCount
    (fun index => RunningTransitionSourceSupport.stateStart + index.val) (by
      intro index
      have indexBound := index.isLt
      change index.val < 11 at indexBound
      rw [RunningTransitionSourceSupport.stateStart_eq,
        Spartan.sourceColumnCount_eq]
      change 28 + index.val < 29336724
      omega)

def outputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  fieldBlock program RunningTransitionSourceSupport.outputCount
    (fun index => RunningTransitionSourceSupport.outputStart + index.val) (by
      intro index
      have indexBound := index.isLt
      change index.val < 49393 at indexBound
      rw [RunningTransitionSourceSupport.outputStart_eq,
        Spartan.sourceColumnCount_eq]
      change 49663 + index.val < 29336724
      omega)

def roundC0Block (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  fieldBlock program productionShape.cubeVariables
    (fun coordinate => PiCCSStarts.roundTranscriptWitnessStart +
      coordinate.val * RunningTransitionInputs.roundStride +
        RunningTransitionInputs.roundSampleC0Offset) (by
      intro coordinate
      have coordinateBound := coordinate.isLt
      change coordinate.val < 28 at coordinateBound
      rw [PiCCSStarts.roundTranscriptWitnessStart_eq,
        Spartan.sourceColumnCount_eq]
      norm_num [RunningTransitionInputs.roundStride,
        RunningTransitionInputs.roundSampleC0Offset]
      omega)

def roundC1Block (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  fieldBlock program productionShape.cubeVariables
    (fun coordinate => PiCCSStarts.roundTranscriptWitnessStart +
      coordinate.val * RunningTransitionInputs.roundStride +
        RunningTransitionInputs.roundSampleC1Offset) (by
      intro coordinate
      have coordinateBound := coordinate.isLt
      change coordinate.val < 28 at coordinateBound
      rw [PiCCSStarts.roundTranscriptWitnessStart_eq,
        Spartan.sourceColumnCount_eq]
      norm_num [RunningTransitionInputs.roundStride,
        RunningTransitionInputs.roundSampleC1Offset]
      omega)

def piDecBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  fieldBlock program RunningTransitionSourceSupport.piDecCount
    (fun index => RunningTransitionSourceSupport.piDecStart + index.val) (by
      intro index
      have indexBound := index.isLt
      change index.val < 49248 at indexBound
      rw [RunningTransitionSourceSupport.piDecStart_eq,
        Spartan.sourceColumnCount_eq]
      change 28973248 + index.val < 29336724
      omega)

def freshCount : Nat := RunningTransitionSourceSupport.physicalEnd -
  RunningTransitionInputs.phaseOffset

def freshBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  fieldBlock program freshCount
    (fun index => RunningTransitionInputs.phaseOffset + index.val) (by
      intro index
      have indexBound := index.isLt
      norm_num [freshCount, RunningTransitionSourceSupport.physicalEnd,
        RunningTransitionInputs.phaseOffset,
        Spartan.SourceColumnCount] at indexBound ⊢
      omega)

@[simp] theorem stateBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (stateBlock program).slotCount = 11 := by
  rfl

@[simp] theorem outputBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (outputBlock program).slotCount = 49393 := by
  rw [outputBlock]
  exact RunningTransitionSourceSupport.outputCount_eq

@[simp] theorem roundC0Block_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (roundC0Block program).slotCount = 28 := by
  rfl

@[simp] theorem roundC1Block_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (roundC1Block program).slotCount = 28 := by
  rfl

@[simp] theorem piDecBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (piDecBlock program).slotCount = 49248 := by
  rw [piDecBlock]
  exact RunningTransitionSourceSupport.piDecCount_eq

@[simp] theorem freshCount_eq : freshCount = 296138 := by
  norm_num [freshCount, RunningTransitionSourceSupport.physicalEnd,
    RunningTransitionInputs.phaseOffset]

@[simp] theorem freshBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (freshBlock program).slotCount = 296138 := by
  rfl

@[simp] theorem fieldBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program)
    (count : Nat) (source : Fin count → Nat)
    (bounded : ∀ index, source index < Spartan.SourceColumnCount) :
    (fieldBlock program count source bounded).coordinateCount = count * 41 := by
  rfl

def retainedSlotCount (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (stateBlock program).slotCount +
    (outputBlock program).slotCount +
    (roundC0Block program).slotCount +
    (roundC1Block program).slotCount +
    (piDecBlock program).slotCount +
    (freshBlock program).slotCount

@[simp] theorem retainedSlotCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedSlotCount program = 394846 := by
  simp [retainedSlotCount]

def retainedCoordinateCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (stateBlock program).coordinateCount +
    (outputBlock program).coordinateCount +
    (roundC0Block program).coordinateCount +
    (roundC1Block program).coordinateCount +
    (piDecBlock program).coordinateCount +
    (freshBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 16188686 := by
  change 11 * 41 + 49393 * 41 + 28 * 41 + 28 * 41 +
    49248 * 41 + 296138 * 41 = 16188686
  norm_num

end NightstreamFPrime.Export.Stage1.RunningTransitionRetainedBlocks
