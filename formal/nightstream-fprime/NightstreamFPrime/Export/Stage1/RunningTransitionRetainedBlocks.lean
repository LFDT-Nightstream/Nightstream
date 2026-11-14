import NightstreamFPrime.Export.Stage1.PiRLCRetainedPreservation
import NightstreamFPrime.Layout.Stage1.RunningTransitionSourceSupportData

namespace NightstreamFPrime.Export.Stage1.RunningTransitionRetainedBlocks

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle

@[implicit_reducible]
def sourceWidth (program : Lifecycle.Stage1.Application.Program) : Nat :=
  PiRLCRetainedGeometry.sourceWidth program

private theorem target_lt_basePackage (source : Nat)
    (bound : source < PackageOrder.SourceColumnCount) :
    PackageOrder.sourceToPackage source <
      PiRLCProductPlan.basePackage.layout.totalColumnCount := by
  have mapped := PackageOrder.sourceToPackage_lt source bound
  have total : PiRLCProductPlan.basePackage.layout.totalColumnCount =
      10059655 :=
    NightstreamFPrime.Export.Stage1.Package.circuitPackage_layout_values.2.2.2.2
  rw [total]
  simpa [PackageOrder.packageColumnCount] using mapped

def packageSourceColumn (program : Lifecycle.Stage1.Application.Program)
    (source : Nat) (bound : source < PackageOrder.SourceColumnCount) :
    Fin (sourceWidth program) :=
  PiRLCRetainedPreservation.baseSourceColumn program <|
    PiRLCProductPlan.shiftedPackageColumn program
      (PackageOrder.sourceToPackage source) (target_lt_basePackage source bound)

def fieldBlock (program : Lifecycle.Stage1.Application.Program)
    (count : Nat) (source : Fin count → Nat)
    (bounded : ∀ index, source index < PackageOrder.SourceColumnCount) :
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
      change index.val < 9 at indexBound
      rw [RunningTransitionSourceSupport.stateStart_eq,
        PackageOrder.sourceColumnCount_eq]
      change 25320 + index.val < 10059654
      omega)

def outputBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  fieldBlock program RunningTransitionSourceSupport.outputCount
    (fun index => RunningTransitionSourceSupport.outputStart + index.val) (by
      intro index
      have indexBound := index.isLt
      change index.val < 25329 at indexBound
      rw [RunningTransitionSourceSupport.outputStart_eq,
        PackageOrder.sourceColumnCount_eq]
      change 25649 + index.val < 10059654
      omega)

def piDecBlock (program : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth program) :=
  fieldBlock program RunningTransitionSourceSupport.piDecCount
    (fun index => RunningTransitionSourceSupport.piDecStart + index.val) (by
      intro index
      have indexBound := index.isLt
      change index.val < 29568 at indexBound
      rw [RunningTransitionSourceSupport.piDecStart_eq,
        PackageOrder.sourceColumnCount_eq]
      change 10029764 + index.val < 10059654
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
        PackageOrder.SourceColumnCount] at indexBound ⊢
      omega)

@[simp] theorem stateBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (stateBlock program).slotCount = 9 := by
  rfl

@[simp] theorem outputBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (outputBlock program).slotCount = 25329 := by
  rw [outputBlock]
  exact RunningTransitionSourceSupport.outputCount_eq

@[simp] theorem piDecBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (piDecBlock program).slotCount = 29568 := by
  rw [piDecBlock]
  exact RunningTransitionSourceSupport.piDecCount_eq

@[simp] theorem freshCount_eq : freshCount = 2 := by
  rfl

@[simp] theorem freshBlock_slotCount
    (program : Lifecycle.Stage1.Application.Program) :
    (freshBlock program).slotCount = 2 := by
  rfl

@[simp] theorem fieldBlock_coordinateCount
    (program : Lifecycle.Stage1.Application.Program)
    (count : Nat) (source : Fin count → Nat)
    (bounded : ∀ index, source index < PackageOrder.SourceColumnCount) :
    (fieldBlock program count source bounded).coordinateCount = count * 41 := by
  rfl

def retainedSlotCount (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (piDecBlock program).slotCount +
    (freshBlock program).slotCount

@[simp] theorem retainedSlotCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedSlotCount program = 29570 := by
  simp [retainedSlotCount]

def retainedCoordinateCount
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  (piDecBlock program).coordinateCount +
    (freshBlock program).coordinateCount

@[simp] theorem retainedCoordinateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount program = 1212370 := by
  simp only [retainedCoordinateCount, piDecBlock,
    freshBlock, fieldBlock_coordinateCount,
    RunningTransitionSourceSupport.piDecCount_eq, freshCount_eq]

end NightstreamFPrime.Export.Stage1.RunningTransitionRetainedBlocks
