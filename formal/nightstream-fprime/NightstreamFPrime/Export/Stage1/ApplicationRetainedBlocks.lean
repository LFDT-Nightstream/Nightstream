import NightstreamFPrime.Export.Stage1.ApplicationDirectSource
import NightstreamFPrime.Layout.LowNormBlock

/-!
Owns the minimal retained field blocks for one verifier-selected application.
It retains only declared application inputs, witnesses, outputs, and local
lowering values.
-/

namespace NightstreamFPrime.Export.Stage1.ApplicationRetainedBlocks

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle

def sourceWidth (application : Lifecycle.Stage1.Application.Program) : Nat :=
  ApplicationDirectSource.sourceWidth application

def localCount (application : Lifecycle.Stage1.Application.Program) : Nat :=
  sourceWidth application - Layout.Stage1.ApplicationInputs.localStart application

theorem localStart_le_sourceWidth
    (application : Lifecycle.Stage1.Application.Program) :
    Layout.Stage1.ApplicationInputs.localStart application ≤
      sourceWidth application := by
  unfold sourceWidth ApplicationDirectSource.sourceWidth
    ApplicationPackage.r1csFreshStart
  omega

theorem witnessStart_le_sourceWidth
    (application : Lifecycle.Stage1.Application.Program) :
    Layout.Stage1.ApplicationInputs.witnessStart ≤ sourceWidth application := by
  exact Nat.le_trans (by
    unfold Layout.Stage1.ApplicationInputs.localStart
    omega) (localStart_le_sourceWidth application)

private theorem inputColumn_lt_sourceWidth
    (application : Lifecycle.Stage1.Application.Program)
    (index : Lifecycle.Stage1.Application.StateIndex) :
    Layout.Stage1.ApplicationInputs.inputColumn index < sourceWidth application := by
  apply Nat.lt_of_lt_of_le _ (witnessStart_le_sourceWidth application)
  rw [Layout.Stage1.ApplicationInputs.inputColumn_value]
  have bound := index.isLt
  norm_num [Layout.Stage1.ApplicationInputs.witnessStart,
    Layout.Stage1.Spartan.privateColumnCount,
    Layout.Stage1.ApplicationInputs.currentWordStart,
    Lifecycle.Stage1.Application.stateWordCount] at bound ⊢
  omega

private theorem outputColumn_lt_sourceWidth
    (application : Lifecycle.Stage1.Application.Program)
    (index : Lifecycle.Stage1.Application.StateIndex) :
    Layout.Stage1.ApplicationInputs.outputColumn index < sourceWidth application := by
  apply Nat.lt_of_lt_of_le _ (witnessStart_le_sourceWidth application)
  rw [Layout.Stage1.ApplicationInputs.outputColumn_value]
  have bound := index.isLt
  norm_num [Layout.Stage1.ApplicationInputs.witnessStart,
    Layout.Stage1.Spartan.privateColumnCount,
    Layout.Stage1.ApplicationInputs.currentWordStart,
    Lifecycle.Stage1.Application.stateWordCount] at bound ⊢
  omega

def inputBlock (application : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth application) where
  kind := .field
  slotCount := Lifecycle.Stage1.Application.stateWordCount
  source := fun index =>
    ⟨Layout.Stage1.ApplicationInputs.inputColumn index,
      inputColumn_lt_sourceWidth application index⟩

def witnessBlock (application : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth application) where
  kind := .field
  slotCount := application.witnessWordCount
  source := fun index =>
    ⟨Layout.Stage1.ApplicationInputs.witnessColumn index, by
      unfold Layout.Stage1.ApplicationInputs.witnessColumn
      have indexBound := index.isLt
      have localLe := localStart_le_sourceWidth application
      unfold Layout.Stage1.ApplicationInputs.localStart at localLe
      omega⟩

def outputBlock (application : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth application) where
  kind := .field
  slotCount := Lifecycle.Stage1.Application.stateWordCount
  source := fun index =>
    ⟨Layout.Stage1.ApplicationInputs.outputColumn index,
      outputColumn_lt_sourceWidth application index⟩

def localBlock (application : Lifecycle.Stage1.Application.Program) :
    LowNormBlock.Block (sourceWidth application) where
  kind := .field
  slotCount := localCount application
  source := fun index =>
    ⟨Layout.Stage1.ApplicationInputs.localStart application + index.val, by
      have indexBound := index.isLt
      have startLe := localStart_le_sourceWidth application
      unfold localCount at indexBound
      omega⟩

@[simp] theorem inputBlock_slotCount
    (application : Lifecycle.Stage1.Application.Program) :
    (inputBlock application).slotCount = 4 := by
  rfl

@[simp] theorem witnessBlock_slotCount
    (application : Lifecycle.Stage1.Application.Program) :
    (witnessBlock application).slotCount = application.witnessWordCount := by
  rfl

@[simp] theorem outputBlock_slotCount
    (application : Lifecycle.Stage1.Application.Program) :
    (outputBlock application).slotCount = 4 := by
  rfl

@[simp] theorem localBlock_slotCount
    (application : Lifecycle.Stage1.Application.Program) :
    (localBlock application).slotCount = localCount application := by
  rfl

def retainedSlotCount
    (application : Lifecycle.Stage1.Application.Program) : Nat :=
  (inputBlock application).slotCount +
    (witnessBlock application).slotCount +
    (outputBlock application).slotCount +
    (localBlock application).slotCount

theorem retainedSlotCount_eq
    (application : Lifecycle.Stage1.Application.Program) :
    retainedSlotCount application =
      8 + application.witnessWordCount + localCount application := by
  simp [retainedSlotCount]
  omega

def retainedCoordinateCount
    (application : Lifecycle.Stage1.Application.Program) : Nat :=
  (inputBlock application).coordinateCount +
    (witnessBlock application).coordinateCount +
    (outputBlock application).coordinateCount +
    (localBlock application).coordinateCount

theorem retainedCoordinateCount_eq
    (application : Lifecycle.Stage1.Application.Program) :
    retainedCoordinateCount application =
      retainedSlotCount application * 41 := by
  simp [retainedCoordinateCount, retainedSlotCount,
    inputBlock, witnessBlock, outputBlock, localBlock,
    LowNormBlock.Block.coordinateCount, LowNormSlot.Kind.width,
    BalancedTernary.width]
  omega

/-- Every permitted application source is owned by exactly one retained
family, up to harmless overlap between fixed input/output values. -/
theorem sourceAllowed_covered
    (application : Lifecycle.Stage1.Application.Program)
    (column : Fin (sourceWidth application))
    (support : ApplicationDirectSource.SourceAllowed application column.val) :
    (∃ index, (inputBlock application).source index = column) ∨
      (∃ index, (witnessBlock application).source index = column) ∨
      (∃ index, (outputBlock application).source index = column) ∨
      ∃ index, (localBlock application).source index = column := by
  rcases support with input | witness | output | localSupport
  · rcases input with ⟨index, equality⟩
    exact Or.inl ⟨index, Fin.ext equality.symm⟩
  · rcases witness with ⟨index, equality⟩
    exact Or.inr (Or.inl ⟨index, Fin.ext equality.symm⟩)
  · rcases output with ⟨index, equality⟩
    exact Or.inr (Or.inr (Or.inl ⟨index, Fin.ext equality.symm⟩))
  · exact Or.inr (Or.inr (Or.inr ⟨
      ⟨column.val - Layout.Stage1.ApplicationInputs.localStart application,
        by
          change column.val -
            Layout.Stage1.ApplicationInputs.localStart application <
              localCount application
          unfold localCount
          omega⟩,
      Fin.ext (by simp [localBlock]; omega)⟩))

end NightstreamFPrime.Export.Stage1.ApplicationRetainedBlocks
