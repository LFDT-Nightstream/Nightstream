import NightstreamFPrime.Export.Stage1.RunningTransitionRetainedGeometry
import NightstreamFPrime.Layout.Stage1.RunningTransitionReducedRows
import NightstreamFPrime.Layout.Stage1.RunningTransitionValues
import NightstreamFPrime.Layout.Stage1.SpartanValues

/-!
Own the two retained sources of the reduced running transition: the existing
inverse field and first scratch value, now encoded as a bit. This module fixes
local block geometry; it does not change the aggregate assignment schedule.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.RunningTransitionReducedRetainedBlocks

open NightstreamFPrime.Spec NightstreamFPrime.Layout NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle

abbrev ApplicationProgram := Stage1.Application.Program

def sourceWidth (program : ApplicationProgram) : Nat :=
  RunningTransitionRetainedBlocks.sourceWidth program

private theorem inverse_bounded :
    RunningTransitionInputs.phaseOffset < Spartan.SourceColumnCount := by
  rw [Spartan.sourceColumnCount_eq]
  change 29040586 < 29336724
  decide

private theorem flag_bounded :
    RunningTransitionReducedRows.flagIndex < Spartan.SourceColumnCount := by
  rw [Spartan.sourceColumnCount_eq]
  change 29040587 < 29336724
  decide

def inverseSource (program : ApplicationProgram) : Fin (sourceWidth program) :=
  RunningTransitionRetainedBlocks.packageSourceColumn program
    RunningTransitionInputs.phaseOffset inverse_bounded

def flagSource (program : ApplicationProgram) : Fin (sourceWidth program) :=
  RunningTransitionRetainedBlocks.packageSourceColumn program
    RunningTransitionReducedRows.flagIndex flag_bounded

def inverseBlock (program : ApplicationProgram) : LowNormBlock.Block (sourceWidth program) where
  kind := .field
  slotCount := 1
  source := fun _ => inverseSource program

def flagBlock (program : ApplicationProgram) : LowNormBlock.Block (sourceWidth program) where
  kind := .bit
  slotCount := 1
  source := fun _ => flagSource program

def inverseStart (program : ApplicationProgram) : Nat :=
  RunningTransitionRetainedGeometry.freshStart program

def flagStart (program : ApplicationProgram) : Nat := inverseStart program + 41

def nextStart (program : ApplicationProgram) : Nat := flagStart program + 1

@[simp] theorem nextStart_eq (program : ApplicationProgram) :
    nextStart program = 127362796 := by
  have old := RunningTransitionRetainedGeometry.completeLogicalWidth_eq program
  simp only [RunningTransitionRetainedGeometry.completeLogicalWidth,
    RunningTransitionRetainedBlocks.freshBlock,
    RunningTransitionRetainedBlocks.fieldBlock_coordinateCount,
    RunningTransitionRetainedBlocks.freshCount_eq] at old
  unfold nextStart flagStart inverseStart
  omega

private theorem packageSourceColumn_val (program : ApplicationProgram) (column : Nat)
    (bound : column < Spartan.SourceColumnCount) :
    (RunningTransitionRetainedBlocks.packageSourceColumn program column bound).val =
      PerApplicationPackage.shiftColumn program (Spartan.sourceToSpartan column) := by
  dsimp only [RunningTransitionRetainedBlocks.packageSourceColumn,
    PiRLCRetainedPreservation.baseSourceColumn, PiRLCFirst54DirectPlan.prefixColumn,
    ProductRetainedBlock.baseColumn, PiRLCProductPlan.shiftedPackageColumn,
    FieldSuffixBlock.baseColumn]

/-- These are the base-package Spartan private addresses. The existing application
shift preserves them because both precede the application insertion boundary;
the nested source embeddings preserve their values as well. -/
theorem source_addresses (program : ApplicationProgram) :
    (inverseSource program).val = 29040308 ∧ (flagSource program).val = 29040309 := by
  have constant := Package.circuitPackage_layout_values.2.2.1
  change PerApplicationPackage.basePackage.layout.constantColumn = 29336446 at constant
  constructor
  · rw [inverseSource, packageSourceColumn_val]
    have phase : RunningTransitionInputs.phaseOffset = 29040586 := by
      have next := RunningTransitionLayout.logicalColumnCount_eq
      change RunningTransitionInputs.phaseOffset + 1 = 29040587 at next
      omega
    have mapped : Spartan.sourceToSpartan RunningTransitionInputs.phaseOffset = 29040308 := by
      rw [phase]
      rfl
    rw [mapped, PerApplicationPackage.shiftColumn_private program _ (by rw [constant]; decide)]
  · rw [flagSource, packageSourceColumn_val]
    have mapped : Spartan.sourceToSpartan RunningTransitionReducedRows.flagIndex = 29040309 := by
      rw [RunningTransitionReducedRows.flagIndex, RunningTransitionLayout.logicalColumnCount_eq]
      rfl
    rw [mapped, PerApplicationPackage.shiftColumn_private program _ (by rw [constant]; decide)]

theorem local_geometry (program : ApplicationProgram) :
    (inverseBlock program).coordinateCount = 41 ∧
      (flagBlock program).coordinateCount = 1 ∧
      nextStart program = inverseStart program + 42 := by
  exact ⟨rfl, rfl, rfl⟩

/-- The inverse retains exactly the first source slot of the old field block. -/
theorem inverseSource_eq_old (program : ApplicationProgram) :
    inverseSource program = (RunningTransitionRetainedBlocks.freshBlock program).source
      ⟨0, by rw [RunningTransitionRetainedBlocks.freshBlock_slotCount]; decide⟩ := by
  rfl

/-- The flag reads the entire old first scratch field, not one of its trits. -/
theorem flagSource_eq_old (program : ApplicationProgram) :
    flagSource program = (RunningTransitionRetainedBlocks.freshBlock program).source
      ⟨1, by rw [RunningTransitionRetainedBlocks.freshBlock_slotCount]; decide⟩ := by
  rfl

end NightstreamFPrime.Export.Stage1.RunningTransitionReducedRetainedBlocks
