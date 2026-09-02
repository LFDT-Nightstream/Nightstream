import NightstreamFPrime.Layout.Stage1.Lowering
import NightstreamFPrime.Export.Stage1.ApplicationDirectSource
import NightstreamFPrime.Export.Stage1.NextPreimagePackage

/-!
Connects the lower-layer Stage 1 suffix lowering to the canonical package
compiler. These are structural equalities; no artifact-sized list is reduced.
-/

namespace NightstreamFPrime.Export.Stage1.Stage1LoweringBridge

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle

theorem applicationOperations_eq
    (program : Lifecycle.Stage1.Application.Program) :
    Lowering.applicationOperations program =
      ApplicationPackage.operations program
        (ApplicationPackage.productionColumns program)
        (ApplicationInputs.localStart program) := by
  rfl

theorem applicationConstraints_eq
    (program : Lifecycle.Stage1.Application.Program) :
    Lowering.applicationConstraints program =
      ApplicationPackage.constraints program
        (ApplicationPackage.productionColumns program)
        (ApplicationInputs.localStart program) := by
  rfl

theorem applicationFirstFresh_eq
    (program : Lifecycle.Stage1.Application.Program) :
    Lowering.applicationFirstFresh program =
      ApplicationPackage.r1csFreshStart program
        (ApplicationPackage.productionColumns program)
        (ApplicationInputs.localStart program) := by
  rfl

theorem applicationRows_eq_sourceRows
    (program : Lifecycle.Stage1.Application.Program) :
    Lowering.applicationRows program =
      ApplicationDirectSource.sourceRows program := by
  rw [ApplicationDirectSource.sourceRows,
    ApplicationPackage.ofProgram_compiledRows_toR1CS]
  rfl

theorem nextPreimageRows_eq_sourceRows :
    Lowering.nextPreimageRows = NextPreimagePackage.sourceRows := by
  rw [NextPreimagePackage.sourceRows_eq]
  rfl

theorem applicationPrivateCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    Lowering.applicationPrivateCount program =
      PerApplicationPackage.directApplicationPrivateCount program := by
  unfold Lowering.applicationPrivateCount
    PerApplicationPackage.directApplicationPrivateCount
  rw [applicationOperations_eq]
  rfl

theorem addedPrivateColumnCount_eq
    (program : Lifecycle.Stage1.Application.Program) :
    Lowering.addedPrivateColumnCount program =
      PerApplicationPackage.addedPrivateColumnCount program := by
  calc
    Lowering.addedPrivateColumnCount program =
        PerApplicationPackage.directAddedPrivateColumnCount program := by
      unfold Lowering.addedPrivateColumnCount
        PerApplicationPackage.directAddedPrivateColumnCount
      rw [applicationPrivateCount_eq]
    _ = PerApplicationPackage.addedPrivateColumnCount program :=
      PerApplicationPackage.directAddedPrivateColumnCount_eq_addedPrivateColumnCount
        program

theorem shiftColumn_eq
    (program : Lifecycle.Stage1.Application.Program) (column : Nat) :
    Lowering.shiftColumn program column =
      PerApplicationPackage.shiftColumn program column := by
  calc
    Lowering.shiftColumn program column =
        PerApplicationPackage.directShiftColumn program column := by
      unfold Lowering.shiftColumn PerApplicationPackage.directShiftColumn
      rw [addedPrivateColumnCount_eq]
      rfl
    _ = PerApplicationPackage.shiftColumn program column :=
      PerApplicationPackage.directShiftColumn_eq_shiftColumn program column

end NightstreamFPrime.Export.Stage1.Stage1LoweringBridge
