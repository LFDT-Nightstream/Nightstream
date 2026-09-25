import NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource
import NightstreamFPrime.Export.Stage1.Wide.ApplicationPackage

/-! Recover each emitted ordinary row, including the insertion of the
application's private columns before the constant and public columns. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixRecovery

open NightstreamFPrime.Layout
open PhysicalRelabel PhysicalMatrixSource

theorem ordinary_row_recovered (extra : Nat) (before after : Rows.CompiledRow)
    (emitted : ordinaryMap.compiledRow before = .ok after) :
    (inverse extra).row? after.toR1CS = some before.toR1CS := by
  obtain ⟨_, equal, supported⟩ := ordinaryMap.compiledRow_correct before after emitted
  rw [equal]
  apply MatrixProgram.SourceProjection.row?_mapColumns_supported
  apply supported.mono
  intro source mapped
  obtain ⟨target, read⟩ := mapped
  have value : Map.columnValue ordinaryMap source = target := by
    simp [Map.columnValue, read, Except.toOption]
  rw [value]
  exact ordinary_column_inverse extra source target read

theorem ordinary_row_recovered_inserted
    (application : Lifecycle.Stage1.Application.Program) (before moved after : Rows.CompiledRow)
    (emitted : ordinaryMap.compiledRow before = .ok moved)
    (inserted : (ApplicationPackage.insertApplication
      (PerApplicationPackage.directAddedPrivateColumnCount application)).compiledRow moved = .ok after) :
    (inverse (PerApplicationPackage.directAddedPrivateColumnCount application)).row? after.toR1CS =
      some (PerApplicationSourceProjection.basePackageRow application before.toR1CS) := by
  obtain ⟨_, movedEqual, supported⟩ := ordinaryMap.compiledRow_correct before moved emitted
  obtain ⟨_, insertedEqual, _⟩ :=
    (ApplicationPackage.insertApplication _).compiledRow_correct moved after inserted
  rw [insertedEqual, movedEqual]
  have composed : ∀ (row : R1CS.Row) (first second : Nat → Nat),
      R1CS.mapRowColumns first (R1CS.mapRowColumns second row) =
        R1CS.mapRowColumns (fun source => first (second source)) row := by
    intro row first second
    cases row
    simp [R1CS.mapRowColumns, R1CS.mapCombinationColumns, List.map_map, Function.comp_def]
  rw [composed]
  apply MatrixProgram.SourceProjection.row?_mapColumns_to
  apply supported.mono
  intro source mapped
  obtain ⟨target, read⟩ := mapped
  have value : Map.columnValue ordinaryMap source = target := by
    simp [Map.columnValue, read, Except.toOption]
  rw [value]
  have recover := inverse_insert (PerApplicationPackage.directAddedPrivateColumnCount application)
    source target (ordinary_column_inverse 0 source target read)
  simpa only [Map.columnValue, ApplicationPackage.insertApplication,
    Layout.Stage1.Wide.SourceOrder.privateColumns_eq, Except.toOption, Option.getD_some,
    ← PerApplicationPackage.directShiftColumn_eq_shiftColumn,
    PerApplicationPackage.directShiftColumn, Data.physicalLayout,
    Layout.Stage1.Spartan.privateColumnCount_eq] using recover

end NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixRecovery
