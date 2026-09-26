import NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixSource
import NightstreamFPrime.Export.Stage1.Wide.ApplicationPackage
import NightstreamFPrime.Export.Stage1.ApplicationDirectSource

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
    Layout.Stage1.Spartan.privateColumnCount_eq] using! recover

private theorem application_column_range (application : Lifecycle.Stage1.Application.Program)
    (column : Nat) (allowed : ApplicationDirectSource.SourceAllowed application column) :
    column < 19512839 ∨
      (28784740 ≤ column ∧
        column < 28784740 + PerApplicationPackage.directAddedPrivateColumnCount application) := by
  have width : ApplicationDirectSource.sourceWidth application =
      28784740 + PerApplicationPackage.directAddedPrivateColumnCount application := by
    simp only [ApplicationDirectSource.sourceWidth, Stage1.ApplicationPackage.r1csFreshStart,
      Stage1.ApplicationPackage.constraints, PerApplicationPackage.directAddedPrivateColumnCount,
      PerApplicationPackage.directApplicationPrivateCount, Layout.Stage1.ApplicationInputs.localStart,
      Layout.Stage1.ApplicationInputs.witnessStart, Layout.Stage1.Spartan.privateColumnCount_eq,
      Nat.add_assoc]
  rcases allowed with ⟨index, rfl⟩ | ⟨index, rfl⟩ | ⟨index, rfl⟩ | ⟨lower, upper⟩
  · left
    rw [Layout.Stage1.ApplicationInputs.inputColumn_value]
    have bound := index.isLt
    change index.val < 4 at bound
    change 35 + index.val < 19512839
    omega
  · right
    have bound := index.isLt
    simp only [Layout.Stage1.ApplicationInputs.witnessColumn,
      Layout.Stage1.ApplicationInputs.witnessStart, Layout.Stage1.Spartan.privateColumnCount_eq,
      PerApplicationPackage.directAddedPrivateColumnCount]
    omega
  · left
    rw [Layout.Stage1.ApplicationInputs.outputColumn_value]
    have bound := index.isLt
    change index.val < 4 at bound
    omega
  · right
    rw [width] at upper
    simp only [Layout.Stage1.ApplicationInputs.localStart,
      Layout.Stage1.ApplicationInputs.witnessStart, Layout.Stage1.Spartan.privateColumnCount_eq] at lower
    exact ⟨by omega, upper⟩

private theorem application_column_inverse (application : Lifecycle.Stage1.Application.Program)
    (source : Nat) (allowed : ApplicationDirectSource.SourceAllowed application source) :
    (inverse (PerApplicationPackage.directAddedPrivateColumnCount application)).column?
      (ApplicationRelocation.column source) = some source := by
  rw [inverse, inverseRanges_eq]
  simp only [ApplicationRelocation.column, Layout.Stage1.Spartan.privateColumnCount_eq,
    Layout.Stage1.Wide.SourceOrder.privateColumns_eq]
  rcases application_column_range application source allowed with shared | ⟨lower, upper⟩
  · rw [if_pos (by omega)]
    apply MatrixProgram.SourceProjection.mapped_three_first_column?
    · simpa using MatrixProgram.SourceProjectionRange.column?_at ⟨0, 0, 19512839⟩ ⟨source, shared⟩
    · exact MatrixProgram.SourceProjectionRange.column?_eq_none_of_before _ _ (by dsimp; omega)
    · exact MatrixProgram.SourceProjectionRange.column?_eq_none_of_before _ _ (by dsimp; omega)
  · rw [if_neg (by omega)]
    apply MatrixProgram.SourceProjection.mapped_three_column?
    · exact MatrixProgram.SourceProjectionRange.column?_eq_none_of_after _ _ (by dsimp; omega)
    · exact MatrixProgram.SourceProjectionRange.column?_eq_none_of_after _ _ (by dsimp; omega)
    · have target : 27859260 + (source - 28784740) = 19646884 + (source - 20572364) := by omega
      rw [target]
      have recovered := MatrixProgram.SourceProjectionRange.column?_at
        ⟨19646884, 20572364, 8212655 + PerApplicationPackage.directAddedPrivateColumnCount application⟩
        ⟨source - 20572364, by dsimp; omega⟩
      simpa only [Nat.add_sub_of_le (show 20572364 ≤ source by omega)] using recovered

/-- Relocation recovers all application rows on the existing source support.
No unrelated private column is admitted by the inverse map. -/
theorem application_row_recovered (application : Lifecycle.Stage1.Application.Program)
    (rowStart : Nat) (before after : Rows.CompiledRow)
    (supported : before.toR1CS.VarsSatisfy (ApplicationDirectSource.SourceAllowed application))
    (emitted : (ApplicationRelocation.mapping rowStart).compiledRow before = .ok after) :
    (inverse (PerApplicationPackage.directAddedPrivateColumnCount application)).row? after.toR1CS =
      some before.toR1CS := by
  obtain ⟨_, equal, _⟩ := (ApplicationRelocation.mapping rowStart).compiledRow_correct before after emitted
  rw [equal]
  apply MatrixProgram.SourceProjection.row?_mapColumns_supported
  apply supported.mono
  intro source allowed
  simpa only [Map.columnValue, ApplicationRelocation.mapping, Except.toOption, Option.getD_some] using
    application_column_inverse application source allowed

end NightstreamFPrime.Export.Stage1.Wide.PhysicalMatrixRecovery
