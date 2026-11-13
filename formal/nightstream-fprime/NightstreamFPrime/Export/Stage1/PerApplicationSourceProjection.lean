import NightstreamFPrime.Export.Stage1.PerApplicationPackage
import NightstreamFPrime.Layout.MatrixProgram.SourceProjection

namespace NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection

open NightstreamFPrime.Layout.R1CS (mapCombinationColumns mapRowColumns mapCombinationColumns_zero mapCombinationColumns_const mapCombinationColumns_one mapCombinationColumns_ofVar mapCombinationColumns_add mapCombinationColumns_scale)

open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle

abbrev Program := Lifecycle.Stage1.Application.Program

@[simp] private theorem basePackage_constantColumn_eq :
    PerApplicationPackage.basePackage.layout.constantColumn = 11414924 := by
  rw [PerApplicationPackage.basePackage, Data.circuitPackage_layout]
  rfl

def finalConstant (program : Program) : Nat :=
  PerApplicationPackage.basePackage.layout.constantColumn +
    PerApplicationPackage.addedPrivateColumnCount program

def directFinalConstant (program : Program) : Nat :=
  11414924 + PerApplicationPackage.directAddedPrivateColumnCount program

theorem directFinalConstant_eq_finalConstant (program : Program) :
    directFinalConstant program = finalConstant program := by
  unfold directFinalConstant finalConstant
  rw [PerApplicationPackage.directAddedPrivateColumnCount_eq_addedPrivateColumnCount,
    basePackage_constantColumn_eq]

@[csimp] theorem finalConstant_eq_directFinalConstant :
    @finalConstant = @directFinalConstant := by
  funext program
  exact (directFinalConstant_eq_finalConstant program).symm

def basePrivateRangeReference (_delay : Unit := ()) : SourceProjectionRange :=
  ⟨0, 0, PerApplicationPackage.basePackage.layout.constantColumn⟩

def basePrivateRange : SourceProjectionRange := ⟨0, 0, 11414924⟩

theorem basePrivateRange_eq_reference :
    basePrivateRange = basePrivateRangeReference () := by
  simp [basePrivateRange, basePrivateRangeReference]

def baseSuffixRange (program : Program) : SourceProjectionRange :=
  ⟨finalConstant program,
    PerApplicationPackage.basePackage.layout.constantColumn,
    PerApplicationPackage.basePackage.layout.totalColumnCount -
      PerApplicationPackage.basePackage.layout.constantColumn⟩

def directBaseSuffixRange (program : Program) : SourceProjectionRange :=
  ⟨directFinalConstant program, 11414924, 329⟩

theorem directBaseSuffixRange_eq_baseSuffixRange (program : Program) :
    directBaseSuffixRange program = baseSuffixRange program := by
  unfold directBaseSuffixRange baseSuffixRange
  rw [directFinalConstant_eq_finalConstant,
    basePackage_constantColumn_eq,
    PerApplicationPackage.basePackage_totalColumnCount_eq]

@[csimp] theorem baseSuffixRange_eq_directBaseSuffixRange :
    @baseSuffixRange = @directBaseSuffixRange := by
  funext program
  exact (directBaseSuffixRange_eq_baseSuffixRange program).symm

def base (program : Program) : SourceProjection :=
  .mapped [basePrivateRange, baseSuffixRange program]

def pilotInputRange : SourceProjectionRange :=
  ⟨0, 0, PackageOrder.pilotInputPrivateColumnCount⟩

def pilotPrivateRange : SourceProjectionRange :=
  ⟨PackageOrder.pilotInputPrivateColumnCount + PackageOrder.proofInputColumnCount,
    PackageOrder.pilotInputPrivateColumnCount,
    PackageOrder.pilotPrivateColumnCount - PackageOrder.pilotInputPrivateColumnCount⟩

def pilotSuffixRange (program : Program) : SourceProjectionRange :=
  ⟨finalConstant program, PilotPackageOrder.privateColumnCount,
    PilotPackageOrder.packageColumnCount - PilotPackageOrder.privateColumnCount⟩

def pilot (program : Program) : SourceProjection :=
  .mapped [pilotInputRange, pilotPrivateRange, pilotSuffixRange program]

def application : SourceProjection := .identity

theorem base_column (program : Program) (column : Nat)
    (bound : column < PerApplicationPackage.basePackage.layout.totalColumnCount) :
    (base program).column?
        (PerApplicationPackage.shiftColumn program column) = some column := by
  by_cases privateColumn :
      column < PerApplicationPackage.basePackage.layout.constantColumn
  · have shifted := PerApplicationPackage.shiftColumn_private program column
      privateColumn
    rw [basePackage_constantColumn_eq] at privateColumn
    have first : basePrivateRange.column? column = some column := by
      simpa [basePrivateRange] using
        (SourceProjectionRange.column?_at basePrivateRange
          ⟨column, by simpa [basePrivateRange] using privateColumn⟩)
    have second : (baseSuffixRange program).column? column = none := by
      apply SourceProjectionRange.column?_eq_none_of_before
      simp [baseSuffixRange, finalConstant]
      omega
    rw [shifted]
    simp [base, SourceProjection.column?, first, second]
  · have suffixColumn :
        PerApplicationPackage.basePackage.layout.constantColumn ≤ column :=
      Nat.le_of_not_gt privateColumn
    have shifted := PerApplicationPackage.shiftColumn_constantOrPublic
      program column suffixColumn
    rw [basePackage_constantColumn_eq] at suffixColumn
    let offset := column -
      PerApplicationPackage.basePackage.layout.constantColumn
    have offsetBound : offset < (baseSuffixRange program).count := by
      rw [PerApplicationPackage.basePackage_totalColumnCount_eq] at bound
      simp [offset, baseSuffixRange, basePackage_constantColumn_eq]
      omega
    have shiftedEq :
        column + PerApplicationPackage.addedPrivateColumnCount program =
          (baseSuffixRange program).packageStart + offset := by
      simp [baseSuffixRange, finalConstant, offset,
        basePackage_constantColumn_eq]
      omega
    have sourceEq :
        (baseSuffixRange program).sourceStart + offset = column := by
      simp [baseSuffixRange, offset, basePackage_constantColumn_eq]
      omega
    have first : basePrivateRange.column?
        (column + PerApplicationPackage.addedPrivateColumnCount program) = none := by
      apply SourceProjectionRange.column?_eq_none_of_after
      simp [basePrivateRange]
      omega
    have second : (baseSuffixRange program).column?
        (column + PerApplicationPackage.addedPrivateColumnCount program) =
          some column := by
      rw [shiftedEq]
      simpa [sourceEq] using
        (SourceProjectionRange.column?_at (baseSuffixRange program)
          ⟨offset, offsetBound⟩)
    rw [shifted]
    simp [base, SourceProjection.column?, first, second]

private theorem pilot_column_input (program : Program) (column : Nat)
    (inputColumn : column < PackageOrder.pilotInputPrivateColumnCount) :
    (pilot program).column?
        (PerApplicationPackage.shiftColumn program
          (PackageOrder.liftPilotColumn column)) = some column := by
  have lifted : PackageOrder.liftPilotColumn column = column := by
    simp [PackageOrder.liftPilotColumn, inputColumn]
  have belowBase :
      column < PerApplicationPackage.basePackage.layout.constantColumn := by
    norm_num [PerApplicationPackage.basePackage,
      Data.circuitPackage_layout, Data.physicalLayout,
      PackageOrder.pilotInputPrivateColumnCount, PackageOrder.constantColumn]
      at inputColumn ⊢
    omega
  have first : pilotInputRange.column? column = some column := by
    simpa [pilotInputRange] using
      (SourceProjectionRange.column?_at pilotInputRange
        ⟨column, by simpa [pilotInputRange] using inputColumn⟩)
  have second : pilotPrivateRange.column? column = none := by
    apply SourceProjectionRange.column?_eq_none_of_before
    norm_num [pilotPrivateRange, PackageOrder.pilotInputPrivateColumnCount,
      PackageOrder.proofInputColumnCount] at inputColumn ⊢
    omega
  have third : (pilotSuffixRange program).column? column = none := by
    have before : column < (pilotSuffixRange program).packageStart := by
      change column < finalConstant program
      exact Nat.lt_of_lt_of_le belowBase (by
        unfold finalConstant
        omega)
    exact SourceProjectionRange.column?_eq_none_of_before
      (pilotSuffixRange program) column before
  have shifted := PerApplicationPackage.shiftColumn_private program column
    belowBase
  rw [lifted, shifted]
  simpa only [pilot] using
    SourceProjection.mapped_three_first_column? pilotInputRange
      pilotPrivateRange (pilotSuffixRange program) column column first second
      third

private theorem pilot_column_private (program : Program) (column : Nat)
    (inputLe : PackageOrder.pilotInputPrivateColumnCount ≤ column)
    (privateColumn : column < PackageOrder.pilotPrivateColumnCount) :
    (pilot program).column?
        (PerApplicationPackage.shiftColumn program
          (PackageOrder.liftPilotColumn column)) = some column := by
  have notInput : ¬ column < PackageOrder.pilotInputPrivateColumnCount := by omega
  have lifted : PackageOrder.liftPilotColumn column =
      column + PackageOrder.proofInputColumnCount := by
    simp only [PackageOrder.liftPilotColumn, ite_eq_right notInput, ite_eq_left privateColumn]
  have belowBase :
      column + PackageOrder.proofInputColumnCount <
        PerApplicationPackage.basePackage.layout.constantColumn := by
    norm_num [PerApplicationPackage.basePackage,
      Data.circuitPackage_layout, Data.physicalLayout,
      PackageOrder.pilotPrivateColumnCount, PackageOrder.proofInputColumnCount,
      PackageOrder.constantColumn] at privateColumn ⊢
    omega
  let offset := column - PackageOrder.pilotInputPrivateColumnCount
  have offsetBound : offset < pilotPrivateRange.count := by
    norm_num [offset, pilotPrivateRange, PackageOrder.pilotPrivateColumnCount,
      PackageOrder.pilotInputPrivateColumnCount] at inputLe privateColumn ⊢
    omega
  have packageEq : column + PackageOrder.proofInputColumnCount =
      pilotPrivateRange.packageStart + offset := by
    norm_num [pilotPrivateRange, offset, PackageOrder.pilotInputPrivateColumnCount,
      PackageOrder.proofInputColumnCount] at inputLe ⊢
    omega
  have sourceEq : pilotPrivateRange.sourceStart + offset = column := by
    norm_num [pilotPrivateRange, offset, PackageOrder.pilotInputPrivateColumnCount]
      at inputLe ⊢
    omega
  have first : pilotInputRange.column?
      (column + PackageOrder.proofInputColumnCount) = none := by
    apply SourceProjectionRange.column?_eq_none_of_after
    norm_num [pilotInputRange, PackageOrder.pilotInputPrivateColumnCount,
      PackageOrder.proofInputColumnCount] at inputLe ⊢
    omega
  have second : pilotPrivateRange.column?
      (column + PackageOrder.proofInputColumnCount) = some column := by
    rw [packageEq]
    simpa [sourceEq] using
      (SourceProjectionRange.column?_at pilotPrivateRange
        ⟨offset, offsetBound⟩)
  have third : (pilotSuffixRange program).column?
      (column + PackageOrder.proofInputColumnCount) = none := by
    have before : column + PackageOrder.proofInputColumnCount <
        (pilotSuffixRange program).packageStart := by
      norm_num [pilotSuffixRange, finalConstant,
        PerApplicationPackage.basePackage, Data.circuitPackage_layout,
        Data.physicalLayout, PackageOrder.constantColumn,
        PackageOrder.pilotPrivateColumnCount, PackageOrder.proofInputColumnCount]
        at privateColumn ⊢
      omega
    exact SourceProjectionRange.column?_eq_none_of_before
      (pilotSuffixRange program) _ before
  have shifted := PerApplicationPackage.shiftColumn_private program
    (column + PackageOrder.proofInputColumnCount) belowBase
  rw [lifted, shifted]
  simpa only [pilot] using
    SourceProjection.mapped_three_second_column? pilotInputRange
      pilotPrivateRange (pilotSuffixRange program)
      (column + PackageOrder.proofInputColumnCount) column first second third

private theorem pilot_column_suffix (program : Program) (column : Nat)
    (privateLe : PackageOrder.pilotPrivateColumnCount ≤ column)
    (bound : column < PilotPackageOrder.packageColumnCount) :
    (pilot program).column?
        (PerApplicationPackage.shiftColumn program
          (PackageOrder.liftPilotColumn column)) = some column := by
  have notInput : ¬ column < PackageOrder.pilotInputPrivateColumnCount := by
    norm_num [PackageOrder.pilotInputPrivateColumnCount,
      PackageOrder.pilotPrivateColumnCount] at privateLe ⊢
    omega
  have notPrivate : ¬ column < PackageOrder.pilotPrivateColumnCount := by omega
  have lifted : PackageOrder.liftPilotColumn column =
      PerApplicationPackage.basePackage.layout.constantColumn +
        (column - PackageOrder.pilotPrivateColumnCount) := by
    unfold PackageOrder.liftPilotColumn
    rw [ite_eq_right notInput, ite_eq_right notPrivate]
    rfl
  let offset := column - PackageOrder.pilotPrivateColumnCount
  have offsetBound : offset < 325 := by
    norm_num [offset, PackageOrder.pilotPrivateColumnCount,
      PilotPackageOrder.packageColumnCount_value] at privateLe bound ⊢
    omega
  have sourceEq : PilotPackageOrder.privateColumnCount + offset = column := by
    norm_num [offset, PackageOrder.pilotPrivateColumnCount,
      PilotPackageOrder.privateColumnCount_value] at privateLe ⊢
    omega
  have baseSuffix :
      PerApplicationPackage.basePackage.layout.constantColumn ≤
        PerApplicationPackage.basePackage.layout.constantColumn + offset := by
    omega
  have first : pilotInputRange.column?
      (finalConstant program + offset) = none := by
    have after : pilotInputRange.packageStart + pilotInputRange.count ≤
        finalConstant program + offset := by
      have inputEndLeBase :
          pilotInputRange.packageStart + pilotInputRange.count ≤
            PerApplicationPackage.basePackage.layout.constantColumn := by
        norm_num [pilotInputRange, PerApplicationPackage.basePackage,
          Data.circuitPackage_layout, Data.physicalLayout,
          PackageOrder.constantColumn, PackageOrder.pilotInputPrivateColumnCount]
      exact inputEndLeBase.trans (by
        unfold finalConstant
        omega)
    exact SourceProjectionRange.column?_eq_none_of_after pilotInputRange _ after
  have second : pilotPrivateRange.column?
      (finalConstant program + offset) = none := by
    apply SourceProjectionRange.column?_eq_none_of_after
    norm_num [pilotPrivateRange, finalConstant,
      PerApplicationPackage.basePackage, Data.circuitPackage_layout,
      Data.physicalLayout, PackageOrder.constantColumn,
      PackageOrder.pilotPrivateColumnCount, PackageOrder.pilotInputPrivateColumnCount,
      PackageOrder.proofInputColumnCount]
    omega
  have third : (pilotSuffixRange program).column?
      (finalConstant program + offset) = some column := by
    have selected := SourceProjectionRange.column?_at
      (pilotSuffixRange program) ⟨offset, by
        simpa [pilotSuffixRange] using! offsetBound⟩
    simpa [pilotSuffixRange, sourceEq] using selected
  have shifted := PerApplicationPackage.shiftColumn_constantOrPublic
    program _ baseSuffix
  have packageEq :
      PerApplicationPackage.basePackage.layout.constantColumn + offset +
          PerApplicationPackage.addedPrivateColumnCount program =
        finalConstant program + offset := by
    simp [finalConstant]
    omega
  rw [lifted, shifted]
  rw [packageEq]
  simpa only [pilot] using
    SourceProjection.mapped_three_column? pilotInputRange pilotPrivateRange
      (pilotSuffixRange program) (finalConstant program + offset) column first
      second third

theorem pilot_column (program : Program) (column : Nat)
    (bound : column < PilotPackageOrder.packageColumnCount) :
    (pilot program).column?
        (PerApplicationPackage.shiftColumn program
          (PackageOrder.liftPilotColumn column)) = some column := by
  by_cases inputColumn : column < PackageOrder.pilotInputPrivateColumnCount
  · exact pilot_column_input program column inputColumn
  · have inputLe : PackageOrder.pilotInputPrivateColumnCount ≤ column :=
      Nat.le_of_not_gt inputColumn
    by_cases privateColumn : column < PackageOrder.pilotPrivateColumnCount
    · exact pilot_column_private program column inputLe privateColumn
    · exact pilot_column_suffix program column
        (Nat.le_of_not_gt privateColumn) bound

def basePackageRow (program : Program) (row : R1CS.Row) : R1CS.Row :=
  mapRowColumns (PerApplicationPackage.shiftColumn program) row

def pilotPackageRow (program : Program) (row : R1CS.Row) : R1CS.Row :=
  mapRowColumns (PerApplicationPackage.shiftColumn program)
    (mapRowColumns PackageOrder.liftPilotColumn row)

theorem base_row (program : Program) (row : R1CS.Row)
    (bounded : row.VarsBelow
      PerApplicationPackage.basePackage.layout.totalColumnCount) :
    (base program).row? (basePackageRow program row) = some row := by
  apply SourceProjection.row?_mapColumns _ _ _ _ bounded
  exact fun column => base_column program column.val column.isLt

theorem pilot_row (program : Program) (row : R1CS.Row)
    (bounded : row.VarsBelow PilotPackageOrder.packageColumnCount) :
    (pilot program).row? (pilotPackageRow program row) = some row := by
  unfold pilotPackageRow
  rw [show mapRowColumns (PerApplicationPackage.shiftColumn program)
        (mapRowColumns PackageOrder.liftPilotColumn row) =
      mapRowColumns
        (fun column => PerApplicationPackage.shiftColumn program
          (PackageOrder.liftPilotColumn column)) row by
    cases row; simp [mapRowColumns, mapCombinationColumns, List.map_map,
      Function.comp_def]]
  apply SourceProjection.row?_mapColumns _ _ _ _ bounded
  exact fun column => pilot_column program column.val column.isLt

@[simp] theorem application_row (row : R1CS.Row) :
    application.row? row = some row := by
  exact SourceProjection.identity_row? row

end NightstreamFPrime.Export.Stage1.PerApplicationSourceProjection
