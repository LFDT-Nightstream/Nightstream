import NightstreamFPrime.Export.Stage1.PerApplicationMatrixProgramSemantics
import NightstreamFPrime.Export.Stage1.PerApplicationPackageSourceRows

/-!
Constructs matrix source-row custody from one Lean-authored per-application
package. No caller supplies rows or selects an application relation.

This module does not claim package identity or production closure.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationPackageSourceCustody

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1

abbrev ApplicationProgram := Lifecycle.Stage1.Application.Program

def sourceRow (application : ApplicationProgram) : Nat → Option R1CS.Row :=
  PackageSourceRows.packageSourceRow?
    (PerApplicationPackage.package application)

theorem runningTransitionSourceRow?_eq_some
    (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (index : Fin (RunningTransitionDirectSource.program
      (PerApplicationMatrixProgramSemantics.relation application fits)).rowCount) :
    sourceRow application (RunningTransitionArithmetic.rowStart + index.val) =
      some (PerApplicationSourceProjection.basePackageRow application
        ((RunningTransitionDirectSource.program
          (PerApplicationMatrixProgramSemantics.relation application fits)).row
            index)) := by
  let relation := PerApplicationMatrixProgramSemantics.relation application fits
  let rows := (RunningTransitionArithmetic.canonicalPlan
    (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application)).rows
  have rowsLength : rows.length =
      (RunningTransitionDirectSource.program relation).rowCount := by
    rw [RunningTransitionDirectSource.program_rowCount relation,
      RunningTransitionArithmetic.Plan.rows_length,
      RunningTransitionArithmetic.canonicalPlan_rowCount relation]
  have rowIndices : rows.map Rows.CompiledRow.rowIndex =
      List.range' RunningTransitionArithmetic.rowStart
        (RunningTransitionDirectSource.program relation).rowCount := by
    calc
      _ = List.range'
          (RunningTransitionArithmetic.canonicalPlan
            (PerApplicationFixedPoint.logicalWidth application)
            (PerApplicationFixedPoint.publicFits application)).rowStart
          rows.length := PiCCSArithmetic.compilePacket_rowIndices _ _ _
      _ = List.range' RunningTransitionArithmetic.rowStart
          (RunningTransitionDirectSource.program relation).rowCount := by
        rw [rowsLength]
        rfl
  have included : ∀ row ∈ rows,
      row ∈ PerApplicationPackageSourceRows.baseRows := by
    intro row member
    rw [PerApplicationPackageSourceRows.baseRows, List.mem_append]
    apply Or.inr
    have rowsEq : rows =
        (RunningTransitionArithmetic.canonicalPlan Data.logicalWidth
          Data.publicFits).rows := by
      rfl
    rw [rowsEq] at member
    unfold Data.arithmeticRows
    simp only [List.mem_append]
    exact Or.inr member
  have exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn (RunningTransitionDirectSource.program relation).row := by
    calc
      _ = RunningTransitionDirectSource.sourceRows
          (PerApplicationFixedPoint.logicalWidth application)
          (PerApplicationFixedPoint.publicFits application) :=
        (RunningTransitionDirectSource.sourceRows_eq_canonicalRows).symm
      _ = List.ofFn (RunningTransitionDirectSource.program relation).row := by
        change _ = List.ofFn (fun position =>
          (RunningTransitionDirectSource.sourceRows
            (PerApplicationFixedPoint.logicalWidth application)
            (PerApplicationFixedPoint.publicFits application)).get position)
        exact (List.ofFn_get _).symm
  exact PerApplicationPackageSourceRows.indexedBasePackageSourceRow?_eq_some
    application rows rowsLength rowIndices included
    (RunningTransitionDirectSource.program relation).row exactRows index

theorem applicationSourceRow?_eq_some
    (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (index : Fin (ApplicationDirectSource.program application
      fits.package).rowCount) :
    sourceRow application
        (PerApplicationPackage.basePackage.layout.rowCount + index.val) =
      some ((ApplicationDirectSource.program application fits.package).row
        index) := by
  let rows := PerApplicationPackageSourceRows.applicationRows application
  have rowsLength : rows.length =
      (ApplicationDirectSource.program application fits.package).rowCount := by
    change rows.length = (ApplicationDirectSource.sourceRows application).length
    unfold rows PerApplicationPackageSourceRows.applicationRows
      ApplicationDirectSource.sourceRows
    rw [List.length_map]
  have rowIndices : rows.map Rows.CompiledRow.rowIndex =
      List.range' PerApplicationPackage.basePackage.layout.rowCount
        (ApplicationDirectSource.program application fits.package).rowCount := by
    rw [PerApplicationPackageSourceRows.applicationRows_rowIndices, rowsLength]
  have exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn (ApplicationDirectSource.program application fits.package).row := by
    change ApplicationDirectSource.sourceRows application =
      List.ofFn (fun position =>
        (ApplicationDirectSource.sourceRows application).get position)
    exact (List.ofFn_get _).symm
  exact
    PerApplicationPackageSourceRows.indexedApplicationPackageSourceRow?_eq_some
      application rows rowsLength rowIndices (fun _ member => member)
      (ApplicationDirectSource.program application fits.package).row exactRows
      index

theorem nextPreimageSourceRow?_eq_some
    (application : ApplicationProgram)
    (index : Fin NextPreimageDirectPlan.program.rowCount) :
    sourceRow application
        (PerApplicationPackage.nextPreimageRowStart application + index.val) =
      some (NextPreimageDirectPlan.program.row index) := by
  let rows := PerApplicationPackageSourceRows.nextPreimageRows application
  have rowsLength : rows.length = NextPreimageDirectPlan.program.rowCount := by
    simpa [rows, PerApplicationPackageSourceRows.nextPreimageRows] using
      NextPreimagePackage.compiledRows_length
        (PerApplicationPackage.nextPreimageRowStart application)
  have rowIndices : rows.map Rows.CompiledRow.rowIndex =
      List.range' (PerApplicationPackage.nextPreimageRowStart application)
        NextPreimageDirectPlan.program.rowCount := by
    rw [PerApplicationPackageSourceRows.nextPreimageRows_rowIndices,
      NextPreimageDirectPlan.program_rowCount]
  have exactRows : rows.map Rows.CompiledRow.toR1CS =
      List.ofFn NextPreimageDirectPlan.program.row := by
    calc
      _ = NextPreimagePackage.sourceRows := by
        simpa [rows, PerApplicationPackageSourceRows.nextPreimageRows] using
          (NextPreimagePackage.compiledRows_toR1CS
            (PerApplicationPackage.nextPreimageRowStart application)).trans
              NextPreimagePackage.sourceRows_eq.symm
      _ = List.ofFn NextPreimageDirectPlan.program.row := by
        change _ = List.ofFn fun position =>
          NextPreimageDirectPlan.sourceRows.get position
        exact (List.ofFn_get _).symm
  exact PerApplicationPackageSourceRows.indexedPackageSourceRow?_eq_some
    application rows rowsLength rowIndices NextPreimageDirectPlan.program.row
    exactRows (fun row => row)
    (PerApplicationPackageSourceRows.nextPreimagePackageSourceRow?_eq_some
      application) index

theorem custody (application : ApplicationProgram)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application) :
    PerApplicationMatrixProgramSemantics.SourceCustody application fits
      (sourceRow application) := by
  let relation := PerApplicationMatrixProgramSemantics.relation application fits
  refine {
    piCcsOrdinary := ?_
    pilotOrdinary := ?_
    samplerOrdinary := ?_
    piDecPublic := ?_
    piDecCommitment := ?_
    piDecEvalK := ?_
    piDecEvalA := ?_
    runningTransition := ?_
    applicationRows := ?_
    nextPreimage := ?_ }
  · intro index sourceIndex selected
    exact PerApplicationPackageSourceRows.piCcsPackageSourceRow?_eq_some
      application relation index sourceIndex selected
  · intro index
    exact PerApplicationPackageSourceRows.pilotPackageSourceRowAt?_eq_some
      application index
  · intro index sourceIndex selected
    exact PerApplicationPackageSourceRows.samplerPackageSourceRow?_eq_some
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application)
      application index sourceIndex selected
  · intro index
    exact PerApplicationPackageSourceRows.piDecPublicPackageSourceRow?_eq_some
      application relation index
  · intro index
    exact
      PerApplicationPackageSourceRows.piDecCommitmentPackageSourceRow?_eq_some
        application relation index
  · intro index
    exact PerApplicationPackageSourceRows.piDecEvalKPackageSourceRow?_eq_some
      application relation index
  · intro index
    exact PerApplicationPackageSourceRows.piDecEvalAPackageSourceRow?_eq_some
      application relation index
  · intro index
    exact runningTransitionSourceRow?_eq_some application fits index
  · intro index
    exact applicationSourceRow?_eq_some application fits index
  · intro index
    exact nextPreimageSourceRow?_eq_some application index

end NightstreamFPrime.Export.Stage1.PerApplicationPackageSourceCustody
