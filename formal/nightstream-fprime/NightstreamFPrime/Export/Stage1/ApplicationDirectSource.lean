import NightstreamFPrime.Export.Stage1.PerApplicationPackage
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
import NightstreamFPrime.Layout.R1CS.Support

/-!
Owns indexed access to the exact Lean-lowered rows of one verifier-selected
Stage 1 application. It does not select an application or retained low-norm
coordinates.
-/

namespace NightstreamFPrime.Export.Stage1.ApplicationDirectSource

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle

def sourceRows (application : Lifecycle.Stage1.Application.Program) :
    List R1CS.Row :=
  (ApplicationPackage.compiledRows application
    (ApplicationPackage.productionColumns application)
    (Layout.Stage1.ApplicationInputs.localStart application)
    PerApplicationPackage.basePackage.layout.rowCount).map
      Rows.CompiledRow.toR1CS

/-- Exclusive source-column bound after application-local R1CS lowering. -/
def sourceWidth (application : Lifecycle.Stage1.Application.Program) : Nat :=
  ApplicationPackage.r1csFreshStart application
      (ApplicationPackage.productionColumns application)
      (Layout.Stage1.ApplicationInputs.localStart application) +
    R1CS.totalFreshCount
      (ApplicationPackage.constraints application
        (ApplicationPackage.productionColumns application)
        (Layout.Stage1.ApplicationInputs.localStart application))

/-- Exact source columns that one selected application is permitted to read. -/
def SourceAllowed (application : Lifecycle.Stage1.Application.Program)
    (column : Nat) : Prop :=
  (∃ index : Lifecycle.Stage1.Application.StateIndex,
    column = Layout.Stage1.ApplicationInputs.inputColumn index) ∨
  (∃ index : Fin application.witnessWordCount,
    column = Layout.Stage1.ApplicationInputs.witnessColumn index) ∨
  (∃ index : Lifecycle.Stage1.Application.StateIndex,
    column = Layout.Stage1.ApplicationInputs.outputColumn index) ∨
  (Layout.Stage1.ApplicationInputs.localStart application ≤ column ∧
    column < sourceWidth application)

theorem sourceRows_length_eq_plan
    (application : Lifecycle.Stage1.Application.Program) :
    (sourceRows application).length =
      (PerApplicationPackage.applicationPlan application).rowCount := by
  simp [sourceRows, PerApplicationPackage.applicationPlan,
    ApplicationPackage.productionPlan, ApplicationPackage.ofProgram]

/-- Every source row is confined to the selected application's exact
caller-and-lowering interval. -/
theorem sourceRows_varsBelow
    (application : Lifecycle.Stage1.Application.Program) :
    ∀ row ∈ sourceRows application, row.VarsBelow (sourceWidth application) := by
  rw [sourceRows, ApplicationPackage.ofProgram_compiledRows_toR1CS]
  have assumptions := application.assumptions
    (Layout.Stage1.ApplicationInputs.interface application)
    (Layout.Stage1.ApplicationInputs.localStart application) (fun _ => 0)
    (Layout.Stage1.ApplicationInputs.externalBelow application)
  have scope := application.scope
    (Layout.Stage1.ApplicationInputs.interface application)
    (Layout.Stage1.ApplicationInputs.localStart application) (fun _ => 0)
    (Layout.Stage1.ApplicationInputs.externalBelow application)
    assumptions
  simpa [sourceWidth, ApplicationPackage.productionColumns] using
    (R1CS.lowerConstraints_rows_varsBelow
      (ApplicationPackage.constraints application
        (ApplicationPackage.productionColumns application)
        (Layout.Stage1.ApplicationInputs.localStart application))
      (ApplicationPackage.r1csFreshStart application
        (ApplicationPackage.productionColumns application)
        (Layout.Stage1.ApplicationInputs.localStart application)) scope)

/-- Lowering preserves the exact application-owned source set. No unrelated
prior private column can enter an application row. -/
theorem sourceRows_varsSatisfy
    (application : Lifecycle.Stage1.Application.Program) :
    ∀ row ∈ sourceRows application, row.VarsSatisfy (SourceAllowed application) := by
  let interface := Layout.Stage1.ApplicationInputs.interface application
  let localStart := Layout.Stage1.ApplicationInputs.localStart application
  let columns := ApplicationPackage.productionColumns application
  let constraints := ApplicationPackage.constraints application columns localStart
  let freshStart := ApplicationPackage.r1csFreshStart application columns localStart
  have assumptions := application.assumptions interface localStart (fun _ => 0)
    (Layout.Stage1.ApplicationInputs.externalBelow application)
  have inputSupport : Lifecycle.Stage1.Application.InputsSupported interface
      localStart (SourceAllowed application) := by
    refine {
      input := fun index => ?_
      witness := fun index => ?_
      output := fun index => ?_ }
    · exact Or.inl ⟨index, rfl⟩
    · exact Or.inr (Or.inl ⟨index, rfl⟩)
    · exact Or.inr (Or.inr (Or.inl ⟨index, rfl⟩))
  have scope : ∀ expression ∈ constraints,
      expression.VarsSatisfy (SourceAllowed application) := by
    apply application.support interface localStart (fun _ => 0)
      (SourceAllowed application) assumptions inputSupport
    intro index lower upper
    exact Or.inr (Or.inr (Or.inr ⟨lower, by
      unfold sourceWidth
      have beforeFresh : index < ApplicationPackage.r1csFreshStart application
          (ApplicationPackage.productionColumns application)
          (Layout.Stage1.ApplicationInputs.localStart application) := upper
      omega⟩))
  have lowered := R1CS.lowerConstraints_rows_varsSatisfy constraints freshStart
    (SourceAllowed application) scope
  rw [sourceRows, ApplicationPackage.ofProgram_compiledRows_toR1CS]
  intro row member
  apply R1CS.Row.VarsSatisfy.mono row (lowered row member)
  intro index support
  rcases support with source | ⟨lower, upper⟩
  · exact source
  · exact Or.inr (Or.inr (Or.inr ⟨by
      unfold freshStart ApplicationPackage.r1csFreshStart at lower
      omega, by
      simpa [sourceWidth, freshStart, constraints, columns, localStart] using
        upper⟩))

theorem sourceRows_rowCount_le
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationPackage.FitsTwoPow28 application) :
    (sourceRows application).length ≤ 2 ^ Lifecycle.cubeVariables := by
  have rows := fits.rows
  rw [PerApplicationPackage.package_rowCount] at rows
  rw [sourceRows_length_eq_plan]
  omega

/-- Proof-oriented indexed access to the exact canonical application rows. -/
def program (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationPackage.FitsTwoPow28 application) :
    OrdinarySourcePlan.Program (sourceWidth application) where
  rowCount := (sourceRows application).length
  rowCount_le := sourceRows_rowCount_le application fits
  row := fun index => (sourceRows application).get index
  bounded := fun index =>
    sourceRows_varsBelow application _ (List.get_mem _ index)

@[simp] theorem program_rowCount
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationPackage.FitsTwoPow28 application) :
    (program application fits).rowCount =
      (PerApplicationPackage.applicationPlan application).rowCount := by
  exact sourceRows_length_eq_plan application

theorem program_holds_iff_rowsHold
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationPackage.FitsTwoPow28 application) (env : Env) :
    (program application fits).Holds env ↔
      R1CS.RowsHold env (sourceRows application) := by
  change (∀ index, ((sourceRows application).get index).Holds env) ↔
    R1CS.RowsHold env (sourceRows application)
  constructor
  · intro holds row member
    rcases List.mem_iff_get.mp member with ⟨index, rfl⟩
    exact holds index
  · intro holds index
    exact holds _ (List.get_mem _ index)

/-- Exact satisfaction of the direct source rows implies the selected Lean
application transition for an arbitrary source assignment. -/
theorem rowsHold_implies_applicationHolds
    (application : Lifecycle.Stage1.Application.Program) (env : Env)
    (rows : R1CS.RowsHold env (sourceRows application)) :
    Lifecycle.Stage1.Application.Holds application.step
      (Layout.Stage1.ApplicationInputs.interface application)
      (Layout.Stage1.ApplicationInputs.localStart application) env := by
  let columns := ApplicationPackage.productionColumns application
  let privateStart := Layout.Stage1.ApplicationInputs.localStart application
  have loweredRows : R1CS.RowsHold env
      (R1CS.lowerConstraints
        (ApplicationPackage.constraints application columns privateStart)
        (ApplicationPackage.r1csFreshStart application columns privateStart)).rows := by
    rw [← ApplicationPackage.ofProgram_compiledRows_toR1CS]
    exact rows
  have flattened : holdsFlat env
      (ApplicationPackage.operations application columns privateStart) :=
    R1CS.lowerConstraints_sound env
      (ApplicationPackage.constraints application columns privateStart)
      (ApplicationPackage.r1csFreshStart application columns privateStart)
      loweredRows
  have assumptions := application.assumptions
    (Layout.Stage1.ApplicationInputs.interface application) privateStart env
    (Layout.Stage1.ApplicationInputs.externalBelow application)
  exact application.soundness
    (Layout.Stage1.ApplicationInputs.interface application) privateStart env
    assumptions (holdsFlat_implies_holds env
      (ApplicationPackage.operations application columns privateStart) flattened)

end NightstreamFPrime.Export.Stage1.ApplicationDirectSource
