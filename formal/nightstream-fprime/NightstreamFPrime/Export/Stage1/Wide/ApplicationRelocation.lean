import NightstreamFPrime.Export.Stage1.Wide.PhysicalRelabel
import NightstreamFPrime.Export.Stage1.PerApplicationPackage

/-! Move the one compiled application slice into the wide physical archive.
The compiler runs at its canonical source coordinates; relocation preserves its exact syntax. -/

namespace NightstreamFPrime.Export.Stage1.Wide.ApplicationRelocation

open NightstreamFPrime.Circuit NightstreamFPrime.Layout NightstreamFPrime.Export.Package

/-- Application-local coordinates move together; shared state coordinates stay fixed. -/
def column (index : Nat) : Nat :=
  if index < Layout.Stage1.Spartan.privateColumnCount then index
  else Layout.Stage1.Wide.SourceOrder.privateColumns + (index - Layout.Stage1.Spartan.privateColumnCount)

def row (start index : Nat) : Nat := start + (index - Data.physicalLayout.rowCount)

def mapping (start : Nat) : PhysicalRelabel.Map := {
  column := fun index => .ok (column index)
  row := fun index => if Data.physicalLayout.rowCount ≤ index then .ok (row start index)
    else .error "row precedes the application interval" }

theorem row_injective (start left right target : Nat)
    (first : (mapping start).row left = .ok target)
    (second : (mapping start).row right = .ok target) : left = right := by
  simp only [mapping] at first second
  split_ifs at first second <;> simp_all [row] <;> omega

private def expression (value : Expr) : Expr := CompactRows.renameExpr column value

private def hint : Hint → Hint
  | .bit source index => .bit (expression source) index
  | .inverseOrZero source => .inverseOrZero (expression source)
  | .quotientFive source => .quotientFive (expression source)
  | .remainderFive source => .remainderFive (expression source)

private def batch (value : WitnessBatch) : WitnessBatch :=
  ⟨column value.start, value.recipes.map expression, value.hints.map hint⟩

private def combination (value : SparseCombination) : SparseCombination :=
  ⟨value.constant, value.terms.map fun term => ⟨column term.column, term.coefficient⟩⟩

def instruction (start : Nat) (value : WitnessInstruction) : WitnessInstruction :=
  ⟨row start value.rowIndex, column value.target, combination value.a, combination value.b⟩

def assertion (start : Nat) (value : SparseRow) : SparseRow :=
  ⟨row start value.rowIndex, combination value.a, combination value.b, combination value.c⟩

private theorem combination_map (start : Nat) (value : SparseCombination) :
    (mapping start).combination value = .ok (combination value) := by
  rcases value with ⟨constant, terms⟩
  have mapped : terms.mapM ((mapping start).term) =
      .ok (terms.map fun term => SparseTerm.mk (column term.column) term.coefficient) := by
    induction terms with
    | nil => rfl
    | cons head tail ih =>
        simp only [List.mapM_cons, ih, List.map_cons]
        rfl
  simp only [PhysicalRelabel.Map.combination, mapped]
  rfl

theorem instruction_map (start : Nat) (value : WitnessInstruction)
    (bound : Data.physicalLayout.rowCount ≤ value.rowIndex) :
    (mapping start).instruction value = .ok (instruction start value) := by
  simp only [PhysicalRelabel.Map.instruction, combination_map, mapping, if_pos bound]
  rfl

theorem assertion_map (start : Nat) (value : SparseRow)
    (bound : Data.physicalLayout.rowCount ≤ value.rowIndex) :
    (mapping start).assertion value = .ok (assertion start value) := by
  simp only [PhysicalRelabel.Map.assertion, combination_map, mapping, if_pos bound]
  rfl

/-- No application-specific compiler case or second lowering is selected here. -/
def plan (program : Lifecycle.Stage1.Application.Program) (start : Nat) : Stage1.ApplicationPackage.Plan :=
  let reference := PerApplicationPackage.directApplicationPlan program
  { reference with
    inputColumns := reference.inputColumns.map column
    witnessColumns := reference.witnessColumns.map column
    outputColumns := reference.outputColumns.map column
    privateStart := column reference.privateStart
    rowStart := start
    witnessBatches := reference.witnessBatches.map batch
    witnessInstructions := reference.witnessInstructions.map (instruction start)
    assertionRows := reference.assertionRows.map (assertion start) }

theorem plan_counts (program : Lifecycle.Stage1.Application.Program) (start : Nat) :
    (plan program start).privateCount = (PerApplicationPackage.applicationPlan program).privateCount ∧
      (plan program start).rowCount = (PerApplicationPackage.applicationPlan program).rowCount := by
  simp [plan, PerApplicationPackage.directApplicationPlan_eq_applicationPlan]

end NightstreamFPrime.Export.Stage1.Wide.ApplicationRelocation
