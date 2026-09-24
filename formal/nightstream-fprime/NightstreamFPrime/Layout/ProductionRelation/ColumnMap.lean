import NightstreamFPrime.Layout.ProductionRelation.PlanComposition
import NightstreamFPrime.Layout.ProductionRelation.SparseEvaluation

/-! Rename retained coordinates without changing row equations. No injectivity
assumption is needed for evaluation; a layout separately proves that retained
coordinates do not collide. -/

namespace NightstreamFPrime.Layout.ProductionRelation

open NightstreamFPrime.Spec
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

namespace SparseForm

def mapColumns {source target : Nat} (column : Fin source → Fin target)
    (form : SparseForm source) : SparseForm target :=
  ⟨form.entries.map fun entry => ⟨column entry.column, entry.coefficient⟩⟩

theorem mapColumns_eval {source target : Nat} (column : Fin source → Fin target)
    (form : SparseForm source) (assignment : Assignment F target) :
    (mapColumns column form).eval assignment = form.eval (assignment ∘ column) := by
  rw [← evalSparse_eq_eval, ← evalSparse_eq_eval]
  simp only [mapColumns, evalSparse, List.foldl_map, Function.comp_def]

theorem mapColumns_length {source target : Nat} (column : Fin source → Fin target)
    (form : SparseForm source) : (mapColumns column form).entries.length = form.entries.length := by
  simp [mapColumns]

/-- Rename stored entries only when each source has a mapping certificate. -/
def mapColumnsChecked {source target : Nat} {predicate : Fin source → Prop}
    (column : ∀ source, predicate source → Fin target) (form : SparseForm source)
    (supported : ∀ entry ∈ form.entries, predicate entry.column) : SparseForm target :=
  ⟨form.entries.pmap (fun entry live => ⟨column entry.column live, entry.coefficient⟩) supported⟩

end SparseForm

namespace Plan

def mapColumns {source target : Nat} (column : Fin source → Fin target)
    (plan : ProductionRelation.Plan source) : ProductionRelation.Plan target where
  rowCount := plan.rowCount
  rowCount_le := plan.rowCount_le
  forms := fun row port => (plan.forms row port).mapColumns column

@[simp] theorem mapColumns_rowCount {source target : Nat} (column : Fin source → Fin target)
    (plan : ProductionRelation.Plan source) : (mapColumns column plan).rowCount = plan.rowCount := rfl

theorem mapColumns_port_eval {source target : Nat} (column : Fin source → Fin target)
    (plan : ProductionRelation.Plan source) (assignment : Assignment F target)
    (row : Fin plan.rowCount) (port : Fin Spec.ProductionRelation.matrixCount) :
    ((mapColumns column plan).portForm row port).eval assignment =
      (plan.portForm row port).eval (assignment ∘ column) := by
  unfold portForm
  cases meaningfulPort? port with
  | none => simp only [SparseForm.empty_eval]
  | some meaningful => exact SparseForm.mapColumns_eval column _ assignment

theorem mapColumns_rowsZero_iff {source target : Nat} (column : Fin source → Fin target)
    (plan : ProductionRelation.Plan source) (assignment : Assignment F target) :
    (mapColumns column plan).RowsZero assignment ↔ plan.RowsZero (assignment ∘ column) := by
  unfold RowsZero
  simp only [rowImage_toVertex, mapColumns_port_eval]
  rfl

/-- A certificate for every stored port is required before a partial map can
be applied to a complete row plan. -/
def mapColumnsChecked {source target : Nat} {predicate : Fin source → Prop}
    (column : ∀ source, predicate source → Fin target) (plan : ProductionRelation.Plan source)
    (supported : ∀ row port entry, entry ∈ (plan.forms row port).entries → predicate entry.column) :
    ProductionRelation.Plan target where
  rowCount := plan.rowCount
  rowCount_le := plan.rowCount_le
  forms := fun row port => (plan.forms row port).mapColumnsChecked column (supported row port)

@[simp] theorem mapColumnsChecked_rowCount {source target : Nat} {predicate : Fin source → Prop}
    (column : ∀ source, predicate source → Fin target) (plan : ProductionRelation.Plan source)
    (supported : ∀ row port entry, entry ∈ (plan.forms row port).entries → predicate entry.column) :
    (mapColumnsChecked column plan supported).rowCount = plan.rowCount := rfl

end Plan

end NightstreamFPrime.Layout.ProductionRelation
