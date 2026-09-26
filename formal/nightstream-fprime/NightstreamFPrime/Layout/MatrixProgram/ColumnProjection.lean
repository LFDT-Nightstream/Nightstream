import NightstreamFPrime.Layout.MatrixProgram.SourceProjection
import NightstreamFPrime.Layout.MatrixProgram.Phi81Product
import NightstreamFPrime.Layout.ProductionRelation.ColumnMap

/-! Checked projection of stored matrix coordinates. Missing and out-of-range
images reject the row, including entries whose coefficient is zero. -/

namespace NightstreamFPrime.Layout.MatrixProgram.SourceProjection

open NightstreamFPrime.Spec NightstreamFPrime.Layout.ProductionRelation

def entry? {source : Nat} (projection : SourceProjection) (target : Nat)
    (entry : SparseEntry source) : Option (SparseEntry target) := do
  let column ← projection.column? entry.column.val
  if bound : column < target then some ⟨⟨column, bound⟩, entry.coefficient⟩ else none

def entries? {source : Nat} (projection : SourceProjection) (target : Nat) :
    List (SparseEntry source) → Option (List (SparseEntry target))
  | [] => some []
  | entry :: rest => do
      let head ← projection.entry? target entry
      let tail ← entries? projection target rest
      pure (head :: tail)

def sparseForm? {source : Nat} (projection : SourceProjection) (target : Nat)
    (form : SparseForm source) : Option (SparseForm target) :=
  (entries? projection target form.entries).map SparseForm.mk

def ports? {source : Nat} (projection : SourceProjection) (target : Nat)
    (forms : RowForms source) : Option (RowForms target) :=
  Phi81Product.loadFin? _ (fun port => sparseForm? projection target (forms port))

theorem entry?_of_image {source target : Nat} (projection : SourceProjection)
    (entry : SparseEntry source) (column : Fin target)
    (mapped : projection.column? entry.column.val = some column.val) :
    projection.entry? target entry = some ⟨column, entry.coefficient⟩ := by
  simp [entry?, mapped, column.isLt]

theorem entry?_unmapped {source target : Nat} (projection : SourceProjection)
    (entry : SparseEntry source) (unmapped : projection.column? entry.column.val = none) :
    projection.entry? target entry = none := by
  simp [entry?, unmapped]

private theorem entries?_checked {source target : Nat} {predicate : Fin source → Prop}
    (projection : SourceProjection) (column : ∀ source, predicate source → Fin target)
    (entries : List (SparseEntry source)) (supported : ∀ entry ∈ entries, predicate entry.column)
    (mapped : ∀ source live, projection.column? source.val = some (column source live).val) :
    entries? projection target entries =
      some (entries.pmap (fun entry live => ⟨column entry.column live, entry.coefficient⟩) supported) := by
  induction entries with
  | nil => rfl
  | cons entry rest ih =>
    rw [entries?, entry?_of_image projection entry (column entry.column (supported _ List.mem_cons_self)) (mapped _ _)]
    rw [ih (fun value member => supported value (List.mem_cons_of_mem entry member))]
    rfl

theorem sparseForm?_checked {source target : Nat} {predicate : Fin source → Prop}
    (projection : SourceProjection) (column : ∀ source, predicate source → Fin target)
    (form : SparseForm source) (supported : ∀ entry ∈ form.entries, predicate entry.column)
    (mapped : ∀ source live, projection.column? source.val = some (column source live).val) :
    projection.sparseForm? target form = some (form.mapColumnsChecked column supported) := by
  rw [sparseForm?, entries?_checked projection column form.entries supported mapped]
  rfl

theorem ports?_checked {source target : Nat} {predicate : Fin source → Prop}
    (projection : SourceProjection) (column : ∀ source, predicate source → Fin target)
    (forms : RowForms source) (supported : ∀ port entry, entry ∈ (forms port).entries → predicate entry.column)
    (mapped : ∀ source live, projection.column? source.val = some (column source live).val) :
    projection.ports? target forms =
      some (fun port => (forms port).mapColumnsChecked column (supported port)) := by
  apply Phi81Product.loadFin?_of_some
  intro port
  exact sparseForm?_checked projection column (forms port) (supported port) mapped

end NightstreamFPrime.Layout.MatrixProgram.SourceProjection
