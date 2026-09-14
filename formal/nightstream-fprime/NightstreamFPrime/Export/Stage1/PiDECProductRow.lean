import NightstreamFPrime.Layout.MatrixProgram.Phi81Product

/-!
Select one existing product-sum row without constructing the other rows.
The selected five terms are stored once before their left/right projections
are read. Retained lookup, term order, padding and rejection remain unchanged.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECProductRow

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

private def prepareGroup {columns : Nat} (group : List (ProductSumPlan.Term columns)) :
    Vector (ProductSumPlan.Term columns) 5 :=
  Vector.ofFn (ProductSumPlan.termAt group)

private theorem prepareGroup_get {columns : Nat}
    (group : List (ProductSumPlan.Term columns)) (lane : Fin 5) :
    (prepareGroup group).get lane = ProductSumPlan.termAt group lane := by
  change (Vector.ofFn (ProductSumPlan.termAt group))[lane.val] = _
  rw [Vector.getElem_ofFn]

/-- Reuse this group's stored terms for every left/right matrix port. -/
def productRow {columns : Nat} (interface : ProductSumPlan.Interface columns)
    (group : Fin (ProductSumPlan.groups interface.terms).length) :
    ProductSumRow.Forms columns :=
  let terms := prepareGroup (ProductSumPlan.groupAt interface group)
  { selector := ProductSumPlan.selector interface
    left := fun lane => (terms.get lane).left
    right := fun lane => (terms.get lane).right
    output := interface.groupOutput group }

theorem productRow_value {columns : Nat} (interface : ProductSumPlan.Interface columns)
    (group : Fin (ProductSumPlan.groups interface.terms).length) :
    productRow interface group = ProductSumPlan.productRow interface group := by
  simp only [productRow, ProductSumPlan.productRow, prepareGroup_get]

/-- Construct only the requested product row or final pin. Bounds use the
existing grouped-term count; partial final groups retain termAt's zero padding. -/
def row? {columns : Nat} (interface : ProductSumPlan.Interface columns)
    (ordinal : Nat) : Option (ProductSumPlan.Row columns) :=
  if within : ordinal < (ProductSumPlan.groups interface.terms).length then
    some (.product (productRow interface ⟨ordinal, within⟩))
  else if ordinal = (ProductSumPlan.groups interface.terms).length then
    some (.pin (ProductSumPlan.finalRow interface))
  else none

/-- Total equality includes the final pin and every out-of-range index. -/
theorem row?_value {columns : Nat} (interface : ProductSumPlan.Interface columns)
    (ordinal : Nat) :
    row? interface ordinal = (ProductSumPlan.rows interface)[ordinal]? := by
  by_cases within : ordinal < (ProductSumPlan.groups interface.terms).length
  · have productBound : ordinal < (ProductSumPlan.productRows interface).length := by
      simpa only [ProductSumPlan.productRows_length] using within
    have mappedBound : ordinal <
        ((ProductSumPlan.productRows interface).map ProductSumPlan.Row.product).length := by
      simpa only [List.length_map] using productBound
    rw [row?, dif_pos within, productRow_value, ProductSumPlan.rows,
      List.getElem?_append_left mappedBound, List.getElem?_map,
      List.getElem?_eq_getElem productBound]
    simp only [ProductSumPlan.productRows, List.getElem_ofFn, Option.map_some]
  · have lower : (ProductSumPlan.groups interface.terms).length ≤ ordinal :=
      Nat.le_of_not_gt within
    have mappedLower :
        ((ProductSumPlan.productRows interface).map ProductSumPlan.Row.product).length ≤
          ordinal := by
      simpa only [List.length_map, ProductSumPlan.productRows_length] using lower
    rw [row?, dif_neg within, ProductSumPlan.rows,
      List.getElem?_append_right mappedLower, List.length_map,
      ProductSumPlan.productRows_length]
    cases difference : ordinal - (ProductSumPlan.groups interface.terms).length with
    | zero =>
        have same : ordinal = (ProductSumPlan.groups interface.terms).length := by omega
        exact if_pos same
    | succ rest =>
        have different : ordinal ≠ (ProductSumPlan.groups interface.terms).length := by omega
        exact if_neg different

/-- Keep the existing opcode guards and interface loader. Only selection
from the complete row list is replaced by the proved direct row lookup. -/
def blockRow? (block : MatrixProgram.Phi81Product.Block) (columns ordinal : Nat) :
    Option (MatrixProgram.RowForms columns) :=
  if ordinal < block.rowCount then do
    let descriptor ← MatrixProgram.Phi81Product.descriptor? block.families (ordinal / 34)
    let interface ← block.interface? columns descriptor
    let selected ← row? interface (ordinal % 34)
    pure selected.meaningfulForm
  else none

theorem blockRow?_value (block : MatrixProgram.Phi81Product.Block)
    (columns ordinal : Nat) :
    blockRow? block columns ordinal = block.row? columns ordinal := by
  simp only [blockRow?, MatrixProgram.Phi81Product.Block.row?, row?_value]

end NightstreamFPrime.Export.Stage1.PiDECProductRow
