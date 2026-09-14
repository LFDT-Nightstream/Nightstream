import NightstreamFPrime.Export.Stage1.PiDECProductInterface

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

/-- Keep the existing opcode guards with the proved direct interface and
row readers. Every loaded value and every rejection remain unchanged. -/
def blockRow? (block : MatrixProgram.Phi81Product.Block) (columns ordinal : Nat) :
    Option (MatrixProgram.RowForms columns) :=
  if ordinal < block.rowCount then do
    let descriptor ← MatrixProgram.Phi81Product.descriptor? block.families (ordinal / 34)
    let interface ← PiDECProductInterface.interface? block columns descriptor
    let selected ← row? interface (ordinal % 34)
    pure selected.meaningfulForm
  else none

theorem blockRow?_value (block : MatrixProgram.Phi81Product.Block)
    (columns ordinal : Nat) :
    blockRow? block columns ordinal = block.row? columns ordinal := by
  simp only [blockRow?, MatrixProgram.Phi81Product.Block.row?,
    PiDECProductInterface.interface?_value, row?_value]

/-- A row selected from a shared invocation interface has the exact canonical
block meaning. Alignment connects the range-local quotient and remainder to
those of the full block row; no interface-validity premise is required. -/
theorem blockRow?_of_grouped_loaded (block : MatrixProgram.Phi81Product.Block)
    (columns firstRow index : Nat)
    (aligned : firstRow % 34 = 0)
    (bound : firstRow + index < block.rowCount)
    (descriptor : MatrixProgram.Phi81Product.Descriptor)
    (descriptorSelected : MatrixProgram.Phi81Product.descriptor? block.families
      (firstRow / 34 + index / 34) = some descriptor)
    (interface : ProductSumPlan.Interface columns)
    (interfaceLoaded : PiDECProductInterface.interface? block columns descriptor =
      some interface)
    (row : ProductSumPlan.Row columns)
    (rowLoaded : row? interface (index % 34) = some row) :
    block.row? columns (firstRow + index) = some row.meaningfulForm := by
  have quotient : (firstRow + index) / 34 = firstRow / 34 + index / 34 := by
    omega
  have remainder : (firstRow + index) % 34 = index % 34 := by
    omega
  refine MatrixProgram.Phi81Product.Block.row?_of_loaded block columns
    (firstRow + index) bound descriptor ?_ interface ?_ row ?_
  · simpa only [quotient] using descriptorSelected
  · rw [← PiDECProductInterface.interface?_value]
    exact interfaceLoaded
  · rw [remainder, ← row?_value]
    exact rowLoaded

end NightstreamFPrime.Export.Stage1.PiDECProductRow
