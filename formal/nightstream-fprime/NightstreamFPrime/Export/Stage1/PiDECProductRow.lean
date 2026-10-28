import NightstreamFPrime.Export.Stage1.PiDECProductInterface

/-!
Select one quotient evaluation row without constructing the other rows.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECProductRow

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def row? {columns : Nat} (interface : Phi81ProductPlan.Interface columns)
    (ordinal : Nat) : Option (OrdinaryRow.Forms columns) :=
  if within : ordinal < 108 then
    some (Phi81ProductPlan.rowAt interface ⟨ordinal, within⟩)
  else none

theorem row?_value {columns : Nat} (interface : Phi81ProductPlan.Interface columns)
    (ordinal : Nat) :
    row? interface ordinal = (Phi81ProductPlan.rows interface)[ordinal]? := by
  simp only [row?, Phi81ProductPlan.rows, List.getElem?_ofFn]

def blockRow? (block : MatrixProgram.Phi81Product.Block) (columns ordinal : Nat) :
    Option (MatrixProgram.RowForms columns) :=
  if ordinal < block.rowCount then do
    let descriptor ← MatrixProgram.Phi81Product.ringDescriptor? block.families (ordinal / 108)
    let interface ← PiDECProductInterface.interface? block columns descriptor
    let selected ← row? interface (ordinal % 108)
    pure selected.meaningfulForm
  else none

theorem blockRow?_value (block : MatrixProgram.Phi81Product.Block)
    (columns ordinal : Nat) :
    blockRow? block columns ordinal = block.row? columns ordinal := by
  simp only [blockRow?, MatrixProgram.Phi81Product.Block.row?,
    PiDECProductInterface.interface?_value, row?_value]

theorem blockRow?_of_grouped_loaded (block : MatrixProgram.Phi81Product.Block)
    (columns firstRow index : Nat)
    (aligned : firstRow % 108 = 0)
    (bound : firstRow + index < block.rowCount)
    (descriptor : MatrixProgram.Phi81Product.Descriptor)
    (descriptorSelected : MatrixProgram.Phi81Product.ringDescriptor? block.families
      (firstRow / 108 + index / 108) = some descriptor)
    (interface : Phi81ProductPlan.Interface columns)
    (interfaceLoaded : PiDECProductInterface.interface? block columns descriptor =
      some interface)
    (row : OrdinaryRow.Forms columns)
    (rowLoaded : row? interface (index % 108) = some row) :
    block.row? columns (firstRow + index) = some row.meaningfulForm := by
  have quotient : (firstRow + index) / 108 = firstRow / 108 + index / 108 := by
    omega
  have remainder : (firstRow + index) % 108 = index % 108 := by
    omega
  refine MatrixProgram.Phi81Product.Block.row?_of_loaded block columns
    (firstRow + index) bound descriptor ?_ interface ?_ row ?_
  · simpa only [quotient] using descriptorSelected
  · rw [← PiDECProductInterface.interface?_value]
    exact interfaceLoaded
  · rw [remainder, ← row?_value]
    exact rowLoaded

end NightstreamFPrime.Export.Stage1.PiDECProductRow
