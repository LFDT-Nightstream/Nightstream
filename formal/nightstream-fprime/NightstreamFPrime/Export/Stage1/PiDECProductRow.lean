import NightstreamFPrime.Export.Stage1.PiDECProductInterface

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECProductRow

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def row? {columns : Nat} (interface : RingProductPlan.Interface columns)
    (ordinal : Nat) : Option (OrdinaryRow.Forms columns) :=
  if within : ordinal < 128 then
    some (RingProductPlan.rowAt interface ⟨ordinal, within⟩)
  else none

theorem row?_value {columns : Nat} (interface : RingProductPlan.Interface columns)
    (ordinal : Nat) :
    row? interface ordinal = (RingProductPlan.rows interface)[ordinal]? := by
  simp only [row?, RingProductPlan.rows, List.getElem?_ofFn]

def blockRow? (block : MatrixProgram.RingProduct.Block) (columns ordinal : Nat) :
    Option (MatrixProgram.RowForms columns) :=
  if ordinal < block.rowCount then do
    let descriptor ← MatrixProgram.RingProduct.ringDescriptor? block.families (ordinal / 128)
    let interface ← PiDECProductInterface.interface? block columns descriptor
    let selected ← row? interface (ordinal % 128)
    pure selected.meaningfulForm
  else none

theorem blockRow?_value (block : MatrixProgram.RingProduct.Block)
    (columns ordinal : Nat) :
    blockRow? block columns ordinal = block.row? columns ordinal := by
  simp only [blockRow?, MatrixProgram.RingProduct.Block.row?,
    PiDECProductInterface.interface?_value, row?_value]

theorem blockRow?_of_grouped_loaded (block : MatrixProgram.RingProduct.Block)
    (columns firstRow index : Nat)
    (aligned : firstRow % 128 = 0)
    (bound : firstRow + index < block.rowCount)
    (descriptor : MatrixProgram.RingProduct.Descriptor)
    (descriptorSelected : MatrixProgram.RingProduct.ringDescriptor? block.families
      (firstRow / 128 + index / 128) = some descriptor)
    (interface : RingProductPlan.Interface columns)
    (interfaceLoaded : PiDECProductInterface.interface? block columns descriptor =
      some interface)
    (row : OrdinaryRow.Forms columns)
    (rowLoaded : row? interface (index % 128) = some row) :
    block.row? columns (firstRow + index) = some row.meaningfulForm := by
  have quotient : (firstRow + index) / 128 = firstRow / 128 + index / 128 := by
    omega
  have remainder : (firstRow + index) % 128 = index % 128 := by
    omega
  refine MatrixProgram.RingProduct.Block.row?_of_loaded block columns
    (firstRow + index) bound descriptor ?_ interface ?_ row ?_
  · simpa only [quotient] using descriptorSelected
  · rw [← PiDECProductInterface.interface?_value]
    exact interfaceLoaded
  · rw [remainder, ← row?_value]
    exact rowLoaded

end NightstreamFPrime.Export.Stage1.PiDECProductRow
