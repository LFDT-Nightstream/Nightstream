import NightstreamFPrime.Layout.MatrixProgram
import NightstreamFPrime.Layout.ProductionRelation.PinFamilyPlan

/-!
Owns the generic executable interpreter for small explicit zero-pin families
in a compact sparse 4-matrix program. The package supplies each value form in
order.

This module does not select Stage 1 pin families or their order.
-/

namespace NightstreamFPrime.Layout.MatrixProgram.Pin

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

/-- Complete package operands for one ordered zero-pin family. -/
structure Block where
  values : List WireForm
deriving Repr, DecidableEq

def Block.rowCount (block : Block) : Nat :=
  block.values.length

/-- Decode one pin row without materializing any other row. -/
def Block.row? (block : Block) (logicalWidth ordinal : Nat) :
    Option (PinRow.Forms logicalWidth) :=
  do
    let encoded ← block.values[ordinal]?
    let value ← encoded.semantic? logicalWidth
    pure { value := value }

/-- Canonical wire block derived from one semantic pin interface. -/
def Block.ofSemantic {logicalWidth rowCount : Nat}
    (interface : PinFamilyPlan.Interface logicalWidth rowCount) : Block where
  values := List.ofFn fun row => WireForm.ofSemantic (interface.value row)

@[simp] theorem Block.ofSemantic_rowCount {logicalWidth rowCount : Nat}
    (interface : PinFamilyPlan.Interface logicalWidth rowCount) :
    (Block.ofSemantic interface).rowCount = rowCount := by
  simp [Block.rowCount, Block.ofSemantic]

/-- The canonical wire block returns the exact semantic pin row. -/
theorem Block.row?_ofSemantic {logicalWidth rowCount : Nat}
    (interface : PinFamilyPlan.Interface logicalWidth rowCount)
    (row : Fin rowCount) :
    (Block.ofSemantic interface).row? logicalWidth row.val =
      some (PinFamilyPlan.forms interface row) := by
  simp [Block.row?, Block.ofSemantic, PinFamilyPlan.forms]

end NightstreamFPrime.Layout.MatrixProgram.Pin
