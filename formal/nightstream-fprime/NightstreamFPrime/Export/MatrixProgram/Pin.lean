import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Layout.ProductionRelation.PinFamilyPlan

/-!
Owns the generic executable interpreter for small explicit zero-pin families
in a compact sparse 14-matrix program. The package supplies the selector
column and each value form in order.

This module does not select Stage 1 pin families or their order.
-/

namespace NightstreamFPrime.Export.MatrixProgram.Pin

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

/-- Complete package operands for one ordered zero-pin family. -/
structure Block where
  oneColumn : Nat
  values : List WireForm
deriving Repr, DecidableEq

def Block.format : Format Block where
  encode := fun block => .array [
    .atom block.oneColumn,
    (list WireForm.format).encode block.values]
  decode
    | .array [.atom oneColumn, values] => do
        pure ⟨oneColumn, ← (list WireForm.format).decode values⟩
    | _ => .error "invalid pin matrix block"
  decode_encode := by
    intro block
    cases block
    simp [Format.decode_encode]

def Block.rowCount (block : Block) : Nat :=
  block.values.length

/-- Decode one pin row without materializing any other row. -/
def Block.row? (block : Block) (logicalWidth ordinal : Nat) :
    Option (PinRow.Forms logicalWidth) :=
  if oneBound : block.oneColumn < logicalWidth then do
    let encoded ← block.values[ordinal]?
    let value ← encoded.semantic? logicalWidth
    pure {
      selector := SparseForm.singleton ⟨block.oneColumn, oneBound⟩ 1
      value := value }
  else
    none

/-- Canonical wire block derived from one semantic pin interface. -/
def Block.ofSemantic {logicalWidth rowCount : Nat}
    (interface : PinFamilyPlan.Interface logicalWidth rowCount) : Block where
  oneColumn := interface.oneColumn.val
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
  simp [Block.row?, Block.ofSemantic, interface.oneColumn.isLt,
    PinFamilyPlan.forms]

end NightstreamFPrime.Export.MatrixProgram.Pin
