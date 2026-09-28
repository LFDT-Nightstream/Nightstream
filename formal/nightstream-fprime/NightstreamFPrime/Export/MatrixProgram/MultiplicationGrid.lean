import NightstreamFPrime.Layout.MatrixProgram.MultiplicationGrid
import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Export.MatrixProgram.AffineGrid

/-! Canonical codecs for the shared Layout MatrixProgram types. -/

namespace NightstreamFPrime.Layout.MatrixProgram.MultiplicationGrid

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

def Shape.format : Format Shape where
  encode := fun shape => .array [
    .atom shape.majorCount,
    .atom shape.middleCount,
    .atom shape.minorCount]
  decode
    | .array [.atom majorCount, .atom middleCount, .atom minorCount] =>
        .ok ⟨majorCount, middleCount, minorCount⟩
    | _ => .error "invalid multiplication-grid shape"
  decode_encode := by
    intro shape
    cases shape
    rfl

def Block.format : Format Block where
  encode := fun block => .array [
    Shape.format.encode block.shape,
    .atom block.oneColumn,
    AffineGrid.Program.format.encode block.left,
    AffineGrid.Program.format.encode block.right,
    AffineGrid.Program.format.encode block.output]
  decode
    | .array [shape, .atom oneColumn, left, right, output] => do
        pure {
          shape := ← Shape.format.decode shape
          oneColumn
          left := ← AffineGrid.Program.format.decode left
          right := ← AffineGrid.Program.format.decode right
          output := ← AffineGrid.Program.format.decode output }
    | _ => .error "invalid multiplication-grid block"
  decode_encode := by
    rintro ⟨shape, oneColumn, left, right, output⟩
    simp only
    rw [Shape.format.decode_encode,
      AffineGrid.Program.format.decode_encode,
      AffineGrid.Program.format.decode_encode,
      AffineGrid.Program.format.decode_encode]
    rfl

end NightstreamFPrime.Layout.MatrixProgram.MultiplicationGrid
