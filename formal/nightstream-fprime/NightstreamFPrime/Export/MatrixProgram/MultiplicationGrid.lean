import Batteries.Data.Fin.Coding
import NightstreamFPrime.Export.MatrixProgram.AffineGrid
import NightstreamFPrime.Layout.ProductionRelation.MultiplicationFamilyPlan

/-!
Owns a generic executable multiplication family over a three-axis grid.
The package supplies the grid shape and affine operand programs. The
interpreter does not select phase dimensions, retained blocks, or formulas.
-/

namespace NightstreamFPrime.Export.MatrixProgram.MultiplicationGrid

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

structure Shape where
  majorCount : Nat
  middleCount : Nat
  minorCount : Nat
deriving Repr, DecidableEq

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

def Shape.rowCount (shape : Shape) : Nat :=
  shape.majorCount * (shape.middleCount * shape.minorCount)

def Shape.coordinate? (shape : Shape) (ordinal : Nat) :
    Option AffineGrid.Coordinate :=
  if bound : ordinal < shape.rowCount then
    let outer : Fin (shape.majorCount *
        (shape.middleCount * shape.minorCount)) := ⟨ordinal, bound⟩
    let decoded := Fin.decodeProd outer
    let inner := Fin.decodeProd decoded.2
    some {
      major := decoded.1.val
      middle := inner.1.val
      minor := inner.2.val }
  else none

theorem Shape.coordinate?_of_coordinates (shape : Shape)
    (major : Fin shape.majorCount) (middle : Fin shape.middleCount)
    (minor : Fin shape.minorCount) :
    shape.coordinate?
        (Fin.encodeProd (major, Fin.encodeProd (middle, minor))).val =
      some { major := major.val, middle := middle.val, minor := minor.val } := by
  unfold coordinate?
  change (if bound :
      (Fin.encodeProd (major, Fin.encodeProd (middle, minor))).val <
        shape.majorCount * (shape.middleCount * shape.minorCount) then
      let outer : Fin (shape.majorCount *
          (shape.middleCount * shape.minorCount)) :=
        ⟨(Fin.encodeProd (major, Fin.encodeProd (middle, minor))).val, bound⟩
      let decoded := Fin.decodeProd outer
      let inner := Fin.decodeProd decoded.2
      some (AffineGrid.Coordinate.mk decoded.1.val inner.1.val inner.2.val)
    else none) = _
  rw [dif_pos (Fin.encodeProd (major,
    Fin.encodeProd (middle, minor))).isLt]
  simp only
  rw [Fin.decodeProd_encodeProd, Fin.decodeProd_encodeProd]

/-- Complete operands for one multiplication grid. -/
structure Block where
  shape : Shape
  oneColumn : Nat
  left : AffineGrid.Program
  right : AffineGrid.Program
  output : AffineGrid.Program
deriving Repr, DecidableEq

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

def Block.rowCount (block : Block) : Nat := block.shape.rowCount

/-- Decode one multiplication row without expanding any other row. -/
def Block.row? (block : Block) (logicalWidth ordinal : Nat) :
    Option (OrdinaryRow.Forms logicalWidth) := do
  if oneBound : block.oneColumn < logicalWidth then
    let coordinate ← block.shape.coordinate? ordinal
    let left ← block.left.form? logicalWidth block.oneColumn coordinate
    let right ← block.right.form? logicalWidth block.oneColumn coordinate
    let output ← block.output.form? logicalWidth block.oneColumn coordinate
    pure {
      selector := SparseForm.singleton ⟨block.oneColumn, oneBound⟩ 1
      a := left
      b := right
      c := output }
  else none

/-- Exact affine operand results produce the standard multiplication-family
row at the same grid coordinate. -/
theorem Block.row?_of_results (block : Block) {logicalWidth : Nat}
    (oneColumn : Fin logicalWidth) (oneEqual : block.oneColumn = oneColumn.val)
    (major : Fin block.shape.majorCount)
    (middle : Fin block.shape.middleCount)
    (minor : Fin block.shape.minorCount)
    (left right output : SparseForm logicalWidth)
    (leftLoaded : block.left.form? logicalWidth block.oneColumn
      { major := major.val, middle := middle.val, minor := minor.val } =
        some left)
    (rightLoaded : block.right.form? logicalWidth block.oneColumn
      { major := major.val, middle := middle.val, minor := minor.val } =
        some right)
    (outputLoaded : block.output.form? logicalWidth block.oneColumn
      { major := major.val, middle := middle.val, minor := minor.val } =
        some output) :
    block.row? logicalWidth
        (Fin.encodeProd (major, Fin.encodeProd (middle, minor))).val =
      some {
        selector := SparseForm.singleton oneColumn 1
        a := left
        b := right
        c := output } := by
  have leftLoaded' : block.left.form? logicalWidth oneColumn.val
      { major := major.val, middle := middle.val, minor := minor.val } =
        some left := by
    simpa [oneEqual] using leftLoaded
  have rightLoaded' : block.right.form? logicalWidth oneColumn.val
      { major := major.val, middle := middle.val, minor := minor.val } =
        some right := by
    simpa [oneEqual] using rightLoaded
  have outputLoaded' : block.output.form? logicalWidth oneColumn.val
      { major := major.val, middle := middle.val, minor := minor.val } =
        some output := by
    simpa [oneEqual] using outputLoaded
  unfold Block.row?
  rw [oneEqual, dif_pos oneColumn.isLt]
  rw [Shape.coordinate?_of_coordinates]
  change (do
    let loadedLeft ← block.left.form? logicalWidth oneColumn.val
      { major := major.val, middle := middle.val, minor := minor.val }
    let loadedRight ← block.right.form? logicalWidth oneColumn.val
      { major := major.val, middle := middle.val, minor := minor.val }
    let loadedOutput ← block.output.form? logicalWidth oneColumn.val
      { major := major.val, middle := middle.val, minor := minor.val }
    pure ({
      selector := SparseForm.singleton oneColumn 1
      a := loadedLeft
      b := loadedRight
      c := loadedOutput } : OrdinaryRow.Forms logicalWidth)) = _
  rw [leftLoaded', rightLoaded', outputLoaded']
  rfl

end NightstreamFPrime.Export.MatrixProgram.MultiplicationGrid
