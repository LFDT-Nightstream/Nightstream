import Nightstream.Implementation.NebulaV2.FixedBits

/-!
Contract: exact placement of a fixed binary parser block in consecutive
public R1CS columns.

Assurance tier: implementation model.

Owns one per-bit placement predicate and exact extraction of every safe
contiguous slice from those columns. The predicate contains no decoded value
or protocol conclusion.

Does not own a concrete codec schema, parser, or R1CS constraints.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.PublicBitBlock

structure Layout where
  publicBitStart : Nat
deriving DecidableEq, Repr

/-- The parser block and public assignment contain the same bit at each exact
physical position. -/
def Placed {total : Nat} (layout : Layout) (assignment : Nat → Nat)
    (block : FixedBits.Word total) : Prop :=
  ∀ index (bound : index < total),
    assignment (layout.publicBitStart + index) =
      block.val[index]'(by simpa [block.property.1] using bound)

def sliceColumns (layout : Layout) (assignment : Nat → Nat)
    (offset width : Nat) : List Nat :=
  (List.range width).map fun index =>
    assignment (layout.publicBitStart + offset + index)

/-- Every safe parser slice is exactly the corresponding consecutive public
column slice. -/
theorem slice_eq_columns
    {total : Nat} {layout : Layout} {assignment : Nat → Nat}
    {block : FixedBits.Word total}
    (placed : Placed layout assignment block)
    (offset width : Nat) (fits : offset + width ≤ total) :
    (FixedBits.slice block offset width fits).val =
      sliceColumns layout assignment offset width := by
  apply List.ext_getElem
  · simp [FixedBits.slice, sliceColumns, block.property.1]
    omega
  · intro index leftBound rightBound
    have indexBound : index < width := by
      simpa [sliceColumns] using rightBound
    simp only [FixedBits.slice, sliceColumns, List.getElem_take,
      List.getElem_drop, List.getElem_map, List.getElem_range]
    rw [← placed (offset + index) (by omega)]
    congr 1
    omega

/-- The full assigned public slice recovers the source parser block. -/
theorem full_slice_eq
    {total : Nat} {layout : Layout} {assignment : Nat → Nat}
    {block : FixedBits.Word total}
    (placed : Placed layout assignment block) :
    sliceColumns layout assignment 0 total = block.val := by
  rw [← slice_eq_columns placed 0 total (by omega)]
  apply List.ext_getElem
  · simp [FixedBits.slice, block.property.1]
  · intro index leftBound rightBound
    simp [FixedBits.slice]

end Nightstream.Implementation.NebulaV2.PublicBitBlock
