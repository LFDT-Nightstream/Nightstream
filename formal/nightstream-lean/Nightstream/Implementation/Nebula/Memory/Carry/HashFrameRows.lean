import Nightstream.Implementation.Nebula.Memory.Carry.HashPackingRows
import Nightstream.Implementation.R1CS.Core.ConstantPins

/-!
Contract: exact 117-field input block for the V2 memory-carry Poseidon2
digest.

Assurance tier: implementation model.

Owns nine fixed profile/frame prefix columns, the exact 108 packed-word
columns, the ordered input-column list, row soundness to the pure `frame`
definition, and honest local completeness.

Does not own the Poseidon2 sponge trace, the outer state hash, absolute
generated columns, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryCarryHashFrameRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.Nebula.MemoryCarryHashFrame
open Nightstream.Implementation.Nebula.MemoryCarryHashPackingRows
open Nightstream.Protocol.Nebula.MemoryWireGeometry

structure Layout where
  packing : MemoryCarryHashPackingRows.Layout
  prefixStart : Nat
deriving DecidableEq, Repr

def Layout.prefixColumn (layout : Layout) (index : Nat) : Nat :=
  layout.prefixStart + index

def prefixPins (layout : Layout) : List (Nat × Nat) :=
  [ (layout.prefixColumn 0, domainTag)
  , (layout.prefixColumn 1, frameVersion)
  , (layout.prefixColumn 2, 2)
  , (layout.prefixColumn 3, 2)
  , (layout.prefixColumn 4, 1)
  , (layout.prefixColumn 5, 1)
  , (layout.prefixColumn 6, carryBits)
  , (layout.prefixColumn 7, wordBitCount)
  , (layout.prefixColumn 8, packedWordCount)
  ]

def prefixColumns (layout : Layout) : List Nat :=
  (prefixPins layout).map Prod.fst

def prefixValues (layout : Layout) : List Nat :=
  (prefixPins layout).map Prod.snd

theorem prefixValues_exact (layout : Layout) :
    prefixValues layout = framePrefix := by
  change
    [domainTag, frameVersion, 2, 2, 1, 1, carryBits, wordBitCount,
      packedWordCount] = framePrefix
  rw [framePrefix_exact, exact_geometry.1]
  norm_num [domainTag, frameVersion, wordBitCount, packedWordCount]

theorem prefixColumns_length (layout : Layout) :
    (prefixColumns layout).length = 9 := by
  simp [prefixColumns, prefixPins]

theorem prefixPins_valuesCanonical (layout : Layout) :
    ConstantPins.ValuesCanonical (prefixPins layout) := by
  have carryExact : carryBits = 3433 := exact_geometry.1
  intro pin member
  simp only [prefixPins, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    norm_num [goldilocksP, domainTag, frameVersion, carryExact, wordBitCount,
      packedWordCount]

def packedColumns (layout : Layout) : List Nat :=
  (List.range packedWordCount).map fun index =>
    layout.packing.packedWordColumn index

theorem packedColumns_length (layout : Layout) :
    (packedColumns layout).length = packedWordCount := by
  simp [packedColumns]

def inputColumns (layout : Layout) : List Nat :=
  prefixColumns layout ++ packedColumns layout

theorem inputColumns_length (layout : Layout) :
    (inputColumns layout).length = 117 := by
  rw [inputColumns, List.length_append, prefixColumns_length,
    packedColumns_length]
  decide

def rows (layout : Layout) : List Row :=
  MemoryCarryHashPackingRows.rows layout.packing ++
    ConstantPins.rows (prefixPins layout)

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 140 := by
  simp [rows, MemoryCarryHashPackingRows.rows_length_exact,
    ConstantPins.rows, prefixPins]

private theorem packing_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryCarryHashPackingRows.rows layout.packing) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem prefix_rows_included (layout : Layout) :
    rowsIncluded (ConstantPins.rows (prefixPins layout)) (rows layout) = true := by
  unfold rowsIncluded
  apply List.all_eq_true.mpr
  intro row member
  exact decide_eq_true (by simp [rows, member])

private theorem prefix_facts
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    ∀ pin ∈ prefixPins layout, assignment pin.1 = pin.2 :=
  ConstantPins.sound (prefixPins_valuesCanonical layout)
    (prefix_rows_included layout) canonical one holds

theorem prefix_column_values
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    (prefixColumns layout).map assignment = framePrefix := by
  rw [← prefixValues_exact layout]
  simp only [prefixColumns, prefixValues, List.map_map]
  apply List.map_congr_left
  intro pin member
  exact prefix_facts canonical one holds pin member

theorem packed_column_values
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.packing.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    (packedColumns layout).map assignment = encodePacked block := by
  simpa [packedColumns, packedColumnValues, List.map_map,
    Function.comp_def] using
    MemoryCarryHashPackingRows.packed_columns_eq_encodePacked
      canonical one placed (packing_rows_hold holds)

/-- The ordered 117 assigned columns are the exact fixed frame of the same
authority-bearing carry block. -/
theorem input_column_values
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.packing.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    (inputColumns layout).map assignment = frame block := by
  rw [inputColumns, List.map_append, prefix_column_values canonical one holds,
    packed_column_values canonical one placed holds]
  rfl

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (block : MemoryCarryParser.Block) : Prop where
  packing : MemoryCarryHashPackingRows.Honest layout.packing assignment block
  prefixPlaced : ∀ pin ∈ prefixPins layout, assignment pin.1 = pin.2

theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (one : assignment 0 = 1)
    (honest : Honest layout assignment block) :
    Satisfies (rows layout) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with packingMember | prefixMember
  · exact MemoryCarryHashPackingRows.rows_complete one honest.packing
      row packingMember
  · exact ConstantPins.complete (prefixPins_valuesCanonical layout) one
      honest.prefixPlaced row prefixMember

end Nightstream.Implementation.Nebula.MemoryCarryHashFrameRows
