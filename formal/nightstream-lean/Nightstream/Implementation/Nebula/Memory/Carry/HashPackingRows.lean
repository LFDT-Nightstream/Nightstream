import Nightstream.Implementation.Nebula.Memory.Carry.HashFrame
import Nightstream.Implementation.Nebula.Core.PublicBitBlock

/-!
Contract: exact R1CS packing of the 3,433 V2 carry bits into the 108
32-bit Poseidon2 frame words.

Assurance tier: implementation model.

Owns 23 zero-padding rows, one linked little-endian recomposition row per
packed word, sound equality with `MemoryCarryHashFrame.encodePacked`, and
honest local completeness. It reuses the parser-owned binary public bits and
does not emit duplicate Boolean rows.

Does not own the 117-field frame placement, Poseidon2 rows, the outer state
hash, absolute generated columns, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryCarryHashPackingRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.Nebula.MemoryCarryHashFrame
open Nightstream.Protocol.Nebula.MemoryWireGeometry

structure Layout where
  publicBitStart : Nat
  packedWordStart : Nat
deriving DecidableEq, Repr

def Layout.publicBits (layout : Layout) : PublicBitBlock.Layout :=
  { publicBitStart := layout.publicBitStart }

def Layout.paddedBitColumn (layout : Layout) (offset : Nat) : Nat :=
  layout.publicBitStart + offset

def Layout.packedWordColumn (layout : Layout) (index : Nat) : Nat :=
  layout.packedWordStart + index

def Layout.word (layout : Layout) (index : Nat) : BoundedWordRows.Layout :=
  { width := wordBitCount
    valueColumn := layout.packedWordColumn index
    bitStart := layout.paddedBitColumn (wordBitCount * index) }

def paddingRows (layout : Layout) : List Row :=
  (List.range highPaddingBitCount).map fun offset =>
    builderLinearRow
      (layout.paddedBitColumn (carryBits + offset)) []

def packingRows (layout : Layout) : List Row :=
  (List.range packedWordCount).map fun index =>
    (layout.word index).recompositionRow

def rows (layout : Layout) : List Row :=
  paddingRows layout ++ packingRows layout

theorem paddingRows_length (layout : Layout) :
    (paddingRows layout).length = highPaddingBitCount := by
  simp [paddingRows]

theorem packingRows_length (layout : Layout) :
    (packingRows layout).length = packedWordCount := by
  simp [packingRows]

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 131 := by
  rw [rows, List.length_append, paddingRows_length, packingRows_length]
  decide

private theorem padding_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (paddingRows layout) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem packing_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (packingRows layout) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

theorem padding_column_zero
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment)
    (offset : Nat) (bound : offset < highPaddingBitCount) :
    assignment (layout.paddedBitColumn (carryBits + offset)) = 0 := by
  have rowMember :
      builderLinearRow
          (layout.paddedBitColumn (carryBits + offset)) [] ∈
        paddingRows layout :=
    List.mem_map.mpr ⟨offset, List.mem_range.mpr bound, rfl⟩
  have defined := builderLinearRow_sound canonical one
    (layout.paddedBitColumn (carryBits + offset)) []
    (by simp [CanonicalTerms])
    (padding_rows_hold holds _ rowMember)
  simpa [lcEval] using defined

private theorem word_row_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment)
    (index : Nat) (bound : index < packedWordCount) :
    RowHolds assignment (layout.word index).recompositionRow := by
  apply packing_rows_hold holds
  exact List.mem_map.mpr ⟨index, List.mem_range.mpr bound, rfl⟩

private theorem padded_index_bound
    {index offset : Nat}
    (indexBound : index < packedWordCount)
    (offsetBound : offset < wordBitCount) :
    wordBitCount * index + offset < paddedBitCount := by
  norm_num [packedWordCount, wordBitCount, paddedBitCount] at indexBound offsetBound ⊢
  omega

private theorem assigned_padded_bit
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (paddingZero : ∀ offset, offset < highPaddingBitCount →
      assignment (layout.paddedBitColumn (carryBits + offset)) = 0)
    (index offset : Nat)
    (indexBound : index < packedWordCount)
    (offsetBound : offset < wordBitCount) :
    bitBool
        (assignment
          (layout.paddedBitColumn (wordBitCount * index + offset))) =
      (paddedWord block).getLsbD (wordBitCount * index + offset) := by
  let global := wordBitCount * index + offset
  have globalPadded : global < paddedBitCount :=
    padded_index_bound indexBound offsetBound
  by_cases logical : global < carryBits
  · have placedBit := placed global logical
    dsimp only [global] at globalPadded logical placedBit ⊢
    rw [show layout.paddedBitColumn global =
        layout.publicBits.publicBitStart + global by rfl,
      placedBit]
    simp [paddedWord, BitVec.getLsbD_setWidth, globalPadded,
      logicalWord, fixedBitsVector, logical]
  · have logicalEnd : carryBits ≤ global := Nat.le_of_not_gt logical
    have padBound : global - carryBits < highPaddingBitCount := by
      have carryExact : carryBits = 3433 := by decide
      have paddedExact : paddedBitCount = 3456 := by decide
      have paddingExact : highPaddingBitCount = 23 := by decide
      omega
    have padZero := paddingZero (global - carryBits) padBound
    have columnEqual :
        layout.paddedBitColumn global =
          layout.paddedBitColumn (carryBits + (global - carryBits)) := by
      simp only [Layout.paddedBitColumn]
      omega
    rw [columnEqual, padZero]
    simp only [bitBool]
    exact (padded_high_bit_zero block global logicalEnd).symm

private theorem word_digits_binary
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (paddingZero : ∀ offset, offset < highPaddingBitCount →
      assignment (layout.paddedBitColumn (carryBits + offset)) = 0)
    (index : Nat) (indexBound : index < packedWordCount) :
    ∀ digit ∈ (layout.word index).digits assignment, digit < 2 := by
  intro digit member
  rcases List.mem_map.mp member with
    ⟨offset, offsetMember, digitEqual⟩
  subst digit
  have offsetBound : offset < wordBitCount := by
    simpa [Layout.word] using List.mem_range.mp offsetMember
  let global := wordBitCount * index + offset
  by_cases logical : global < carryBits
  · have placedBit := placed global logical
    rw [show (layout.word index).bitColumn offset =
        layout.publicBits.publicBitStart + global by
      simp [Layout.word, Layout.paddedBitColumn,
        Layout.publicBits, BoundedWordRows.Layout.bitColumn, global]
      omega]
    rw [placedBit]
    exact block.property.2 _
      (List.get_mem _ ⟨global, by simpa [block.property.1] using logical⟩)
  · have globalPadded : global < paddedBitCount :=
      padded_index_bound indexBound offsetBound
    have logicalEnd : carryBits ≤ global := Nat.le_of_not_gt logical
    have padBound : global - carryBits < highPaddingBitCount := by
      have carryExact : carryBits = 3433 := by decide
      have paddedExact : paddedBitCount = 3456 := by decide
      have paddingExact : highPaddingBitCount = 23 := by decide
      omega
    rw [show (layout.word index).bitColumn offset =
        layout.paddedBitColumn (carryBits + (global - carryBits)) by
      simp [Layout.word, BoundedWordRows.Layout.bitColumn,
        Layout.paddedBitColumn, global]
      omega]
    rw [paddingZero (global - carryBits) padBound]
    decide

private theorem word_vector_eq
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (paddingZero : ∀ offset, offset < highPaddingBitCount →
      assignment (layout.paddedBitColumn (carryBits + offset)) = 0)
    (index : Nat) (indexBound : index < packedWordCount) :
    fixedBitsVector
        (⟨(layout.word index).digits assignment,
          (layout.word index).digits_length assignment,
          word_digits_binary placed paddingZero index indexBound⟩ :
          FixedBits.Word wordBitCount) =
      packedWords block ⟨index, indexBound⟩ := by
  apply BitVec.eq_of_getLsbD_eq
  intro offset offsetBound
  have assigned := assigned_padded_bit placed paddingZero index offset
    indexBound offsetBound
  simp only [fixedBitsVector, BitVec.getLsbD_ofFnLE]
  rw [dif_pos (by simpa [Layout.word] using offsetBound)]
  simp only [BoundedWordRows.Layout.digits]
  rw [show
      ((List.range (layout.word index).width).map fun current =>
          assignment ((layout.word index).bitColumn current)).get
          ⟨offset, by simpa [Layout.word] using offsetBound⟩ =
        assignment ((layout.word index).bitColumn offset) by
    simp]
  change
    bitBool
        (assignment
          ((layout.word index).bitColumn offset)) =
      (packedWords block ⟨index, indexBound⟩).getLsbD offset
  rw [show (layout.word index).bitColumn offset =
      layout.paddedBitColumn (wordBitCount * index + offset) by
    simp [Layout.word, BoundedWordRows.Layout.bitColumn,
      Layout.paddedBitColumn]
    omega]
  rw [assigned]
  simp [packedWords, splitWords, BitVec.getLsbD_extractLsb', offsetBound]

theorem word_decoded_eq_packed
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (paddingZero : ∀ offset, offset < highPaddingBitCount →
      assignment (layout.paddedBitColumn (carryBits + offset)) = 0)
    (index : Nat) (indexBound : index < packedWordCount) :
    BoundedWordRows.decoded (layout.word index) assignment =
      (packedWords block ⟨index, indexBound⟩).toNat := by
  let word : FixedBits.Word wordBitCount :=
    ⟨(layout.word index).digits assignment,
      (layout.word index).digits_length assignment,
      word_digits_binary placed paddingZero index indexBound⟩
  calc
    BoundedWordRows.decoded (layout.word index) assignment =
        FixedBits.decode word := rfl
    _ = (fixedBitsVector word).toNat :=
      (fixedBitsVector_toNat word).symm
    _ = (packedWords block ⟨index, indexBound⟩).toNat :=
      congrArg BitVec.toNat
        (word_vector_eq placed paddingZero index indexBound)

/-- Every packed-word column is derived from the same 3,433 accepted carry
bits. There is no independent packed-word placement assumption. -/
theorem packed_word_column_eq
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment)
    (index : Nat) (indexBound : index < packedWordCount) :
    assignment (layout.packedWordColumn index) =
      (packedWords block ⟨index, indexBound⟩).toNat := by
  have paddingZero := padding_column_zero canonical one holds
  have binary := word_digits_binary placed paddingZero index indexBound
  have decodedBound :
      BoundedWordRows.decoded (layout.word index) assignment <
        2 ^ wordBitCount := by
    exact Nat.ofDigits_lt_base_pow_length (by decide : 1 < 2) binary
  have recomposed :=
    BoundedWordRows.recompositionRow_sound_of_decoded_bound
      (layout := layout.word index) (by
        simpa [Layout.word] using
          (show 2 ^ wordBitCount ≤ goldilocksP by decide))
      canonical one decodedBound
      (word_row_holds holds index indexBound)
  exact recomposed.trans
    (word_decoded_eq_packed placed paddingZero index indexBound)

def packedColumnValues (layout : Layout) (assignment : Nat → Nat) :
    List Nat :=
  (List.range packedWordCount).map fun index =>
    assignment (layout.packedWordColumn index)

/-- The 108 R1CS output columns equal the exact lossless frame encoder. -/
theorem packed_columns_eq_encodePacked
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    packedColumnValues layout assignment = encodePacked block := by
  apply List.ext_getElem
  · simp [packedColumnValues, encodePacked]
  · intro index leftBound rightBound
    have indexBound : index < packedWordCount := by
      simpa [packedColumnValues] using leftBound
    simp only [packedColumnValues, List.getElem_map, List.getElem_range,
      encodePacked, List.getElem_ofFn]
    exact packed_word_column_eq canonical one placed holds index indexBound

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (block : MemoryCarryParser.Block) : Prop where
  placed : PublicBitBlock.Placed layout.publicBits assignment block
  paddingZero : ∀ offset, offset < highPaddingBitCount →
    assignment (layout.paddedBitColumn (carryBits + offset)) = 0
  packedPlaced : ∀ (index : Nat) (bound : index < packedWordCount),
    assignment (layout.packedWordColumn index) =
      (packedWords block ⟨index, bound⟩).toNat

/-- The exact carry bits, zero padding, and packed outputs satisfy all 131
local rows. -/
theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (one : assignment 0 = 1)
    (honest : Honest layout assignment block) :
    Satisfies (rows layout) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with padMember | wordMember
  · rcases List.mem_map.mp padMember with
      ⟨offset, offsetMember, rfl⟩
    apply builderLinearRow_complete one _ [] (by simp [CanonicalTerms])
    simp [lcEval, honest.paddingZero offset
      (List.mem_range.mp offsetMember)]
  · rcases List.mem_map.mp wordMember with
      ⟨index, indexMember, rfl⟩
    have indexBound := List.mem_range.mp indexMember
    have binary := word_digits_binary honest.placed honest.paddingZero
      index indexBound
    have decodedBound :
        BoundedWordRows.decoded (layout.word index) assignment <
          2 ^ wordBitCount :=
      Nat.ofDigits_lt_base_pow_length (by decide : 1 < 2) binary
    apply BoundedWordRows.recompositionRow_complete_of_decoded
      (by
        simpa [Layout.word] using
          (show 2 ^ wordBitCount ≤ goldilocksP by decide))
      one _ decodedBound
    change assignment (layout.packedWordColumn index) =
      BoundedWordRows.decoded (layout.word index) assignment
    rw [honest.packedPlaced index indexBound]
    exact word_decoded_eq_packed honest.placed honest.paddingZero
      index indexBound |>.symm

end Nightstream.Implementation.Nebula.MemoryCarryHashPackingRows
