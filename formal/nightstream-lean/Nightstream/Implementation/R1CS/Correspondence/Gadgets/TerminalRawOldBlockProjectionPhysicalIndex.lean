import Nightstream.Implementation.R1CS.Correspondence.Gadgets.TerminalRawOldBlockProjectionCompiler

/-!
Compact physical indexing for the direct terminal raw-old-block projection.

The production program has a ragged tensor prefix, followed by a lane-major
rectangle of coordinate-product rows and a lane-major terminal family.  This
leaf gives proof-sized encoders and inverses for those three families.  It
never constructs the production-sized row list.
-/

namespace Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionPhysicalIndex

open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler

/-! ## Ragged tensor multiplication slots -/

inductive TensorSlot : List TensorLevel -> Type
  | here {level tail} : Fin level.multiplicationCount ->
      TensorSlot (level :: tail)
  | there {level tail} : TensorSlot tail -> TensorSlot (level :: tail)

def tensorSlotCount : List TensorLevel -> Nat
  | [] => 0
  | level :: tail => level.multiplicationCount + tensorSlotCount tail

def TensorSlot.at : {levels : List TensorLevel} ->
    (level : Fin levels.length) ->
    Fin (levels.get level).multiplicationCount -> TensorSlot levels
  | [], level, _ => Fin.elim0 level
  | _ :: _, level, multiplication =>
      Fin.cases
        (fun current => .here current)
        (fun next current => .there (TensorSlot.at next current))
        level multiplication

def TensorSlot.address : {levels : List TensorLevel} -> TensorSlot levels ->
    Sigma fun level : Fin levels.length =>
      Fin (levels.get level).multiplicationCount
  | _ :: _, .here multiplication => ⟨0, multiplication⟩
  | _ :: _, .there slot =>
      let address := slot.address
      ⟨address.1.succ, address.2⟩

@[simp] theorem TensorSlot.at_address
    {levels : List TensorLevel} (slot : TensorSlot levels) :
    TensorSlot.at slot.address.1 slot.address.2 = slot := by
  induction slot with
  | here multiplication => rfl
  | there slot inductionHypothesis =>
      simp [TensorSlot.address, TensorSlot.at, inductionHypothesis]

@[simp] theorem TensorSlot.address_at
    {levels : List TensorLevel} (level : Fin levels.length)
    (multiplication : Fin (levels.get level).multiplicationCount) :
    (TensorSlot.at level multiplication).address =
      ⟨level, multiplication⟩ := by
  induction levels with
  | nil => exact Fin.elim0 level
  | cons head tail inductionHypothesis =>
      revert multiplication
      refine Fin.cases ?_ (fun next => ?_) level
      · intro current
        rfl
      · intro current
        have tailAddress := inductionHypothesis next current
        let liftAddress :
            (Sigma fun tailLevel : Fin tail.length =>
              Fin (tail.get tailLevel).multiplicationCount) ->
            (Sigma fun wholeLevel : Fin (head :: tail).length =>
              Fin ((head :: tail).get wholeLevel).multiplicationCount) :=
          fun address => ⟨address.1.succ, address.2⟩
        change liftAddress (TensorSlot.at next current).address =
          liftAddress ⟨next, current⟩
        exact congrArg liftAddress tailAddress

def TensorSlot.toFin : {levels : List TensorLevel} ->
    TensorSlot levels -> Fin (tensorSlotCount levels)
  | _ :: _, .here multiplication =>
      ⟨multiplication.val, by
        simp only [tensorSlotCount]
        omega⟩
  | level :: _, .there slot =>
      ⟨level.multiplicationCount + slot.toFin.val, by
        simp only [tensorSlotCount]
        omega⟩

def TensorSlot.ofFin : {levels : List TensorLevel} ->
    Fin (tensorSlotCount levels) -> TensorSlot levels
  | [], index => Fin.elim0 index
  | level :: tail, index =>
      if within : index.val < level.multiplicationCount then
        .here ⟨index.val, within⟩
      else
        .there (TensorSlot.ofFin
          ⟨index.val - level.multiplicationCount, by
            have indexBound :
                index.val < level.multiplicationCount +
                  tensorSlotCount tail := by
              simpa only [tensorSlotCount] using index.isLt
            omega⟩)

@[simp] theorem TensorSlot.ofFin_toFin
    {levels : List TensorLevel} (slot : TensorSlot levels) :
    TensorSlot.ofFin slot.toFin = slot := by
  induction slot with
  | @here level tail multiplication =>
      simp [TensorSlot.toFin, TensorSlot.ofFin, multiplication.isLt]
  | @there level tail slot inductionHypothesis =>
      have outside :
          ¬((level.multiplicationCount + slot.toFin.val) <
            level.multiplicationCount) := by omega
      simp [TensorSlot.toFin, TensorSlot.ofFin, outside,
        inductionHypothesis]

@[simp] theorem TensorSlot.toFin_ofFin
    {levels : List TensorLevel} (index : Fin (tensorSlotCount levels)) :
    (TensorSlot.ofFin index).toFin = index := by
  induction levels with
  | nil => exact Fin.elim0 index
  | cons level tail inductionHypothesis =>
      by_cases within : index.val < level.multiplicationCount
      · apply Fin.ext
        simp [TensorSlot.ofFin, TensorSlot.toFin, within]
      · let tailIndex : Fin (tensorSlotCount tail) :=
          ⟨index.val - level.multiplicationCount, by
            have indexBound :
                index.val < level.multiplicationCount +
                  tensorSlotCount tail := by
              simpa only [tensorSlotCount] using index.isLt
            omega⟩
        have tailRoundTrip := inductionHypothesis tailIndex
        apply Fin.ext
        simp only [TensorSlot.ofFin, within, ↓reduceDIte,
          TensorSlot.toFin]
        rw [show (TensorSlot.ofFin tailIndex).toFin = tailIndex from
          tailRoundTrip]
        simp only [tailIndex]
        omega

private theorem foldl_tensorCount (levels : List TensorLevel)
    (initial : Nat) :
    levels.foldl (fun count level => count + level.multiplicationCount)
        initial =
      initial + tensorSlotCount levels := by
  induction levels generalizing initial with
  | nil => simp [tensorSlotCount]
  | cons level tail inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      simp [tensorSlotCount, Nat.add_assoc]

/-- The compact ordinal of a tensor multiplication is the exact count of all
preceding level slots plus its index within the current level. -/
theorem TensorSlot.toFin_at_val {levels : List TensorLevel}
    (level : Fin levels.length)
    (multiplication : Fin (levels.get level).multiplicationCount) :
    (TensorSlot.at level multiplication).toFin.val =
      (levels.take level.val).foldl
          (fun count current => count + current.multiplicationCount) 0 +
        multiplication.val := by
  induction levels with
  | nil => exact Fin.elim0 level
  | cons head tail inductionHypothesis =>
      revert multiplication
      refine Fin.cases ?_ (fun next => ?_) level
      · intro current
        change current.val = 0 + current.val
        omega
      · intro current
        change head.multiplicationCount +
            (TensorSlot.at next current).toFin.val =
          ((head :: tail).take (next.val + 1)).foldl
              (fun count level => count + level.multiplicationCount) 0 +
            current.val
        rw [List.take_succ_cons, List.foldl_cons]
        rw [inductionHypothesis next current]
        rw [foldl_tensorCount, foldl_tensorCount]
        simp [Nat.add_assoc]

theorem tensorMultiplicationCount_eq_slotCount (layout : Layout) :
    tensorMultiplicationCount layout = tensorSlotCount layout.tensorLevels := by
  unfold tensorMultiplicationCount
  simpa using foldl_tensorCount layout.tensorLevels 0

/-! ## Small rectangular encoders -/

def pairToFin {leftCount rightCount : Nat}
    (left : Fin leftCount) (right : Fin rightCount) :
    Fin (leftCount * rightCount) :=
  ⟨left.val * rightCount + right.val, by
    have nextLeft : left.val + 1 <= leftCount := by omega
    have currentBlock :
        (left.val + 1) * rightCount <= leftCount * rightCount :=
      Nat.mul_le_mul_right rightCount nextLeft
    have withinBlock :
        left.val * rightCount + right.val <
          (left.val + 1) * rightCount := by
      rw [Nat.add_mul]
      simp only [Nat.one_mul]
      omega
    exact Nat.lt_of_lt_of_le withinBlock currentBlock⟩

def finToPair {leftCount rightCount : Nat} (positiveRight : 0 < rightCount)
    (index : Fin (leftCount * rightCount)) :
    Fin leftCount × Fin rightCount :=
  (⟨index.val / rightCount, by
      rw [Nat.div_lt_iff_lt_mul positiveRight]
      exact index.isLt⟩,
   ⟨index.val % rightCount, Nat.mod_lt _ positiveRight⟩)

@[simp] theorem finToPair_pairToFin
    {leftCount rightCount : Nat} (positiveRight : 0 < rightCount)
    (left : Fin leftCount) (right : Fin rightCount) :
    finToPair positiveRight (pairToFin left right) = (left, right) := by
  apply Prod.ext <;> apply Fin.ext
  · change (left.val * rightCount + right.val) / rightCount = left.val
    rw [Nat.mul_comm left.val rightCount,
      Nat.mul_add_div positiveRight, Nat.div_eq_of_lt right.isLt]
    omega
  · simp [finToPair, pairToFin, Nat.mul_add_mod_of_lt right.isLt]

@[simp] theorem pairToFin_finToPair
    {leftCount rightCount : Nat} (positiveRight : 0 < rightCount)
    (index : Fin (leftCount * rightCount)) :
    pairToFin (finToPair positiveRight index).1
        (finToPair positiveRight index).2 = index := by
  apply Fin.ext
  change index.val / rightCount * rightCount +
      index.val % rightCount = index.val
  rw [Nat.mul_comm (index.val / rightCount) rightCount]
  exact Nat.div_add_mod index.val rightCount

/-! ## Existing compiler indices -/

def tensorRowToFin {layout : Layout} (index : TensorRowIndex layout) :
    Fin (tensorSlotCount layout.tensorLevels * 5) :=
  let slot := TensorSlot.at index.level index.multiplication
  pairToFin slot.toFin index.definition

def tensorRowOfFin {layout : Layout}
    (index : Fin (tensorSlotCount layout.tensorLevels * 5)) :
    TensorRowIndex layout :=
  let pair := finToPair (by decide : 0 < 5) index
  let address := (TensorSlot.ofFin pair.1).address
  { level := address.1
    multiplication := address.2
    definition := pair.2 }

@[simp] theorem tensorRowOfFin_toFin {layout : Layout}
    (index : TensorRowIndex layout) :
    tensorRowOfFin (tensorRowToFin index) = index := by
  rcases index with ⟨level, multiplication, definition⟩
  simp only [tensorRowToFin, tensorRowOfFin]
  rw [finToPair_pairToFin, TensorSlot.ofFin_toFin,
    TensorSlot.address_at]

@[simp] theorem tensorRowToFin_ofFin {layout : Layout}
    (index : Fin (tensorSlotCount layout.tensorLevels * 5)) :
    tensorRowToFin (tensorRowOfFin index) = index := by
  apply Fin.ext
  simp [tensorRowToFin, tensorRowOfFin]

/-- Block-major logical coordinate to the production lane-major rectangle. -/
def coordinateToLaneMajor {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (coordinate : Fin layout.logicalWidth) : Fin layout.logicalWidth :=
  let blockMajor : Fin (blockCount layout * layout.activeLanes) :=
    Fin.cast rectangle coordinate
  let pair := finToPair positiveLanes blockMajor
  Fin.cast (by rw [rectangle, Nat.mul_comm])
    (pairToFin pair.2 pair.1)

/-- Inverse lane-major rectangle decoder. -/
def coordinateOfLaneMajor {layout : Layout}
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (ordinal : Fin layout.logicalWidth) : Fin layout.logicalWidth :=
  let laneMajor : Fin (layout.activeLanes * blockCount layout) :=
    Fin.cast (by rw [rectangle, Nat.mul_comm]) ordinal
  let pair := finToPair positiveBlocks laneMajor
  Fin.cast rectangle.symm (pairToFin pair.2 pair.1)

@[simp] theorem coordinateOfLaneMajor_toLaneMajor {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (coordinate : Fin layout.logicalWidth) :
    coordinateOfLaneMajor positiveBlocks rectangle
        (coordinateToLaneMajor positiveLanes rectangle coordinate) =
      coordinate := by
  simp [coordinateOfLaneMajor, coordinateToLaneMajor]

@[simp] theorem coordinateToLaneMajor_ofLaneMajor {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (ordinal : Fin layout.logicalWidth) :
    coordinateToLaneMajor positiveLanes rectangle
        (coordinateOfLaneMajor positiveBlocks rectangle ordinal) =
      ordinal := by
  simp [coordinateOfLaneMajor, coordinateToLaneMajor]

/-! ## Full tensor/product/terminal physical order -/

def coordinateRowToFin {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (coordinate : Fin layout.logicalWidth) (limb : Fin 2) :
    Fin (2 * layout.logicalWidth) :=
  Fin.cast (Nat.mul_comm layout.logicalWidth 2)
    (pairToFin (coordinateToLaneMajor positiveLanes rectangle coordinate) limb)

def coordinateRowOfFin {layout : Layout}
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (index : Fin (2 * layout.logicalWidth)) :
    Fin layout.logicalWidth × Fin 2 :=
  let pair := finToPair (by decide : 0 < 2)
    (Fin.cast (Nat.mul_comm 2 layout.logicalWidth) index)
  (coordinateOfLaneMajor positiveBlocks rectangle pair.1,
    pair.2)

@[simp] theorem coordinateRowOfFin_toFin {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (coordinate : Fin layout.logicalWidth) (limb : Fin 2) :
    coordinateRowOfFin positiveBlocks rectangle
        (coordinateRowToFin positiveLanes rectangle coordinate limb) =
      (coordinate, limb) := by
  simp [coordinateRowOfFin, coordinateRowToFin]

@[simp] theorem coordinateRowToFin_ofFin {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (index : Fin (2 * layout.logicalWidth)) :
    coordinateRowToFin positiveLanes rectangle
        (coordinateRowOfFin positiveBlocks rectangle index).1
        (coordinateRowOfFin positiveBlocks rectangle index).2 =
      index := by
  simp [coordinateRowOfFin, coordinateRowToFin]

def terminalRowToFin {layout : Layout}
    (lane : Fin layout.activeLanes) (limb : Fin 2) :
    Fin (2 * layout.activeLanes) :=
  Fin.cast (Nat.mul_comm layout.activeLanes 2) (pairToFin lane limb)

def terminalRowOfFin {layout : Layout}
    (index : Fin (2 * layout.activeLanes)) : Fin layout.activeLanes × Fin 2 :=
  finToPair (by decide : 0 < 2)
    (Fin.cast (Nat.mul_comm 2 layout.activeLanes) index)

@[simp] theorem terminalRowOfFin_toFin {layout : Layout}
    (lane : Fin layout.activeLanes) (limb : Fin 2) :
    terminalRowOfFin (terminalRowToFin lane limb) = (lane, limb) := by
  simp [terminalRowOfFin, terminalRowToFin]

@[simp] theorem terminalRowToFin_ofFin {layout : Layout}
    (index : Fin (2 * layout.activeLanes)) :
    terminalRowToFin (terminalRowOfFin index).1
        (terminalRowOfFin index).2 = index := by
  simp [terminalRowOfFin, terminalRowToFin]

private def tensorRows (layout : Layout) : Nat :=
  tensorSlotCount layout.tensorLevels * 5

private def coordinateRows (layout : Layout) : Nat :=
  2 * layout.logicalWidth

private def terminalRows (layout : Layout) : Nat :=
  2 * layout.activeLanes

private def segmentedRowCount (layout : Layout) : Nat :=
  (tensorRows layout + coordinateRows layout) + terminalRows layout

private theorem rowCount_eq_segmented (layout : Layout) :
    rowCount layout = segmentedRowCount layout := by
  unfold rowCount segmentedRowCount tensorRows coordinateRows terminalRows
  rw [tensorMultiplicationCount_eq_slotCount]
  omega

private def segmentedPhysicalIndex {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes) :
    RowIndex layout -> Fin (segmentedRowCount layout)
  | .tensor index =>
      Fin.castAdd (terminalRows layout)
        (Fin.castAdd (coordinateRows layout) (tensorRowToFin index))
  | .coordinate coordinate limb =>
      Fin.castAdd (terminalRows layout)
        (Fin.natAdd (tensorRows layout)
          (coordinateRowToFin positiveLanes rectangle coordinate limb))
  | .terminal lane limb =>
      Fin.natAdd (tensorRows layout + coordinateRows layout)
        (terminalRowToFin lane limb)

private def segmentedPhysicalOwner {layout : Layout}
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (index : Fin (segmentedRowCount layout)) : RowIndex layout :=
  Fin.addCases
    (fun prefixIndex =>
      Fin.addCases
        (fun tensor => .tensor (tensorRowOfFin tensor))
        (fun coordinate =>
          let decoded := coordinateRowOfFin positiveBlocks rectangle coordinate
          .coordinate decoded.1 decoded.2)
        prefixIndex)
    (fun terminal =>
      let decoded := terminalRowOfFin terminal
      .terminal decoded.1 decoded.2)
    index

@[simp] private theorem segmentedPhysicalOwner_index {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (index : RowIndex layout) :
    segmentedPhysicalOwner positiveBlocks rectangle
        (segmentedPhysicalIndex positiveLanes rectangle index) = index := by
  cases index <;> simp [segmentedPhysicalOwner, segmentedPhysicalIndex]

@[simp] private theorem segmentedPhysicalIndex_owner {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (index : Fin (segmentedRowCount layout)) :
    segmentedPhysicalIndex positiveLanes rectangle
        (segmentedPhysicalOwner positiveBlocks rectangle index) =
      index := by
  refine Fin.addCases ?_ ?_ index
  · intro prefixIndex
    refine Fin.addCases ?_ ?_ prefixIndex
    · intro tensor
      simp [segmentedPhysicalOwner, segmentedPhysicalIndex]
    · intro coordinate
      simp [segmentedPhysicalOwner, segmentedPhysicalIndex]
  · intro terminal
    simp [segmentedPhysicalOwner, segmentedPhysicalIndex]

/-- Exact Rust physical order: round-major ragged tensor rows, then
lane-major/block-minor product rows, then lane/limb terminal rows. -/
def physicalIndex {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (index : RowIndex layout) : Fin (rowCount layout) :=
  Fin.cast (rowCount_eq_segmented layout).symm
    (segmentedPhysicalIndex positiveLanes rectangle index)

/-- Numeric row offset of one tensor definition in the exact Rust order. -/
theorem physicalIndex_tensor_val {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (index : TensorRowIndex layout) :
    (physicalIndex positiveLanes rectangle (.tensor index)).val =
      (TensorSlot.at index.level index.multiplication).toFin.val * 5 +
        index.definition.val := by
  rfl

/-- Numeric row offset of one block-major coordinate after Rust's
lane-major permutation. -/
theorem physicalIndex_coordinate_val {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (coordinate : Fin layout.logicalWidth) (limb : Fin 2) :
    (physicalIndex positiveLanes rectangle (.coordinate coordinate limb)).val =
      tensorMultiplicationCount layout * 5 +
        2 * ((coordinate.val % layout.activeLanes) * blockCount layout +
          coordinate.val / layout.activeLanes) + limb.val := by
  simp [physicalIndex, segmentedPhysicalIndex, coordinateRowToFin,
    coordinateToLaneMajor, finToPair, pairToFin, tensorRows,
    tensorMultiplicationCount_eq_slotCount]
  omega

/-- Numeric row offset of one terminal parent-lane equation. -/
theorem physicalIndex_terminal_val {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (lane : Fin layout.activeLanes) (limb : Fin 2) :
    (physicalIndex positiveLanes rectangle (.terminal lane limb)).val =
      tensorMultiplicationCount layout * 5 + 2 * layout.logicalWidth +
        2 * lane.val + limb.val := by
  simp [physicalIndex, segmentedPhysicalIndex, terminalRowToFin,
    pairToFin, tensorRows, coordinateRows,
    tensorMultiplicationCount_eq_slotCount]
  omega

def physicalOwner {layout : Layout}
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (index : Fin (rowCount layout)) : RowIndex layout :=
  segmentedPhysicalOwner positiveBlocks rectangle
    (Fin.cast (rowCount_eq_segmented layout) index)

@[simp] theorem physicalOwner_index {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (index : RowIndex layout) :
    physicalOwner positiveBlocks rectangle
        (physicalIndex positiveLanes rectangle index) = index := by
  simp [physicalOwner, physicalIndex]

@[simp] theorem physicalIndex_owner {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes)
    (index : Fin (rowCount layout)) :
    physicalIndex positiveLanes rectangle
        (physicalOwner positiveBlocks rectangle index) = index := by
  simp [physicalOwner, physicalIndex]

theorem physicalIndex_injective {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes) :
    Function.Injective (physicalIndex positiveLanes rectangle) := by
  intro left right equal
  have := congrArg
    (physicalOwner positiveBlocks rectangle) equal
  simpa using this

theorem physicalIndex_surjective {layout : Layout}
    (positiveLanes : 0 < layout.activeLanes)
    (positiveBlocks : 0 < blockCount layout)
    (rectangle : layout.logicalWidth = blockCount layout * layout.activeLanes) :
    Function.Surjective (physicalIndex positiveLanes rectangle) := by
  intro index
  exact ⟨physicalOwner positiveBlocks rectangle index,
    physicalIndex_owner positiveLanes positiveBlocks rectangle index⟩

end Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionPhysicalIndex
