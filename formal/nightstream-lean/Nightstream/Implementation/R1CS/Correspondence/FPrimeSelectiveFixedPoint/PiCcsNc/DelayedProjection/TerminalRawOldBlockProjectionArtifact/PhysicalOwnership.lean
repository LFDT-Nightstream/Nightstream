import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.Schedule

/-!
Physical ownership for the generated final-round-factorized production
raw-old-block projection.

This leaf owns the exact conceptual-to-physical row permutation and the unique
generated owner of every tensor, coordinate-product, final-scale, and terminal
row.  In particular, the final 108 rows are the parent-versus-scale-output
checks; this module contains no owner for the deleted direct parent-versus-
prefix-sum equations.

Owns: the four-family row census, the generated conceptual-to-physical
permutation, and existence and uniqueness of one owner for every row.

Does not own: row coefficients, column placement, witness assignments, row
satisfaction, semantic acceptance, security events, or row-removal authority.

Emits constraints: no.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.projection_rows.census` | 1,310,715 tensor, 22,874,076 product, 270 scale, and 108 terminal rows sum to 24,185,169 | derived |
| `f_prime.pi_ccs_nc.delayed.projection_rows.permutation` | the generated row ordinal equals the optimized compiler physical index | derived |
| `f_prime.pi_ccs_nc.delayed.projection_rows.unique_owner` | every physical artifact row has exactly one coherent four-family owner | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionPhysicalIndex
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt

private def productionTensorRows : Nat :=
  tensorSlotCount productionLayout.tensorLevels * 5

private def productionCoordinateRows : Nat :=
  2 * productionLayout.logicalWidth

private def productionScaleRows : Nat :=
  productionFactoredLayout.base.activeLanes * 5

private def productionTerminalRows : Nat :=
  2 * productionFactoredLayout.base.activeLanes

private def productionSegmentedRowCount : Nat :=
  ((productionTensorRows + productionCoordinateRows) + productionScaleRows) +
    productionTerminalRows

private theorem productionRowCount_eq_segmented :
    rowCount productionFactoredLayout = productionSegmentedRowCount := by
  unfold rowCount tensorMultiplicationCount productionSegmentedRowCount
    productionTensorRows productionCoordinateRows productionScaleRows
    productionTerminalRows
  rw [tensorMultiplicationCount_eq_slotCount]
  simp only [productionFactoredBase]
  omega

/-! The four constructors below deliberately mirror Rust's emitter order. -/

private def productionSegmentedPhysicalIndex :
    RowIndex productionFactoredLayout -> Fin productionSegmentedRowCount
  | .tensor index =>
      Fin.castAdd productionTerminalRows
        (Fin.castAdd productionScaleRows
          (Fin.castAdd productionCoordinateRows (tensorRowToFin index)))
  | .coordinate coordinate limb =>
      Fin.castAdd productionTerminalRows
        (Fin.castAdd productionScaleRows
          (Fin.natAdd productionTensorRows
            (coordinateRowToFin productionPositiveLanes productionRectangle
              coordinate limb)))
  | .scale lane definition =>
      Fin.castAdd productionTerminalRows
        (Fin.natAdd (productionTensorRows + productionCoordinateRows)
          (pairToFin lane definition))
  | .terminal lane limb =>
      Fin.natAdd
        ((productionTensorRows + productionCoordinateRows) +
          productionScaleRows)
        (terminalRowToFin lane limb)

private def productionSegmentedPhysicalOwner
    (index : Fin productionSegmentedRowCount) :
    RowIndex productionFactoredLayout :=
  Fin.addCases
    (fun beforeTerminal =>
      Fin.addCases
        (fun beforeScale =>
          Fin.addCases
            (fun tensor => .tensor (tensorRowOfFin tensor))
            (fun coordinate =>
              let decoded := coordinateRowOfFin productionPositiveBlocks
                productionRectangle coordinate
              .coordinate decoded.1 decoded.2)
            beforeScale)
        (fun scale =>
          let decoded := finToPair (by decide : 0 < 5) scale
          .scale decoded.1 decoded.2)
        beforeTerminal)
    (fun terminal =>
      let decoded := terminalRowOfFin terminal
      .terminal decoded.1 decoded.2)
    index

@[simp] private theorem productionSegmentedPhysicalOwner_index
    (index : RowIndex productionFactoredLayout) :
    productionSegmentedPhysicalOwner
        (productionSegmentedPhysicalIndex index) = index := by
  cases index with
  | tensor index =>
      unfold productionSegmentedPhysicalOwner
        productionSegmentedPhysicalIndex
      simp only [Fin.addCases_left]
      simpa [productionFactoredLayout] using tensorRowOfFin_toFin index
  | coordinate coordinate limb =>
      unfold productionSegmentedPhysicalOwner
        productionSegmentedPhysicalIndex
      simp only [Fin.addCases_left, Fin.addCases_right]
      exact congrArg
        (fun decoded : Fin productionLayout.logicalWidth × Fin 2 =>
          RowIndex.coordinate (layout := productionFactoredLayout)
            (Fin.cast (by rfl) decoded.1) decoded.2)
        (coordinateRowOfFin_toFin productionPositiveLanes
          productionPositiveBlocks productionRectangle coordinate limb)
  | scale lane definition =>
      unfold productionSegmentedPhysicalOwner
        productionSegmentedPhysicalIndex
      simp only [Fin.addCases_left, Fin.addCases_right]
      exact congrArg
        (fun decoded => RowIndex.scale decoded.1 decoded.2)
        (finToPair_pairToFin (by decide : 0 < 5) lane definition)
  | terminal lane limb =>
      unfold productionSegmentedPhysicalOwner
        productionSegmentedPhysicalIndex
      simp only [Fin.addCases_right]
      exact congrArg
        (fun decoded => RowIndex.terminal decoded.1 decoded.2)
        (terminalRowOfFin_toFin lane limb)

@[simp] private theorem productionSegmentedPhysicalIndex_owner
    (index : Fin productionSegmentedRowCount) :
    productionSegmentedPhysicalIndex
        (productionSegmentedPhysicalOwner index) = index := by
  refine Fin.addCases ?_ ?_ index
  · intro beforeTerminal
    refine Fin.addCases ?_ ?_ beforeTerminal
    · intro beforeScale
      refine Fin.addCases ?_ ?_ beforeScale
      · intro tensor
        unfold productionSegmentedPhysicalOwner
          productionSegmentedPhysicalIndex
        simp only [Fin.addCases_left]
        exact congrArg
          (fun value => Fin.castAdd productionTerminalRows
            (Fin.castAdd productionScaleRows
              (Fin.castAdd productionCoordinateRows value)))
          (tensorRowToFin_ofFin tensor)
      · intro coordinate
        unfold productionSegmentedPhysicalOwner
          productionSegmentedPhysicalIndex
        simp only [Fin.addCases_left, Fin.addCases_right]
        rw [coordinateRowToFin_ofFin]
    · intro scale
      unfold productionSegmentedPhysicalOwner
        productionSegmentedPhysicalIndex
      simp only [Fin.addCases_left, Fin.addCases_right]
      rw [pairToFin_finToPair]
  · intro terminal
    unfold productionSegmentedPhysicalOwner
      productionSegmentedPhysicalIndex
    simp only [Fin.addCases_right]
    rw [terminalRowToFin_ofFin]

/-- Exact optimized compiler-to-artifact order: tensor, product, final scale,
then parent-versus-scale terminal rows. -/
def productionPhysicalIndex :
    RowIndex productionFactoredLayout -> Fin (rowCount productionFactoredLayout) :=
  fun index => Fin.cast productionRowCount_eq_segmented.symm
    (productionSegmentedPhysicalIndex index)

private def productionPhysicalOwner
    (index : Fin (rowCount productionFactoredLayout)) :
    RowIndex productionFactoredLayout :=
  productionSegmentedPhysicalOwner
    (Fin.cast productionRowCount_eq_segmented index)

@[simp] theorem productionPhysicalOwner_index
    (index : RowIndex productionFactoredLayout) :
    productionPhysicalOwner (productionPhysicalIndex index) = index := by
  simp [productionPhysicalOwner, productionPhysicalIndex]

@[simp] theorem productionPhysicalIndex_owner
    (index : Fin (rowCount productionFactoredLayout)) :
    productionPhysicalIndex (productionPhysicalOwner index) = index := by
  simp [productionPhysicalOwner, productionPhysicalIndex]

theorem productionPhysicalIndex_injective :
    Function.Injective productionPhysicalIndex := by
  intro left right equal
  have := congrArg productionPhysicalOwner equal
  simpa using this

theorem productionPhysicalIndex_surjective :
    Function.Surjective productionPhysicalIndex := by
  intro index
  exact ⟨productionPhysicalOwner index,
    productionPhysicalIndex_owner index⟩

/-- Exact multiplicity of each generated family. -/
theorem productionFourFamilyCensus :
    productionTensorRows = 1310715 /\
    productionCoordinateRows = 22874076 /\
    productionScaleRows = 270 /\
    productionTerminalRows = 108 /\
    productionSegmentedRowCount = 24185169 := by
  decide

private theorem fin18_cases {predicate : Fin 18 -> Prop}
    (case0 : predicate 0) (case1 : predicate 1)
    (case2 : predicate 2) (case3 : predicate 3)
    (case4 : predicate 4) (case5 : predicate 5)
    (case6 : predicate 6) (case7 : predicate 7)
    (case8 : predicate 8) (case9 : predicate 9)
    (case10 : predicate 10) (case11 : predicate 11)
    (case12 : predicate 12) (case13 : predicate 13)
    (case14 : predicate 14) (case15 : predicate 15)
    (case16 : predicate 16) (case17 : predicate 17) :
    forall index, predicate index := by
  intro index
  refine Fin.cases case0 ?_ index
  intro index
  refine Fin.cases case1 ?_ index
  intro index
  refine Fin.cases case2 ?_ index
  intro index
  refine Fin.cases case3 ?_ index
  intro index
  refine Fin.cases case4 ?_ index
  intro index
  refine Fin.cases case5 ?_ index
  intro index
  refine Fin.cases case6 ?_ index
  intro index
  refine Fin.cases case7 ?_ index
  intro index
  refine Fin.cases case8 ?_ index
  intro index
  refine Fin.cases case9 ?_ index
  intro index
  refine Fin.cases case10 ?_ index
  intro index
  refine Fin.cases case11 ?_ index
  intro index
  refine Fin.cases case12 ?_ index
  intro index
  refine Fin.cases case13 ?_ index
  intro index
  refine Fin.cases case14 ?_ index
  intro index
  refine Fin.cases case15 ?_ index
  intro index
  refine Fin.cases case16 ?_ index
  intro index
  have valueZero : index.val = 0 :=
    Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ index.isLt)
  have indexZero : index = (0 : Fin 1) := Fin.ext valueZero
  subst index
  exact case17

private theorem productionTensorSlotOrdinal
    (level : Fin productionLayout.tensorLevels.length)
    (multiplication : Fin
      (productionLayout.tensorLevels.get level).multiplicationCount) :
    (TensorSlot.at level multiplication).toFin.val =
      tensorMulOrdinal level.val multiplication.val := by
  rw [TensorSlot.toFin_at_val]
  change
    (productionTensorLevels.take level.val).foldl
        (fun count current => count + current.multiplicationCount) 0 +
      multiplication.val =
    tensorRoundMulStart level.val + multiplication.val
  congr 1
  unfold productionTensorLevels tensorRoundMulStart
  rw [← List.map_take]
  change
    List.foldl (fun count current => count + current.multiplicationCount) 0
        (List.map productionTensorLevel
          (List.take level.val (List.range 18))) =
      List.foldl (fun count prior => count + tensorRoundMulCount prior) 0
        (List.range level.val)
  have levelLt : level.val < 18 := by
    have := level.isLt
    change level.val < 18 at this
    exact this
  rw [List.take_range, Nat.min_eq_left (Nat.le_of_lt levelLt)]
  rw [List.foldl_map]
  rfl

private theorem tensorOwner_case
    (round start count : Nat)
    (startEq : tensorRoundMulStart round = start)
    (countEq : tensorRoundMulCount round = count)
    (owns : forall parent, parent < count ->
      tensorOwner (start + parent) = (round, parent))
    (parent : Fin (tensorRoundMulCount round)) :
    tensorOwner (tensorMulOrdinal round parent.val) =
      (round, parent.val) := by
  have parentLt : parent.val < count := by
    calc
      parent.val < tensorRoundMulCount round := parent.isLt
      _ = count := countEq
  change tensorOwner (tensorRoundMulStart round + parent.val) = _
  rw [startEq]
  exact owns parent.val parentLt

private theorem generatedTensorOwnerAt (level : Fin 18)
    (parent : Fin (tensorRoundMulCount level.val)) :
    tensorOwner (tensorMulOrdinal level.val parent.val) =
      (level.val, parent.val) := by
  revert parent level
  apply fin18_cases <;> intro parent
  · exact tensorOwner_case 0 0 1 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 1 1 2 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 2 3 4 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 3 7 8 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 4 15 16 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 5 31 32 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 6 63 64 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 7 127 128 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 8 255 256 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 9 511 512 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 10 1023 1024 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 11 2047 2048 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 12 4095 4096 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 13 8191 8192 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 14 16383 16384 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 15 32767 32768 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 16 65535 65536 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent
  · exact tensorOwner_case 17 131071 131072 (by decide) (by decide)
      (by intro value valueLt; unfold tensorOwner; grind) parent

private theorem productionTensorOwnerAt
    (level : Fin productionLayout.tensorLevels.length)
    (multiplication : Fin
      (productionLayout.tensorLevels.get level).multiplicationCount) :
    tensorOwner (tensorMulOrdinal level.val multiplication.val) =
      (level.val, multiplication.val) := by
  let round : Fin 18 := ⟨level.val, by
    have levelLt := level.isLt
    change level.val < 18 at levelLt
    exact levelLt⟩
  let parent : Fin (tensorRoundMulCount round.val) :=
    ⟨multiplication.val, by
      have multiplicationLt := multiplication.isLt
      simpa [round, productionLayout, productionTensorLevels,
        productionTensorLevel] using multiplicationLt⟩
  simpa [round, parent] using generatedTensorOwnerAt round parent

private theorem productionPhysicalIndex_tensor
    (index : PrefixTensorRowIndex productionFactoredLayout.base) :
    (productionPhysicalIndex (.tensor index)).val =
      tensorPhysicalRow index.level.val index.multiplication.val
        index.definition.val := by
  rcases index with ⟨level, multiplication, definition⟩
  have ordinalEq :
      (TensorSlot.at level multiplication).toFin.val =
        tensorMulOrdinal level.val multiplication.val := by
    simpa [productionFactoredLayout] using
      productionTensorSlotOrdinal level multiplication
  change
    (TensorSlot.at level multiplication).toFin.val * 5 + definition.val =
      tensorPhysicalRow level.val multiplication.val definition.val
  unfold tensorPhysicalRow
  rw [ordinalEq]
  omega

private theorem productionPhysicalIndex_coordinate
    (coordinate : Fin productionFactoredLayout.base.logicalWidth)
    (limb : Fin 2) :
    (productionPhysicalIndex (.coordinate coordinate limb)).val =
      productPhysicalRow (coordinate.val % 54) (coordinate.val / 54)
        limb.val := by
  change
    tensorSlotCount productionLayout.tensorLevels * 5 +
        (((coordinate.val % productionLayout.activeLanes) *
          Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.blockCount
            productionLayout +
          coordinate.val / productionLayout.activeLanes) * 2 + limb.val) =
      productPhysicalRow (coordinate.val % 54) (coordinate.val / 54)
        limb.val
  rw [show tensorSlotCount productionLayout.tensorLevels = 262143 by
    rw [← tensorMultiplicationCount_eq_slotCount]
    exact productionTensorMultiplicationCount]
  rw [productionActiveLanes, productionBlockCount]
  unfold productPhysicalRow witnessOffset productRowFirst
  simp only [Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockCount]
  have coordinateLt := coordinate.isLt
  change coordinate.val < 11437038 at coordinateLt
  omega

private theorem productionPhysicalIndex_scale
    (lane : Fin productionFactoredLayout.base.activeLanes)
    (definition : Fin 5) :
    (productionPhysicalIndex (.scale lane definition)).val =
      finalScalePhysicalRow lane.val definition.val := by
  change
    tensorSlotCount productionLayout.tensorLevels * 5 +
        2 * productionLayout.logicalWidth +
        (lane.val * 5 + definition.val) =
      finalScalePhysicalRow lane.val definition.val
  rw [show tensorSlotCount productionLayout.tensorLevels = 262143 by
    rw [← tensorMultiplicationCount_eq_slotCount]
    exact productionTensorMultiplicationCount]
  rw [productionLogicalWidth]
  unfold finalScalePhysicalRow finalScaleRowFirst
  omega

private theorem productionPhysicalIndex_terminal
    (lane : Fin productionFactoredLayout.base.activeLanes) (limb : Fin 2) :
    (productionPhysicalIndex (.terminal lane limb)).val =
      terminalPhysicalRow lane.val limb.val := by
  change
    tensorSlotCount productionLayout.tensorLevels * 5 +
        2 * productionLayout.logicalWidth +
        productionFactoredLayout.base.activeLanes * 5 +
        (lane.val * 2 + limb.val) =
      terminalPhysicalRow lane.val limb.val
  rw [show tensorSlotCount productionLayout.tensorLevels = 262143 by
    rw [← tensorMultiplicationCount_eq_slotCount]
    exact productionTensorMultiplicationCount]
  rw [productionLogicalWidth]
  simp only [productionFactoredBase, productionActiveLanes]
  unfold terminalPhysicalRow terminalRowFirst
  omega

/-- Every generated tensor ordinal decodes to exactly one valid round and
parent.  This is the inverse direction needed by the list-free production SSA
program; it is derived from the already proved physical-row permutation rather
than by normalizing Rust's full 18-way decoder. -/
theorem productionTensorOwner_valid
    (ordinal : Nat) (inRange : ordinal < 262143) :
    let owner := tensorOwner ordinal
    owner.1 < 18 /\
      owner.2 < tensorRoundMulCount owner.1 /\
      tensorMulOrdinal owner.1 owner.2 = ordinal := by
  let row : Fin (rowCount productionFactoredLayout) :=
    ⟨5 * ordinal, by
      rw [productionRowCount_exact]
      omega⟩
  obtain ⟨index, indexEq⟩ := productionPhysicalIndex_surjective row
  cases index with
  | tensor index =>
      rcases index with ⟨level, multiplication, definition⟩
      have valueEq := congrArg Fin.val indexEq
      rw [productionPhysicalIndex_tensor] at valueEq
      have definitionLt : definition.val < 5 := definition.isLt
      have ordinalEq :
          tensorMulOrdinal level.val multiplication.val = ordinal := by
        dsimp [row] at valueEq
        unfold tensorPhysicalRow at valueEq
        omega
      have levelLt : level.val < 18 := by
        have value := level.isLt
        change level.val < 18 at value
        exact value
      have multiplicationLt :
          multiplication.val < tensorRoundMulCount level.val := by
        have value := multiplication.isLt
        simpa [productionLayout, productionTensorLevels,
          productionTensorLevel] using value
      dsimp
      rw [← ordinalEq, productionTensorOwnerAt]
      exact ⟨levelLt, multiplicationLt, rfl⟩
  | coordinate coordinate limb =>
      have valueEq := congrArg Fin.val indexEq
      rw [productionPhysicalIndex_coordinate] at valueEq
      have rowBefore : row.val < productRowFirst := by
        dsimp [row]
        unfold productRowFirst
        omega
      have ownerAfter :
          productRowFirst <=
            productPhysicalRow (coordinate.val % 54)
              (coordinate.val / 54) limb.val := by
        unfold productPhysicalRow
        omega
      omega
  | scale lane definition =>
      have valueEq := congrArg Fin.val indexEq
      rw [productionPhysicalIndex_scale] at valueEq
      have rowBefore : row.val < finalScaleRowFirst := by
        dsimp [row]
        unfold finalScaleRowFirst
        omega
      have ownerAfter :
          finalScaleRowFirst <=
            finalScalePhysicalRow lane.val definition.val := by
        unfold finalScalePhysicalRow
        omega
      omega
  | terminal lane limb =>
      have valueEq := congrArg Fin.val indexEq
      rw [productionPhysicalIndex_terminal] at valueEq
      have rowBefore : row.val < terminalRowFirst := by
        dsimp [row]
        unfold terminalRowFirst
        omega
      have ownerAfter :
          terminalRowFirst <= terminalPhysicalRow lane.val limb.val := by
        unfold terminalPhysicalRow
        omega
      omega

theorem productionOwner_tensor
    (index : PrefixTensorRowIndex productionFactoredLayout.base) :
    ownerAt (productionPhysicalIndex (.tensor index)) =
      RowOwner.tensor index.level.val index.multiplication.val
        index.definition.val := by
  rcases index with ⟨level, multiplication, definition⟩
  change ownerAtNat (productionPhysicalIndex
      (.tensor ⟨level, multiplication, definition⟩)).val = _
  rw [productionPhysicalIndex_tensor]
  have slotLt := (TensorSlot.at level multiplication).toFin.isLt
  have ordinalLt :
      tensorMulOrdinal level.val multiplication.val < 262143 := by
    calc
      tensorMulOrdinal level.val multiplication.val =
          (TensorSlot.at level multiplication).toFin.val :=
        (productionTensorSlotOrdinal level multiplication).symm
      _ < tensorSlotCount productionLayout.tensorLevels := slotLt
      _ = 262143 := by
        rw [← tensorMultiplicationCount_eq_slotCount]
        exact productionTensorMultiplicationCount
  have definitionLt : definition.val < 5 := definition.isLt
  have rowLt :
      tensorPhysicalRow level.val multiplication.val definition.val <
        tensorRows := by
    unfold tensorPhysicalRow tensorRows
    omega
  have rowDiv :
      tensorPhysicalRow level.val multiplication.val definition.val / 5 =
        tensorMulOrdinal level.val multiplication.val := by
    simp [tensorPhysicalRow]
    omega
  have rowMod :
      tensorPhysicalRow level.val multiplication.val definition.val % 5 =
        definition.val := by
    simp [tensorPhysicalRow, Nat.mod_eq_of_lt definitionLt]
  unfold ownerAtNat
  rw [if_pos rowLt, rowDiv, rowMod]
  change
    RowOwner.tensor
        (tensorOwner (tensorMulOrdinal level.val multiplication.val)).1
        (tensorOwner (tensorMulOrdinal level.val multiplication.val)).2
        definition.val =
      RowOwner.tensor level.val multiplication.val definition.val
  rw [productionTensorOwnerAt]

theorem productionOwner_coordinate
    (coordinate : Fin productionFactoredLayout.base.logicalWidth)
    (limb : Fin 2) :
    ownerAt (productionPhysicalIndex (.coordinate coordinate limb)) =
      RowOwner.product (coordinate.val % 54) (coordinate.val / 54)
        limb.val := by
  have coordinateLt : coordinate.val < 11437038 := by
    change coordinate.val < 11437038
    exact coordinate.isLt
  have blockLt : coordinate.val / 54 < 211797 :=
    (Nat.div_lt_iff_lt_mul (by decide : 0 < 54)).2 coordinateLt
  have limbLt : limb.val < 2 := limb.isLt
  change ownerAtNat (productionPhysicalIndex
      (.coordinate coordinate limb)).val = _
  rw [productionPhysicalIndex_coordinate]
  let ordinal :=
    (coordinate.val % 54) * 211797 + coordinate.val / 54
  have offsetEq :
      productPhysicalRow (coordinate.val % 54) (coordinate.val / 54)
          limb.val - productRowFirst =
        2 * ordinal + limb.val := by
    simp [productPhysicalRow, witnessOffset, productRowFirst, ordinal,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockCount]
    omega
  have productEq : (2 * ordinal + limb.val) / 2 = ordinal := by omega
  have limbEq : (2 * ordinal + limb.val) % 2 = limb.val := by omega
  have laneEq : ordinal / 211797 = coordinate.val % 54 := by
    change
      (coordinate.val % 54 * 211797 + coordinate.val / 54) /
          211797 = coordinate.val % 54
    rw [Nat.mul_comm (coordinate.val % 54) 211797,
      Nat.mul_add_div (by decide : 0 < 211797),
      Nat.div_eq_of_lt blockLt, Nat.add_zero]
  have blockEq : ordinal % 211797 = coordinate.val / 54 :=
    Nat.mul_add_mod_of_lt blockLt
  have notTensor :
      ¬ productPhysicalRow (coordinate.val % 54)
          (coordinate.val / 54) limb.val < tensorRows := by
    change
      ¬ (1310715 + 2 * ((coordinate.val % 54) * 211797 +
        coordinate.val / 54) + limb.val < 1310715)
    omega
  have beforeScale :
      productPhysicalRow (coordinate.val % 54)
          (coordinate.val / 54) limb.val < finalScaleRowFirst := by
    have laneLt : coordinate.val % 54 < 54 :=
      Nat.mod_lt _ (by decide)
    change
      1310715 + 2 * ((coordinate.val % 54) * 211797 +
        coordinate.val / 54) + limb.val < 24184791
    omega
  unfold ownerAtNat
  rw [if_neg notTensor, if_pos beforeScale]
  change
    RowOwner.product
        (((productPhysicalRow (coordinate.val % 54)
            (coordinate.val / 54) limb.val - productRowFirst) / 2) /
          211797)
        (((productPhysicalRow (coordinate.val % 54)
            (coordinate.val / 54) limb.val - productRowFirst) / 2) %
          211797)
        ((productPhysicalRow (coordinate.val % 54)
            (coordinate.val / 54) limb.val - productRowFirst) % 2) =
      RowOwner.product (coordinate.val % 54) (coordinate.val / 54)
        limb.val
  rw [offsetEq, productEq]
  change
    RowOwner.product (ordinal / 211797) (ordinal % 211797)
        ((2 * ordinal + limb.val) % 2) =
      RowOwner.product (coordinate.val % 54) (coordinate.val / 54)
        limb.val
  rw [laneEq, blockEq, limbEq]

theorem productionOwner_scale
    (lane : Fin productionFactoredLayout.base.activeLanes)
    (definition : Fin 5) :
    ownerAt (productionPhysicalIndex (.scale lane definition)) =
      RowOwner.finalScale lane.val definition.val := by
  change ownerAtNat (productionPhysicalIndex
      (.scale lane definition)).val = _
  rw [productionPhysicalIndex_scale]
  have laneLt : lane.val < 54 := by
    have := lane.isLt
    change lane.val < 54 at this
    exact this
  have definitionLt : definition.val < 5 := definition.isLt
  have offsetEq :
      finalScalePhysicalRow lane.val definition.val - finalScaleRowFirst =
        5 * lane.val + definition.val := by
    simp [finalScalePhysicalRow, finalScaleRowFirst]
    omega
  have laneEq : (5 * lane.val + definition.val) / 5 = lane.val := by
    rw [Nat.mul_add_div (by decide : 0 < 5),
      Nat.div_eq_of_lt definitionLt, Nat.add_zero]
  have definitionEq :
      (5 * lane.val + definition.val) % 5 = definition.val := by
    simpa [Nat.mod_eq_of_lt definitionLt] using
      Nat.mul_add_mod_self_right lane.val 5 definition.val
  unfold ownerAtNat
  rw [if_neg (by
    change ¬ (24184791 + 5 * lane.val + definition.val < 1310715)
    omega)]
  rw [if_neg (by
    change ¬ (24184791 + 5 * lane.val + definition.val < 24184791)
    omega)]
  rw [if_pos (by
    change 24184791 + 5 * lane.val + definition.val < 24185061
    omega)]
  change
    RowOwner.finalScale
      ((finalScalePhysicalRow lane.val definition.val -
          finalScaleRowFirst) / 5)
      ((finalScalePhysicalRow lane.val definition.val -
          finalScaleRowFirst) % 5) = _
  rw [offsetEq, laneEq, definitionEq]

theorem productionOwner_terminal
    (lane : Fin productionFactoredLayout.base.activeLanes) (limb : Fin 2) :
    ownerAt (productionPhysicalIndex (.terminal lane limb)) =
      RowOwner.terminal lane.val limb.val := by
  change ownerAtNat (productionPhysicalIndex (.terminal lane limb)).val = _
  rw [productionPhysicalIndex_terminal]
  have laneLt : lane.val < 54 := by
    have := lane.isLt
    change lane.val < 54 at this
    exact this
  have limbLt : limb.val < 2 := limb.isLt
  let offset := 2 * lane.val + limb.val
  have offsetEq :
      terminalPhysicalRow lane.val limb.val - terminalRowFirst = offset := by
    simp [terminalPhysicalRow, terminalRowFirst, offset]
    omega
  have laneEq : offset / 2 = lane.val := by omega
  have limbEq : offset % 2 = limb.val := by omega
  unfold ownerAtNat
  rw [if_neg (by
    change ¬ (24185061 + 2 * lane.val + limb.val < 1310715)
    omega)]
  rw [if_neg (by
    change ¬ (24185061 + 2 * lane.val + limb.val < 24184791)
    omega)]
  rw [if_neg (by
    change ¬ (24185061 + 2 * lane.val + limb.val < 24185061)
    omega)]
  change
    RowOwner.terminal
        ((terminalPhysicalRow lane.val limb.val - terminalRowFirst) / 2)
        ((terminalPhysicalRow lane.val limb.val - terminalRowFirst) % 2) = _
  rw [offsetEq, laneEq, limbEq]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
