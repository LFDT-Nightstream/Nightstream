import Batteries.Data.Fin.Coding
import NightstreamFPrime.Export.MatrixProgram
import NightstreamFPrime.Export.MatrixProgram.PoseidonInput
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedFamily

/-!
Owns the generic executable interpreter for an invocation-major family of
94-row Poseidon2 matrix plans. The package supplies retained field geometry;
the caller supplies one decoded eight-lane input state per invocation.

This module does not select transcript actions, payloads, or Stage 1 order.
-/

namespace NightstreamFPrime.Export.MatrixProgram.Poseidon

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

abbrev InputState (logicalWidth : Nat) :=
  PoseidonSboxPlan.State logicalWidth

structure Block where
  invocationCount : Nat
  oneColumn : Nat
  retained : RetainedBlock
  input : PoseidonInput.Program
deriving Repr, DecidableEq

def Block.format : Format Block where
  encode := fun block => .array [
    .atom block.invocationCount,
    .atom block.oneColumn,
    RetainedBlock.format.encode block.retained,
    PoseidonInput.Program.format.encode block.input]
  decode
    | .array [.atom invocationCount, .atom oneColumn, retained, input] => do
        pure ⟨invocationCount, oneColumn,
          ← RetainedBlock.format.decode retained,
          ← PoseidonInput.Program.format.decode input⟩
    | _ => .error "invalid Poseidon2 matrix block"
  decode_encode := by
    intro block
    cases block
    simp [RetainedBlock.format.decode_encode,
      PoseidonInput.Program.format.decode_encode]
    rfl

def Block.rowCount (block : Block) : Nat :=
  block.invocationCount * 94

private def retainedSchedule (block : Block)
    (slotCountEq : block.retained.slotCount = block.invocationCount * 86) :
    PoseidonRetainedFamily.Schedule block.retained.slotCount
      block.invocationCount where
  block := block.retained.semantic
  slotCount_eq := by
    simpa only [PoseidonRetainedSlots.rows_length] using slotCountEq

def Block.invocationInterface (block : Block) (logicalWidth : Nat)
    (oneBound : block.oneColumn < logicalWidth)
    (slotCountEq : block.retained.slotCount = block.invocationCount * 86)
    (retainedFits : block.retained.start +
      block.retained.coordinateCount ≤ logicalWidth)
    (input : InputState logicalWidth)
    (invocation : Fin block.invocationCount) :
    PoseidonSboxPlan.Interface logicalWidth :=
  let schedule := retainedSchedule block slotCountEq
  let fits : block.retained.start +
      schedule.block.coordinateCount ≤ logicalWidth := by
    simpa only [schedule, retainedSchedule,
      RetainedBlock.semantic_coordinateCount] using retainedFits
  { oneColumn := ⟨block.oneColumn, oneBound⟩
    input := input
    sboxOutput := PoseidonRetainedFamily.form schedule
      block.retained.start fits invocation
    output := PoseidonRetainedFamily.outputState schedule
      block.retained.start fits invocation }

/-- Decode one Poseidon2 family row. Every malformed geometry returns none. -/
def Block.rowWithInput? (block : Block) (logicalWidth : Nat)
    (inputState : Nat → Option (InputState logicalWidth))
    (ordinal : Nat) : Option (RowForms logicalWidth) :=
  if rowBound : ordinal < block.rowCount then
    if oneBound : block.oneColumn < logicalWidth then
      if fieldKind : block.retained.kind = .field then
        if slotCountEq : block.retained.slotCount =
            block.invocationCount * 86 then
          if retainedFits : block.retained.start +
              block.retained.coordinateCount ≤ logicalWidth then do
            let decoded : Fin block.invocationCount × Fin 94 :=
              Fin.decodeProd ⟨ordinal, rowBound⟩
            let input ← inputState decoded.1.val
            let interface := block.invocationInterface logicalWidth oneBound
              slotCountEq retainedFits input decoded.1
            pure fun port =>
              ((PoseidonSboxPlan.rows interface).get
                ⟨decoded.2.val, by
                  rw [PoseidonSboxPlan.rows_length]
                  exact decoded.2.isLt⟩).meaningfulForm port
          else
            none
        else
          none
      else
        none
    else
      none
  else
    none

/-- Package-only row decoding through the block-owned input program. -/
def Block.row? (block : Block) (logicalWidth ordinal : Nat) :
    Option (RowForms logicalWidth) :=
  block.rowWithInput? logicalWidth
    (block.input.state? logicalWidth block.oneColumn) ordinal

def Block.ofSemantic {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth invocationCount)
    (retainedStart : Nat) (oneColumn : Fin logicalWidth)
    (input : PoseidonInput.Program) : Block where
  invocationCount := invocationCount
  oneColumn := oneColumn.val
  retained := RetainedBlock.ofSemantic schedule.block retainedStart
  input := input

/-- Once all wire guards hold and the input state is present, row decoding is
exactly selection from the generated 94-row invocation interface. -/
theorem Block.rowWithInput?_of_valid (block : Block) (logicalWidth : Nat)
    (inputState : Nat → Option (InputState logicalWidth)) (ordinal : Nat)
    (rowBound : ordinal < block.rowCount)
    (oneBound : block.oneColumn < logicalWidth)
    (fieldKind : block.retained.kind = .field)
    (slotCountEq : block.retained.slotCount = block.invocationCount * 86)
    (retainedFits : block.retained.start +
      block.retained.coordinateCount ≤ logicalWidth)
    (input : InputState logicalWidth)
    (loaded : inputState
      (Fin.decodeProd (⟨ordinal, rowBound⟩ :
        Fin (block.invocationCount * 94))).1.val = some input) :
    block.rowWithInput? logicalWidth inputState ordinal =
      let decoded : Fin block.invocationCount × Fin 94 :=
        Fin.decodeProd ⟨ordinal, rowBound⟩
      some fun port =>
        ((PoseidonSboxPlan.rows
          (block.invocationInterface logicalWidth oneBound slotCountEq
            retainedFits input decoded.1)).get
          ⟨decoded.2.val, by
            rw [PoseidonSboxPlan.rows_length]
            exact decoded.2.isLt⟩).meaningfulForm port := by
  unfold Block.rowWithInput?
  rw [dif_pos rowBound, dif_pos oneBound, dif_pos fieldKind,
    dif_pos slotCountEq, dif_pos retainedFits]
  dsimp only
  rw [loaded]
  rfl

/-- Erasing a semantic retained block preserves the exact Poseidon2
per-invocation interface. -/
theorem Block.invocationInterface_ofSemantic
    {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth invocationCount)
    (retainedStart : Nat) (oneColumn : Fin logicalWidth)
    (inputProgram : PoseidonInput.Program)
    (fits : retainedStart + schedule.block.coordinateCount ≤ logicalWidth)
    (input : Fin invocationCount → InputState logicalWidth)
    (invocation : Fin invocationCount) :
    let block := Block.ofSemantic schedule retainedStart oneColumn inputProgram
    let slotCountEq : block.retained.slotCount = invocationCount * 86 := by
      simpa only [block, Block.ofSemantic, RetainedBlock.ofSemantic,
        PoseidonRetainedSlots.rows_length] using schedule.slotCount_eq
    let retainedFits : block.retained.start +
        block.retained.coordinateCount ≤ logicalWidth := by
      simpa only [block, Block.ofSemantic, RetainedBlock.ofSemantic,
        RetainedBlock.coordinateCount,
        LowNormBlock.Block.coordinateCount] using fits
    block.invocationInterface logicalWidth oneColumn.isLt slotCountEq
        retainedFits (input invocation) invocation =
      PoseidonRetainedFamily.invocationInterface schedule retainedStart fits
        oneColumn input invocation := by
  rfl

/-- A canonical wire block returns every row of the exact semantic
invocation-major Poseidon2 family. -/
theorem Block.rowWithInput?_ofSemantic
    {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth invocationCount)
    (fieldKind : schedule.block.kind = .field)
    (retainedStart : Nat) (oneColumn : Fin logicalWidth)
    (inputProgram : PoseidonInput.Program)
    (fits : retainedStart + schedule.block.coordinateCount ≤ logicalWidth)
    (input : Fin invocationCount → InputState logicalWidth)
    (inputState : Nat → Option (InputState logicalWidth))
    (inputExact : ∀ invocation,
      inputState invocation.val = some (input invocation))
    (global : Fin (invocationCount * 94)) :
    let block := Block.ofSemantic schedule retainedStart oneColumn inputProgram
    block.rowWithInput? logicalWidth inputState
        global.val =
      let decoded : Fin invocationCount × Fin 94 := Fin.decodeProd global
      some (PoseidonSboxFamilyPlan.rowForms
        (PoseidonRetainedFamily.familyInterface schedule retainedStart fits
          oneColumn input) decoded.1 decoded.2) := by
  let block := Block.ofSemantic schedule retainedStart oneColumn inputProgram
  have rowBound : global.val < block.rowCount := by
    exact global.isLt
  have wireKind : block.retained.kind = .field := by
    exact fieldKind
  have slotCountEq : block.retained.slotCount = invocationCount * 86 := by
    simpa only [block, Block.ofSemantic, RetainedBlock.ofSemantic,
      PoseidonRetainedSlots.rows_length] using schedule.slotCount_eq
  have retainedFits : block.retained.start +
      block.retained.coordinateCount ≤ logicalWidth := by
    simpa only [block, Block.ofSemantic, RetainedBlock.ofSemantic,
      RetainedBlock.coordinateCount, LowNormBlock.Block.coordinateCount]
      using fits
  have globalEq :
      (⟨global.val, rowBound⟩ :
        Fin (block.invocationCount * 94)) = global := by
    apply Fin.ext
    rfl
  let decoded : Fin invocationCount × Fin 94 := Fin.decodeProd global
  have loaded : inputState
      (Fin.decodeProd (⟨global.val, rowBound⟩ :
        Fin (block.invocationCount * 94))).1.val =
        some (input decoded.1) := by
    rw [globalEq]
    exact inputExact decoded.1
  calc
    block.rowWithInput? logicalWidth inputState global.val =
        some (fun port =>
          ((PoseidonSboxPlan.rows
            (block.invocationInterface logicalWidth oneColumn.isLt
              slotCountEq retainedFits (input decoded.1) decoded.1)).get
            ⟨decoded.2.val, by
              rw [PoseidonSboxPlan.rows_length]
              exact decoded.2.isLt⟩).meaningfulForm port) := by
      have selected :=
        Block.rowWithInput?_of_valid block logicalWidth inputState
          global.val rowBound oneColumn.isLt wireKind slotCountEq retainedFits
          (input decoded.1) loaded
      rw [globalEq] at selected
      exact selected
    _ = some (PoseidonSboxFamilyPlan.rowForms
          (PoseidonRetainedFamily.familyInterface schedule retainedStart fits
            oneColumn input) decoded.1 decoded.2) := by
      rw [Block.invocationInterface_ofSemantic schedule retainedStart
        oneColumn inputProgram fits input decoded.1]
      rfl

/-- The package-owned input program and retained block generate the exact
semantic Poseidon2 family rows. -/
theorem Block.row?_ofSemantic
    {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth invocationCount)
    (fieldKind : schedule.block.kind = .field)
    (retainedStart : Nat) (oneColumn : Fin logicalWidth)
    (inputProgram : PoseidonInput.Program)
    (fits : retainedStart + schedule.block.coordinateCount ≤ logicalWidth)
    (input : Fin invocationCount → InputState logicalWidth)
    (inputExact : ∀ invocation,
      inputProgram.state? logicalWidth oneColumn.val invocation.val =
        some (input invocation))
    (global : Fin (invocationCount * 94)) :
    let block := Block.ofSemantic schedule retainedStart oneColumn inputProgram
    block.row? logicalWidth global.val =
      let decoded : Fin invocationCount × Fin 94 := Fin.decodeProd global
      some (PoseidonSboxFamilyPlan.rowForms
        (PoseidonRetainedFamily.familyInterface schedule retainedStart fits
          oneColumn input) decoded.1 decoded.2) := by
  unfold Block.row?
  exact Block.rowWithInput?_ofSemantic schedule fieldKind retainedStart
    oneColumn inputProgram fits input
    (inputProgram.state? logicalWidth oneColumn.val) inputExact global

@[simp] theorem Block.ofSemantic_rowCount
    {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth invocationCount)
    (retainedStart : Nat) (oneColumn : Fin logicalWidth)
    (input : PoseidonInput.Program) :
    (Block.ofSemantic schedule retainedStart oneColumn input).rowCount =
      invocationCount * 94 := by
  rfl

end NightstreamFPrime.Export.MatrixProgram.Poseidon
