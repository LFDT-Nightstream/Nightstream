import NightstreamFPrime.Export.MatrixProgram.Ordinary
import NightstreamFPrime.Export.MatrixProgram.MultiplicationGrid
import NightstreamFPrime.Export.MatrixProgram.Phi81Product
import NightstreamFPrime.Export.MatrixProgram.Pin
import NightstreamFPrime.Export.MatrixProgram.Poseidon

/-!
Owns the generic ordered interpreter for a compact sparse 14-matrix program.
Each decoded row returns the 13 meaningful sparse ports. Matrix slot 13
remains zero through `ProductionRelation.Plan.portForm`.

The interpreter does not select Stage 1 phases, row order, or source rows.
-/

namespace NightstreamFPrime.Export.MatrixProgram

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation

/-- Generic block vocabulary. New constructors must carry all operands that
determine their rows; a consumer may not select hidden phase data. -/
inductive Block where
  | ordinary (block : Ordinary.Block)
  | multiplicationGrid (block : MultiplicationGrid.Block)
  | phi81Product (block : Phi81Product.Block)
  | pin (block : Pin.Block)
  | poseidon (block : Poseidon.Block)
deriving Repr, DecidableEq

def Block.format : Format Block where
  encode
    | .ordinary block =>
        .array [.atom 0, Ordinary.Block.format.encode block]
    | .multiplicationGrid block =>
        .array [.atom 4, MultiplicationGrid.Block.format.encode block]
    | .phi81Product block =>
        .array [.atom 3, Phi81Product.Block.format.encode block]
    | .pin block =>
        .array [.atom 1, Pin.Block.format.encode block]
    | .poseidon block =>
        .array [.atom 2, Poseidon.Block.format.encode block]
  decode
    | .array [.atom 0, block] => do
        pure (.ordinary (← Ordinary.Block.format.decode block))
    | .array [.atom 4, block] => do
        pure (.multiplicationGrid
          (← MultiplicationGrid.Block.format.decode block))
    | .array [.atom 3, block] => do
        pure (.phi81Product (← Phi81Product.Block.format.decode block))
    | .array [.atom 1, block] => do
        pure (.pin (← Pin.Block.format.decode block))
    | .array [.atom 2, block] => do
        pure (.poseidon (← Poseidon.Block.format.decode block))
    | _ => .error "invalid production matrix block"
  decode_encode := by
    intro block
    cases block <;>
      simp [Ordinary.Block.format.decode_encode,
        MultiplicationGrid.Block.format.decode_encode,
        Phi81Product.Block.format.decode_encode,
        Pin.Block.format.decode_encode, Poseidon.Block.format.decode_encode]

def Block.rowCount : Block → Nat
  | .ordinary block => block.rowCount
  | .multiplicationGrid block => block.rowCount
  | .phi81Product block => block.rowCount
  | .pin block => block.rowCount
  | .poseidon block => block.rowCount

/-- Decode one block row. `sourceRow` is the identity-checked package R1CS
row accessor and is used only by ordinary blocks. -/
def Block.row? (block : Block) (logicalWidth : Nat)
    (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat) :
    Option (RowForms logicalWidth) :=
  match block with
  | .ordinary ordinaryBlock => do
      let forms ← ordinaryBlock.row? logicalWidth sourceRow ordinal
      pure forms.meaningfulForm
  | .multiplicationGrid multiplicationBlock => do
      let forms ← multiplicationBlock.row? logicalWidth ordinal
      pure forms.meaningfulForm
  | .phi81Product productBlock =>
      productBlock.row? logicalWidth ordinal
  | .pin pinBlock => do
      let forms ← pinBlock.row? logicalWidth ordinal
      pure forms.meaningfulForm
  | .poseidon poseidonBlock =>
      poseidonBlock.row? logicalWidth ordinal

/-- One identity-bound ordered matrix program. -/
structure Program where
  blocks : List Block
deriving Repr, DecidableEq

def Program.format : Format Program where
  encode := fun program => (list Block.format).encode program.blocks
  decode := fun value => do
    pure ⟨← (list Block.format).decode value⟩
  decode_encode := by
    intro program
    cases program
    simp [Format.decode_encode]

def Program.rowCount (program : Program) : Nat :=
  (program.blocks.map Block.rowCount).sum

/-- Canonical ordered concatenation of two compact matrix programs. -/
def Program.append (left right : Program) : Program where
  blocks := left.blocks ++ right.blocks

/-- Select one row without expanding any block. -/
def Program.row? (program : Program) (logicalWidth : Nat)
    (sourceRow : Nat → Option R1CS.Row) : Nat →
    Option (RowForms logicalWidth)
  | ordinal => select program.blocks ordinal
where
  select : List Block → Nat → Option (RowForms logicalWidth)
    | [], _ => none
    | block :: rest, ordinal =>
        if ordinal < block.rowCount then
          block.row? logicalWidth sourceRow ordinal
        else
          select rest (ordinal - block.rowCount)

@[simp] theorem Program.append_rowCount (left right : Program) :
    (left.append right).rowCount = left.rowCount + right.rowCount := by
  simp [Program.append, Program.rowCount]

private theorem Program.select_append_left (blocks tail : List Block)
    (logicalWidth : Nat) (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat)
    (bound : ordinal < (blocks.map Block.rowCount).sum) :
    Program.row?.select logicalWidth sourceRow (blocks ++ tail) ordinal =
      Program.row?.select logicalWidth sourceRow blocks ordinal := by
  induction blocks generalizing ordinal with
  | nil => simp at bound
  | cons block rest ih =>
      simp only [List.map_cons, List.sum_cons] at bound
      simp only [List.cons_append, Program.row?.select]
      by_cases first : ordinal < block.rowCount
      · simp [first]
      · have nextBound :
            ordinal - block.rowCount < (rest.map Block.rowCount).sum := by
          omega
        simp [first, ih (ordinal - block.rowCount) nextBound]

private theorem Program.select_append_right (blocks tail : List Block)
    (logicalWidth : Nat) (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat) :
    Program.row?.select logicalWidth sourceRow (blocks ++ tail)
        ((blocks.map Block.rowCount).sum + ordinal) =
      Program.row?.select logicalWidth sourceRow tail ordinal := by
  induction blocks with
  | nil => simp [Program.row?.select]
  | cons block rest ih =>
      simp only [List.map_cons, List.sum_cons, List.cons_append]
      rw [Program.row?.select]
      have skip :
          ¬ (block.rowCount + (rest.map Block.rowCount).sum + ordinal <
            block.rowCount) := by
        omega
      rw [if_neg skip]
      have shifted :
          block.rowCount + (rest.map Block.rowCount).sum + ordinal -
              block.rowCount =
            (rest.map Block.rowCount).sum + ordinal := by
        omega
      rw [shifted]
      exact ih

/-- A row in the left child keeps its exact compact interpretation. -/
theorem Program.append_left_row? (left right : Program)
    (logicalWidth : Nat) (sourceRow : Nat → Option R1CS.Row)
    (ordinal : Nat) (bound : ordinal < left.rowCount) :
    (left.append right).row? logicalWidth sourceRow ordinal =
      left.row? logicalWidth sourceRow ordinal := by
  exact Program.select_append_left left.blocks right.blocks logicalWidth
    sourceRow ordinal bound

/-- A row in the right child is selected after the exact left row count. -/
theorem Program.append_right_row? (left right : Program)
    (logicalWidth : Nat) (sourceRow : Nat → Option R1CS.Row)
    (ordinal : Nat) :
    (left.append right).row? logicalWidth sourceRow
        (left.rowCount + ordinal) =
      right.row? logicalWidth sourceRow ordinal := by
  exact Program.select_append_right left.blocks right.blocks logicalWidth
    sourceRow ordinal

@[simp] theorem Program.singleton_rowCount (block : Block) :
    (Program.mk [block]).rowCount = block.rowCount := by
  simp [Program.rowCount]

@[simp] theorem Program.two_rowCount (left right : Block) :
    (Program.mk [left, right]).rowCount = left.rowCount + right.rowCount := by
  simp [Program.rowCount]

@[simp] theorem Program.singleton_row? (block : Block) (logicalWidth : Nat)
    (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat) :
    (Program.mk [block]).row? logicalWidth sourceRow ordinal =
      if ordinal < block.rowCount then
        block.row? logicalWidth sourceRow ordinal
      else
        none := by
  simp [Program.row?, Program.row?.select]

theorem Program.cons_first_row? (block : Block) (rest : List Block)
    (logicalWidth : Nat) (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat)
    (bound : ordinal < block.rowCount) :
    (Program.mk (block :: rest)).row? logicalWidth sourceRow ordinal =
      block.row? logicalWidth sourceRow ordinal := by
  simp [Program.row?, Program.row?.select, bound]

theorem Program.cons_rest_row? (block : Block) (rest : List Block)
    (logicalWidth : Nat) (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat)
    (bound : block.rowCount ≤ ordinal) :
    (Program.mk (block :: rest)).row? logicalWidth sourceRow ordinal =
      (Program.mk rest).row? logicalWidth sourceRow
        (ordinal - block.rowCount) := by
  simp [Program.row?, Program.row?.select, Nat.not_lt.mpr bound]

theorem Program.two_first_row? (left right : Block) (logicalWidth : Nat)
    (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat)
    (bound : ordinal < left.rowCount) :
    (Program.mk [left, right]).row? logicalWidth sourceRow ordinal =
      left.row? logicalWidth sourceRow ordinal := by
  simp [Program.row?, Program.row?.select, bound]

theorem Program.two_second_row? (left right : Block) (logicalWidth : Nat)
    (sourceRow : Nat → Option R1CS.Row) (ordinal : Nat)
    (leftBound : left.rowCount ≤ ordinal)
    (rightBound : ordinal - left.rowCount < right.rowCount) :
    (Program.mk [left, right]).row? logicalWidth sourceRow ordinal =
      right.row? logicalWidth sourceRow (ordinal - left.rowCount) := by
  simp [Program.row?, Program.row?.select, Nat.not_lt.mpr leftBound,
    rightBound]

end NightstreamFPrime.Export.MatrixProgram
