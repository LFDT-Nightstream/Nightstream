import NightstreamFPrime.Layout.MatrixProgram.Exact

/-! Ordered fixed-size matrix blocks. Selection is structural in the block
list and does not enumerate the matrices or their nonzero entries. -/

namespace NightstreamFPrime.Layout.MatrixProgram

open ProductionRelation

def Program.indexed {count : Nat} (block : Fin count → Block) : Program :=
  ⟨List.ofFn block⟩

private theorem uniform_count (blocks : List Block) (rows : Nat)
    (counts : ∀ block ∈ blocks, block.rowCount = rows) :
    (Program.mk blocks).rowCount = blocks.length * rows := by
  induction blocks with
  | nil => simp [Program.rowCount]
  | cons block rest ih =>
    have tail : ∀ child ∈ rest, child.rowCount = rows := fun child member =>
      counts child (List.mem_cons_of_mem block member)
    have head := counts block (by simp)
    simpa [Program.rowCount, head, Nat.add_mul, Nat.add_comm] using congrArg (fun n => rows + n) (ih tail)

private theorem uniform_row {columns : Nat} (blocks : List Block) (rows : Nat)
    (counts : ∀ block ∈ blocks, block.rowCount = rows)
    (sourceRow : Nat → Option R1CS.Row) (index : Fin blocks.length) (row : Fin rows) :
    (Program.mk blocks).row? columns sourceRow (rows * index.val + row.val) =
      (blocks.get index).row? columns sourceRow row.val := by
  induction blocks with
  | nil => exact Fin.elim0 index
  | cons block rest ih =>
    have head := counts block (by simp)
    have tail : ∀ child ∈ rest, child.rowCount = rows := fun child member =>
      counts child (List.mem_cons_of_mem block member)
    refine Fin.cases ?_ (fun i => ?_) index
    · simp only [Fin.val_zero, Nat.mul_zero, Nat.zero_add, List.get_cons_zero]
      exact Program.cons_first_row? block rest columns sourceRow row.val (by rw [head]; exact row.isLt)
    · simp only [Fin.val_succ, Nat.mul_add, Nat.mul_one]
      rw [show rows * i.val + rows + row.val = block.rowCount + (rows * i.val + row.val) by omega]
      rw [Program.cons_rest_row?]
      · simpa only [Nat.add_sub_cancel_left, List.get_cons_succ] using ih tail i
      · omega

theorem Program.indexed_count {count : Nat} (block : Fin count → Block) (rows : Nat)
    (counts : ∀ index, (block index).rowCount = rows) :
    (Program.indexed block).rowCount = count * rows := by
  have bounded : ∀ child ∈ List.ofFn block, child.rowCount = rows := by
    intro child member
    obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
    exact counts index
  simpa only [List.length_ofFn] using uniform_count (List.ofFn block) rows bounded

theorem Program.indexed_row {columns count rows : Nat} (block : Fin count → Block)
    (counts : ∀ index, (block index).rowCount = rows)
    (sourceRow : Nat → Option R1CS.Row) (index : Fin count) (row : Fin rows) :
    (Program.indexed block).row? columns sourceRow (Fin.encodeProd (index, row)).val =
      (block index).row? columns sourceRow row.val := by
  have bounded : ∀ child ∈ List.ofFn block, child.rowCount = rows := by
    intro child member
    obtain ⟨i, rfl⟩ := List.mem_ofFn.mp member
    exact counts i
  have found := uniform_row (columns := columns) (List.ofFn block) rows bounded sourceRow
    ⟨index.val, by simp⟩ row
  simpa [Program.indexed, Fin.encodeProd] using found

theorem Exact.indexed {columns count rows : Nat} (block : Fin count → Block)
    (forms : Fin count → Fin rows → RowForms columns)
    (sourceRow : Nat → Option R1CS.Row)
    (counts : ∀ index, (block index).rowCount = rows)
    (decoded : ∀ index row, (block index).row? columns sourceRow row.val = some (forms index row))
    (fits : count * rows ≤ 2 ^ NightstreamFPrime.Lifecycle.cubeVariables) :
    Exact (Program.indexed block) (ProductionRelation.Plan.indexed forms fits) sourceRow := by
  refine ⟨Program.indexed_count block rows counts, ?_⟩
  intro global
  have encoded : Fin.encodeProd (Fin.decodeProd global) = global := Fin.encodeProd_decodeProd global
  rw [← encoded, Program.indexed_row block counts, decoded]
  simp only [ProductionRelation.Plan.indexed, Fin.decodeProd_encodeProd]

end NightstreamFPrime.Layout.MatrixProgram
