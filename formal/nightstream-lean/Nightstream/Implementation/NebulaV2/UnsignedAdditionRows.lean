import Nightstream.Implementation.NebulaV2.BoundedWordRows

/-!
Contract: exact linked R1CS equation `left + right = output` for bounded
unsigned integers.

Assurance tier: implementation model.

Owns one linear row, no-wrap soundness from existing operand bounds, and
honest completeness.

Does not emit range rows, own absolute generated columns, or select a
protocol-specific width.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.UnsignedAdditionRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

structure Layout where
  leftWidth : Nat
  rightWidth : Nat
  leftColumn : Nat
  rightColumn : Nat
  outputColumn : Nat
deriving DecidableEq, Repr

structure Layout.Valid (layout : Layout) : Prop where
  sumFits : 2 ^ layout.leftWidth + 2 ^ layout.rightWidth ≤ goldilocksP

def Layout.sumRow (layout : Layout) : Row :=
  builderLinearRow layout.outputColumn
    [(layout.leftColumn, 1), (layout.rightColumn, 1)]

def rows (layout : Layout) : List Row := [layout.sumRow]

theorem rows_length (layout : Layout) : (rows layout).length = 1 := by
  simp [rows]

theorem terms_canonical (layout : Layout) :
    CanonicalTerms [(layout.leftColumn, 1), (layout.rightColumn, 1)] := by
  intro term member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;> norm_num [goldilocksP]

private theorem sum_row_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    RowHolds assignment layout.sumRow :=
  holds _ (by simp [rows])

private theorem lcEval_eq_sum
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    (leftBound : assignment layout.leftColumn < 2 ^ layout.leftWidth)
    (rightBound : assignment layout.rightColumn < 2 ^ layout.rightWidth) :
    lcEval assignment
        [(layout.leftColumn, 1), (layout.rightColumn, 1)] =
      assignment layout.leftColumn + assignment layout.rightColumn := by
  have sumBound :
      assignment layout.leftColumn + assignment layout.rightColumn <
        goldilocksP := by
    have fits := valid.sumFits
    omega
  simp only [lcEval, List.foldl_cons, List.foldl_nil]
  norm_num only [Nat.zero_add, Nat.one_mul]
  exact Nat.mod_eq_of_lt sumBound

/-- The linked field equation is the intended integer equation. -/
theorem output_eq_add
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    (leftBound : assignment layout.leftColumn < 2 ^ layout.leftWidth)
    (rightBound : assignment layout.rightColumn < 2 ^ layout.rightWidth)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.outputColumn =
      assignment layout.leftColumn + assignment layout.rightColumn := by
  have fieldEqual := builderLinearRow_sound canonical one
    layout.outputColumn
    [(layout.leftColumn, 1), (layout.rightColumn, 1)]
    (terms_canonical layout) (sum_row_holds holds)
  rw [lcEval_eq_sum valid leftBound rightBound] at fieldEqual
  exact fieldEqual

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (left right : Nat) : Prop where
  leftBound : left < 2 ^ layout.leftWidth
  rightBound : right < 2 ^ layout.rightWidth
  leftPlaced : assignment layout.leftColumn = left
  rightPlaced : assignment layout.rightColumn = right
  outputPlaced : assignment layout.outputColumn = left + right

/-- Honest bounded addition satisfies the one exact local row. -/
theorem rows_complete
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {left right : Nat}
    (one : assignment 0 = 1)
    (honest : Honest layout assignment left right) :
    Satisfies (rows layout) assignment := by
  intro row member
  simp only [rows, List.mem_singleton] at member
  subst row
  apply builderLinearRow_complete one layout.outputColumn
    [(layout.leftColumn, 1), (layout.rightColumn, 1)]
    (terms_canonical layout)
  rw [lcEval_eq_sum valid (by simpa [honest.leftPlaced] using honest.leftBound)
    (by simpa [honest.rightPlaced] using honest.rightBound)]
  rw [honest.leftPlaced, honest.rightPlaced]
  exact honest.outputPlaced

end Nightstream.Implementation.NebulaV2.UnsignedAdditionRows
