import Nightstream.Implementation.Nebula.Core.LessThanConstantRows

/-!
Contract: strict constant bound for a value that already has an exact
bounded-word block.

Assurance tier: implementation model.

Owns one bounded slack word and one no-wrap sum row. Soundness reuses the
independently proved bound on the existing value word, so it does not emit a
second copy of the value-bit and recomposition rows.

Does not own the existing value-word rows, absolute generated columns, or a
concrete protocol schema.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.LessThanConstantLinkedRows

open Nightstream.Implementation.R1CS

structure Layout where
  width : Nat
  limit : Nat
  valueColumn : Nat
  slackColumn : Nat
  slackBitStart : Nat
deriving DecidableEq, Repr

structure Layout.Valid (layout : Layout) : Prop where
  limitPositive : 0 < layout.limit
  limitFits : layout.limit ≤ 2 ^ layout.width
  sumFits : 2 ^ (layout.width + 1) ≤ goldilocksP

def Layout.slackWord (layout : Layout) : BoundedWordRows.Layout :=
  { width := layout.width
    valueColumn := layout.slackColumn
    bitStart := layout.slackBitStart }

def Layout.sumRow (layout : Layout) : Row :=
  ⟨[(layout.valueColumn, 1), (layout.slackColumn, 1)],
    [(0, 1)], [(0, layout.limit - 1)]⟩

def rows (layout : Layout) : List Row :=
  BoundedWordRows.rows layout.slackWord ++ [layout.sumRow]

theorem rows_length (layout : Layout) :
    (rows layout).length = layout.width + 2 := by
  simp [rows, BoundedWordRows.rows_length, Layout.slackWord]

private theorem slack_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (BoundedWordRows.rows layout.slackWord) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem sum_row_holds
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    RowHolds assignment layout.sumRow :=
  holds _ (by simp [rows])

theorem slack_bound
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.slackColumn < 2 ^ layout.width := by
  have wordFits : 2 ^ layout.width ≤ goldilocksP := by
    have doubleFits :
        2 ^ layout.width + 2 ^ layout.width ≤ goldilocksP := by
      simpa [pow_succ, Nat.mul_two] using valid.sumFits
    omega
  simpa [Layout.slackWord] using
    BoundedWordRows.value_lt_twoPower wordFits canonical one
      (slack_rows_hold holds)

theorem value_add_slack_eq
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    (valueBound : assignment layout.valueColumn < 2 ^ layout.width)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.valueColumn + assignment layout.slackColumn =
      layout.limit - 1 := by
  have slackBound := slack_bound valid canonical one holds
  have sumBound :
      assignment layout.valueColumn + assignment layout.slackColumn <
        goldilocksP := by
    have doubleFits :
        2 ^ layout.width + 2 ^ layout.width ≤ goldilocksP := by
      simpa [pow_succ, Nat.mul_two] using valid.sumFits
    omega
  have limitBound : layout.limit - 1 < goldilocksP := by
    have widthFits : 2 ^ layout.width ≤ goldilocksP := by
      have doubleFits :
          2 ^ layout.width + 2 ^ layout.width ≤ goldilocksP := by
        simpa [pow_succ, Nat.mul_two] using valid.sumFits
      omega
    exact (Nat.sub_lt valid.limitPositive (by decide)).trans_le
      (valid.limitFits.trans widthFits)
  have fieldEqual := sum_row_holds holds
  simp only [Layout.sumRow, RowHolds, lcEval, List.foldl_cons,
    List.foldl_nil, one] at fieldEqual
  norm_num only [Nat.zero_add, Nat.one_mul, Nat.mul_one] at fieldEqual
  simp only [Nat.mod_eq_of_lt (by decide : 1 < goldilocksP),
    Nat.mul_one, Nat.mod_mod] at fieldEqual
  change
    (assignment layout.valueColumn + assignment layout.slackColumn) %
        goldilocksP =
      (layout.limit - 1) % goldilocksP at fieldEqual
  rw [Nat.mod_eq_of_lt sumBound, Nat.mod_eq_of_lt limitBound] at fieldEqual
  exact fieldEqual

/-- The strict bound follows from the existing value-word bound, the new
slack-word rows, and the no-wrap sum row. -/
theorem value_lt_limit
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    (valueBound : assignment layout.valueColumn < 2 ^ layout.width)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.valueColumn < layout.limit := by
  have exactSum := value_add_slack_eq valid valueBound canonical one holds
  have valueLeSum :
      assignment layout.valueColumn ≤
        assignment layout.valueColumn + assignment layout.slackColumn :=
    Nat.le_add_right _ _
  rw [exactSum] at valueLeSum
  have predLt : layout.limit - 1 < layout.limit :=
    Nat.sub_lt valid.limitPositive (by decide)
  exact valueLeSum.trans_lt predLt

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (value : Nat) : Prop where
  valueLt : value < layout.limit
  valuePlaced : assignment layout.valueColumn = value
  slackWord : BoundedWordRows.Honest layout.slackWord assignment
    (layout.limit - 1 - value)

/-- Honest bounded values satisfy the linked strict-bound block. -/
theorem rows_complete
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {value : Nat}
    (one : assignment 0 = 1)
    (honest : Honest layout assignment value) :
    Satisfies (rows layout) assignment := by
  have widthFits : 2 ^ layout.width ≤ goldilocksP := by
    have doubleFits :
        2 ^ layout.width + 2 ^ layout.width ≤ goldilocksP := by
      simpa [pow_succ, Nat.mul_two] using valid.sumFits
    omega
  have slackRows := BoundedWordRows.rows_complete widthFits one
    honest.slackWord
  intro row member
  simp only [rows, List.mem_append, List.mem_singleton] at member
  rcases member with slackMember | sumEqual
  · exact slackRows row slackMember
  · subst row
    have slackPlaced := honest.slackWord.valuePlaced
    simp only [Layout.slackWord] at slackPlaced
    have valueLt := honest.valueLt
    have exactSum : value + (layout.limit - 1 - value) =
        layout.limit - 1 := by
      omega
    simp only [Layout.sumRow, RowHolds, lcEval, List.foldl_cons,
      List.foldl_nil, one]
    norm_num only [Nat.zero_add, Nat.one_mul, Nat.mul_one]
    simp only [Nat.mod_eq_of_lt (by decide : 1 < goldilocksP),
      Nat.mul_one, Nat.mod_mod]
    change
      (assignment layout.valueColumn + assignment layout.slackColumn) %
          goldilocksP =
        (layout.limit - 1) % goldilocksP
    rw [honest.valuePlaced, slackPlaced, exactSum]

end Nightstream.Implementation.Nebula.LessThanConstantLinkedRows
