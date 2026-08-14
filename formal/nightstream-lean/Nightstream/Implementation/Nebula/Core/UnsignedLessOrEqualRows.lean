import Nightstream.Implementation.Nebula.Core.BoundedWordRows

/-!
Contract: exact linked R1CS proof that one bounded unsigned integer is at
most another bounded unsigned integer.

Assurance tier: implementation model.

Owns one bounded slack word, the equation `left + slack = right`, sound
integer extraction without Goldilocks wrap, and honest completeness.

Does not own the existing range rows for `left` or `right`, absolute generated
columns, or a concrete protocol transition.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.UnsignedLessOrEqualRows

open Nightstream.Implementation.R1CS

structure Layout where
  width : Nat
  leftColumn : Nat
  rightColumn : Nat
  slackColumn : Nat
  slackBitStart : Nat
deriving DecidableEq, Repr

structure Layout.Valid (layout : Layout) : Prop where
  sumFits : 2 ^ (layout.width + 1) ≤ goldilocksP

def Layout.slackWord (layout : Layout) : BoundedWordRows.Layout :=
  { width := layout.width
    valueColumn := layout.slackColumn
    bitStart := layout.slackBitStart }

def Layout.sumRow (layout : Layout) : Row :=
  ⟨[(layout.leftColumn, 1), (layout.slackColumn, 1)],
    [(0, 1)], [(layout.rightColumn, 1)]⟩

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

private theorem word_fits
    {layout : Layout} (valid : layout.Valid) :
    2 ^ layout.width ≤ goldilocksP := by
  have doubled : 2 ^ layout.width + 2 ^ layout.width ≤ goldilocksP := by
    simpa [pow_succ, Nat.mul_two] using valid.sumFits
  omega

theorem slack_bound
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.slackColumn < 2 ^ layout.width := by
  simpa [Layout.slackWord] using
    BoundedWordRows.value_lt_twoPower (word_fits valid) canonical one
      (slack_rows_hold holds)

/-- The field row is an integer equation because both addends are bounded
below `2^width`, and the declared doubled range fits below Goldilocks. -/
theorem left_add_slack_eq_right
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    (leftBound : assignment layout.leftColumn < 2 ^ layout.width)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.leftColumn + assignment layout.slackColumn =
      assignment layout.rightColumn := by
  have slackBound := slack_bound valid canonical one holds
  have sumBound :
      assignment layout.leftColumn + assignment layout.slackColumn <
        goldilocksP := by
    have doubled : 2 ^ layout.width + 2 ^ layout.width ≤ goldilocksP := by
      simpa [pow_succ, Nat.mul_two] using valid.sumFits
    omega
  have fieldEqual := sum_row_holds holds
  simp only [Layout.sumRow, RowHolds, lcEval, List.foldl_cons,
    List.foldl_nil, one] at fieldEqual
  norm_num only [Nat.zero_add, Nat.one_mul, Nat.mul_one] at fieldEqual
  simp only [Nat.mod_eq_of_lt (by decide : 1 < goldilocksP),
    Nat.mul_one, Nat.mod_mod] at fieldEqual
  change
    (assignment layout.leftColumn + assignment layout.slackColumn) %
        goldilocksP =
      assignment layout.rightColumn % goldilocksP at fieldEqual
  rw [Nat.mod_eq_of_lt sumBound,
    Nat.mod_eq_of_lt (canonical layout.rightColumn)] at fieldEqual
  exact fieldEqual

/-- Satisfying rows prove the intended unsigned comparison. -/
theorem left_le_right
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat}
    (leftBound : assignment layout.leftColumn < 2 ^ layout.width)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.leftColumn ≤ assignment layout.rightColumn := by
  rw [← left_add_slack_eq_right valid leftBound canonical one holds]
  exact Nat.le_add_right _ _

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (left right : Nat) : Prop where
  leftBound : left < 2 ^ layout.width
  rightBound : right < 2 ^ layout.width
  ordered : left ≤ right
  leftPlaced : assignment layout.leftColumn = left
  rightPlaced : assignment layout.rightColumn = right
  slackWord : BoundedWordRows.Honest layout.slackWord assignment
    (right - left)

/-- Honest bounded ordered values satisfy the exact local comparison block. -/
theorem rows_complete
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {left right : Nat}
    (one : assignment 0 = 1)
    (honest : Honest layout assignment left right) :
    Satisfies (rows layout) assignment := by
  have slackRows := BoundedWordRows.rows_complete (word_fits valid) one
    honest.slackWord
  intro row member
  simp only [rows, List.mem_append, List.mem_singleton] at member
  rcases member with slackMember | sumEqual
  · exact slackRows row slackMember
  · subst row
    have slackPlaced := honest.slackWord.valuePlaced
    simp only [Layout.slackWord] at slackPlaced
    have ordered := honest.ordered
    have exactSum : left + (right - left) = right := by omega
    simp only [Layout.sumRow, RowHolds, lcEval, List.foldl_cons,
      List.foldl_nil, one]
    norm_num only [Nat.zero_add, Nat.one_mul, Nat.mul_one]
    simp only [Nat.mod_eq_of_lt (by decide : 1 < goldilocksP),
      Nat.mul_one, Nat.mod_mod]
    change
      (assignment layout.leftColumn + assignment layout.slackColumn) %
          goldilocksP =
        assignment layout.rightColumn % goldilocksP
    rw [honest.leftPlaced, honest.rightPlaced, slackPlaced, exactSum]

end Nightstream.Implementation.Nebula.UnsignedLessOrEqualRows
