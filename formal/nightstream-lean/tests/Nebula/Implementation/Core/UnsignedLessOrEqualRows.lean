import Nightstream.Implementation.Nebula.Core.UnsignedLessOrEqualRows

/-! Focused soundness gate for linked unsigned comparison rows. -/

set_option autoImplicit false

namespace tests.NebulaUnsignedLessOrEqualRows

open Nightstream.Implementation.Nebula.UnsignedLessOrEqualRows
open Nightstream.Implementation.R1CS

def layout : Layout :=
  { width := 23
    leftColumn := 1
    rightColumn := 2
    slackColumn := 3
    slackBitStart := 4 }

theorem layout_valid : layout.Valid where
  sumFits := by norm_num [layout, goldilocksP]

theorem exact_row_count : (rows layout).length = 25 := by
  norm_num [rows_length, layout]

theorem satisfying_rows_imply_integer_order
    (assignment : Nat → Nat)
    (leftBound : assignment 1 < 2 ^ 23)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment 1 ≤ assignment 2 := by
  exact left_le_right layout_valid leftBound canonical one holds

end tests.NebulaUnsignedLessOrEqualRows
