import Nightstream.Implementation.Nebula.Core.UnsignedAdditionRows

/-! Focused soundness gate for no-wrap unsigned addition rows. -/

set_option autoImplicit false

namespace tests.NebulaUnsignedAdditionRows

open Nightstream.Implementation.Nebula.UnsignedAdditionRows
open Nightstream.Implementation.R1CS

def layout : Layout :=
  { leftWidth := 23
    rightWidth := 6
    leftColumn := 1
    rightColumn := 2
    outputColumn := 3 }

theorem layout_valid : layout.Valid where
  sumFits := by norm_num [layout, goldilocksP]

theorem exact_row_count : (rows layout).length = 1 := rows_length layout

theorem satisfying_row_is_exact_integer_addition
    (assignment : Nat → Nat)
    (leftBound : assignment 1 < 2 ^ 23)
    (rightBound : assignment 2 < 2 ^ 6)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment 3 = assignment 1 + assignment 2 := by
  exact output_eq_add layout_valid leftBound rightBound canonical one holds

end tests.NebulaUnsignedAdditionRows
