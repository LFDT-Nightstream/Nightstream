import Nightstream.Implementation.NebulaV2.BoundedWordRows

set_option autoImplicit false

namespace tests.NebulaV2BoundedWordRows

open Nightstream.Implementation.NebulaV2.BoundedWordRows
open Nightstream.Implementation.R1CS

def layout : Layout :=
  { width := 23
    valueColumn := 1
    bitStart := 2 }

theorem exact_row_count : (rows layout).length = 24 := by
  norm_num [rows_length, layout]

/-- The row theorem itself, and not a typed wrapper, supplies the exact
23-bit timestamp bound. -/
theorem timestamp_bound_comes_from_rows
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.valueColumn < 2 ^ 23 := by
  exact value_lt_twoPower (layout := layout) (by decide)
    canonical one holds

end tests.NebulaV2BoundedWordRows
