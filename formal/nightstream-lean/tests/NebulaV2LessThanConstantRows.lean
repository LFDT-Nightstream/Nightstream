import Nightstream.Implementation.NebulaV2.LessThanConstantRows

set_option autoImplicit false

namespace tests.NebulaV2LessThanConstantRows

open Nightstream.Implementation.NebulaV2.LessThanConstantRows
open Nightstream.Implementation.R1CS

def layout : Layout :=
  { width := 11
    limit := 1088
    valueColumn := 1
    valueBitStart := 10
    slackColumn := 2
    slackBitStart := 21 }

def valid : layout.Valid where
  limitPositive := by decide
  limitFits := by decide
  sumFits := by decide

theorem exact_row_count : (rows layout).length = 25 := by
  norm_num [rows_length, layout]

/-- The boundary value 1,088 fits in 11 bits but cannot satisfy the strict
step-index rows. -/
theorem first_invalid_step_is_rejected
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (atLimit : assignment layout.valueColumn = 1088) :
    ¬ Satisfies (rows layout) assignment := by
  intro holds
  have below := value_lt_limit valid canonical one holds
  have belowExact : assignment layout.valueColumn < 1088 := by
    simpa [layout] using below
  rw [atLimit] at belowExact
  omega

end tests.NebulaV2LessThanConstantRows
