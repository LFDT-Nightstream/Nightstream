import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Hostile countermodel for an omitted recursive NIFS output-carrier row.

The retained row binds column 1 but does not mention output column 2. Two
canonical assignments can therefore satisfy the retained relation while
disagreeing on the output. Adding the output row rejects the bad assignment.
-/

set_option autoImplicit false

namespace tests.NebulaProductionNifsOutputCarrierCountermodels

open Nightstream.Implementation.R1CS

def retainedRow : Row :=
  ⟨[(1, 1)], [(0, 1)], [(1, 1)]⟩

def outputRow : Row :=
  ⟨[(2, 1)], [(0, 1)], [(0, 5)]⟩

def retainedRows : List Row := [retainedRow]

def completeRows : List Row := [retainedRow, outputRow]

def goodAssignment : Nat -> Nat
  | 0 => 1
  | 1 => 9
  | 2 => 5
  | _ => 0

def badAssignment : Nat -> Nat
  | 0 => 1
  | 1 => 9
  | 2 => 6
  | _ => 0

/-- Row-count and upstream-row satisfaction do not bind a missing output
coordinate. The exact output row is necessary. -/
theorem omitted_output_row_allows_wrong_carrier :
    Satisfies retainedRows goodAssignment /\
      Satisfies retainedRows badAssignment /\
      goodAssignment 2 ≠ badAssignment 2 /\
      Satisfies completeRows goodAssignment /\
      ¬ Satisfies completeRows badAssignment := by
  decide

end tests.NebulaProductionNifsOutputCarrierCountermodels
