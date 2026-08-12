import Nightstream.Implementation.NebulaV2.FPrime.Core.IterationInputRows
import Nightstream.Implementation.NebulaV2.Production.FPrime.Fresh.FPrimeBranchRows

set_option autoImplicit false

namespace Tests.NebulaV2ProductionFreshFPrimeBranchRows

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS

#check IterationZeroRows.selector_eq_one_iff_iteration_eq_zero
#check IterationZeroRows.selector_eq_zero_iff_iteration_ne_zero
#check ProductionFreshFPrimeBranchRows.sound
#check ProductionFreshFPrimeBranchRows.complete_base
#check ProductionFreshFPrimeBranchRows.complete_recursive

/-! Without the zero-test rows, a free selector can choose the base arm at a
nonzero iteration. This is the exact countermodel excluded by `sound`. -/

def falseRecursiveRows : List Row := [bitRow 3]

def freeSelectorAssignment : Nat -> Nat
  | 0 => 1
  | 1 => 7
  | 2 => 1
  | 3 => 2
  | 4 => 2
  | _ => 0

def freeRecursiveGate : SelectorGatedRows.Layout falseRecursiveRows where
  selectorColumn := 2
  productColumn := fun _ => 4
  outputColumn := fun _ => 5

/-- If the zero-test rows are omitted, the prover can set the recursive
selector to the unselected value at a nonzero iteration. The gated rows then
hold although the recursive source row is false. -/
theorem omitted_zero_test_allows_wrong_branch :
    freeSelectorAssignment 1 = 7 /\
      freeSelectorAssignment freeRecursiveGate.selectorColumn = 1 /\
      Satisfies (SelectorGatedRows.rows .zero freeRecursiveGate)
        freeSelectorAssignment /\
      ¬ Satisfies falseRecursiveRows freeSelectorAssignment := by
  norm_num [freeSelectorAssignment, freeRecursiveGate, falseRecursiveRows,
    SelectorGatedRows.rows, SelectorGatedRows.blockRows,
    SelectorGatedRows.productRow, SelectorGatedRows.outputRow,
    SelectorGatedRows.selectorRow, ConditionalEqualityRows.row,
    Satisfies, RowHolds, bitRow, lcEval, goldilocksP]

/-! A base-only zero row does not bind a branch relation that reads a
different iteration column. The verifier-owned static column equality is
therefore necessary. -/

def splitIterationAssignment : Nat -> Nat
  | 0 => 1
  | 4 => 0
  | 5 => 7
  | _ => 0

def baseIterationLayout : FPrimeIterationInputRows.Layout where
  iterationColumn := 4

theorem unbound_iteration_columns_allow_nonzero_branch_input :
    Satisfies (FPrimeIterationInputRows.rows baseIterationLayout)
        splitIterationAssignment /\
      splitIterationAssignment 5 = 7 := by
  constructor
  · exact FPrimeIterationInputRows.complete rfl rfl
  · rfl

end Tests.NebulaV2ProductionFreshFPrimeBranchRows
