import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: the authoritative F-prime input iteration for the base branch.

The base branch has no prior recursive state from which to read the iteration.
This module owns one row that fixes the input iteration column to zero. The
recursive branch must instead alias the same source column to the verified
prior-state invocation index; that static alias is owned by the F-prime
manifest authority.

Assurance tier: implementation model.

Emits constraints: exactly one row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.FPrimeIterationInputRows

open Nightstream.Implementation.R1CS

structure Layout where
  iterationColumn : Nat
deriving DecidableEq, Repr

/-- `iteration * 1 = 0`. Canonical field decoding makes this integer zero. -/
def row (layout : Layout) : Row :=
  { a := [(layout.iterationColumn, 1)]
    b := [(0, 1)]
    c := [] }

def rows (layout : Layout) : List Row := [row layout]

@[simp] theorem rows_length (layout : Layout) :
    (rows layout).length = 1 := by
  simp [rows]

theorem sound
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment layout.iterationColumn = 0 := by
  have holds := satisfied (row layout) (by simp [rows])
  have reduced : assignment layout.iterationColumn % goldilocksP = 0 := by
    simpa [row, RowHolds, lcEval, one, goldilocksP] using holds
  rw [Nat.mod_eq_of_lt (canonical layout.iterationColumn)] at reduced
  exact reduced

theorem complete
    {layout : Layout} {assignment : Nat -> Nat}
    (one : assignment 0 = 1)
    (iterationZero : assignment layout.iterationColumn = 0) :
    Satisfies (rows layout) assignment := by
  intro candidate member
  have exact : candidate = row layout := by simpa [rows] using member
  subst candidate
  simp [row, RowHolds, lcEval, one, iterationZero, goldilocksP]

end Nightstream.Implementation.Nebula.FPrimeIterationInputRows
