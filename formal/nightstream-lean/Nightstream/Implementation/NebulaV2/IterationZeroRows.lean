import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: exact numeric zero test for the F-prime iteration coordinate.

The three rows enforce

```text
iteration * inverse  = 1 - selector
iteration * selector = 0
selector * (selector - 1) = 0.
```

For canonical Goldilocks assignments with column zero equal to one, row
satisfaction proves `selector = 1` exactly when `iteration = 0`. The selector
is therefore not prover authority.

Owns the zero-test rows, sound branch extraction, and honest auxiliary
completion. Does not own the base or recursive branch rows, the iteration
column's protocol placement, or generated-artifact containment.

Assurance tier: implementation model.

Emits constraints: exactly three rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.IterationZeroRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField

structure Layout where
  iterationColumn : Nat
  inverseColumn : Nat
  selectorColumn : Nat
deriving DecidableEq, Repr

def inverseRow (layout : Layout) : Row :=
  { a := [(layout.iterationColumn, 1)]
    b := [(layout.inverseColumn, 1)]
    c := [(0, 1), (layout.selectorColumn, goldilocksP - 1)] }

def annihilatorRow (layout : Layout) : Row :=
  { a := [(layout.iterationColumn, 1)]
    b := [(layout.selectorColumn, 1)]
    c := [] }

def rows (layout : Layout) : List Row :=
  [inverseRow layout, annihilatorRow layout, bitRow layout.selectorColumn]

@[simp] theorem rows_length (layout : Layout) :
    (rows layout).length = 3 := by
  simp [rows]

private theorem inverse_equation
    {layout : Layout} {assignment : Nat -> Nat}
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.iterationColumn * assignment layout.inverseColumn %
          goldilocksP =
        (1 + (goldilocksP - 1) * assignment layout.selectorColumn) %
          goldilocksP := by
  have rowHolds := holds (inverseRow layout) (by simp [rows])
  simpa [inverseRow, RowHolds, lcEval, one, goldilocksP] using rowHolds

private theorem annihilator_equation
    {layout : Layout} {assignment : Nat -> Nat}
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.iterationColumn * assignment layout.selectorColumn %
      goldilocksP = 0 := by
  have rowHolds := holds (annihilatorRow layout) (by simp [rows])
  simpa [annihilatorRow, RowHolds, lcEval, goldilocksP] using rowHolds

private theorem selector_le_one
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.selectorColumn <= 1 := by
  apply bitRow_le_one goldilocks_euclidPrime
    (canonical layout.selectorColumn) one
  exact holds (bitRow layout.selectorColumn) (by simp [rows])

/-- Satisfied rows derive the selector value. Neither direction is an input. -/
theorem selector_eq_one_iff_iteration_eq_zero
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.selectorColumn = 1 ↔
      assignment layout.iterationColumn = 0 := by
  have selectorBound := selector_le_one canonical one holds
  have inverseEquation := inverse_equation one holds
  have annihilatorEquation := annihilator_equation holds
  constructor
  · intro selectorOne
    rw [selectorOne, Nat.mul_one, Nat.mod_eq_of_lt
      (canonical layout.iterationColumn)] at annihilatorEquation
    exact annihilatorEquation
  · intro iterationZero
    have alternatives : assignment layout.selectorColumn = 0 ∨
        assignment layout.selectorColumn = 1 := by
      omega
    rcases alternatives with selectorZero | selectorOne
    · have impossible : (0 : Nat) = 1 := by
        simpa [iterationZero, selectorZero, goldilocksP] using inverseEquation
      omega
    · exact selectorOne

/-- The complementary branch is selected exactly for a nonzero iteration. -/
theorem selector_eq_zero_iff_iteration_ne_zero
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.selectorColumn = 0 ↔
      assignment layout.iterationColumn ≠ 0 := by
  have selectorBound := selector_le_one canonical one holds
  have selectorOneIff :=
    selector_eq_one_iff_iteration_eq_zero canonical one holds
  constructor
  · intro selectorZero iterationZero
    have selectorOne := selectorOneIff.mpr iterationZero
    omega
  · intro iterationNonzero
    have alternatives : assignment layout.selectorColumn = 0 ∨
        assignment layout.selectorColumn = 1 := by
      omega
    rcases alternatives with selectorZero | selectorOne
    · exact selectorZero
    · exact False.elim (iterationNonzero (selectorOneIff.mp selectorOne))

/-- Exact auxiliary values used by an honest zero-test witness. -/
structure AuxiliariesPlaced (layout : Layout)
    (assignment : Nat -> Nat) : Prop where
  selector : assignment layout.selectorColumn =
    if assignment layout.iterationColumn = 0 then 1 else 0
  inverse : assignment layout.iterationColumn *
        assignment layout.inverseColumn % goldilocksP =
      (1 + (goldilocksP - 1) * assignment layout.selectorColumn) %
        goldilocksP

/-- The exact auxiliary equations complete all three rows. -/
theorem complete
    {layout : Layout} {assignment : Nat -> Nat}
    (one : assignment 0 = 1)
    (placed : AuxiliariesPlaced layout assignment) :
    Satisfies (rows layout) assignment := by
  intro row member
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · simpa [inverseRow, RowHolds, lcEval, one, goldilocksP] using placed.inverse
  · by_cases iterationZero : assignment layout.iterationColumn = 0
    · simp [annihilatorRow, RowHolds, lcEval, iterationZero, goldilocksP]
    · have selectorZero : assignment layout.selectorColumn = 0 := by
        simpa [iterationZero] using placed.selector
      simp [annihilatorRow, RowHolds, lcEval, selectorZero, goldilocksP]
  · by_cases iterationZero : assignment layout.iterationColumn = 0
    · have selectorOne : assignment layout.selectorColumn = 1 := by
        simpa [iterationZero] using placed.selector
      simp [bitRow, RowHolds, lcEval, one, selectorOne, goldilocksP]
    · have selectorZero : assignment layout.selectorColumn = 0 := by
        simpa [iterationZero] using placed.selector
      simp [bitRow, RowHolds, lcEval, one, selectorZero, goldilocksP]

end Nightstream.Implementation.NebulaV2.IterationZeroRows
