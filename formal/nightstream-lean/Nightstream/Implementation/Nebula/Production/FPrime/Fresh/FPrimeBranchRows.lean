import Nightstream.Implementation.Nebula.Core.IterationZeroRows
import Nightstream.Implementation.Nebula.Core.SelectorGatedRows

/-!
Contract: one fixed numeric base/recursive branch relation for a fresh
F-prime claim.

The iteration zero-test owns the selector. The base rows are enabled exactly
at iteration zero. The recursive rows are enabled exactly at every nonzero
iteration. A satisfying assignment therefore selects one branch by the
authoritative iteration coordinate; it cannot select a branch independently.

Owns the ordered row composition, branch soundness, disjointness, and honest
completeness. Does not identify either branch with a generated V2 manifest,
bind the iteration column to a state parser, or prove artifact containment.

Assurance tier: implementation model.

Emits constraints: three zero-test rows plus three rows per source branch row.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionFreshFPrimeBranchRows

open Nightstream.Implementation.R1CS

structure Layout (baseRows recursiveRows : List Row) where
  iterationZero : IterationZeroRows.Layout
  baseGate : SelectorGatedRows.Layout baseRows
  recursiveGate : SelectorGatedRows.Layout recursiveRows
  baseSelectorExact : baseGate.selectorColumn = iterationZero.selectorColumn
  recursiveSelectorExact :
    recursiveGate.selectorColumn = iterationZero.selectorColumn

/-- Verifier-owned branch program. Both arms and all selector columns are
part of the relation identity. -/
structure Program where
  baseRows : List Row
  recursiveRows : List Row
  layout : Layout baseRows recursiveRows

def rows {baseRows recursiveRows : List Row}
    (layout : Layout baseRows recursiveRows) : List Row :=
  IterationZeroRows.rows layout.iterationZero ++
    SelectorGatedRows.rows .one layout.baseGate ++
    SelectorGatedRows.rows .zero layout.recursiveGate

def Program.rows (program : Program) : List Row :=
  ProductionFreshFPrimeBranchRows.rows program.layout

theorem rows_length {baseRows recursiveRows : List Row}
    (layout : Layout baseRows recursiveRows) :
    (rows layout).length =
      3 + 3 * baseRows.length + 3 * recursiveRows.length := by
  simp [rows, SelectorGatedRows.rows_length]
  omega

theorem Program.rows_length (program : Program) :
    program.rows.length =
      3 + 3 * program.baseRows.length + 3 * program.recursiveRows.length :=
  ProductionFreshFPrimeBranchRows.rows_length program.layout

private theorem zeroRows_satisfied
    {baseRows recursiveRows : List Row}
    {layout : Layout baseRows recursiveRows} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (IterationZeroRows.rows layout.iterationZero) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

private theorem baseGate_satisfied
    {baseRows recursiveRows : List Row}
    {layout : Layout baseRows recursiveRows} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (SelectorGatedRows.rows .one layout.baseGate) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

private theorem recursiveGate_satisfied
    {baseRows recursiveRows : List Row}
    {layout : Layout baseRows recursiveRows} {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (SelectorGatedRows.rows .zero layout.recursiveGate) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

/-- Satisfying the fixed row program derives exactly one semantic arm. -/
theorem sound
    {baseRows recursiveRows : List Row}
    {layout : Layout baseRows recursiveRows} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    (assignment layout.iterationZero.iterationColumn = 0 /\
        Satisfies baseRows assignment) \/
      (0 < assignment layout.iterationZero.iterationColumn /\
        Satisfies recursiveRows assignment) := by
  have zeroHolds := zeroRows_satisfied satisfied
  by_cases iterationZero :
      assignment layout.iterationZero.iterationColumn = 0
  · apply Or.inl
    refine ⟨iterationZero, ?_⟩
    apply SelectorGatedRows.rows_sound_selected (when := .one) canonical one
    · rw [layout.baseSelectorExact]
      exact
        (IterationZeroRows.selector_eq_one_iff_iteration_eq_zero
          canonical one zeroHolds).mpr iterationZero
    · exact baseGate_satisfied satisfied
  · apply Or.inr
    refine ⟨Nat.pos_of_ne_zero iterationZero, ?_⟩
    apply SelectorGatedRows.rows_sound_selected (when := .zero) canonical one
    · rw [layout.recursiveSelectorExact]
      exact
        (IterationZeroRows.selector_eq_zero_iff_iteration_ne_zero
          canonical one zeroHolds).mpr iterationZero
    · exact recursiveGate_satisfied satisfied

theorem Program.sound
    (program : Program) {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies program.rows assignment) :
    (assignment program.layout.iterationZero.iterationColumn = 0 /\
        Satisfies program.baseRows assignment) \/
      (0 < assignment program.layout.iterationZero.iterationColumn /\
        Satisfies program.recursiveRows assignment) :=
  ProductionFreshFPrimeBranchRows.sound canonical one satisfied

/-- The two branch predicates cannot both hold for the same iteration. -/
theorem branches_disjoint
    {baseRows recursiveRows : List Row}
    {layout : Layout baseRows recursiveRows} {assignment : Nat -> Nat} :
    ¬ (assignment layout.iterationZero.iterationColumn = 0 /\
      0 < assignment layout.iterationZero.iterationColumn) := by
  omega

structure AuxiliariesPlaced
    {baseRows recursiveRows : List Row}
    (layout : Layout baseRows recursiveRows)
    (assignment : Nat -> Nat) : Prop where
  zero : IterationZeroRows.AuxiliariesPlaced layout.iterationZero assignment
  base : SelectorGatedRows.AuxiliariesPlaced layout.baseGate assignment
  recursive :
    SelectorGatedRows.AuxiliariesPlaced layout.recursiveGate assignment

/-- Honest base execution satisfies the complete fixed branch relation. -/
theorem complete_base
    {baseRows recursiveRows : List Row}
    {layout : Layout baseRows recursiveRows} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (iterationZero : assignment layout.iterationZero.iterationColumn = 0)
    (baseSatisfied : Satisfies baseRows assignment)
    (placed : AuxiliariesPlaced layout assignment) :
    Satisfies (rows layout) assignment := by
  have selectorOne : assignment layout.iterationZero.selectorColumn = 1 := by
    simpa [iterationZero] using placed.zero.selector
  have baseSelector : assignment layout.baseGate.selectorColumn = 1 := by
    rw [layout.baseSelectorExact]
    exact selectorOne
  have recursiveSelector :
      assignment layout.recursiveGate.selectorColumn = 1 := by
    rw [layout.recursiveSelectorExact]
    exact selectorOne
  intro row member
  simp only [rows, List.mem_append] at member
  rcases member with zeroOrBase | recursiveMember
  · rcases zeroOrBase with zeroMember | baseMember
    · exact IterationZeroRows.complete one placed.zero row zeroMember
    · exact SelectorGatedRows.rows_complete_selected (when := .one)
        canonical one baseSelector baseSatisfied placed.base row baseMember
  · exact SelectorGatedRows.rows_complete_unselected (when := .zero)
      canonical one recursiveSelector placed.recursive row recursiveMember

/-- Honest recursive execution satisfies the complete fixed branch relation. -/
theorem complete_recursive
    {baseRows recursiveRows : List Row}
    {layout : Layout baseRows recursiveRows} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (iterationNonzero :
      assignment layout.iterationZero.iterationColumn ≠ 0)
    (recursiveSatisfied : Satisfies recursiveRows assignment)
    (placed : AuxiliariesPlaced layout assignment) :
    Satisfies (rows layout) assignment := by
  have selectorZero : assignment layout.iterationZero.selectorColumn = 0 := by
    simpa [iterationNonzero] using placed.zero.selector
  have baseSelector : assignment layout.baseGate.selectorColumn = 0 := by
    rw [layout.baseSelectorExact]
    exact selectorZero
  have recursiveSelector :
      assignment layout.recursiveGate.selectorColumn = 0 := by
    rw [layout.recursiveSelectorExact]
    exact selectorZero
  intro row member
  simp only [rows, List.mem_append] at member
  rcases member with zeroOrBase | recursiveMember
  · rcases zeroOrBase with zeroMember | baseMember
    · exact IterationZeroRows.complete one placed.zero row zeroMember
    · exact SelectorGatedRows.rows_complete_unselected (when := .one)
        canonical one baseSelector placed.base row baseMember
  · exact SelectorGatedRows.rows_complete_selected (when := .zero)
      canonical one recursiveSelector recursiveSatisfied placed.recursive row
        recursiveMember

theorem Program.complete_base
    (program : Program) {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (iterationZero :
      assignment program.layout.iterationZero.iterationColumn = 0)
    (baseSatisfied : Satisfies program.baseRows assignment)
    (placed : AuxiliariesPlaced program.layout assignment) :
    Satisfies program.rows assignment :=
  ProductionFreshFPrimeBranchRows.complete_base canonical one iterationZero
    baseSatisfied placed

theorem Program.complete_recursive
    (program : Program) {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (iterationNonzero :
      assignment program.layout.iterationZero.iterationColumn ≠ 0)
    (recursiveSatisfied : Satisfies program.recursiveRows assignment)
    (placed : AuxiliariesPlaced program.layout assignment) :
    Satisfies program.rows assignment :=
  ProductionFreshFPrimeBranchRows.complete_recursive canonical one
    iterationNonzero recursiveSatisfied placed

end Nightstream.Implementation.Nebula.ProductionFreshFPrimeBranchRows
