import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: list-free materialization for a consecutive straight-line program.

`SequentialProgram first count` describes exactly one definition for each
column `first + index`, with every right-hand-side reference strictly before
that output.  `materialize` executes those definitions by structural
recursion on the number of completed steps; it never constructs a
`count`-sized list.

This module owns only generic SSA materialization.  It does not classify
production columns, decode an artifact, or assert that a particular Rust
emitter supplies such a program.
-/

namespace Nightstream.Implementation.R1CS

open Program

/-- An indexed SSA block whose outputs are the consecutive columns
`first, ..., first + count - 1`. -/
structure SequentialProgram (first count : Nat) where
  definitionAt : Nat → Definition
  output_eq :
    ∀ index, index < count →
      (definitionAt index).output = first + index
  references_before :
    ∀ index, index < count →
      ∀ column, column ∈ (definitionAt index).rhs.refs →
        column < first + index

namespace SequentialProgram

private theorem rhsEval_lt (assignment : Nat → Nat) (rhs : Rhs) :
    rhs.eval assignment < goldilocksP := by
  have modulusPositive : 0 < goldilocksP := by decide
  cases rhs with
  | linear terms =>
      rw [Rhs.eval, Program.lcEval_eq_raw_mod]
      exact Nat.mod_lt _ modulusPositive
  | product left right =>
      simp only [Rhs.eval]
      exact Nat.mod_lt _ modulusPositive

private theorem lcEval_congr_refs
    {left right : Nat → Nat} (terms : List (Nat × Nat))
    (agreement : ∀ term ∈ terms, left term.1 = right term.1) :
    lcEval left terms = lcEval right terms := by
  unfold lcEval
  have foldAgree : ∀ initial,
      terms.foldl (fun acc term => acc + term.2 * left term.1) initial =
        terms.foldl (fun acc term => acc + term.2 * right term.1) initial := by
    intro initial
    induction terms generalizing initial with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.foldl]
        rw [agreement head (by simp)]
        apply inductionHypothesis
        intro term member
        exact agreement term (by simp [member])
  rw [foldAgree 0]

private theorem rhsEval_congr_refs
    {left right : Nat → Nat} (rhs : Rhs)
    (agreement : ∀ column ∈ rhs.refs, left column = right column) :
    rhs.eval left = rhs.eval right := by
  cases rhs with
  | linear terms =>
      apply lcEval_congr_refs terms
      intro term member
      apply agreement term.1
      exact List.mem_map.mpr ⟨term, member, rfl⟩
  | product lhs rhs =>
      simp only [Rhs.eval]
      rw [lcEval_congr_refs lhs (by
        intro term member
        apply agreement term.1
        apply List.mem_append_left
        exact List.mem_map.mpr ⟨term, member, rfl⟩)]
      rw [lcEval_congr_refs rhs (by
        intro term member
        apply agreement term.1
        apply List.mem_append_right
        exact List.mem_map.mpr ⟨term, member, rfl⟩)]

/-- State after `steps` indexed definitions.  Steps beyond `count` are no-ops,
which keeps the recursion total while `materialize` selects exactly `count`
steps. -/
def stateAt {first count : Nat}
    (program : SequentialProgram first count) (source : Nat → Nat) :
    Nat → Nat → Nat
  | 0 => source
  | steps + 1 =>
      if steps < count then
        Program.execute (stateAt program source steps)
          (program.definitionAt steps)
      else
        stateAt program source steps

/-- Execute all indexed definitions without first materializing them as a
list. -/
def materialize {first count : Nat}
    (program : SequentialProgram first count) (source : Nat → Nat) :
    Nat → Nat :=
  stateAt program source count

private theorem stateAt_source_below
    {first count : Nat} (program : SequentialProgram first count)
    (source : Nat → Nat) :
    ∀ steps column, column < first →
      stateAt program source steps column = source column := by
  intro steps
  induction steps with
  | zero =>
      intro column below
      rfl
  | succ steps inductionHypothesis =>
      intro column below
      by_cases inProgram : steps < count
      · have different : column ≠ (program.definitionAt steps).output := by
          intro equal
          have output := program.output_eq steps inProgram
          rw [output] at equal
          omega
        rw [stateAt, if_pos inProgram, Program.execute,
          Program.setColumn_other _ different]
        exact inductionHypothesis column below
      · rw [stateAt, if_neg inProgram]
        exact inductionHypothesis column below

private theorem stateAt_canonical
    {first count : Nat} (program : SequentialProgram first count)
    (source : Nat → Nat) (sourceCanonical : ∀ column, source column < goldilocksP) :
    ∀ steps column, stateAt program source steps column < goldilocksP := by
  intro steps
  induction steps with
  | zero => exact sourceCanonical
  | succ steps inductionHypothesis =>
      intro column
      by_cases inProgram : steps < count
      · by_cases isOutput : column = (program.definitionAt steps).output
        · subst column
          simp only [stateAt, if_pos inProgram, Program.execute,
            Program.setColumn_same]
          exact rhsEval_lt (stateAt program source steps)
            (program.definitionAt steps).rhs
        · rw [stateAt, if_pos inProgram, Program.execute,
            Program.setColumn_other _ isOutput]
          exact inductionHypothesis column
      · rw [stateAt, if_neg inProgram]
        exact inductionHypothesis column

private theorem stateAt_succ_definition_holds
    {first count : Nat} (program : SequentialProgram first count)
    (source : Nat → Nat) (index : Nat) (inProgram : index < count) :
    Definition.Holds (stateAt program source (index + 1))
      (program.definitionAt index) := by
  have referencesUnchanged :
      ∀ column ∈ (program.definitionAt index).rhs.refs,
        Program.execute (stateAt program source index)
            (program.definitionAt index) column =
          stateAt program source index column := by
    intro column member
    have different : column ≠ (program.definitionAt index).output := by
      intro equal
      have referenceBefore :=
        program.references_before index inProgram column member
      have output := program.output_eq index inProgram
      rw [output] at equal
      omega
    rw [Program.execute, Program.setColumn_other _ different]
  unfold Definition.Holds
  simp only [stateAt, if_pos inProgram, Program.execute,
    Program.setColumn_same]
  exact (rhsEval_congr_refs (program.definitionAt index).rhs
    referencesUnchanged).symm

private theorem stateAt_add_preserves_before
    {first count : Nat} (program : SequentialProgram first count)
    (source : Nat → Nat) (base extra column : Nat)
    (before : column < first + base) :
    stateAt program source (base + extra) column =
      stateAt program source base column := by
  induction extra with
  | zero => rfl
  | succ extra inductionHypothesis =>
      rw [Nat.add_succ, stateAt]
      by_cases inProgram : base + extra < count
      · rw [if_pos inProgram]
        have different :
            column ≠ (program.definitionAt (base + extra)).output := by
          intro equal
          have output := program.output_eq (base + extra) inProgram
          rw [output] at equal
          omega
        rw [Program.execute, Program.setColumn_other _ different]
        exact inductionHypothesis
      · rw [if_neg inProgram]
        exact inductionHypothesis

/-- Materialization preserves every source column strictly below the first
derived output. -/
theorem materialize_source_below
    {first count : Nat} (program : SequentialProgram first count)
    (source : Nat → Nat) {column : Nat} (below : column < first) :
    materialize program source column = source column := by
  exact stateAt_source_below program source count column below

/-- A canonical source assignment produces a canonical final assignment. -/
theorem materialize_canonical
    {first count : Nat} (program : SequentialProgram first count)
    (source : Nat → Nat) (sourceCanonical : ∀ column, source column < goldilocksP) :
    ∀ column, materialize program source column < goldilocksP := by
  exact stateAt_canonical program source sourceCanonical count

/-- Every indexed SSA equation holds in the final materialized assignment.
This conclusion is computed from `definitionAt`; it is not a caller-supplied
row-satisfaction premise. -/
theorem materialize_definition_holds
    {first count : Nat} (program : SequentialProgram first count)
    (source : Nat → Nat) (index : Nat) (inProgram : index < count) :
    Definition.Holds (materialize program source)
      (program.definitionAt index) := by
  have stepHolds :=
    stateAt_succ_definition_holds program source index inProgram
  have baseLe : index + 1 ≤ count := Nat.succ_le_iff.mpr inProgram
  have split : index + 1 + (count - (index + 1)) = count := by
    omega
  have preserved : ∀ column, column < first + (index + 1) →
      materialize program source column =
        stateAt program source (index + 1) column := by
    intro column before
    unfold materialize
    have tailPreserves := stateAt_add_preserves_before program source
      (index + 1) (count - (index + 1)) column before
    exact (congrArg (fun steps => stateAt program source steps column) split).symm.trans
      tailPreserves
  have outputBefore :
      (program.definitionAt index).output < first + (index + 1) := by
    rw [program.output_eq index inProgram]
    omega
  have rhsPreserved :
      (program.definitionAt index).rhs.eval (materialize program source) =
        (program.definitionAt index).rhs.eval
          (stateAt program source (index + 1)) := by
    apply rhsEval_congr_refs
    intro column member
    apply preserved column
    have referenceBefore :=
      program.references_before index inProgram column member
    omega
  unfold Definition.Holds at stepHolds ⊢
  rw [preserved _ outputBefore, rhsPreserved]
  exact stepHolds

end SequentialProgram

end Nightstream.Implementation.R1CS
