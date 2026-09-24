import NightstreamFPrime.Layout.PiRlcWideSampler.Rows
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan

/-! Certified direct CCS compiler for the checked range gadget. Four source
columns are caller-owned inputs; 1,404 temporary columns have no retained
slot; the last 617 columns are the checked witness. The
compiler rejects an unsupported expression or an out-of-range source.
The temporary columns have no row reads and reconstruct as zero. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.RangePlan

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Sampling
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def interface : WideReduction.Interface where
  source := fun lane _ => .var lane.val

def constraints : List Expr :=
  flatConstraints (WideReduction.Program.operations interface 4)

theorem constraints_length : constraints.length = 681 :=
  WideReduction.Program.rowCount_eq interface 4

/-- Matrix rows may read only the caller inputs and checked values. -/
def Active (row : R1CS.Row) : Prop :=
  (∀ term ∈ row.a.terms, term.1 < 4 ∨ 1408 ≤ term.1) ∧
  (∀ term ∈ row.b.terms, term.1 < 4 ∨ 1408 ≤ term.1) ∧
  (∀ term ∈ row.c.terms, term.1 < 4 ∨ 1408 ≤ term.1)

private instance (row : R1CS.Row) : Decidable (Active row) := by
  unfold Active
  infer_instance

structure Compiled where
  rows : List R1CS.Row
  correct : Rows.compile? 1408 constraints = some rows
  bounded : ∀ row ∈ rows, SourceCompiler.RowBounded 2025 row
  active : ∀ row ∈ rows, Active row

private instance (row : R1CS.Row) : Decidable (SourceCompiler.RowBounded 2025 row) :=
  @instDecidableAnd _ _ (SourceCompiler.combinationBoundedDecidable 2025 row.a)
    (@instDecidableAnd _ _ (SourceCompiler.combinationBoundedDecidable 2025 row.b)
      (SourceCompiler.combinationBoundedDecidable 2025 row.c))

def compile? : Option Compiled :=
  match correct : Rows.compile? 1408 constraints with
  | none => none
  | some rows =>
      if bounded : ∀ row ∈ rows, SourceCompiler.RowBounded 2025 row then
        if active : ∀ row ∈ rows, Active row then some ⟨rows, correct, bounded, active⟩ else none
      else none

namespace Compiled

theorem rowCount (compiled : Compiled) : compiled.rows.length = 681 := by
  rw [Rows.compile?_length 1408 constraints compiled.rows compiled.correct, constraints_length]

def program (compiled : Compiled) : OrdinarySourcePlan.Program 2025 where
  rowCount := compiled.rows.length
  rowCount_le := by rw [compiled.rowCount]; decide
  row := compiled.rows.get
  bounded := fun index => compiled.bounded _ (List.get_mem _ _)

theorem holds_iff (compiled : Compiled) (env : Env) :
    compiled.program.Holds env ↔ ConstraintsHold env constraints := by
  rw [← Rows.compile?_correct 1408 constraints compiled.rows compiled.correct env]
  constructor
  · intro all row member
    obtain ⟨index, same⟩ := List.mem_iff_get.mp member
    rw [← same]
    exact all index
  · intro all index
    exact all _ (List.get_mem _ _)

theorem plan_rowCount (compiled : Compiled) {columns : Nat}
    (inputs : compiled.program.Inputs columns) :
    (compiled.program.compile inputs).toPlan.rowCount = 681 := compiled.rowCount

/-- Sparse substitution preserves both directions of the checked relation.
A caller supplies the actual Poseidon endpoint forms and retained slots. -/
theorem plan_rows_iff (compiled : Compiled) {columns : Nat}
    (inputs : compiled.program.Inputs columns) (assignment : Assignment F columns)
    (env : Env) (one : assignment inputs.oneColumn = 1)
    (preserves : ∀ index, (inputs.sourceMap index).Preserves assignment env) :
    (compiled.program.compile inputs).toPlan.RowsZero assignment ↔ ConstraintsHold env constraints := by
  rw [← compiled.holds_iff env]
  apply OrdinarySourcePlan.Program.rowsZero_iff _ _ _ _ one
  intro index
  exact ⟨fun term member => preserves index ⟨term.1, (compiled.program.bounded index).1 term member⟩,
    fun term member => preserves index ⟨term.1, (compiled.program.bounded index).2.1 term member⟩,
    fun term member => preserves index ⟨term.1, (compiled.program.bounded index).2.2 term member⟩⟩

end Compiled

end NightstreamFPrime.Layout.PiRlcWideSampler.RangePlan
