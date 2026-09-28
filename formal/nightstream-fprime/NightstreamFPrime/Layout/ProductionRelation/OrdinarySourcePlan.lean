import NightstreamFPrime.Layout.ProductionRelation.PlanComposition
import NightstreamFPrime.Layout.ProductionRelation.SourceCompiler

/-!
Owns the indexed source-to-matrix bridge for ordinary R1CS rows. Each row can
use its own sparse source resolver. Preservation is required only for source
columns that occur in that row.

This module does not select production rows or retained assignment columns.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

namespace SourceMap

/-- Exact reconstruction for the source terms used by one combination. -/
def PreservesCombination {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded sourceWidth combination) :
    Prop :=
  ∀ term, ∀ member : term ∈ combination.terms,
    (sourceMap.form ⟨term.1, bounded term member⟩).eval assignment =
      source term.1

/-- Exact reconstruction for all source terms used by one row. -/
def PreservesRow {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (row : R1CS.Row) (bounded : SourceCompiler.RowBounded sourceWidth row) :
    Prop :=
  PreservesCombination sourceMap assignment source row.a bounded.1 ∧
    PreservesCombination sourceMap assignment source row.b bounded.2.1 ∧
      PreservesCombination sourceMap assignment source row.c bounded.2.2

end SourceMap

private theorem compileTerms_eval_local {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (terms : List (Nat × F))
    (bounded : ∀ term ∈ terms, term.1 < sourceWidth)
    (preserves : ∀ term, ∀ member : term ∈ terms,
      (sourceMap.form ⟨term.1, bounded term member⟩).eval assignment =
        source term.1) :
    (SourceCompiler.compileTerms sourceMap terms bounded).eval assignment =
      (terms.map fun term => term.2 * source term.1).sum := by
  induction terms with
  | nil => simp [SourceCompiler.compileTerms]
  | cons term rest inductionHypothesis =>
      have headBound : term.1 < sourceWidth := bounded term (by simp)
      have restBound : ∀ candidate ∈ rest, candidate.1 < sourceWidth :=
        fun candidate member => bounded candidate (by simp [member])
      have headPreserves :
          (sourceMap.form ⟨term.1, headBound⟩).eval assignment =
            source term.1 := by
        simpa only using preserves term (by simp)
      have restPreserves : ∀ candidate, ∀ member : candidate ∈ rest,
          (sourceMap.form ⟨candidate.1, restBound candidate member⟩).eval
              assignment = source candidate.1 := by
        intro candidate member
        simpa only using preserves candidate (by simp [member])
      simp only [SourceCompiler.compileTerms, SparseForm.add_eval,
        SparseForm.scale_eval, List.map_cons, List.sum_cons]
      rw [headPreserves, inductionHypothesis restBound restPreserves]

private theorem compileCombination_eval_local
    {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (combination : R1CS.LinearCombination)
    (bounded : SourceCompiler.CombinationBounded sourceWidth combination)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (one : assignment oneColumn = 1)
    (preserves : SourceMap.PreservesCombination sourceMap assignment source
      combination bounded) :
    (SourceCompiler.compileCombination sourceMap oneColumn combination bounded).eval
        assignment = combination.eval source := by
  rw [SourceCompiler.compileCombination, SparseForm.add_eval,
    SparseForm.singleton_eval, one, mul_one,
    compileTerms_eval_local sourceMap assignment source combination.terms
      bounded preserves]
  rfl

/-- Local term preservation is sufficient for the exact ordinary-row bridge. -/
theorem compileRow_preserves_local {sourceWidth logicalWidth : Nat}
    (sourceMap : SourceCompiler.SourceMap sourceWidth logicalWidth)
    (oneColumn : Fin logicalWidth) (row : R1CS.Row)
    (bounded : SourceCompiler.RowBounded sourceWidth row)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (one : assignment oneColumn = 1)
    (preserves : SourceMap.PreservesRow sourceMap assignment source row
      bounded) :
    (SourceCompiler.compileRow sourceMap oneColumn row bounded).Preserves
      assignment source row := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [SourceCompiler.compileRow, one]
  · exact compileCombination_eval_local sourceMap oneColumn row.a bounded.1
      assignment source one preserves.1
  · exact compileCombination_eval_local sourceMap oneColumn row.b bounded.2.1
      assignment source one preserves.2.1
  · exact compileCombination_eval_local sourceMap oneColumn row.c bounded.2.2
      assignment source one preserves.2.2

/-- Canonical indexed source rows before final sparse-form substitution. -/
structure Program (sourceWidth : Nat) where
  rowCount : Nat
  rowCount_le : rowCount ≤ 2 ^ NightstreamFPrime.Lifecycle.cubeVariables
  row : Fin rowCount → R1CS.Row
  bounded : ∀ index, SourceCompiler.RowBounded sourceWidth (row index)

namespace Program

/-- Final sparse source resolver for each indexed row. -/
structure Inputs {sourceWidth : Nat} (program : Program sourceWidth)
    (logicalWidth : Nat) where
  oneColumn : Fin logicalWidth
  sourceMap : Fin program.rowCount →
    SourceCompiler.SourceMap sourceWidth logicalWidth

/-- Exact source-row satisfaction in canonical row order. -/
def Holds {sourceWidth : Nat} (program : Program sourceWidth)
    (source : Circuit.Env) : Prop :=
  ∀ index, (program.row index).Holds source

/-- Compile every indexed source row into the sole 14-matrix row relation. -/
def compile {sourceWidth logicalWidth : Nat} (program : Program sourceWidth)
    (inputs : program.Inputs logicalWidth) : OrdinaryRow.Program logicalWidth where
  rowCount := program.rowCount
  rowCount_le := program.rowCount_le
  row := fun index =>
    { source := program.row index
      forms := SourceCompiler.compileRow (inputs.sourceMap index)
        inputs.oneColumn (program.row index) (program.bounded index) }

@[simp] theorem compile_rowCount {sourceWidth logicalWidth : Nat}
    (program : Program sourceWidth) (inputs : program.Inputs logicalWidth) :
    (program.compile inputs).rowCount = program.rowCount := by
  rfl

/-- One compiled plan row is exactly the source compiler output for that
indexed source row. This theorem keeps concrete row programs opaque. -/
theorem compile_toPlan_forms {sourceWidth logicalWidth : Nat}
    (program : Program sourceWidth) (inputs : program.Inputs logicalWidth)
    (index : Fin program.rowCount) :
    (program.compile inputs).toPlan.forms index =
      (SourceCompiler.compileRow (inputs.sourceMap index) inputs.oneColumn
        (program.row index) (program.bounded index)).meaningfulForm := by
  rfl

/-- Row-local source preservation gives preservation for the full indexed
ordinary program. -/
theorem compile_preserves {sourceWidth logicalWidth : Nat}
    (program : Program sourceWidth) (inputs : program.Inputs logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (one : assignment inputs.oneColumn = 1)
    (preserves : ∀ index,
      SourceMap.PreservesRow (inputs.sourceMap index) assignment source
        (program.row index) (program.bounded index)) :
    (program.compile inputs).Preserves assignment source := by
  intro index
  exact compileRow_preserves_local (inputs.sourceMap index) inputs.oneColumn
    (program.row index) (program.bounded index) assignment source one
    (preserves index)

/-- The final 14-matrix plan accepts exactly the indexed source R1CS rows. -/
theorem rowsZero_iff {sourceWidth logicalWidth : Nat}
    (program : Program sourceWidth) (inputs : program.Inputs logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (one : assignment inputs.oneColumn = 1)
    (preserves : ∀ index,
      SourceMap.PreservesRow (inputs.sourceMap index) assignment source
        (program.row index) (program.bounded index)) :
    (program.compile inputs).toPlan.RowsZero assignment ↔
      program.Holds source := by
  have compiledPreserves :=
    program.compile_preserves inputs assignment source one preserves
  constructor
  · intro rows index
    exact ((program.compile inputs).residualAt_live_zero_iff assignment source
      compiledPreserves index).mp (rows index)
  · intro holds index
    exact ((program.compile inputs).residualAt_live_zero_iff assignment source
      compiledPreserves index).mpr (holds index)

end Program

end NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
