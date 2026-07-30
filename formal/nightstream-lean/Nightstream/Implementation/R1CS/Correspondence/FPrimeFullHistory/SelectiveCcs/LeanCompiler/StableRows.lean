import Nightstream.Implementation.Lowering.Goldilocks.Compiler
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.DirectRows

/-!
Contract: compile stable-column Goldilocks rows into the Lean-owned direct
selective CCS relation.

Assurance tier: model-level.

Owns: coefficient deposition from structural `ColumnId` values into a finite
matrix width, exact equality between dense and sparse linear-combination
evaluation, occurrence-preserving row translation, and the equivalence between
the compiled CCS zero set and `Goldilocks.Satisfies` on the pulled assignment.

Does not own: selection of a concrete column index, proof that a chosen index
is injective on allocated columns, construction of an honest indexed
assignment, low-norm encoding, fixed-point shape, Rust, or generated
artifacts.

Emits constraints: no new R1CS rows. It compiles each supplied physical row
to one selective product row.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.StableRows

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.Implementation.Lowering

namespace Goldilocks

abbrev ColumnId :=
  Nightstream.Implementation.Lowering.Goldilocks.ColumnId

abbrev LinearCombination :=
  Nightstream.Implementation.Lowering.Goldilocks.LinearCombination

abbrev Row :=
  Nightstream.Implementation.Lowering.Goldilocks.Row

abbrev OwnedRow :=
  Nightstream.Implementation.Lowering.Goldilocks.OwnedRow

def Satisfies :=
  Nightstream.Implementation.Lowering.Goldilocks.Satisfies

end Goldilocks

private theorem fadd_assoc (left middle right : F) :
    (left + middle) + right = left + (middle + right) := by
  apply Fin.ext
  simp only [Fin.val_add]
  rw [Nat.mod_add_mod, Nat.add_mod_mod, Nat.add_assoc]

private theorem fadd_comm (left right : F) :
    left + right = right + left := by
  apply Fin.ext
  simp only [Fin.val_add, Nat.add_comm]

private theorem fadd_mul (left middle right : F) :
    (left + middle) * right = left * right + middle * right := by
  apply Fin.ext
  simp only [Fin.val_add, Fin.val_mul]
  rw [Nat.mod_mul_mod, Nat.add_mul, ← Nat.add_mod]

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨fadd_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨fadd_comm⟩

private theorem foldl_pointwise_add
    {columns : Nat}
    (indices : List (Fin columns))
    (left right : Fin columns → F)
    (assignment : Fin columns → F)
    (leftInitial rightInitial : F) :
    indices.foldl
        (fun accumulated column =>
          accumulated + (left column + right column) * assignment column)
        (leftInitial + rightInitial) =
      indices.foldl
          (fun accumulated column =>
            accumulated + left column * assignment column)
          leftInitial +
        indices.foldl
          (fun accumulated column =>
            accumulated + right column * assignment column)
          rightInitial := by
  induction indices generalizing leftInitial rightInitial with
  | nil =>
      rfl
  | cons column tail inductionHypothesis =>
      simp only [List.foldl_cons]
      have firstStep :
          (leftInitial + rightInitial) +
              (left column + right column) * assignment column =
            (leftInitial + left column * assignment column) +
              (rightInitial + right column * assignment column) := by
        rw [fadd_mul]
        ac_rfl
      rw [firstStep]
      exact inductionHypothesis
        (leftInitial + left column * assignment column)
        (rightInitial + right column * assignment column)

private theorem eval_pointwise_add
    {columns : Nat}
    (left right : DirectRows.LinearCombination columns)
    (assignment : Fin columns → F) :
    DirectRows.LinearCombination.eval
        (fun column => left column + right column) assignment =
      DirectRows.LinearCombination.eval left assignment +
        DirectRows.LinearCombination.eval right assignment := by
  unfold DirectRows.LinearCombination.eval
  simpa only [Fin.zero_add] using
    (foldl_pointwise_add (canonicalFinIndices columns) left right assignment
      0 0)

private theorem eval_zero
    {columns : Nat}
    (assignment : Fin columns → F) :
    DirectRows.LinearCombination.eval (fun _ => 0) assignment = 0 := by
  unfold DirectRows.LinearCombination.eval
  generalize canonicalFinIndices columns = indices
  induction indices with
  | nil =>
      rfl
  | cons column tail inductionHypothesis =>
      rw [List.foldl_cons]
      change
        List.foldl
            (fun accumulated next =>
              accumulated + 0 * assignment next)
            (0 + 0 * assignment column) tail =
          0
      change
        List.foldl
            (fun accumulated next =>
              accumulated + 0 * assignment next)
            0 tail =
          0 at inductionHypothesis
      rw [Fin.zero_mul, Fin.add_zero]
      exact inductionHypothesis

private theorem foldl_absent_single
    {Index : Type}
    [DecidableEq Index]
    (indices : List Index)
    (selected : Index)
    (term : Index → F)
    (absent : selected ∉ indices)
    (initial : F) :
    indices.foldl
        (fun accumulated index =>
          accumulated +
            if index = selected then term index else 0)
        initial =
      initial := by
  induction indices generalizing initial with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      have headNe : head ≠ selected := by
        intro equal
        apply absent
        simp [equal]
      have absentTail : selected ∉ tail := by
        intro member
        exact absent (by simp [member])
      simp only [List.foldl_cons, if_neg headNe, Fin.add_zero]
      exact inductionHypothesis absentTail initial

private theorem foldl_single
    {Index : Type}
    [DecidableEq Index]
    (indices : List Index)
    (selected : Index)
    (term : Index → F)
    (nodup : indices.Nodup)
    (member : selected ∈ indices) :
    indices.foldl
        (fun accumulated index =>
          accumulated +
            if index = selected then term index else 0)
        0 =
      term selected := by
  induction indices with
  | nil =>
      simp at member
  | cons head tail inductionHypothesis =>
      simp only [List.foldl_cons]
      by_cases headEq : head = selected
      · subst head
        rw [if_pos rfl, Fin.zero_add]
        exact foldl_absent_single tail selected term
          (List.nodup_cons.mp nodup).1 (term selected)
      · have memberTail : selected ∈ tail := by
          simpa [Ne.symm headEq] using member
        rw [if_neg headEq, Fin.add_zero]
        exact inductionHypothesis
          (List.nodup_cons.mp nodup).2 memberTail

private def single
    {columns : Nat}
    (selected : Fin columns)
    (coefficient : F) :
    DirectRows.LinearCombination columns :=
  fun column => if column = selected then coefficient else 0

private theorem eval_single
    {columns : Nat}
    (selected : Fin columns)
    (coefficient : F)
    (assignment : Fin columns → F) :
    DirectRows.LinearCombination.eval
        (single selected coefficient) assignment =
      coefficient * assignment selected := by
  unfold DirectRows.LinearCombination.eval
  have contributionFunction :
      (fun accumulated column =>
        accumulated + single selected coefficient column *
          assignment column) =
        (fun accumulated column =>
          accumulated +
            if column = selected then
              coefficient * assignment column
            else
              0) := by
    funext accumulated column
    by_cases equal : column = selected
    · simp [single, equal]
    · simp [single, equal, Fin.zero_mul]
  rw [contributionFunction]
  exact foldl_single
    (canonicalFinIndices columns) selected
    (fun column => coefficient * assignment column)
    (canonicalFinIndices_nodup columns)
    (by simp [canonicalFinIndices])

/-- Deposit every sparse term at the finite index selected for its structural
column. Different structural columns may share an index at this layer; the
soundness theorem then uses the corresponding pulled assignment. Honest
completeness later requires an injective selected index. -/
def denseCombination
    {columns : Nat}
    (columnIndex : Goldilocks.ColumnId → Fin columns) :
    Goldilocks.LinearCombination →
      DirectRows.LinearCombination columns
  | [] => fun _ => 0
  | term :: tail =>
      fun column =>
        single (columnIndex term.column) term.coefficient column +
          denseCombination columnIndex tail column

/-- Read a stable-column assignment through the selected finite index. -/
def pulledAssignment
    {columns : Nat}
    (columnIndex : Goldilocks.ColumnId → Fin columns)
    (assignment : Fin columns → F) :
    Goldilocks.ColumnId → F :=
  fun column => assignment (columnIndex column)

/-- Coefficient deposition preserves the sparse linear-combination value
exactly. This is the semantic center of the stable-column compiler. -/
theorem denseCombination_eval
    {columns : Nat}
    (columnIndex : Goldilocks.ColumnId → Fin columns)
    (source : Goldilocks.LinearCombination)
    (assignment : Fin columns → F) :
    DirectRows.LinearCombination.eval
        (denseCombination columnIndex source) assignment =
      Nightstream.Implementation.Lowering.Goldilocks.LinearCombination.eval
        (pulledAssignment columnIndex assignment) source := by
  induction source with
  | nil =>
      exact eval_zero assignment
  | cons term tail inductionHypothesis =>
      change
        DirectRows.LinearCombination.eval
            (fun column =>
              single (columnIndex term.column) term.coefficient column +
                denseCombination columnIndex tail column)
            assignment =
          term.coefficient *
              assignment (columnIndex term.column) +
            Nightstream.Implementation.Lowering.Goldilocks.LinearCombination.eval
              (pulledAssignment columnIndex assignment) tail
      rw [eval_pointwise_add, eval_single, inductionHypothesis]

/-- Translate one stable physical equation without changing its arithmetic
meaning. -/
def row
    {columns : Nat}
    (columnIndex : Goldilocks.ColumnId → Fin columns)
    (source : Goldilocks.Row) :
    DirectRows.SourceRow columns where
  a := denseCombination columnIndex source.a
  b := denseCombination columnIndex source.b
  c := denseCombination columnIndex source.c

theorem row_holds_iff
    {columns : Nat}
    (columnIndex : Goldilocks.ColumnId → Fin columns)
    (source : Goldilocks.Row)
    (assignment : Fin columns → F) :
    (row columnIndex source).Holds assignment ↔
      Nightstream.Implementation.Lowering.Goldilocks.Row.Holds
        (pulledAssignment columnIndex assignment) source := by
  unfold DirectRows.SourceRow.Holds
  unfold Nightstream.Implementation.Lowering.Goldilocks.Row.Holds
  simp only [row]
  rw [denseCombination_eval, denseCombination_eval, denseCombination_eval]

/-- Preserve source occurrence order. Row identities remain on the source
program and are not replaced with syntactic row equality. -/
def program
    {columns : Nat}
    (columnIndex : Goldilocks.ColumnId → Fin columns)
    (rows : List Goldilocks.OwnedRow) :
    List (DirectRows.SourceRow columns) :=
  rows.map fun owned => row columnIndex owned.row

theorem program_holds_iff
    {columns : Nat}
    (columnIndex : Goldilocks.ColumnId → Fin columns)
    (rows : List Goldilocks.OwnedRow)
    (assignment : Fin columns → F) :
    (∀ index : Fin (program columnIndex rows).length,
        ((program columnIndex rows).get index).Holds assignment) ↔
      Goldilocks.Satisfies rows
        (pulledAssignment columnIndex assignment) := by
  induction rows with
  | nil =>
      constructor
      · intro _
        exact True.intro
      · intro _ index
        exact Fin.elim0 index
  | cons head tail inductionHypothesis =>
      constructor
      · intro all
        have headHolds :
            (row columnIndex head.row).Holds assignment := by
          simpa [program] using all ⟨0, by simp [program]⟩
        have tailHolds :
            ∀ index : Fin (program columnIndex tail).length,
              ((program columnIndex tail).get index).Holds assignment := by
          intro index
          simpa [program] using all (Fin.succ index)
        exact ⟨
          (row_holds_iff columnIndex head.row assignment).mp headHolds,
          inductionHypothesis.mp tailHolds
        ⟩
      · rintro ⟨headHolds, tailHolds⟩ index
        refine Fin.cases ?_ (fun tailIndex => ?_) index
        · simpa [program] using
            (row_holds_iff columnIndex head.row assignment).mpr headHolds
        · have allTail := inductionHypothesis.mpr tailHolds
          simpa [program] using allTail tailIndex

/-- The Lean-owned selective relation accepts exactly the supplied stable
physical rows on the pulled assignment. -/
theorem constraintSatisfied_iff
    {columns : Nat}
    (columnIndex : Goldilocks.ColumnId → Fin columns)
    (one : Goldilocks.ColumnId)
    (rows : List Goldilocks.OwnedRow)
    (profile :
      RelationProfile.Profile (program columnIndex rows).length columns)
    (assignment : Fin columns → F)
    (constantOne : assignment (columnIndex one) = 1) :
    ConstraintSatisfied baseOps
        (DirectRows.paperSystem
          (DirectRows.relation (columnIndex one)
            (program columnIndex rows))
          profile)
        assignment ↔
      Goldilocks.Satisfies rows
        (pulledAssignment columnIndex assignment) := by
  rw [DirectRows.constraintSatisfied_iff
    (columnIndex one) (program columnIndex rows) profile assignment
    constantOne]
  exact program_holds_iff columnIndex rows assignment

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.StableRows
