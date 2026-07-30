import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Rows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.RelationProfile

/-!
Contract: compile dense Goldilocks R1CS equations into the fixed thirteen-port
selective CCS relation.

Assurance tier: model-level.

Owns: the source equation type, its finite dot-product semantics, the exact
thirteen matrix rows for every source equation, padding rows, and the
soundness/completeness equivalence between source equations and the compiled
CCS zero set.

Does not own: stable-column indexing, sparse manifest decoding, low-norm
assignment encoding, canonical ternary openings, branch selection, fixed-point
shape discovery, Rust, or generated artifacts.

Emits constraints: one selective product row for each supplied source
equation. Matrix arity and the polynomial come from `RelationProfile`.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.DirectRows

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Ports

/-- One dense linear combination over a fixed source-column domain. -/
abbrev LinearCombination (columns : Nat) :=
  Fin columns → F

namespace LinearCombination

/-- Canonical finite dot product in the same increasing column order as the
paper matrix-vector product. -/
def eval {columns : Nat}
    (combination : LinearCombination columns)
    (assignment : Assignment F columns) : F :=
  (canonicalFinIndices columns).foldl
    (fun accumulated column =>
      accumulated + combination column * assignment column)
    0

end LinearCombination

/-- One source R1CS equation `(A z) * (B z) = C z`. -/
structure SourceRow (columns : Nat) where
  a : LinearCombination columns
  b : LinearCombination columns
  c : LinearCombination columns

namespace SourceRow

def Holds {columns : Nat}
    (row : SourceRow columns)
    (assignment : Assignment F columns) : Prop :=
  row.a.eval assignment * row.b.eval assignment =
    row.c.eval assignment

end SourceRow

/-- Dense coefficient row for one selective matrix role. The constant-one
selector is an identity row at `one`; all unused ports are exactly zero. -/
def roleRow {columns : Nat}
    (one : Fin columns)
    (source : SourceRow columns) :
    Role → Fin columns → F
  | .generalSelector =>
      fun column => if column = one then 1 else 0
  | .a => source.a
  | .b => source.b
  | .c => source.c
  | _ => fun _ => 0

/-- Lean-owned finite matrices for the supplied source program. -/
def relation {columns : Nat}
    (one : Fin columns)
    (program : List (SourceRow columns)) :
    RelationProfile.FiniteRelation program.length columns where
  matrices := fun role row => roleRow one (program.get row) role

theorem matrixVectorAt_roleRow_generalSelector
    {variables columns : Nat}
    (one : Fin columns)
    (source : SourceRow columns)
    (assignment : Assignment F columns)
    (constantOne : assignment one = 1)
    (vertex : BooleanVertex variables) :
    matrixVectorAt baseOps
        (fun _ => roleRow one source .generalSelector)
        assignment vertex =
      1 := by
  rw [matrixVectorAt_identityRow baseOps baseLaws
    (fun _ => roleRow one source .generalSelector)
    assignment vertex one]
  · exact constantOne
  · intro column
    simp [roleRow, baseOps]

theorem matrixVectorAt_roleRow_source
    {variables columns : Nat}
    (one : Fin columns)
    (source : SourceRow columns)
    (assignment : Assignment F columns)
    (vertex : BooleanVertex variables) :
    (matrixVectorAt baseOps
        (fun _ => roleRow one source .a) assignment vertex =
          source.a.eval assignment) ∧
      (matrixVectorAt baseOps
        (fun _ => roleRow one source .b) assignment vertex =
          source.b.eval assignment) ∧
      (matrixVectorAt baseOps
        (fun _ => roleRow one source .c) assignment vertex =
          source.c.eval assignment) := by
  constructor
  · rfl
  · constructor <;> rfl

private theorem matrixVectorAt_zeroMatrix
    {variables columns : Nat}
    (assignment : Assignment F columns)
    (vertex : BooleanVertex variables) :
    matrixVectorAt baseOps (fun _ _ => 0) assignment vertex = 0 := by
  unfold matrixVectorAt
  generalize canonicalFinIndices columns = indices
  induction indices with
  | nil => rfl
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

theorem matrixVectorAt_roleRow_unused
    {variables columns : Nat}
    (one : Fin columns)
    (source : SourceRow columns)
    (assignment : Assignment F columns)
    (vertex : BooleanVertex variables)
    (role : Role)
    (unused :
      role ≠ .generalSelector ∧ role ≠ .a ∧ role ≠ .b ∧ role ≠ .c) :
    matrixVectorAt baseOps
        (fun _ => roleRow one source role) assignment vertex =
      0 := by
  have roleZero : roleRow one source role = fun _ => 0 := by
    funext column
    cases role <;> simp_all [roleRow]
  rw [roleZero]
  exact matrixVectorAt_zeroMatrix assignment vertex

/-- Paper CCS view of one Lean-owned finite relation. -/
def paperSystem
    {rows columns : Nat}
    (finite : RelationProfile.FiniteRelation rows columns)
    (profile : RelationProfile.Profile rows columns) :
    CCSResidualTable.Structure F
      (RelationProfile.Profile.shape profile).sourceShape columns where
  matrices := fun port =>
    RowPadding.padRows (finite.matrixAt port)
  constraintPolynomial :=
    (finite.toStructure profile).constraintPolynomial

theorem matrixVectorAt_padRows_numeric
    {rows variables columns : Nat}
    (matrix : RowPadding.NumericMatrix F rows columns)
    (covers : rows ≤ 2 ^ variables)
    (assignment : Assignment F columns)
    (row : Fin rows) :
    matrixVectorAt baseOps (RowPadding.padRows matrix) assignment
        (RowPadding.numericRowVertex covers row) =
      matrixVectorAt baseOps (fun _ => matrix row) assignment
        (RowPadding.numericRowVertex covers row) := by
  unfold matrixVectorAt
  have stepsEqual :
      (fun accumulated column =>
        baseOps.add accumulated
          (baseOps.mul
            (RowPadding.padRows matrix
              (RowPadding.numericRowVertex covers row) column)
            (assignment column))) =
        (fun accumulated column =>
          baseOps.add accumulated
            (baseOps.mul (matrix row column) (assignment column))) := by
    funext accumulated column
    rw [RowPadding.padRows_at_numericRow]
  rw [stepsEqual]

theorem matrixVectorAt_padRows_padding
    {rows variables columns : Nat}
    (matrix : RowPadding.NumericMatrix F rows columns)
    (assignment : Assignment F columns)
    (vertex : BooleanVertex variables)
    (padding : rows ≤ rowIndex vertex) :
    matrixVectorAt baseOps (RowPadding.padRows matrix) assignment vertex =
      0 := by
  have matrixZero :
      matrixVectorAt baseOps (RowPadding.padRows matrix) assignment vertex =
        matrixVectorAt baseOps (fun _ _ => 0) assignment vertex := by
    unfold matrixVectorAt
    have stepsEqual :
        (fun accumulated column =>
          baseOps.add accumulated
            (baseOps.mul (RowPadding.padRows matrix vertex column)
              (assignment column))) =
          (fun accumulated column =>
            baseOps.add accumulated
              (baseOps.mul 0 (assignment column))) := by
      funext accumulated column
      rw [RowPadding.padRows_atPadding matrix vertex column padding]
    rw [stepsEqual]
  rw [matrixZero]
  exact matrixVectorAt_zeroMatrix assignment vertex

/-- At one declared source row, the compiled thirteen matrix images are
exactly the selective product point. -/
theorem matrixImagesAt_sourceRow
    {columns : Nat}
    (one : Fin columns)
    (program : List (SourceRow columns))
    (profile : RelationProfile.Profile program.length columns)
    (assignment : Assignment F columns)
    (constantOne : assignment one = 1)
    (row : Fin program.length) :
    matrixImagesAt baseOps
        (paperSystem (relation one program) profile)
        assignment
        (RowPadding.numericRowVertex profile.rows_covered row) =
      Rows.productPoint 1
        ((program.get row).a.eval assignment)
        ((program.get row).b.eval assignment)
        ((program.get row).c.eval assignment) := by
  funext port
  let role := Role.ofIndex port
  have portEq : role.index = port := Role.index_ofIndex port
  change
    matrixVectorAt baseOps
        ((paperSystem (relation one program) profile).matrices port)
        assignment
        (RowPadding.numericRowVertex profile.rows_covered row) =
      Rows.productPoint 1
        ((program.get row).a.eval assignment)
        ((program.get row).b.eval assignment)
        ((program.get row).c.eval assignment) port
  rw [← portEq]
  simp only [paperSystem, relation,
    RelationProfile.FiniteRelation.matrixAt_role]
  rw [matrixVectorAt_padRows_numeric]
  cases role with
  | bit =>
      change matrixVectorAt baseOps
          (fun _ => roleRow one (program.get row) .bit) assignment
          (RowPadding.numericRowVertex profile.rows_covered row) = 0
      exact matrixVectorAt_roleRow_unused one (program.get row) assignment
        (RowPadding.numericRowVertex profile.rows_covered row) .bit (by decide)
  | generalSelector =>
      change matrixVectorAt baseOps
          (fun _ => roleRow one (program.get row) .generalSelector) assignment
          (RowPadding.numericRowVertex profile.rows_covered row) = 1
      exact matrixVectorAt_roleRow_generalSelector one (program.get row)
        assignment constantOne
        (RowPadding.numericRowVertex profile.rows_covered row)
  | a =>
      exact (matrixVectorAt_roleRow_source one (program.get row) assignment
        (RowPadding.numericRowVertex profile.rows_covered row)).1
  | b =>
      exact (matrixVectorAt_roleRow_source one (program.get row) assignment
        (RowPadding.numericRowVertex profile.rows_covered row)).2.1
  | c =>
      exact (matrixVectorAt_roleRow_source one (program.get row) assignment
        (RowPadding.numericRowVertex profile.rows_covered row)).2.2
  | sboxInput =>
      simpa [Rows.productPoint, Rows.sparsePoint, Role.index] using
        (matrixVectorAt_roleRow_unused one (program.get row) assignment
          (RowPadding.numericRowVertex profile.rows_covered row) .sboxInput
          (by decide))
  | centeredUnit =>
      simpa [Rows.productPoint, Rows.sparsePoint, Role.index] using
        (matrixVectorAt_roleRow_unused one (program.get row) assignment
          (RowPadding.numericRowVertex profile.rows_covered row) .centeredUnit
          (by decide))
  | evalSelector =>
      simpa [Rows.productPoint, Rows.sparsePoint, Role.index] using
        (matrixVectorAt_roleRow_unused one (program.get row) assignment
          (RowPadding.numericRowVertex profile.rows_covered row) .evalSelector
          (by decide))
  | canonicalDigit =>
      simpa [Rows.productPoint, Rows.sparsePoint, Role.index] using
        (matrixVectorAt_roleRow_unused one (program.get row) assignment
          (RowPadding.numericRowVertex profile.rows_covered row)
          .canonicalDigit (by decide))
  | canonicalBorrow =>
      simpa [Rows.productPoint, Rows.sparsePoint, Role.index] using
        (matrixVectorAt_roleRow_unused one (program.get row) assignment
          (RowPadding.numericRowVertex profile.rows_covered row)
          .canonicalBorrow (by decide))
  | canonicalNextBorrow =>
      simpa [Rows.productPoint, Rows.sparsePoint, Role.index] using
        (matrixVectorAt_roleRow_unused one (program.get row) assignment
          (RowPadding.numericRowVertex profile.rows_covered row)
          .canonicalNextBorrow (by decide))
  | canonicalBoundDigit =>
      simpa [Rows.productPoint, Rows.sparsePoint, Role.index] using
        (matrixVectorAt_roleRow_unused one (program.get row) assignment
          (RowPadding.numericRowVertex profile.rows_covered row)
          .canonicalBoundDigit (by decide))
  | evalTailRight =>
      simpa [Rows.productPoint, Rows.sparsePoint, Role.index] using
        (matrixVectorAt_roleRow_unused one (program.get row) assignment
          (RowPadding.numericRowVertex profile.rows_covered row) .evalTailRight
          (by decide))

theorem residualAt_sourceRow
    {columns : Nat}
    (one : Fin columns)
    (program : List (SourceRow columns))
    (profile : RelationProfile.Profile program.length columns)
    (assignment : Assignment F columns)
    (constantOne : assignment one = 1)
    (row : Fin program.length) :
    residualAt baseOps
        (paperSystem (relation one program) profile)
        assignment
        (RowPadding.numericRowVertex profile.rows_covered row) =
      Polynomial.Components.productResidual
        (Rows.productPoint (1 : F)
          ((program.get row).a.eval assignment)
          ((program.get row).b.eval assignment)
          ((program.get row).c.eval assignment)) := by
  unfold residualAt
  rw [matrixImagesAt_sourceRow one program profile assignment constantOne row]
  exact Rows.evaluate_productPoint _ _ _ _

theorem residualAt_sourceRow_eq_zero_iff
    {columns : Nat}
    (one : Fin columns)
    (program : List (SourceRow columns))
    (profile : RelationProfile.Profile program.length columns)
    (assignment : Assignment F columns)
    (constantOne : assignment one = 1)
    (row : Fin program.length) :
    residualAt baseOps
        (paperSystem (relation one program) profile)
        assignment
        (RowPadding.numericRowVertex profile.rows_covered row) = 0 ↔
      (program.get row).Holds assignment := by
  rw [residualAt_sourceRow one program profile assignment constantOne row]
  have expanded :
      Polynomial.Components.productResidual
          (Rows.productPoint (1 : F)
            ((program.get row).a.eval assignment)
            ((program.get row).b.eval assignment)
            ((program.get row).c.eval assignment)) =
        (program.get row).a.eval assignment *
            (program.get row).b.eval assignment +
          -((program.get row).c.eval assignment) := by
    simp [Polynomial.Components.productResidual, Rows.productPoint,
      Rows.sparsePoint, Role.index, Fin.one_mul]
  rw [expanded]
  simpa only [Fin.sub_eq_add_neg] using
    (Lean.Grind.AddCommGroup.sub_eq_zero_iff :
      (program.get row).a.eval assignment *
            (program.get row).b.eval assignment -
          (program.get row).c.eval assignment = 0 ↔
        (program.get row).a.eval assignment *
            (program.get row).b.eval assignment =
          (program.get row).c.eval assignment)

theorem matrixImagesAt_padding
    {columns : Nat}
    (one : Fin columns)
    (program : List (SourceRow columns))
    (profile : RelationProfile.Profile program.length columns)
    (assignment : Assignment F columns)
    (vertex : BooleanVertex profile.rowVariables)
    (padding : program.length ≤ rowIndex vertex) :
    matrixImagesAt baseOps
        (paperSystem (relation one program) profile)
        assignment vertex =
      fun _ => 0 := by
  funext port
  unfold matrixImagesAt
  exact matrixVectorAt_padRows_padding
    ((relation one program).matrixAt port) assignment vertex padding

theorem residualAt_padding
    {columns : Nat}
    (one : Fin columns)
    (program : List (SourceRow columns))
    (profile : RelationProfile.Profile program.length columns)
    (assignment : Assignment F columns)
    (vertex : BooleanVertex profile.rowVariables)
    (padding : program.length ≤ rowIndex vertex) :
    residualAt baseOps
        (paperSystem (relation one program) profile)
        assignment vertex =
      0 := by
  unfold residualAt
  rw [matrixImagesAt_padding one program profile assignment vertex padding]
  change Polynomial.Semantics.evaluate (fun _ => 0) = 0
  have productZero := Rows.evaluate_productPoint 0 0 0 0
  simpa [Rows.productPoint, Rows.sparsePoint, Role.index,
    Polynomial.Components.productResidual] using productZero

/-- Exact source-program/compiled-CCS correspondence. This theorem gives
soundness and honest completeness as one equivalence because the compiled
assignment is the same finite assignment used by the source equations. -/
theorem constraintSatisfied_iff
    {columns : Nat}
    (one : Fin columns)
    (program : List (SourceRow columns))
    (profile : RelationProfile.Profile program.length columns)
    (assignment : Assignment F columns)
    (constantOne : assignment one = 1) :
    ConstraintSatisfied baseOps
        (paperSystem (relation one program) profile)
        assignment ↔
      ∀ row : Fin program.length,
        (program.get row).Holds assignment := by
  constructor
  · intro satisfied row
    exact (residualAt_sourceRow_eq_zero_iff one program profile assignment
      constantOne row).mp
      (satisfied (RowPadding.numericRowVertex profile.rows_covered row))
  · intro sourceHolds vertex
    by_cases live : rowIndex vertex < program.length
    · let row : Fin program.length := ⟨rowIndex vertex, live⟩
      have vertexEq :
          RowPadding.numericRowVertex profile.rows_covered row = vertex := by
        unfold RowPadding.numericRowVertex
        simpa [row] using rowVertex_rowIndex vertex
      rw [← vertexEq]
      exact (residualAt_sourceRow_eq_zero_iff one program profile assignment
        constantOne row).mpr (sourceHolds row)
    · exact residualAt_padding one program profile assignment vertex
        (Nat.le_of_not_gt live)

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.DirectRows
