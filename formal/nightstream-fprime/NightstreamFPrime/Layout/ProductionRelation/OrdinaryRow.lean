import NightstreamFPrime.Layout.ProductionRelation
import NightstreamFPrime.Layout.R1CS
import NightstreamFPrime.Spec.ProductionRelation.RowSemantics

/-!
Owns the ordinary-row branch of the production selective compiler. One source
R1CS equation supplies only the selector, `A`, `B`, and `C` matrix images.
The fixed 74-term polynomial then checks exactly the source equation.

Programs are indexed functions, not artifact-sized lists. This module does
not choose retained source slots or construct their low-norm substitutions.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.OrdinaryRow

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Lifecycle

/-- The four live sparse forms of one ordinary source equation. -/
structure Forms (logicalWidth : Nat) where
  selector : SparseForm logicalWidth
  a : SparseForm logicalWidth
  b : SparseForm logicalWidth
  c : SparseForm logicalWidth
deriving Repr, DecidableEq

namespace Forms

/-- Exact meaningful-port placement for an ordinary multiplication row. -/
def meaningfulForm {logicalWidth : Nat} (forms : Forms logicalWidth)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    SparseForm logicalWidth :=
  match port.val with
  | 1 => forms.selector
  | 2 => forms.a
  | 3 => forms.b
  | 4 => forms.c
  | _ => .empty

/-- Complete 14-port view. Slot 13 is zero through `meaningfulPort?`. -/
def portForm {logicalWidth : Nat} (forms : Forms logicalWidth)
    (port : Fin Spec.ProductionRelation.matrixCount) : SparseForm logicalWidth :=
  match ProductionRelation.meaningfulPort? port with
  | some meaningful => forms.meaningfulForm meaningful
  | none => .empty

def portImages {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) :
    Fin Spec.ProductionRelation.matrixCount → F :=
  fun port => (forms.portForm port).eval assignment

private theorem portImages_eq_multiplication {logicalWidth : Nat}
    (forms : Forms logicalWidth) (assignment : Assignment F logicalWidth) :
    forms.portImages assignment =
      (Spec.ProductionRelation.RowSemantics.multiplication
        (forms.selector.eval assignment)
        (forms.a.eval assignment)
        (forms.b.eval assignment)
        (forms.c.eval assignment)).get := by
  funext port
  fin_cases port <;>
    simp [portImages, portForm, meaningfulForm,
      ProductionRelation.meaningfulPort?,
      Spec.ProductionRelation.RowSemantics.multiplication,
      Spec.ProductionRelation.RowSemantics.general,
      Spec.ProductionRelation.RowSemantics.PortValues.get]

/-- Residual of the sole production polynomial on this compiled row. -/
def residual {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) : F :=
  evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
    (forms.portImages assignment)

/-- The complete 74-term polynomial reduces to the selected source residual. -/
theorem residual_eq {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) :
    forms.residual assignment =
      forms.selector.eval assignment *
        (forms.a.eval assignment * forms.b.eval assignment -
          forms.c.eval assignment) := by
  unfold residual
  rw [portImages_eq_multiplication]
  exact Spec.ProductionRelation.RowSemantics.evaluate_multiplication _ _ _ _

/-- The compiled forms reconstruct one source row under these two assignments. -/
def Preserves {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (row : R1CS.Row) : Prop :=
  forms.selector.eval assignment = 1 ∧
    forms.a.eval assignment = row.a.eval source ∧
    forms.b.eval assignment = row.b.eval source ∧
    forms.c.eval assignment = row.c.eval source

/-- A preserving ordinary-row compilation accepts exactly when its source
R1CS equation holds. -/
theorem residual_zero_iff {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (row : R1CS.Row) (preserves : forms.Preserves assignment source row) :
    forms.residual assignment = 0 ↔ row.Holds source := by
  rcases preserves with ⟨selector, a, b, c⟩
  rw [residual_eq, selector, a, b, c, one_mul]
  change
    row.a.eval source * row.b.eval source - row.c.eval source = 0 ↔
      row.a.eval source * row.b.eval source = row.c.eval source
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff

end Forms

/-- A canonical plan built directly from row-local ordinary forms. -/
def planOfForms {logicalWidth rowCount : Nat}
    (rowCount_le : rowCount ≤ 2 ^ cubeVariables)
    (forms : Fin rowCount → Forms logicalWidth) :
    ProductionRelation.Plan logicalWidth where
  rowCount := rowCount
  rowCount_le := rowCount_le
  forms := fun row port => (forms row).meaningfulForm port

private theorem planOfForms_portForm {logicalWidth rowCount : Nat}
    (rowCount_le : rowCount ≤ 2 ^ cubeVariables)
    (forms : Fin rowCount → Forms logicalWidth) (row : Fin rowCount)
    (port : Fin Spec.ProductionRelation.matrixCount) :
    (planOfForms rowCount_le forms).portForm row port =
      (forms row).portForm port := by
  rfl

/-- One direct ordinary-form row accepts exactly its preserved source row. -/
theorem planOfForms_residual_zero_iff {logicalWidth rowCount : Nat}
    (rowCount_le : rowCount ≤ 2 ^ cubeVariables)
    (forms : Fin rowCount → Forms logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env)
    (row : Fin rowCount) (sourceRow : R1CS.Row)
    (preserves : (forms row).Preserves assignment source sourceRow) :
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
        ((planOfForms rowCount_le forms).rowImage assignment
          ((planOfForms rowCount_le forms).rowLayout.toVertex row)) = 0 ↔
      sourceRow.Holds source := by
  have images :
      (planOfForms rowCount_le forms).rowImage assignment
          ((planOfForms rowCount_le forms).rowLayout.toVertex row) =
        (forms row).portImages assignment := by
    funext port
    unfold ProductionRelation.Plan.rowImage Forms.portImages
    rw [(planOfForms rowCount_le forms).rowLayout.toColumn_toVertex]
    exact congrArg (fun form => form.eval assignment)
      (planOfForms_portForm rowCount_le forms row port)
  rw [images]
  exact Forms.residual_zero_iff _ _ _ _ preserves

/-- One source equation together with its final sparse matrix forms. -/
structure Row (logicalWidth : Nat) where
  source : R1CS.Row
  forms : Forms logicalWidth

/-- Generative ordinary-row program. The row bound is carried once and does
not require construction of a full row list in the kernel. -/
structure Program (logicalWidth : Nat) where
  rowCount : Nat
  rowCount_le : rowCount ≤ 2 ^ cubeVariables
  row : Fin rowCount → Row logicalWidth

namespace Program

def toPlan {logicalWidth : Nat} (program : Program logicalWidth) :
    ProductionRelation.Plan logicalWidth where
  rowCount := program.rowCount
  rowCount_le := program.rowCount_le
  forms := fun row port => (program.row row).forms.meaningfulForm port

private theorem toPlan_portForm {logicalWidth : Nat}
    (program : Program logicalWidth) (row : Fin program.rowCount)
    (port : Fin Spec.ProductionRelation.matrixCount) :
    program.toPlan.portForm row port = (program.row row).forms.portForm port := by
  rfl

private theorem rowImage_at {logicalWidth : Nat}
    (program : Program logicalWidth) (assignment : Assignment F logicalWidth)
    (row : Fin program.rowCount)
    (port : Fin Spec.ProductionRelation.matrixCount) :
    program.toPlan.rowImage assignment
        (program.toPlan.rowLayout.toVertex row) port =
      (program.row row).forms.portImages assignment port := by
  unfold ProductionRelation.Plan.rowImage Forms.portImages
  rw [program.toPlan.rowLayout.toColumn_toVertex]
  exact congrArg (fun form => form.eval assignment)
    (toPlan_portForm program row port)

/-- Every row of a program reconstructs its named source equation. -/
def Preserves {logicalWidth : Nat} (program : Program logicalWidth)
    (assignment : Assignment F logicalWidth) (source : Circuit.Env) : Prop :=
  ∀ row,
    (program.row row).forms.Preserves assignment source
      (program.row row).source

/-- A selected live row needs only its own preservation proof. -/
theorem residualAt_live_zero_iff_row {logicalWidth : Nat}
    (program : Program logicalWidth) (assignment : Assignment F logicalWidth)
    (source : Circuit.Env) (row : Fin program.rowCount)
    (preserves : (program.row row).forms.Preserves assignment source
      (program.row row).source) :
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
        (program.toPlan.rowImage assignment
          (program.toPlan.rowLayout.toVertex row)) = 0 ↔
      (program.row row).source.Holds source := by
  have images :
      program.toPlan.rowImage assignment
          (program.toPlan.rowLayout.toVertex row) =
        (program.row row).forms.portImages assignment := by
    funext port
    exact rowImage_at program assignment row port
  rw [images]
  exact Forms.residual_zero_iff _ _ _ _ preserves

/-- At every live Boolean row, the exact plan residual vanishes exactly when
the corresponding source R1CS equation holds. -/
theorem residualAt_live_zero_iff {logicalWidth : Nat}
    (program : Program logicalWidth) (assignment : Assignment F logicalWidth)
    (source : Circuit.Env) (preserves : program.Preserves assignment source)
    (row : Fin program.rowCount) :
    evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
        (program.toPlan.rowImage assignment
          (program.toPlan.rowLayout.toVertex row)) = 0 ↔
      (program.row row).source.Holds source := by
  exact residualAt_live_zero_iff_row program assignment source row
    (preserves row)

end Program

end NightstreamFPrime.Layout.ProductionRelation.OrdinaryRow
