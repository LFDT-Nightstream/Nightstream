import NightstreamFPrime.Layout.ProductionRelation
import NightstreamFPrime.Spec.ProductionRelation.RowSemantics

/-!
Owns one selective row for a sum of five products. The row uses the exact
evaluation-selector port family of the fixed 74-term production polynomial.

This module does not group products or select assignment columns.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.ProductSumRow

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- The twelve live sparse forms of one five-product row. -/
structure Forms (logicalWidth : Nat) where
  selector : SparseForm logicalWidth
  left : Fin 5 → SparseForm logicalWidth
  right : Fin 5 → SparseForm logicalWidth
  output : SparseForm logicalWidth

namespace Forms

def meaningfulForm {logicalWidth : Nat} (forms : Forms logicalWidth)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    SparseForm logicalWidth :=
  match port.val with
  | 0 => forms.left 0
  | 1 => .empty
  | 2 => forms.right 0
  | 3 => forms.left 1
  | 4 => forms.output
  | 5 => forms.right 1
  | 6 => forms.left 2
  | 7 => forms.selector
  | 8 => forms.right 2
  | 9 => forms.left 3
  | 10 => forms.right 3
  | 11 => forms.left 4
  | 12 => forms.right 4
  | _ => .empty

def portForm {logicalWidth : Nat} (forms : Forms logicalWidth)
    (port : Fin Spec.ProductionRelation.matrixCount) : SparseForm logicalWidth :=
  match ProductionRelation.meaningfulPort? port with
  | some meaningful => forms.meaningfulForm meaningful
  | none => .empty

def portImages {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) :
    Fin Spec.ProductionRelation.matrixCount → F :=
  fun port => (forms.portForm port).eval assignment

private theorem portImages_eq_productSum {logicalWidth : Nat}
    (forms : Forms logicalWidth) (assignment : Assignment F logicalWidth) :
    forms.portImages assignment =
      (Spec.ProductionRelation.RowSemantics.productSum
        (forms.selector.eval assignment)
        (fun lane => (forms.left lane).eval assignment)
        (fun lane => (forms.right lane).eval assignment)
        (forms.output.eval assignment)).get := by
  funext port
  fin_cases port <;>
    simp [portImages, portForm, meaningfulForm,
      ProductionRelation.meaningfulPort?,
      Spec.ProductionRelation.RowSemantics.productSum,
      Spec.ProductionRelation.RowSemantics.PortValues.get]

def residual {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) : F :=
  evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
    (forms.portImages assignment)

theorem residual_eq {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) :
    forms.residual assignment =
      forms.selector.eval assignment *
        (Spec.ProductionRelation.RowSemantics.productTotal
            (fun lane => (forms.left lane).eval assignment)
            (fun lane => (forms.right lane).eval assignment) -
          forms.output.eval assignment) := by
  unfold residual
  rw [portImages_eq_productSum]
  exact Spec.ProductionRelation.RowSemantics.evaluate_productSum _ _ _ _

/-- Exact source-value preservation premise for one five-product row. -/
def Preserves {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth)
    (left right : Fin 5 → F) (output : F) : Prop :=
  forms.selector.eval assignment = 1 ∧
    (∀ lane, (forms.left lane).eval assignment = left lane) ∧
    (∀ lane, (forms.right lane).eval assignment = right lane) ∧
    forms.output.eval assignment = output

/-- A preserving row vanishes exactly for the five-product equation. -/
theorem residual_zero_iff {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth)
    (left right : Fin 5 → F) (output : F)
    (preserves : forms.Preserves assignment left right output) :
    forms.residual assignment = 0 ↔
      Spec.ProductionRelation.RowSemantics.productTotal left right = output := by
  rcases preserves with ⟨selector, leftEqual, rightEqual, outputEqual⟩
  rw [residual_eq, selector, one_mul, outputEqual]
  have leftFunctions :
      (fun lane => (forms.left lane).eval assignment) = left := by
    funext lane
    exact leftEqual lane
  have rightFunctions :
      (fun lane => (forms.right lane).eval assignment) = right := by
    funext lane
    exact rightEqual lane
  rw [leftFunctions, rightFunctions]
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff

end Forms

end NightstreamFPrime.Layout.ProductionRelation.ProductSumRow
