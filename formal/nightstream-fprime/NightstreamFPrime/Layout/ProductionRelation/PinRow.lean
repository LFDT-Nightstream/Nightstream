import NightstreamFPrime.Layout.ProductionRelation
import NightstreamFPrime.Spec.ProductionRelation.RowSemantics

/-!
Owns the selective zero-pin row used for direct linear-output rewrites. The
row places only the general selector and one value form in the `C` port.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PinRow

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

structure Forms (logicalWidth : Nat) where
  selector : SparseForm logicalWidth
  value : SparseForm logicalWidth
deriving Repr, DecidableEq

namespace Forms

def meaningfulForm {logicalWidth : Nat} (forms : Forms logicalWidth)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    SparseForm logicalWidth :=
  match port.val with
  | 1 => forms.selector
  | 4 => forms.value
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

private theorem portImages_eq_pin {logicalWidth : Nat}
    (forms : Forms logicalWidth) (assignment : Assignment F logicalWidth) :
    forms.portImages assignment =
      (Spec.ProductionRelation.RowSemantics.pin
        (forms.selector.eval assignment)
        (forms.value.eval assignment)).get := by
  funext port
  fin_cases port <;>
    simp [portImages, portForm, meaningfulForm,
      ProductionRelation.meaningfulPort?,
      Spec.ProductionRelation.RowSemantics.pin,
      Spec.ProductionRelation.RowSemantics.multiplication,
      Spec.ProductionRelation.RowSemantics.general,
      Spec.ProductionRelation.RowSemantics.PortValues.get]

def residual {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) : F :=
  evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
    (forms.portImages assignment)

theorem residual_eq {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) :
    forms.residual assignment =
      -(forms.selector.eval assignment * forms.value.eval assignment) := by
  unfold residual
  rw [portImages_eq_pin]
  exact Spec.ProductionRelation.RowSemantics.evaluate_pin _ _

def Preserves {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) (value : F) : Prop :=
  forms.selector.eval assignment = 1 ∧ forms.value.eval assignment = value

/-- A preserving pin row vanishes exactly when its value is zero. -/
theorem residual_zero_iff {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) (value : F)
    (preserves : forms.Preserves assignment value) :
    forms.residual assignment = 0 ↔ value = 0 := by
  rcases preserves with ⟨selector, valueEqual⟩
  rw [residual_eq, selector, valueEqual, one_mul]
  constructor
  · intro equal
    have := congrArg Neg.neg equal
    simpa using this
  · rintro rfl
    exact Lean.Grind.AddCommGroup.neg_zero

end Forms

end NightstreamFPrime.Layout.ProductionRelation.PinRow
