import NightstreamFPrime.Layout.ProductionRelation
import NightstreamFPrime.Spec.ProductionRelation.RowSemantics

/-!
Owns the selective S-box row used by the fixed Poseidon2 trace compiler. One
row places only the general selector, seventh-power input, and output forms.

This module does not select Poseidon2 trace positions or source columns.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.SboxRow

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.CCSResidualTable
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- The three live sparse forms of one seventh-power row. -/
structure Forms (logicalWidth : Nat) where
  selector : SparseForm logicalWidth
  input : SparseForm logicalWidth
  output : SparseForm logicalWidth
deriving Repr, DecidableEq

namespace Forms

def meaningfulForm {logicalWidth : Nat} (forms : Forms logicalWidth)
    (port : Fin Spec.ProductionRelation.meaningfulPortCount) :
    SparseForm logicalWidth :=
  match port.val with
  | 1 => forms.selector
  | 4 => forms.output
  | 5 => forms.input
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

private theorem portImages_eq_sbox {logicalWidth : Nat}
    (forms : Forms logicalWidth) (assignment : Assignment F logicalWidth) :
    forms.portImages assignment =
      (Spec.ProductionRelation.RowSemantics.sbox
        (forms.selector.eval assignment)
        (forms.input.eval assignment)
        (forms.output.eval assignment)).get := by
  funext port
  fin_cases port <;>
    simp [portImages, portForm, meaningfulForm,
      ProductionRelation.meaningfulPort?,
      Spec.ProductionRelation.RowSemantics.sbox,
      Spec.ProductionRelation.RowSemantics.general,
      Spec.ProductionRelation.RowSemantics.PortValues.get]

def residual {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) : F :=
  evaluatePolynomial baseOps Spec.ProductionRelation.polynomial
    (forms.portImages assignment)

/-- The complete 74-term polynomial reduces to the selected S-box residual. -/
theorem residual_eq {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) :
    forms.residual assignment =
      forms.selector.eval assignment *
        (Spec.ProductionRelation.RowSemantics.seventhPower
            (forms.input.eval assignment) -
          forms.output.eval assignment) := by
  unfold residual
  rw [portImages_eq_sbox]
  exact Spec.ProductionRelation.RowSemantics.evaluate_sbox _ _ _

/-- Exact source-value preservation premise for one collapsed S-box trace. -/
def Preserves {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) (input output : F) : Prop :=
  forms.selector.eval assignment = 1 ∧
    forms.input.eval assignment = input ∧
    forms.output.eval assignment = output

/-- A preserving S-box row vanishes exactly for the seventh-power equation. -/
theorem residual_zero_iff {logicalWidth : Nat} (forms : Forms logicalWidth)
    (assignment : Assignment F logicalWidth) (input output : F)
    (preserves : forms.Preserves assignment input output) :
    forms.residual assignment = 0 ↔
      Spec.ProductionRelation.RowSemantics.seventhPower input = output := by
  rcases preserves with ⟨selector, inputEqual, outputEqual⟩
  rw [residual_eq, selector, inputEqual, outputEqual, one_mul]
  exact Lean.Grind.AddCommGroup.sub_eq_zero_iff

end Forms

end NightstreamFPrime.Layout.ProductionRelation.SboxRow
