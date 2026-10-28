import NightstreamFPrime.Layout.ProductionAssignment
import NightstreamFPrime.Layout.ProductionRelation

/-!
Owns the sparse matrix form that reconstructs one retained low-norm source
slot. Bit and centered slots use one coordinate. General field slots use the
exact 41-coordinate balanced-ternary Horner form.

This module does not select which source values are retained.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.RetainedSlot

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix

def coordinateForms {sourceWidth : Nat}
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (slot : Fin slots.length) :
    List (SparseForm (ProductionAssignment.logicalWidth slots)) :=
  List.ofFn fun coordinate : Fin (slots.get slot).width =>
    SparseForm.singleton
      (ProductionAssignment.privateColumn
        (LowNormAssignment.coordinateIndex slots slot coordinate)) 1

def recomposeForms {columns : Nat} : List (SparseForm columns) →
    SparseForm columns
  | [] => .empty
  | head :: tail =>
      SparseForm.add head
        (SparseForm.scale (fieldOfNat 3) (recomposeForms tail))

theorem recomposeForms_eval {columns : Nat} (forms : List (SparseForm columns))
    (assignment : Assignment F columns) :
    (recomposeForms forms).eval assignment =
      BalancedTernary.recompose (forms.map fun form => form.eval assignment) := by
  induction forms with
  | nil => simp [recomposeForms, BalancedTernary.recompose]
  | cons head tail inductionHypothesis =>
      simp [recomposeForms, BalancedTernary.recompose, inductionHypothesis]

theorem coordinateForms_eval {sourceWidth : Nat}
    (publicInput : Fin ProductionAssignment.publicWidth → F)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F) (slot : Fin slots.length) :
    (coordinateForms slots slot).map (fun form =>
        form.eval
          (ProductionAssignment.logicalAssignment publicInput slots source)) =
      (slots.get slot).encode source := by
  calc
    (coordinateForms slots slot).map (fun form =>
        form.eval
          (ProductionAssignment.logicalAssignment publicInput slots source)) =
        List.ofFn
          (LowNormSlot.coordinate (slots.get slot).kind
            (source (slots.get slot).source)) := by
      rw [coordinateForms, List.map_ofFn]
      apply congrArg List.ofFn
      funext coordinate
      simp only [Function.comp_apply, SparseForm.singleton_eval, one_mul]
      change
        ProductionAssignment.logicalAssignment publicInput slots source
            (ProductionAssignment.privateColumn
              (LowNormAssignment.coordinateIndex slots slot coordinate)) =
          ((slots.get slot).encode source).getD coordinate.val 0
      exact ProductionAssignment.logicalAssignment_slotCoordinate
        publicInput slots source slot coordinate
    _ = (slots.get slot).encode source := by
      exact LowNormSlot.coordinateList_eq_encode
        (slots.get slot).kind (source (slots.get slot).source)

def form {sourceWidth : Nat}
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (slot : Fin slots.length) :
    SparseForm (ProductionAssignment.logicalWidth slots) :=
  recomposeForms (coordinateForms slots slot)

theorem form_eval {sourceWidth : Nat}
    (publicInput : Fin ProductionAssignment.publicWidth → F)
    (slots : List (LowNormAssignment.Slot sourceWidth))
    (source : Fin sourceWidth → F) (slot : Fin slots.length) :
    (form slots slot).eval
        (ProductionAssignment.logicalAssignment publicInput slots source) =
      source (slots.get slot).source := by
  rw [form, recomposeForms_eval, coordinateForms_eval]
  exact LowNormSlot.recompose_encode _ _

end NightstreamFPrime.Layout.ProductionRelation.RetainedSlot
