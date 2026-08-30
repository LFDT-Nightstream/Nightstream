import NightstreamFPrime.Gadgets.Poseidon2.Layer
import NightstreamFPrime.Layout.ProductionRelation

/-!
Owns the Poseidon2 linear-layer formulas over final sparse matrix forms. The
evaluation theorems connect these forms to the existing field-level
`Gadgets.Poseidon2.Layer` authority lane by lane.

This module contains no S-box relation and allocates no assignment columns.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.SparseLayer

open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev State (logicalWidth : Nat) := Fin 8 → SparseForm logicalWidth

private theorem poseidonOfNat_two :
    Spec.Poseidon2.ofNat 2 = (2 : F) := by
  apply Fin.ext
  rfl

private theorem poseidonOfNat_three :
    Spec.Poseidon2.ofNat 3 = (3 : F) := by
  apply Fin.ext
  rfl

def evalState {logicalWidth : Nat} (assignment : Assignment F logicalWidth)
    (state : State logicalWidth) : Layer.FState :=
  fun lane => (state lane).eval assignment

def get {logicalWidth : Nat} (state : State logicalWidth) (index : Nat) :
    SparseForm logicalWidth :=
  if bounded : index < 8 then state ⟨index, bounded⟩ else .empty

def add {logicalWidth : Nat} :
    SparseForm logicalWidth → SparseForm logicalWidth →
      SparseForm logicalWidth :=
  SparseForm.add

def scale {logicalWidth : Nat} (value : Nat) :
    SparseForm logicalWidth → SparseForm logicalWidth :=
  SparseForm.scale (Spec.Poseidon2.ofNat value)

def constant {logicalWidth : Nat} (oneColumn : Fin logicalWidth)
    (value : F) : SparseForm logicalWidth :=
  SparseForm.singleton oneColumn value

def addConstant {logicalWidth : Nat} (oneColumn : Fin logicalWidth)
    (form : SparseForm logicalWidth) (value : F) : SparseForm logicalWidth :=
  add form (constant oneColumn value)

def mat4 {logicalWidth : Nat} (state : State logicalWidth)
    (base lane : Nat) : SparseForm logicalWidth :=
  match lane with
  | 0 => add (add (add (scale 2 (get state base))
      (scale 3 (get state (base + 1)))) (get state (base + 2)))
      (get state (base + 3))
  | 1 => add (add (add (get state base)
      (scale 2 (get state (base + 1))))
      (scale 3 (get state (base + 2)))) (get state (base + 3))
  | 2 => add (add (add (get state base) (get state (base + 1)))
      (scale 2 (get state (base + 2))))
      (scale 3 (get state (base + 3)))
  | _ => add (add (add (scale 3 (get state base))
      (get state (base + 1))) (get state (base + 2)))
      (scale 2 (get state (base + 3)))

def block {logicalWidth : Nat} (state : State logicalWidth) (index : Nat) :
    SparseForm logicalWidth :=
  mat4 state (if index < 4 then 0 else 4) (index % 4)

def external {logicalWidth : Nat} (state : State logicalWidth) :
    State logicalWidth :=
  fun lane => add (add (block state lane.val) (block state (lane.val % 4)))
    (block state (lane.val % 4 + 4))

def sum {logicalWidth : Nat} (state : State logicalWidth) :
    SparseForm logicalWidth :=
  add (add (add (add (add (add (add (get state 0) (get state 1))
    (get state 2)) (get state 3)) (get state 4)) (get state 5))
    (get state 6)) (get state 7)

def internal {logicalWidth : Nat} (state : State logicalWidth) :
    State logicalWidth :=
  fun lane => add
    (SparseForm.scale (Spec.Poseidon2.ofNat
      (Spec.Poseidon2.internalDiagonal.getD lane.val 0)) (state lane))
    (sum state)

@[simp] theorem eval_get {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) (state : State logicalWidth)
    (index : Nat) :
    (get state index).eval assignment =
      Layer.getF (evalState assignment state) index := by
  unfold get Layer.getF evalState
  split <;> simp

@[simp] theorem eval_constant {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) (oneColumn : Fin logicalWidth)
    (one : assignment oneColumn = 1) (value : F) :
    (constant oneColumn value).eval assignment = value := by
  simp [constant, one]

@[simp] theorem eval_addConstant {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) (oneColumn : Fin logicalWidth)
    (one : assignment oneColumn = 1) (form : SparseForm logicalWidth)
    (value : F) :
    (addConstant oneColumn form value).eval assignment =
      form.eval assignment + value := by
  simp [addConstant, add, eval_constant assignment oneColumn one]

@[simp] theorem eval_mat4 {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) (state : State logicalWidth)
    (base lane : Nat) :
    (mat4 state base lane).eval assignment =
      Layer.mat4F (evalState assignment state) base lane := by
  rcases lane with _ | _ | _ | lane <;>
    simp [mat4, Layer.mat4F, add, scale, poseidonOfNat_two,
      poseidonOfNat_three]

@[simp] theorem eval_external {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) (state : State logicalWidth)
    (lane : Fin 8) :
    (external state lane).eval assignment =
      Layer.externalF (evalState assignment state) lane := by
  simp [external, Layer.externalF, block, Layer.blockF, add]

@[simp] theorem eval_sum {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) (state : State logicalWidth) :
    (sum state).eval assignment =
      Layer.sumF (evalState assignment state) := by
  simp [sum, Layer.sumF, add]

@[simp] theorem eval_internal {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) (state : State logicalWidth)
    (lane : Fin 8) :
    (internal state lane).eval assignment =
      Layer.internalF (evalState assignment state) lane := by
  simp [internal, Layer.internalF, evalState, add]

end NightstreamFPrime.Layout.ProductionRelation.SparseLayer
