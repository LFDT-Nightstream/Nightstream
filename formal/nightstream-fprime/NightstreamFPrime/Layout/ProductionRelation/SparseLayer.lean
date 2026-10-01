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

abbrev State (logicalWidth : Nat) := Fin 16 → SparseForm logicalWidth

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
  if bounded : index < 16 then state ⟨index, bounded⟩ else .empty

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
  mat4 state (4 * (index / 4)) (index % 4)

/-- Left-nested sum of forms, mirroring the field-level `foldl`. -/
def sumForms {logicalWidth : Nat} (forms : List (SparseForm logicalWidth)) :
    SparseForm logicalWidth :=
  forms.foldl add .empty

def column {logicalWidth : Nat} (state : State logicalWidth) (index : Nat) :
    SparseForm logicalWidth :=
  sumForms ((List.range 4).map fun b => block state (4 * b + index % 4))

def external {logicalWidth : Nat} (state : State logicalWidth) :
    State logicalWidth :=
  fun lane => add (block state lane.val) (column state lane.val)

def sum {logicalWidth : Nat} (state : State logicalWidth) :
    SparseForm logicalWidth :=
  sumForms ((List.range 16).map (get state))

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

private theorem eval_foldl_add {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (forms : List (SparseForm logicalWidth)) (initial : SparseForm logicalWidth) :
    (forms.foldl add initial).eval assignment =
      (forms.map fun form => form.eval assignment).foldl (· + ·)
        (initial.eval assignment) := by
  induction forms generalizing initial with
  | nil => rfl
  | cons form forms inductionHypothesis =>
      simp only [List.foldl_cons, List.map_cons]
      rw [inductionHypothesis]
      simp [add]

theorem eval_sumForms {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth)
    (forms : List (SparseForm logicalWidth)) :
    (sumForms forms).eval assignment =
      (forms.map fun form => form.eval assignment).foldl (· + ·) 0 := by
  unfold sumForms
  rw [eval_foldl_add]
  simp

@[simp] theorem eval_block {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) (state : State logicalWidth)
    (index : Nat) :
    (block state index).eval assignment =
      Layer.blockF (evalState assignment state) index := by
  simp [block, Layer.blockF]

@[simp] theorem eval_external {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) (state : State logicalWidth)
    (lane : Fin 16) :
    (external state lane).eval assignment =
      Layer.externalF (evalState assignment state) lane := by
  simp only [external, Layer.externalF, column, Layer.columnF, add,
    SparseForm.add_eval, eval_block, eval_sumForms, List.map_map]
  congr 2
  apply List.map_congr_left
  intro block _
  exact eval_block assignment state _

@[simp] theorem eval_sum {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) (state : State logicalWidth) :
    (sum state).eval assignment =
      Layer.sumF (evalState assignment state) := by
  simp only [sum, Layer.sumF, eval_sumForms, List.map_map]
  congr 1

@[simp] theorem eval_internal {logicalWidth : Nat}
    (assignment : Assignment F logicalWidth) (state : State logicalWidth)
    (lane : Fin 16) :
    (internal state lane).eval assignment =
      Layer.internalF (evalState assignment state) lane := by
  simp [internal, Layer.internalF, evalState, add]

end NightstreamFPrime.Layout.ProductionRelation.SparseLayer
