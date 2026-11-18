import NightstreamFPrime.Circuit.Segmented
import NightstreamFPrime.Gadgets.Poseidon2.Permutation

/-! The mark of a compiled permutation (SPEC.md section 4.4). A gadget that places the permutation
of `state` at `start` marks it with `markOf start state`, and the permutation's rows are the
mark's rows `markWindow` (`markWindow_markOf`). The exporter checks these rows against the
canonical template with the mark's lanes and locals (`Export.Generic.PermutationWindow`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Gadgets.Poseidon2.Permutation

open NightstreamFPrime.Circuit

/-- The input state of a mark's lanes. -/
def laneState (lanes : List Expr) : EState := fun lane => lanes.getD lane.val 0

/-- The rows of a marked permutation: the recipe rows of the permutation of its lanes from its
first local. -/
def markWindow (mark : PermutationMark) : List Expr :=
  recipeConstraints mark.start (compile mark.start (laneState mark.lanes) schedule).recipes

/-- The mark of the permutation of `state` at `start`. -/
def markOf (start : Nat) (state : EState) : PermutationMark := ⟨start, List.ofFn state⟩

theorem laneState_ofFn (state : EState) : laneState (List.ofFn state) = state := by
  funext lane
  rw [laneState, List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]

theorem markWindow_markOf (start : Nat) (state : EState) :
    markWindow (markOf start state) = recipeConstraints start (compile start state schedule).recipes := by
  simp only [markWindow, markOf, laneState_ofFn]

@[simp] theorem markOf_start (start : Nat) (state : EState) : (markOf start state).start = start :=
  rfl

@[simp] theorem markOf_lanes (start : Nat) (state : EState) :
    (markOf start state).lanes = List.ofFn state :=
  rfl

/-- The lanes of a mark: 16 lanes, affine when the state is, and reading only the columns that
the state reads. -/
theorem markOf_lanes_facts {live : Nat → Prop} (start : Nat) (state : EState)
    (affine : ∀ lane, (state lane).Affine)
    (reads : ∀ lane, (state lane).VarsSatisfy fun column => column < start ∧ live column) :
    (markOf start state).Lanes live where
  length := by simp
  affine := by
    intro lane member
    obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
    exact affine index
  reads := by
    intro lane member
    obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
    exact reads index

end NightstreamFPrime.Gadgets.Poseidon2.Permutation
