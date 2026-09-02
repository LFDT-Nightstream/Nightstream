import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Lifecycle.Stage1.RunningTransition

/-!
Paper authority: HyperNova Construction 2, step 5.
Obligation: the next state-hash preimage keeps `z0` and increments `i`.

Inputs:
- the prior and output iteration words;
- the four prior and output initial-state words.

Constraint groups:
- C1: `outputIteration = priorIteration + 1`;
- C2: `outputInitialState[index] = priorInitialState[index]`.

This zero-copy leaf allocates no variable. The application and running
transition own the other fields of the next preimage.
-/

namespace NightstreamFPrime.Lifecycle.Stage1.NextPreimage

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle.Stage1

structure Interface where
  priorIteration : Nat → Expr
  outputIteration : Nat → Expr
  priorInitialState : Nat → RunningTransition.StateIndex → Expr
  outputInitialState : Nat → RunningTransition.StateIndex → Expr

def iterationAssertion (interface : Interface) (offset : Nat) : Expr :=
  interface.outputIteration offset -
    (interface.priorIteration offset + Expr.const 1)

def initialStateAssertions (interface : Interface) (offset : Nat) : List Expr :=
  (List.finRange RunningTransition.stateWordCount).map fun index =>
    interface.outputInitialState offset index -
      interface.priorInitialState offset index

def assertions (interface : Interface) (offset : Nat) : List Expr :=
  iterationAssertion interface offset :: initialStateAssertions interface offset

structure SpecHolds (interface : Interface) (offset : Nat) (env : Env) : Prop where
  iteration : (interface.outputIteration offset).eval env =
    (interface.priorIteration offset).eval env + 1
  initialState : ∀ index,
    (interface.outputInitialState offset index).eval env =
      (interface.priorInitialState offset index).eval env

def opsAt (interface : Interface) (offset : Nat) : List Op :=
  (assertions interface offset).map Op.assertZero

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset, opsAt interface offset)

@[simp] theorem main_ops (interface : Interface) (offset : Nat) :
    Circuit.ops (main interface) offset = opsAt interface offset := by
  rfl

private theorem flatConstraints_assertions (expressions : List Expr) :
    flatConstraints (expressions.map Op.assertZero) = expressions := by
  induction expressions with
  | nil => rfl
  | cons expression rest inductionHypothesis =>
      change [expression] ++ flatConstraints (rest.map Op.assertZero) =
        expression :: rest
      rw [inductionHypothesis]
      rfl

@[simp] theorem flatConstraints_opsAt (interface : Interface) (offset : Nat) :
    flatConstraints (opsAt interface offset) = assertions interface offset := by
  exact flatConstraints_assertions _

private theorem assertion_holds_iff (left right : Expr) (env : Env) :
    (left - right).eval env = 0 ↔ left.eval env = right.eval env := by
  constructor
  · intro row
    exact sub_eq_zero.mp (by simpa using row)
  · intro equal
    simpa using sub_eq_zero.mpr equal

theorem soundness (interface : Interface) (env : Env) (offset : Nat)
    (rows : holds env (Circuit.ops (main interface) offset)) :
    SpecHolds interface offset env := by
  rw [main_ops] at rows
  have rowOfMember : ∀ expression ∈ assertions interface offset,
      expression.eval env = 0 := by
    intro expression member
    exact rows (Op.assertZero expression) (by
      rw [opsAt, List.mem_map]
      exact ⟨expression, member, rfl⟩)
  refine ⟨?_, ?_⟩
  · have row := rowOfMember (iterationAssertion interface offset) (by
      simp [assertions])
    exact (assertion_holds_iff _ _ env).mp row
  · intro index
    have row := rowOfMember
      (interface.outputInitialState offset index -
        interface.priorInitialState offset index) (by
          simp [assertions, initialStateAssertions])
    exact (assertion_holds_iff _ _ env).mp row

theorem completeness (interface : Interface) (env : Env) (offset : Nat)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (main interface) offset)) ∧
      holdsFlat completed (Circuit.ops (main interface) offset) := by
  refine ⟨env, ?_, ?_⟩
  · intro _ _
    rfl
  · rw [main_ops]
    change ConstraintsHold env (flatConstraints (opsAt interface offset))
    rw [flatConstraints_opsAt]
    intro expression member
    rw [assertions, List.mem_cons] at member
    rcases member with rfl | initialMember
    · apply (assertion_holds_iff _ _ env).mpr
      exact specification.iteration
    · rw [initialStateAssertions, List.mem_map] at initialMember
      rcases initialMember with ⟨index, _indexMember, rfl⟩
      apply (assertion_holds_iff _ _ env).mpr
      exact specification.initialState index

structure Assumptions (interface : Interface) (offset : Nat)
    (_env : Env) : Prop where
  priorIteration : (interface.priorIteration offset).VarsBelow offset
  outputIteration : (interface.outputIteration offset).VarsBelow offset
  priorInitialState : ∀ index,
    (interface.priorInitialState offset index).VarsBelow offset
  outputInitialState : ∀ index,
    (interface.outputInitialState offset index).VarsBelow offset

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (main interface) offset),
      expression.VarsBelow offset := by
  intro expression member
  rw [main_ops, flatConstraints_opsAt, assertions, List.mem_cons] at member
  rcases member with rfl | initialMember
  · exact Expr.VarsBelow.sub _ _ _ assumptions.outputIteration
      (Expr.VarsBelow.add _ _ _ assumptions.priorIteration trivial)
  · rw [initialStateAssertions, List.mem_map] at initialMember
    rcases initialMember with ⟨index, _indexMember, rfl⟩
    exact Expr.VarsBelow.sub _ _ _ (assumptions.outputInitialState index)
      (assumptions.priorInitialState index)

private theorem localLength_assertions (expressions : List Expr) :
    localLength (expressions.map Op.assertZero) = 0 := by
  induction expressions with
  | nil => rfl
  | cons _ rest inductionHypothesis =>
      change 0 + localLength (rest.map Op.assertZero) = 0
      simpa using inductionHypothesis

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (main interface) offset) = 0 := by
  rw [main_ops, opsAt, localLength_assertions]

theorem operations_length (interface : Interface) (offset : Nat) :
    (Circuit.ops (main interface) offset).length = 5 := by
  rw [main_ops]
  simp only [opsAt, List.length_map, assertions, List.length_cons,
    initialStateAssertions, List.length_finRange]
  rfl

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (main interface) offset)).length = 5 := by
  rw [main_ops, flatConstraints_opsAt]
  simp only [assertions, List.length_cons, initialStateAssertions,
    List.length_map, List.length_finRange]
  rfl

def circuit (interface : Interface) : FormalCircuit where
  main := main interface
  assumptions := Assumptions interface
  spec := SpecHolds interface
  soundness := by
    intro env offset _assumptions rows
    exact soundness interface env offset rows
  completeness := by
    intro env offset _assumptions specification
    exact completeness interface env offset specification

end NightstreamFPrime.Lifecycle.Stage1.NextPreimage
