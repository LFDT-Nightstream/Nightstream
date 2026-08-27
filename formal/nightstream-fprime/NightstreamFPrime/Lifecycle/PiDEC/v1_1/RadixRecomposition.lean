import NightstreamFPrime.Circuit.StraightLine
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix

/-!
Paper authority: SuperNeo v1.1, Section 7.5, verifier Step 2, equations
`c = sum_i b^i c_i`, `y = sum_i b^i y_i`, and
`y_j = sum_i b^i y_{i,j}`.

Obligation: one coordinate of a typed parent family equals the fixed-radix
recomposition of its sixteen child coordinates.

Inputs:
- one parent expression per coordinate;
- sixteen child expressions per coordinate, in `Radix.ChildIndex` order.

Outputs: none. The verifier-owned weights are constants.

Constraint group:
- R1: one affine recomposition equation per coordinate.

Parent coverage:
- `PiDEC.PaperVerifier.Accepted.commitmentEquation`;
- `PiDEC.PaperVerifier.Accepted.evaluationEquation`.

This leaf does not own public-input digit range checks, child construction,
family-specific coordinate encodings, output binding, or physical columns.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

structure Interface (coordinateCount : Nat) where
  parent : Nat → Fin coordinateCount → Expr
  child : Nat → Radix.ChildIndex → Fin coordinateCount → Expr

def parentValues {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) (env : Env) :
    Fin coordinateCount → F :=
  fun coordinate => (interface.parent offset coordinate).eval env

def childValues {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) (env : Env) :
    Radix.ChildIndex → Fin coordinateCount → F :=
  fun child coordinate => (interface.child offset child coordinate).eval env

def recomposeExpr {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat)
    (coordinate : Fin coordinateCount) : Expr :=
  ((List.ofFn fun child : Radix.ChildIndex =>
      interface.child offset child coordinate).zip
    (List.ofFn fun child : Radix.ChildIndex =>
      EvaluationHomomorphism.PiDEC.radixWeight child)).foldr
    (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0

def constraint {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat)
    (coordinate : Fin coordinateCount) : Expr :=
  interface.parent offset coordinate -
    recomposeExpr interface offset coordinate

def constraints {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) : List Expr :=
  List.ofFn (constraint interface offset)

def operations {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) : List Op :=
  (constraints interface offset).map .assertZero

def main {coordinateCount : Nat}
    (interface : Interface coordinateCount) : Circuit Unit := fun offset =>
  ((), offset, operations interface offset)

structure Assumptions {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) (_env : Env) : Prop where
  parentBelow : ∀ coordinate,
    (interface.parent offset coordinate).VarsBelow offset
  childBelow : ∀ child coordinate,
    (interface.child offset child coordinate).VarsBelow offset

def SpecHolds {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) (env : Env) : Prop :=
  ∀ coordinate,
    parentValues interface offset env coordinate =
      Radix.recomposeScalar
        (fun child => childValues interface offset env child coordinate)

private theorem weightedFold_eval (env : Env) :
    ∀ (values : List Expr) (weights : List F),
      ((values.zip weights).foldr
          (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0).eval env =
        ((values.map fun value => value.eval env).zip weights).foldr
          (fun pair suffix => pair.2 * pair.1 + suffix) 0
  | [], _ => by rfl
  | _ :: _, [] => by rfl
  | value :: values, weight :: weights => by
      change weight * value.eval env +
          ((values.zip weights).foldr
            (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0).eval env =
        weight * value.eval env +
          (((values.map fun item => item.eval env).zip weights).foldr
            (fun pair suffix => pair.2 * pair.1 + suffix) 0)
      rw [weightedFold_eval env values weights]

theorem recomposeExpr_eval {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) (env : Env)
    (coordinate : Fin coordinateCount) :
    (recomposeExpr interface offset coordinate).eval env =
      Radix.recomposeScalar
        (fun child => childValues interface offset env child coordinate) := by
  unfold recomposeExpr
  rw [weightedFold_eval, List.map_ofFn]
  exact Radix.recomposeScalarList_eq
    (fun child => childValues interface offset env child coordinate)

theorem constraint_eval {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) (env : Env)
    (coordinate : Fin coordinateCount) :
    (constraint interface offset coordinate).eval env =
      parentValues interface offset env coordinate -
        Radix.recomposeScalar
          (fun child => childValues interface offset env child coordinate) := by
  simp only [constraint, Expr.eval_sub, parentValues, recomposeExpr_eval]

private theorem flatConstraints_assertions (items : List Expr) :
    flatConstraints (items.map .assertZero) = items := by
  induction items with
  | nil => rfl
  | cons expression rest inductionHypothesis =>
      change expression :: flatConstraints (rest.map .assertZero) =
        expression :: rest
      rw [inductionHypothesis]

theorem flatConstraints_operations {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) :
    flatConstraints (operations interface offset) =
      constraints interface offset := by
  exact flatConstraints_assertions _

theorem localLength_eq {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) :
    localLength (operations interface offset) = 0 := by
  change localLength ((constraints interface offset).map .assertZero) = 0
  induction constraints interface offset with
  | nil => rfl
  | cons _ rest inductionHypothesis =>
      change 0 + localLength (rest.map .assertZero) = 0
      simpa using inductionHypothesis

theorem flatConstraints_length_eq {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) :
    (flatConstraints (operations interface offset)).length = coordinateCount := by
  rw [flatConstraints_operations]
  simp [constraints]

private theorem weightedFold_varsBelow (bound : Nat) :
    ∀ (values : List Expr) (weights : List F),
      (∀ value ∈ values, value.VarsBelow bound) →
      ((values.zip weights).foldr
        (fun pair suffix => Expr.const pair.2 * pair.1 + suffix) 0).VarsBelow bound
  | [], _, _ => trivial
  | _ :: _, [], _ => trivial
  | value :: values, _ :: weights, below =>
      Expr.VarsBelow.add _ _ bound
        (Expr.VarsBelow.mul _ _ bound trivial (below value (by simp)))
        (weightedFold_varsBelow bound values weights
          (fun item member => below item (by simp [member])))

theorem recomposeExpr_varsBelow {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (coordinate : Fin coordinateCount) :
    (recomposeExpr interface offset coordinate).VarsBelow offset := by
  unfold recomposeExpr
  apply weightedFold_varsBelow
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨child, rfl⟩
  exact assumptions.childBelow child coordinate

theorem flatConstraints_varsBelow {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsBelow offset := by
  rw [flatConstraints_operations]
  intro expression member
  rcases List.mem_ofFn.mp member with ⟨coordinate, rfl⟩
  exact Expr.VarsBelow.sub _ _ offset
    (assumptions.parentBelow coordinate)
    (recomposeExpr_varsBelow interface offset env assumptions coordinate)

private theorem constraintsHold_of_holds {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) (env : Env)
    (rows : holds env (operations interface offset)) :
    ConstraintsHold env (constraints interface offset) := by
  intro expression member
  exact rows (.assertZero expression) (by
    simp [operations, member])

theorem soundness {coordinateCount : Nat}
    (interface : Interface coordinateCount) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (rows : holds env (operations interface offset)) :
    SpecHolds interface offset env := by
  have constraintRows := constraintsHold_of_holds interface offset env rows
  intro coordinate
  have zero := constraintRows (constraint interface offset coordinate)
    (List.mem_ofFn.mpr ⟨coordinate, rfl⟩)
  rw [constraint_eval] at zero
  exact sub_eq_zero.mp zero

theorem completeness {coordinateCount : Nat}
    (interface : Interface coordinateCount) (env : Env) (offset : Nat)
    (_assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) := by
  refine ⟨env, ?_, ?_⟩
  · rw [localLength_eq]
    intro _ _
    rfl
  · unfold holdsFlat
    rw [flatConstraints_operations]
    intro expression member
    rcases List.mem_ofFn.mp member with ⟨coordinate, rfl⟩
    rw [constraint_eval, specification coordinate, sub_self]

def circuit {coordinateCount : Nat}
    (interface : Interface coordinateCount) : FormalCircuit :=
  { main := main interface
    assumptions := Assumptions interface
    spec := SpecHolds interface
    privateCount := fun _ => 0
    rowCount := fun _ => coordinateCount
    privateCount_eq := by
      intro offset
      exact localLength_eq interface offset
    rowCount_eq := by
      intro offset
      exact flatConstraints_length_eq interface offset
    soundness := by
      intro env offset assumptions rows
      exact soundness interface env offset assumptions rows
    completeness := by
      intro env offset assumptions specification
      exact completeness interface env offset assumptions specification }

@[simp] theorem circuit_ops {coordinateCount : Nat}
    (interface : Interface coordinateCount) (offset : Nat) :
    Circuit.ops (circuit interface).main offset = operations interface offset := by
  rfl

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition
