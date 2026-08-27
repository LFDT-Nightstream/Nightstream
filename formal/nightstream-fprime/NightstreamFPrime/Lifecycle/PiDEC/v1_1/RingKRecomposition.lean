import Mathlib.Logic.Equiv.Fin.Basic
import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.BaseLinear

/-!
Paper authority: SuperNeo v1.1, Section 7.5, verifier evaluation equations.

Obligation: enforce fixed-radix recomposition for a typed family of `RingK`
evaluations. Each `K` coefficient uses the canonical `c0, c1` field order.

Inputs:
- one parent `RingK` per block;
- sixteen child `RingK` values per block.

Outputs: none.

Constraint group:
- E1: one affine `RadixRecomposition` row per block, ring lane, and `K` cell.

Parent coverage:
- the separate Pad `Eval_K` equation;
- each of the 14 separate matrix `Eval_A` equations.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

def cellCount : Nat := 2

def coordinateCount (blockCount : Nat) : Nat :=
  blockCount * (ringDegree * cellCount)

def coordinates {blockCount : Nat}
    (index : Fin (coordinateCount blockCount)) :
    Fin blockCount × Fin ringDegree × Fin cellCount :=
  let outer : Fin (blockCount * (ringDegree * cellCount)) := index
  let pair := finProdFinEquiv.symm outer
  (pair.1, finProdFinEquiv.symm pair.2)

def indexOf {blockCount : Nat}
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount) :
    Fin (coordinateCount blockCount) :=
  finProdFinEquiv (block, finProdFinEquiv (lane, cell))

@[simp] theorem coordinates_indexOf {blockCount : Nat}
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount) :
    coordinates (indexOf block lane cell) = (block, lane, cell) := by
  simp [coordinates, indexOf]

def kCell (cell : Fin cellCount) (value : K) : F :=
  if cell.val = 0 then value.c0 else value.c1

def expressionCell (cell : Fin cellCount) (value : KExpr) : Expr :=
  if cell.val = 0 then value.c0 else value.c1

def c0Cell : Fin cellCount := ⟨0, by decide⟩
def c1Cell : Fin cellCount := ⟨1, by decide⟩

@[simp] theorem expressionCell_eval (cell : Fin cellCount)
    (value : KExpr) (env : Env) :
    (expressionCell cell value).eval env = kCell cell (value.eval env) := by
  unfold expressionCell kCell KExpr.eval
  split <;> rfl

theorem expressionCell_varsBelow (cell : Fin cellCount)
    (value : KExpr) (bound : Nat) (below : value.VarsBelow bound) :
    (expressionCell cell value).VarsBelow bound := by
  fin_cases cell
  · simpa [expressionCell, cellCount] using below.1
  · simpa [expressionCell, cellCount] using below.2

@[simp] private theorem kCell_zero (cell : Fin cellCount) :
    kCell cell K.zero = 0 := by
  unfold kCell K.zero
  split <;> rfl

@[simp] private theorem kCell_add (cell : Fin cellCount) (left right : K) :
    kCell cell (K.add left right) = kCell cell left + kCell cell right := by
  unfold kCell K.add
  split <;> rfl

@[simp] private theorem kCell_mul_embed
    (cell : Fin cellCount) (scalar : F) (value : K) :
    kCell cell (K.mul (K.embed scalar) value) = scalar * kCell cell value := by
  unfold kCell K.mul K.embed
  split <;> simp

structure Interface (blockCount : Nat) where
  parent : Nat → Fin blockCount → Fin ringDegree → KExpr
  child : Nat → Radix.ChildIndex → Fin blockCount → Fin ringDegree → KExpr

def scalarInterface {blockCount : Nat} (interface : Interface blockCount) :
    RadixRecomposition.Interface (coordinateCount blockCount) where
  parent := fun offset coordinate =>
    expressionCell (coordinates coordinate).2.2
      (interface.parent offset (coordinates coordinate).1
        (coordinates coordinate).2.1)
  child := fun offset child coordinate =>
    expressionCell (coordinates coordinate).2.2
      (interface.child offset child (coordinates coordinate).1
        (coordinates coordinate).2.1)

abbrev Assumptions {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env) :=
  RadixRecomposition.Assumptions (scalarInterface interface) offset env

abbrev SpecHolds {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env) :=
  RadixRecomposition.SpecHolds (scalarInterface interface) offset env

def evalParent {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env) : Fin blockCount → RingK :=
  fun block lane => (interface.parent offset block lane).eval env

def evalChildren {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env) : Radix.ChildIndex → Fin blockCount → RingK :=
  fun child block lane => (interface.child offset child block lane).eval env

private theorem combineEvaluations_cell {count : Nat}
    (weights : Fin count → F) (values : Fin count → RingK)
    (cell : Fin cellCount) (lane : Fin ringDegree) :
    kCell cell (BaseLinear.combineEvaluations weights values lane) =
      ((List.ofFn fun child => kCell cell (values child lane)).zip
        (List.ofFn weights)).foldr
        (fun pair suffix => pair.2 * pair.1 + suffix) 0 := by
  induction count with
  | zero => exact kCell_zero cell
  | succ count inductionHypothesis =>
      simp only [BaseLinear.combineEvaluations, BaseLinear.evaluationAdd,
        BaseLinear.evaluationScale, kCell_add, kCell_mul_embed,
        List.ofFn_succ, List.zip_cons_cons, List.foldr_cons]
      rw [inductionHypothesis]

theorem recomposeEvaluation_cell
    (values : Radix.ChildIndex → RingK)
    (cell : Fin cellCount) (lane : Fin ringDegree) :
    kCell cell
        (BaseLinear.combineEvaluations
          EvaluationHomomorphism.PiDEC.radixWeight values lane) =
      Radix.recomposeScalar (fun child => kCell cell (values child lane)) := by
  calc
    _ = Radix.recomposeScalarList
        (fun child => kCell cell (values child lane)) :=
      combineEvaluations_cell
        EvaluationHomomorphism.PiDEC.radixWeight values cell lane
    _ = _ := Radix.recomposeScalarList_eq _

private theorem parentCoverage_cell {blockCount : Nat}
    (interface : Interface blockCount) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env)
    (block : Fin blockCount) (lane : Fin ringDegree) (cell : Fin cellCount) :
    kCell cell (evalParent interface offset env block lane) =
      kCell cell
        (BaseLinear.combineEvaluations
          EvaluationHomomorphism.PiDEC.radixWeight
          (fun child => evalChildren interface offset env child block) lane) := by
  have coordinate := specification (indexOf block lane cell)
  calc
    kCell cell (evalParent interface offset env block lane) =
        Radix.recomposeScalar (fun child =>
          kCell cell (evalChildren interface offset env child block lane)) := by
      simpa [evalParent, evalChildren, scalarInterface,
        RadixRecomposition.parentValues, RadixRecomposition.childValues] using
        coordinate
    _ = _ := (recomposeEvaluation_cell
      (fun child => evalChildren interface offset env child block) cell lane).symm

theorem parentCoverage {blockCount : Nat}
    (interface : Interface blockCount) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) :
    evalParent interface offset env = fun block =>
      BaseLinear.combineEvaluations
        EvaluationHomomorphism.PiDEC.radixWeight
        (fun child => evalChildren interface offset env child block) := by
  funext block lane
  exact congrArg₂ K.mk
    (by simpa [kCell, c0Cell] using
      parentCoverage_cell interface offset env specification block lane c0Cell)
    (by simpa [kCell, c1Cell] using
      parentCoverage_cell interface offset env specification block lane c1Cell)

theorem specHolds_of_parentCoverage {blockCount : Nat}
    (interface : Interface blockCount) (offset : Nat) (env : Env)
    (equation : evalParent interface offset env = fun block =>
      BaseLinear.combineEvaluations
        EvaluationHomomorphism.PiDEC.radixWeight
        (fun child => evalChildren interface offset env child block)) :
    SpecHolds interface offset env := by
  intro coordinate
  have ringValue := congrFun
    (congrFun equation (coordinates coordinate).1)
    (coordinates coordinate).2.1
  have cellValue := congrArg (kCell (coordinates coordinate).2.2) ringValue
  rw [recomposeEvaluation_cell] at cellValue
  simpa [evalParent, evalChildren, scalarInterface,
    RadixRecomposition.parentValues,
    RadixRecomposition.childValues] using cellValue

abbrev circuit {blockCount : Nat} (interface : Interface blockCount) :
    FormalCircuit :=
  RadixRecomposition.circuit (scalarInterface interface)

theorem localLength_eq {blockCount : Nat}
    (interface : Interface blockCount) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 0 :=
  (circuit interface).privateCount_eq offset

theorem flatConstraints_length {blockCount : Nat}
    (interface : Interface blockCount) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      coordinateCount blockCount :=
  (circuit interface).rowCount_eq offset

theorem flatConstraints_varsBelow {blockCount : Nat}
    (interface : Interface blockCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow offset := by
  rw [RadixRecomposition.circuit_ops]
  exact RadixRecomposition.flatConstraints_varsBelow
    (scalarInterface interface) offset env assumptions

theorem soundness {blockCount : Nat}
    (interface : Interface blockCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  (circuit interface).soundness env offset assumptions rows

theorem completeness {blockCount : Nat}
    (interface : Interface blockCount) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  (circuit interface).completeness env offset assumptions specification

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition
