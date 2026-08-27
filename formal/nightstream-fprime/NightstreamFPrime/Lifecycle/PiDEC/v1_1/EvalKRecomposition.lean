import NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.RingKRecomposition

/-!
Paper authority: SuperNeo v1.1, Section 7.5, verifier equation
`y = sum_i b^i y_i` for the separate Pad evaluation family.

Obligation: enforce fixed-radix recomposition of the one 54-coefficient
`Eval_K` ring. `Eval_K` is not matrix zero and is not merged with `Eval_A`.

Inputs:
- one parent `Eval_K`;
- sixteen child `Eval_K` values.

Outputs: none.

Constraint group:
- K1: 54 coefficients × 2 extension-field cells = 108 affine rows.

Parent coverage:
- the Pad field of `PiDEC.PaperVerifier.Accepted.evaluationEquation`.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism

def blockCount : Nat := 1
def block : Fin blockCount := ⟨0, by decide⟩

theorem coefficientCount_eq :
    productionShape.coefficientCount = ringDegree := by
  rfl

def coefficient (lane : Fin ringDegree) :
    Fin productionShape.coefficientCount :=
  Fin.cast coefficientCount_eq.symm lane

structure Interface where
  parent : Nat → Fin productionShape.coefficientCount → KExpr
  child : Nat →
    NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex →
    Fin productionShape.coefficientCount → KExpr

def ringInterface (interface : Interface) :
    RingKRecomposition.Interface blockCount where
  parent := fun offset _ lane => interface.parent offset (coefficient lane)
  child := fun offset child _ lane =>
    interface.child offset child (coefficient lane)

abbrev Assumptions (interface : Interface) (offset : Nat) (env : Env) :=
  RingKRecomposition.Assumptions (ringInterface interface) offset env

abbrev SpecHolds (interface : Interface) (offset : Nat) (env : Env) :=
  RingKRecomposition.SpecHolds (ringInterface interface) offset env

def evalParent (interface : Interface) (offset : Nat) (env : Env) : RingK :=
  fun lane => (interface.parent offset (coefficient lane)).eval env

def evalChildren (interface : Interface) (offset : Nat) (env : Env) :
    NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex → RingK :=
  fun child lane =>
    (interface.child offset child (coefficient lane)).eval env

theorem parentCoverage (interface : Interface) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) :
    evalParent interface offset env =
      BaseLinear.combineEvaluations
        NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
        (evalChildren interface offset env) := by
  have all := RingKRecomposition.parentCoverage
    (ringInterface interface) offset env specification
  have selected := congrFun all block
  simpa [evalParent, evalChildren, RingKRecomposition.evalParent,
    RingKRecomposition.evalChildren, ringInterface, block] using selected

theorem specHolds_of_parentCoverage
    (interface : Interface) (offset : Nat) (env : Env)
    (equation : evalParent interface offset env =
      BaseLinear.combineEvaluations
        NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
        (evalChildren interface offset env)) :
    SpecHolds interface offset env := by
  apply RingKRecomposition.specHolds_of_parentCoverage
    (blockCount := blockCount) (ringInterface interface) offset env
  funext selected
  have selected_eq : selected = block := by
    apply Fin.ext
    change selected.val = 0
    have selected_lt := selected.isLt
    simp only [blockCount] at selected_lt
    omega
  subst selected
  simpa [evalParent, evalChildren, RingKRecomposition.evalParent,
    RingKRecomposition.evalChildren, ringInterface, block] using equation

theorem coordinateCount_eq :
    RingKRecomposition.coordinateCount blockCount = 108 := by
  norm_num [RingKRecomposition.coordinateCount,
    RingKRecomposition.cellCount, blockCount, ringDegree]

abbrev circuit (interface : Interface) : FormalCircuit :=
  RingKRecomposition.circuit (ringInterface interface)

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 0 :=
  (circuit interface).privateCount_eq offset

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      108 := by
  calc
    _ = RingKRecomposition.coordinateCount blockCount :=
      (circuit interface).rowCount_eq offset
    _ = 108 := coordinateCount_eq

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow offset :=
  RingKRecomposition.flatConstraints_varsBelow
    (ringInterface interface) offset env assumptions

theorem soundness (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  (circuit interface).soundness env offset assumptions rows

theorem completeness (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  (circuit interface).completeness env offset assumptions specification

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition
