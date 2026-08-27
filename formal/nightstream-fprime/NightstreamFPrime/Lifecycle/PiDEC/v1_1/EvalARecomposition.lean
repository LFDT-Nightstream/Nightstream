import NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalKRecomposition

/-!
Paper authority: SuperNeo v1.1, Section 7.5, the separate matrix-evaluation
equations `y_A,j = sum_i b^i y_A,j,i`.

Obligation: enforce fixed-radix recomposition for every one of the 14
production matrix evaluations. `Eval_A` remains separate from Pad `Eval_K`.

Inputs:
- one parent `Eval_A` ring for each production matrix;
- sixteen child `Eval_A` rings for each production matrix.

Outputs: none.

Constraint group:
- A1: 14 matrices × 54 coefficients × 2 extension cells = 1,512 affine rows.

Parent coverage:
- the matrix field of `PiDEC.PaperVerifier.Accepted.evaluationEquation`.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalARecomposition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism

abbrev blockCount : Nat := productionShape.matrixCount

structure Interface where
  parent : Nat → Fin productionShape.matrixCount →
    Fin productionShape.coefficientCount → KExpr
  child : Nat →
    NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex →
    Fin productionShape.matrixCount →
    Fin productionShape.coefficientCount → KExpr

def ringInterface (interface : Interface) :
    RingKRecomposition.Interface blockCount where
  parent := fun offset matrix lane =>
    interface.parent offset matrix (EvalKRecomposition.coefficient lane)
  child := fun offset child matrix lane =>
    interface.child offset child matrix
      (EvalKRecomposition.coefficient lane)

abbrev Assumptions (interface : Interface) (offset : Nat) (env : Env) :=
  RingKRecomposition.Assumptions (ringInterface interface) offset env

abbrev SpecHolds (interface : Interface) (offset : Nat) (env : Env) :=
  RingKRecomposition.SpecHolds (ringInterface interface) offset env

def evalParent (interface : Interface) (offset : Nat) (env : Env) :
    Fin productionShape.matrixCount → RingK :=
  fun matrix lane =>
    (interface.parent offset matrix
      (EvalKRecomposition.coefficient lane)).eval env

def evalChildren (interface : Interface) (offset : Nat) (env : Env) :
    NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix.ChildIndex →
      Fin productionShape.matrixCount → RingK :=
  fun child matrix lane =>
    (interface.child offset child matrix
      (EvalKRecomposition.coefficient lane)).eval env

theorem parentCoverage (interface : Interface) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) :
    evalParent interface offset env = fun matrix =>
      BaseLinear.combineEvaluations
        NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
        (fun child => evalChildren interface offset env child matrix) := by
  simpa [evalParent, evalChildren, RingKRecomposition.evalParent,
    RingKRecomposition.evalChildren, ringInterface] using
    RingKRecomposition.parentCoverage
      (ringInterface interface) offset env specification

theorem specHolds_of_parentCoverage
    (interface : Interface) (offset : Nat) (env : Env)
    (equation : evalParent interface offset env = fun matrix =>
      BaseLinear.combineEvaluations
        NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiDEC.radixWeight
        (fun child => evalChildren interface offset env child matrix)) :
    SpecHolds interface offset env := by
  apply RingKRecomposition.specHolds_of_parentCoverage
    (blockCount := blockCount) (ringInterface interface) offset env
  simpa [evalParent, evalChildren, RingKRecomposition.evalParent,
    RingKRecomposition.evalChildren, ringInterface] using equation

theorem coordinateCount_eq :
    RingKRecomposition.coordinateCount blockCount = 1512 := by
  norm_num [RingKRecomposition.coordinateCount,
    RingKRecomposition.cellCount, blockCount, productionShape,
    Phi81MatrixSource.phi81Shape, productionProfile, ringDegree]

abbrev circuit (interface : Interface) : FormalCircuit :=
  RingKRecomposition.circuit (ringInterface interface)

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 0 :=
  (circuit interface).privateCount_eq offset

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      1512 := by
  calc
    _ = RingKRecomposition.coordinateCount blockCount :=
      (circuit interface).rowCount_eq offset
    _ = 1512 := coordinateCount_eq

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

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.EvalARecomposition
