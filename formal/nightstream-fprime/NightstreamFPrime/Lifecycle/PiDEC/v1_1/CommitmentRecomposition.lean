import Mathlib.Logic.Equiv.Fin.Basic
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.RadixRecomposition
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Commitment

/-!
Paper authority: SuperNeo v1.1, Section 7.5, verifier commitment equation
`c = sum_i b^i c_i`.

Obligation: enforce the fixed-radix recomposition of all 22×54 coefficients
of the typed Ajtai commitment.

Inputs:
- one parent commitment;
- sixteen child commitments.

Outputs: none.

Constraint group:
- C1: one affine `RadixRecomposition` row per commitment coefficient.

Parent coverage:
- `PiDEC.PaperVerifier.Accepted.commitmentEquation`.
-/

namespace NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition

open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

def rowCount : Nat := productionProfile.commitmentWidth
def coordinateCount : Nat := rowCount * ringDegree

theorem rowCount_eq : rowCount = 22 := by
  rfl

theorem coordinateCount_eq : coordinateCount = 1188 := by
  norm_num [coordinateCount, rowCount, productionProfile, ringDegree]

def coordinates (index : Fin coordinateCount) :
    Fin rowCount × Fin ringDegree :=
  finProdFinEquiv.symm index

def indexOf (row : Fin rowCount) (lane : Fin ringDegree) :
    Fin coordinateCount :=
  finProdFinEquiv (row, lane)

@[simp] theorem coordinates_indexOf
    (row : Fin rowCount) (lane : Fin ringDegree) :
    coordinates (indexOf row lane) = (row, lane) := by
  simp [coordinates, indexOf]

structure Interface where
  parent : Nat → Fin rowCount → Fin ringDegree → Expr
  child : Nat → Radix.ChildIndex → Fin rowCount → Fin ringDegree → Expr

def scalarInterface (interface : Interface) :
    RadixRecomposition.Interface coordinateCount where
  parent := fun offset coordinate =>
    interface.parent offset (coordinates coordinate).1 (coordinates coordinate).2
  child := fun offset child coordinate =>
    interface.child offset child
      (coordinates coordinate).1 (coordinates coordinate).2

abbrev Assumptions (interface : Interface) (offset : Nat) (env : Env) :=
  RadixRecomposition.Assumptions (scalarInterface interface) offset env

abbrev SpecHolds (interface : Interface) (offset : Nat) (env : Env) :=
  RadixRecomposition.SpecHolds (scalarInterface interface) offset env

def evalParent (interface : Interface) (offset : Nat) (env : Env) :
    PiRLCAlgebra.Commitment.Value rowCount :=
  fun row lane => (interface.parent offset row lane).eval env

def evalChildren (interface : Interface) (offset : Nat) (env : Env) :
    Radix.ChildIndex → PiRLCAlgebra.Commitment.Value rowCount :=
  fun child row lane => (interface.child offset child row lane).eval env

private theorem combineCommitments_apply {count : Nat}
    (weights : Fin count → F)
    (values : Fin count → PiRLCAlgebra.Commitment.Value rowCount)
    (row : Fin rowCount) (lane : Fin ringDegree) :
    Commitment.combineCommitments weights values row lane =
      ((List.ofFn fun child => values child row lane).zip
        (List.ofFn weights)).foldr
        (fun pair suffix => pair.2 * pair.1 + suffix) 0 := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [Commitment.combineCommitments,
        PiRLCAlgebra.Commitment.commitmentAdd,
        Commitment.commitmentScale,
        EvaluationHomomorphism.CarrierAction.ringFScale, ringFAdd,
        List.ofFn_succ, List.zip_cons_cons, List.foldr_cons]
      rw [inductionHypothesis]

theorem recomposeCommitment_apply
    (values : Radix.ChildIndex → PiRLCAlgebra.Commitment.Value rowCount)
    (row : Fin rowCount) (lane : Fin ringDegree) :
    Commitment.recomposeCommitment values row lane =
      Radix.recomposeScalar (fun child => values child row lane) := by
  calc
    Commitment.recomposeCommitment values row lane =
        Radix.recomposeScalarList (fun child => values child row lane) := by
      exact combineCommitments_apply
        EvaluationHomomorphism.PiDEC.radixWeight values row lane
    _ = Radix.recomposeScalar (fun child => values child row lane) :=
      Radix.recomposeScalarList_eq _

theorem parentCoverage (interface : Interface) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) :
    evalParent interface offset env =
      Commitment.recomposeCommitment (evalChildren interface offset env) := by
  funext row lane
  have coordinate := specification (indexOf row lane)
  calc
    evalParent interface offset env row lane =
        Radix.recomposeScalar
          (fun child => evalChildren interface offset env child row lane) := by
      simpa [evalParent, evalChildren, scalarInterface,
        RadixRecomposition.parentValues,
        RadixRecomposition.childValues] using coordinate
    _ = Commitment.recomposeCommitment
          (evalChildren interface offset env) row lane :=
      (recomposeCommitment_apply
        (evalChildren interface offset env) row lane).symm

theorem specHolds_of_parentCoverage
    (interface : Interface) (offset : Nat) (env : Env)
    (equation : evalParent interface offset env =
      Commitment.recomposeCommitment (evalChildren interface offset env)) :
    SpecHolds interface offset env := by
  intro coordinate
  have value := congrFun
    (congrFun equation (coordinates coordinate).1)
    (coordinates coordinate).2
  rw [recomposeCommitment_apply] at value
  simpa [evalParent, evalChildren, scalarInterface,
    RadixRecomposition.parentValues,
    RadixRecomposition.childValues] using value

abbrev circuit (interface : Interface) : FormalCircuit :=
  RadixRecomposition.circuit (scalarInterface interface)

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 0 := by
  exact (circuit interface).privateCount_eq offset

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      1188 := by
  calc
    _ = coordinateCount := (circuit interface).rowCount_eq offset
    _ = 1188 := coordinateCount_eq

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow offset := by
  rw [RadixRecomposition.circuit_ops]
  exact RadixRecomposition.flatConstraints_varsBelow
    (scalarInterface interface) offset env assumptions

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

end NightstreamFPrime.Lifecycle.PiDEC.v1_1.CommitmentRecomposition
