import NightstreamFPrime.Spec.Profile
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.BaseLinear
import NightstreamFPrime.Spec.Phi81Relation.Semantics
import NightstreamFPrime.Spec.Folding.PiDEC

/-!
Production-parameter evaluation recomposition for the typed Phi81 `PiDEC`
algebra.

Owns: the verifier-fixed `b = 2`, `k = 16` base-field weights; raw width-only
and typed recomposition of complete assignments; their exact equality;
fixed-shape recomposition of all matrix-evaluation arrays; and the concrete
theorem with the exact type of `PiDEC.Algebra.evaluations_hom`.

Does not own: digit splitting, digit norm bounds, commitment or public-input
homomorphisms, construction of a complete `PiDEC.Algebra`, Rust/R1CS
refinement, row removal, or constraint counts.
-/

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiDEC

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def radixWeight
    (index : Fin productionGlobalParams.k) : F :=
  ⟨productionGlobalParams.b ^ index.val % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

namespace Raw

end Raw

def recomposeAssignment {shape : Shape}
    (assignments : Fin productionGlobalParams.k -> Assignment shape) :
    Assignment shape :=
  BaseLinear.combineAssignments radixWeight assignments

def recomposeEvaluations {shape : Shape}
    (items : Fin productionGlobalParams.k -> Array Evaluation) :
    Array Evaluation :=
  Array.ofFn fun matrix : Fin shape.matrixCount =>
    BaseLinear.combineEvaluations radixWeight fun index =>
      (items index).getD matrix.val BaseLinear.evaluationZero

private theorem semantic_getD
    {shape : Shape}
    (system : Structure shape) (assignment : Assignment shape)
    (point : Point shape) (matrix : Fin shape.matrixCount) :
    (evaluations system assignment point).getD matrix.val
        BaseLinear.evaluationZero =
      matrixEvaluation system assignment point matrix := by
  have bound : matrix.val < (evaluations system assignment point).size := by
    rw [evaluations_size]
    exact matrix.isLt
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem bound]
  exact evaluations_get system assignment point matrix

theorem evaluations_hom
    {shape : Shape}
    (system : Structure shape) (point : Point shape)
    (assignments : Fin productionGlobalParams.k -> Assignment shape) :
    evaluations system (recomposeAssignment assignments) point =
      recomposeEvaluations (shape := shape) fun index =>
        evaluations system (assignments index) point := by
  rw [recomposeAssignment, BaseLinear.evaluations_combine]
  apply Array.ext
  · simp [recomposeEvaluations]
  · intro matrix leftLt rightLt
    let typedMatrix : Fin shape.matrixCount := ⟨matrix, by
      simpa [recomposeEvaluations] using rightLt⟩
    simp only [recomposeEvaluations, Array.getElem_ofFn]
    apply congrArg (BaseLinear.combineEvaluations radixWeight)
    funext index
    exact (semantic_getD system (assignments index) point typedMatrix).symm

theorem relation_evaluations_hom
    {shape : Shape} {Commitment : Type}
    (commit : Assignment shape -> Commitment) :
    forall (system : Structure shape) (point : Point shape) assignments,
      (relationSemantics commit).evaluations system
          (recomposeAssignment assignments) point =
        recomposeEvaluations (shape := shape) fun index =>
          (relationSemantics commit).evaluations system
            (assignments index) point := by
  intro system point assignments
  exact evaluations_hom system point assignments

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiDEC
