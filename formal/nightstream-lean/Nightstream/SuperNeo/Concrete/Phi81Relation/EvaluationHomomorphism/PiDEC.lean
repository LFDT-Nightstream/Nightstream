import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.BaseLinear
import Nightstream.SuperNeo.Concrete.Phi81Relation.Semantics
import Nightstream.SuperNeo.Folding.PiDEC

/-!
Production-parameter evaluation recomposition for the typed Phi81 `PiDEC`
algebra.

Owns: the verifier-fixed `b = 2`, `k = 14` base-field weights; raw width-only
and typed recomposition of complete assignments; their exact equality;
fixed-shape recomposition of all matrix-evaluation arrays; and the concrete
theorem with the exact type of `PiDEC.Algebra.evaluations_hom`.

Does not own: digit splitting, digit norm bounds, commitment or public-input
homomorphisms, construction of a complete `PiDEC.Algebra`, Rust/R1CS
refinement, row removal, or constraint counts.

Emits constraints: no.

Authority boundary: every output array has exactly the verifier-owned matrix
count. Default reads appear only in the total operation on arbitrary public
arrays; the homomorphism proof eliminates them because independently derived
semantic evaluation arrays have the exact typed size.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.verify.recomposition.weights` | child `i` has verifier-fixed weight `2^i` in `F` | computed | `radixWeight` |
| `nifs.pi_dec.verify.recomposition.assignment` | recomposition covers every coordinate of the complete Phi81 assignment | computed | `recomposeAssignment` |
| `nifs.pi_dec.verify.recomposition.assignment.raw_refinement` | width-only recomposition is exactly the typed relation recomposition | derived | `raw_recomposeAssignment_eq` |
| `nifs.pi_dec.verify.recomposition.evaluations` | every matrix and all 54 lanes use the same `2^i` weights | computed | `evaluations_hom` |
| `nifs.pi_dec.verify.algebra.evaluations_hom` | the concrete theorem has the exact `PiDEC.Algebra` field signature | derived | `relation_evaluations_hom` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Production `PiDEC` radix weight in canonical Goldilocks representation. -/
def radixWeight
    (index : Fin productionGlobalParams.k) : F :=
  ⟨productionGlobalParams.b ^ index.val % goldilocksModulus,
    Nat.mod_lt _ (by decide)⟩

namespace Raw

/-- Recompose every raw assignment coordinate with the production `PiDEC`
radix weights. -/
def recomposeAssignment {columns : Nat}
    (assignments : Fin productionGlobalParams.k ->
      PaperLinearAlgebra.Assignment F columns) :
    PaperLinearAlgebra.Assignment F columns :=
  BaseLinear.Raw.combineAssignments radixWeight assignments

end Raw

/-- Recompose every typed relation-assignment coordinate with the production
`PiDEC` radix weights. -/
def recomposeAssignment {shape : Shape}
    (assignments : Fin productionGlobalParams.k -> Assignment shape) :
    Assignment shape :=
  BaseLinear.combineAssignments radixWeight assignments

/-- The width-only recomposition used by independent packed semantics is
exactly the typed relation recomposition. -/
theorem raw_recomposeAssignment_eq
    {shape : Shape}
    (assignments : Fin productionGlobalParams.k -> Assignment shape) :
    Raw.recomposeAssignment assignments =
      recomposeAssignment assignments := by
  unfold Raw.recomposeAssignment recomposeAssignment
  exact BaseLinear.raw_combineAssignments_eq radixWeight assignments

/-- Recompose arbitrary public evaluation arrays into the verifier-owned
matrix shape. Semantic arrays never exercise the default branch. -/
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

/-- Exact production-parameter evaluation law. This is the concrete
`PiDEC.Algebra.evaluations_hom` field, not merely an adjacent linearity lemma. -/
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

/-- The same theorem stated through the independently defined relation
semantics, with exactly the field type expected by a future concrete
`PiDEC.Algebra` instance. -/
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

end Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC
