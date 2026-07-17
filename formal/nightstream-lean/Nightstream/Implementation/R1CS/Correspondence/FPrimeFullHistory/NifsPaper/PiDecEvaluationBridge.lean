import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiDec.Weights
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiDEC

/-!
Production public-carrier refinement for `PiDEC` evaluation recomposition.

Owns: equality between the independently defined typed-Phi81 radix weights
and the fixed public-carrier weights, plus equality between the corresponding
54-lane evaluation recomposition functions.

Does not own: assignment decoding, complete-carrier alignment, digit
splitting, private CE membership, commitment/public-input recomposition,
`y_zcol`, Rust/R1CS row satisfaction, row removal, or costs.

Emits constraints: no.

Authority boundary: the implementation operation is compared with the
independently proved semantic operation. Neither side is accepted merely
because its own equations are self-consistent.

| Stage path | Refinement fact | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_dec.verify.refinement.weights` | public-carrier list equals typed semantic weights for every `i : Fin 14` | checked | `radixWeights_eq` |
| `nifs.pi_dec.verify.refinement.evaluations` | implementation componentwise recomposition equals semantic `RingK` recomposition | checked | `semantic_evaluations_hom` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDecEvaluationBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.Weights

private theorem combineEvaluations_c0
    {count : Nat} (weights : Fin count -> F)
    (items : Fin count -> Evaluation) (coefficient : Fin ringDegree) :
    (BaseLinear.combineEvaluations weights items coefficient).c0 =
      combineScalars weights fun index => (items index coefficient).c0 := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [BaseLinear.combineEvaluations, BaseLinear.evaluationAdd,
        BaseLinear.evaluationScale, combineScalars, K.add, K.mul, K.embed,
        inductionHypothesis]
      rw [show (7 : F) * 0 = 0 by exact Fin.mul_zero _,
        show (0 : F) * (items 0 coefficient).c1 = 0 by
          exact Fin.zero_mul _, Fin.add_zero]

private theorem combineEvaluations_c1
    {count : Nat} (weights : Fin count -> F)
    (items : Fin count -> Evaluation) (coefficient : Fin ringDegree) :
    (BaseLinear.combineEvaluations weights items coefficient).c1 =
      combineScalars weights fun index => (items index coefficient).c1 := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [BaseLinear.combineEvaluations, BaseLinear.evaluationAdd,
        BaseLinear.evaluationScale, combineScalars, K.add, K.mul, K.embed,
        inductionHypothesis]
      rw [show (0 : F) * (items 0 coefficient).c0 = 0 by
        exact Fin.zero_mul _, Fin.add_zero]

/-- The production public-carrier operation on one 54-lane evaluation is the
same operation used by the typed Phi81 `PiDEC` homomorphism theorem. -/
theorem combineEvaluation_eq
    (items : Fin productionGlobalParams.k -> Evaluation) :
    combineEvaluation items =
      BaseLinear.combineEvaluations PiDEC.radixWeight items := by
  funext coefficient
  apply k_eq_of_coeffs
  · change combineScalar (fun index => (items index coefficient).c0) =
        (BaseLinear.combineEvaluations PiDEC.radixWeight items coefficient).c0
    rw [combineScalar_eq]
    exact (combineEvaluations_c0 PiDEC.radixWeight items coefficient).symm
  · change combineScalar (fun index => (items index coefficient).c1) =
        (BaseLinear.combineEvaluations PiDEC.radixWeight items coefficient).c1
    rw [combineScalar_eq]
    exact (combineEvaluations_c1 PiDEC.radixWeight items coefficient).symm

/-- Fixed-size array form of `combineEvaluation_eq`. -/
theorem combineEvaluations_eq_of_size
    {shape : Phi81Relation.Shape}
    (items : Fin productionGlobalParams.k -> Array Evaluation)
    (sizes : forall index, (items index).size = shape.matrixCount) :
    combineEvaluations items =
      PiDEC.recomposeEvaluations (shape := shape) items := by
  apply Array.ext
  · simp [combineEvaluations, PiDEC.recomposeEvaluations,
      sizes firstIndex]
  · intro row leftLt rightLt
    have rowLt : row < shape.matrixCount := by
      simpa [PiDEC.recomposeEvaluations] using rightLt
    simp only [combineEvaluations, PiDEC.recomposeEvaluations,
      Array.getElem_ofFn, List.getElem_toArray, List.getElem_map,
      List.getElem_range]
    exact combineEvaluation_eq fun index =>
      (items index).getD row BaseLinear.evaluationZero

/-- Concrete bridge from the independently derived complete assignment to
the exact public evaluation recomposition operation consumed by the current
fixed-profile `PiDEC` correspondence. -/
theorem semantic_evaluations_hom
    {shape : Phi81Relation.Shape}
    (system : Phi81Relation.Structure shape)
    (point : Phi81Relation.Point shape)
    (assignments : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment shape) :
    Phi81Relation.evaluations system
        (PiDEC.recomposeAssignment assignments) point =
      combineEvaluations fun index =>
        Phi81Relation.evaluations system (assignments index) point := by
  calc
    _ = PiDEC.recomposeEvaluations (shape := shape) (fun index =>
          Phi81Relation.evaluations system (assignments index) point) :=
      PiDEC.evaluations_hom system point assignments
    _ = combineEvaluations (fun index =>
          Phi81Relation.evaluations system (assignments index) point) := by
      symm
      apply combineEvaluations_eq_of_size
      intro index
      exact Phi81Relation.evaluations_size system (assignments index) point

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDecEvaluationBridge
