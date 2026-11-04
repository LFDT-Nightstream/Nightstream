import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiRLC
import NightstreamFPrime.Spec.Phi81Relation.Semantics
import NightstreamFPrime.Spec.Folding.PiRLC

/-! SuperNeo v1.2 Theorem 11. -/

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

namespace Raw

def combineAssignments {logicalWidth : Nat} :
    {count : Nat} ->
      (Fin count -> RingF) ->
      (Fin count -> CarrierAction.CompleteAssignment logicalWidth) ->
      CarrierAction.CompleteAssignment logicalWidth
  | 0, _, _ => BaseLinear.Raw.assignmentZero
  | _ + 1, challenges, assignments =>
      BaseLinear.Raw.assignmentAdd
        (CarrierAction.act (challenges 0) (assignments 0))
        (combineAssignments
          (fun index => challenges index.succ)
          (fun index => assignments index.succ))

end Raw

def combineAssignments {shape : Shape} :
    {count : Nat} ->
      (Fin count -> RingF) ->
      (Fin count -> Assignment shape) -> Assignment shape
  | 0, _, _ => BaseLinear.assignmentZero
  | _ + 1, challenges, assignments =>
      BaseLinear.assignmentAdd
        (CarrierAction.act (challenges 0) (assignments 0))
        (combineAssignments
          (fun index => challenges index.succ)
          (fun index => assignments index.succ))

theorem raw_combineAssignments_eq
    {shape : Shape} {count : Nat}
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape) :
    Raw.combineAssignments challenges assignments =
      combineAssignments challenges assignments := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [Raw.combineAssignments, combineAssignments]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => assignments index.succ)]
      rfl

def combineEvaluation :
    {count : Nat} ->
      (Fin count -> RingF) -> (Fin count -> Evaluation) -> Evaluation
  | 0, _, _ => BaseLinear.evaluationZero
  | _ + 1, challenges, items =>
      ringKAdd
        (ringKMul (RingKAction.embedChallenge (challenges 0)) (items 0))
        (combineEvaluation
          (fun index => challenges index.succ)
          (fun index => items index.succ))

def combineEvaluations {shape : Shape} {count : Nat}
    (challenges : Fin count -> RingF)
    (items : Fin count -> Array Evaluation) : Array Evaluation :=
  Array.ofFn fun matrix : Fin shape.matrixCount =>
    combineEvaluation challenges fun index =>
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

theorem matrixEvaluation_combine
    {shape : Shape} {count : Nat}
    (system : Structure shape)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape)
    (point : Point shape) (matrix : Fin shape.matrixCount) :
    matrixEvaluation system (combineAssignments challenges assignments)
        point matrix =
      combineEvaluation challenges fun index =>
        matrixEvaluation system (assignments index) point matrix := by
  induction count with
  | zero =>
      exact BaseLinear.matrixEvaluation_zero system point matrix
  | succ count inductionHypothesis =>
      rw [combineAssignments, combineEvaluation]
      calc
        matrixEvaluation system
            (BaseLinear.assignmentAdd
              (CarrierAction.act (challenges 0) (assignments 0))
              (combineAssignments
                (fun index => challenges index.succ)
                (fun index => assignments index.succ)))
            point matrix =
          BaseLinear.evaluationAdd
            (matrixEvaluation system
              (CarrierAction.act (challenges 0) (assignments 0)) point matrix)
            (matrixEvaluation system
              (combineAssignments
                (fun index => challenges index.succ)
                (fun index => assignments index.succ)) point matrix) :=
          BaseLinear.matrixEvaluation_add system _ _ point matrix
        _ = _ := by
          rw [PiRLC.matrixEvaluation_act system (challenges 0)
            (PiRLC.productOrderLaw (challenges 0))]
          rw [inductionHypothesis
              (fun index => challenges index.succ)
              (fun index => assignments index.succ)]
          rfl

theorem explicitMatrixEvaluation_combine
    {shape : Shape} {count : Nat}
    (system : Structure shape)
    (matrix : PaperLinearAlgebra.BooleanMatrix F shape.rowVariables
      shape.carrierWidth)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape)
    (point : Point shape) :
    PiRLC.ExplicitMatrix.evaluate system matrix
        (combineAssignments challenges assignments) point =
      combineEvaluation challenges fun index =>
        PiRLC.ExplicitMatrix.evaluate system matrix
          (assignments index) point := by
  induction count with
  | zero => exact PiRLC.ExplicitMatrix.evaluate_zero system matrix point
  | succ count inductionHypothesis =>
      rw [combineAssignments, combineEvaluation,
        PiRLC.ExplicitMatrix.evaluate_add,
        PiRLC.ExplicitMatrix.evaluate_act system matrix (challenges 0)
          (PiRLC.productOrderLaw (challenges 0)),
        inductionHypothesis
          (fun index => challenges index.succ)
          (fun index => assignments index.succ)]
      rfl

theorem evaluations_hom
    {shape : Shape} {count : Nat}
    (system : Structure shape) (point : Point shape)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape) :
    evaluations system (combineAssignments challenges assignments) point =
      combineEvaluations (shape := shape) challenges fun index =>
        evaluations system (assignments index) point := by
  apply Array.ext
  · simp [combineEvaluations]
  · intro matrix leftLt rightLt
    let typedMatrix : Fin shape.matrixCount :=
      ⟨matrix, by simpa [combineEvaluations] using rightLt⟩
    simp only [combineEvaluations, Array.getElem_ofFn]
    calc
      (evaluations system (combineAssignments challenges assignments) point)[matrix] =
          matrixEvaluation system
            (combineAssignments challenges assignments) point typedMatrix := by
        simp [evaluations, typedMatrix]
      _ = combineEvaluation challenges fun index =>
            matrixEvaluation system (assignments index) point typedMatrix :=
        matrixEvaluation_combine system challenges assignments point
          typedMatrix
      _ = combineEvaluation challenges fun index =>
            (evaluations system (assignments index) point).getD matrix
              BaseLinear.evaluationZero := by
        apply congrArg (combineEvaluation challenges)
        funext index
        exact (semantic_getD system (assignments index) point typedMatrix).symm

theorem relation_evaluations_hom
    {shape : Shape} {Commitment : Type}
    (commit : Assignment shape -> Commitment)
    {count : Nat} (system : Structure shape) (point : Point shape)
    (challenges : Fin count -> RingF)
    (assignments : Fin count -> Assignment shape) :
    (relationSemantics commit).evaluations system
        (combineAssignments challenges assignments) point =
      combineEvaluations (shape := shape) challenges fun index =>
          (relationSemantics commit).evaluations system
          (assignments index) point := by
  exact evaluations_hom system point challenges assignments

end NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiRLCFinite
