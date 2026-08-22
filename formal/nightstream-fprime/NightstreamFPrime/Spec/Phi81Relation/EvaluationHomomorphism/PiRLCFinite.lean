import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiRLC
import NightstreamFPrime.Spec.Phi81Relation.Semantics
import NightstreamFPrime.Spec.Folding.PiRLC

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Concrete/Phi81Relation/EvaluationHomomorphism/PiRLCFinite.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Finite-batch evaluation homomorphism for the typed Phi81 `Pi_RLC` action.

Protocol: SuperNeo Theorem 5, evaluation-homomorphism branch of `Pi_RLC`.
Phase: canonical finite challenge combination after the one-source action law.
Constraint family: semantic evaluation only; this file emits no rows.

Owns: raw width-only and typed head-first finite `RingF` combinations of
complete assignments; their exact equality; the matching head-first finite
combination of one `RingK` evaluation; the verifier-owned fixed-matrix array
operation; and the theorem with exactly the same system/point/challenge shape as
`Folding.PiRLC.Algebra.evaluations_hom`.

Does not own: the quotient-ring proof behind the per-challenge product-order
law, commitments, public-input projection, norm preservation, a complete
`PiRLC.Algebra`, transcript derivation, Rust/R1CS refinement, row removal, or
counts.

Emits constraints: no.

Authority boundary: all finite combinations are computed in canonical
head-first `Fin n` order. Output arrays have exactly the typed matrix count.
Default reads exist only to totalize the operation on arbitrary public arrays;
the theorem eliminates them because semantic arrays have the exact typed size.
Every challenge's product-order law is derived internally by
`PiRLC.productOrderLaw`; no algebra oracle or caller-supplied law remains.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.evaluation_hom.finite.assignment` | every complete-carrier coordinate uses the same canonical challenge fold | computed | `combineAssignments` |
| `nifs.pi_rlc.verify.evaluation_hom.finite.assignment.raw_refinement` | the independent width-only fold is exactly the typed relation fold | derived | `raw_combineAssignments_eq` |
| `nifs.pi_rlc.verify.evaluation_hom.finite.evaluation` | one 54-lane evaluation uses the identical challenge fold | computed | `combineEvaluation` |
| `nifs.pi_rlc.verify.evaluation_hom.finite.arrays` | output has exactly one combined evaluation per canonical matrix | computed | `combineEvaluations` |
| `nifs.pi_rlc.verify.evaluation_hom.finite.matrix` | one matrix evaluation commutes with the finite challenge fold | derived | `matrixEvaluation_combine` |
| `nifs.pi_rlc.verify.evaluation_hom.algebra` | the concrete theorem has the exact algebra-field signature | derived | `relation_evaluations_hom` |
-/

namespace NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism

namespace Raw

/-- Canonical head-first finite `RingF` combination of raw completed
assignments. Only logical width participates. -/
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

/-- Canonical head-first finite `RingF` combination at the typed relation
boundary. Keeping this owner typed prevents unrelated relation dimensions from
being inferred by downstream protocol proofs. -/
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

/-- The width-only assignment fold is exactly the typed relation fold. The
packed-output candidate may therefore reason independently over the raw
carrier without introducing a second ΠRLC transition. -/
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

/-- The identical head-first finite challenge combination on one `RingK`
evaluation. -/
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

/-- Combine arbitrary public evaluation arrays into the verifier-owned matrix
shape. Semantic arrays never exercise the default branch. -/
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

/-- One canonical matrix evaluation commutes with the finite `RingF`
combination. Each local product-order obligation is discharged by the exact
Phi81 quotient-ring proof. -/
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

/-- Array-level finite evaluation law: every canonical matrix and all 54
lanes use the same challenge sequence as the complete assignment. -/
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

/-- The same theorem stated through the independent relation semantics, with
exactly the field type expected by a future concrete
`Folding.PiRLC.Algebra`. -/
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
